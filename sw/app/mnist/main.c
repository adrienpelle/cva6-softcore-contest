#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "cpp_utils.h"
#include "env.h"
#include "Network.h"
#include "util.h"
#include "mac4.h"

// Standalone correctness check for the MAC4 coprocessor, independent of the
// real inference path (macsOnRange() is untouched by this ticket).
static int mac4_smoke_test(void) {
    const uint32_t packed_inputs  = 0x04030201u;  // lanes (uint8): 1, 2, 3, 4
    const uint32_t packed_weights = 0xF807FA05u;  // lanes (int8): 5, -6, 7, -8
    const int32_t expected = 1*5 + 2*(-6) + 3*7 + 4*(-8);  // -18

    const int32_t got = mac4(packed_inputs, packed_weights);
    const int pass = (got == expected);
    printf("MAC4 smoke test: %s (expected %d, got %d)\n", pass ? "PASS" : "FAIL", (int)expected, (int)got);
    return pass;
}

// Standalone correctness check for the MAC4 v2 coprocessor extensions
// (stationary buffer + T=8 accumulators), independent of the real inference
// path (macsOnRange()/NetworkPropagate.c are untouched by this ticket).
// Covers all 4 new instructions on a hand-computed example: two stationary
// words loaded at different mots, MAC-tuilé accumulating into two different
// slots (slot 0 across two mots, slot 1 once), lire-accumulateur reading
// both back, then réinitialiser-accumulateurs and re-reading to confirm the
// reset actually zeroed every slot.
static int mac4_v2_smoke_test(void) {
    const uint32_t word_a  = 0x04030201u;  // lanes (uint8): 1, 2, 3, 4
    const uint32_t word_b  = 0x08070605u;  // lanes (uint8): 5, 6, 7, 8
    const uint32_t weight1 = 0xF807FA05u;  // lanes (int8): 5, -6, 7, -8
    const uint32_t weight2 = 0x01010101u;  // lanes (int8): 1, 1, 1, 1

    const int32_t mac_a  = 1*5 + 2*(-6) + 3*7 + 4*(-8);  // -18
    const int32_t mac_b  = 5*5 + 6*(-6) + 7*7 + 8*(-8);  // -26
    const int32_t mac_c  = 1*1 + 2*1 + 3*1 + 4*1;        //  10
    const int32_t expected_slot0 = mac_a + mac_b;        // -44
    const int32_t expected_slot1 = mac_c;                //  10

    mac4_reset_acc();
    mac4_load_stationary(word_a, 0);
    mac4_load_stationary(word_b, 1);
    mac4_tiled(weight1, 0, 0);  // acc[0] += mac_a
    mac4_tiled(weight1, 1, 0);  // acc[0] += mac_b
    mac4_tiled(weight2, 0, 1);  // acc[1] += mac_c

    const int32_t got_slot0 = mac4_read_acc(0);
    const int32_t got_slot1 = mac4_read_acc(1);
    int pass = (got_slot0 == expected_slot0) && (got_slot1 == expected_slot1);

    mac4_reset_acc();
    const int32_t got_slot0_after_reset = mac4_read_acc(0);
    const int32_t got_slot1_after_reset = mac4_read_acc(1);
    pass = pass && (got_slot0_after_reset == 0) && (got_slot1_after_reset == 0);

    printf("MAC4 v2 smoke test: %s (expected acc[0]=%d acc[1]=%d, got acc[0]=%d acc[1]=%d, "
           "post-reset acc[0]=%d acc[1]=%d)\n",
           pass ? "PASS" : "FAIL",
           (int)expected_slot0, (int)expected_slot1,
           (int)got_slot0, (int)got_slot1,
           (int)got_slot0_after_reset, (int)got_slot1_after_reset);
    return pass;
}

// Standalone correctness check for the widened MAC4 v3 addressing (map #2,
// ticket "Coprocesseur MAC4 v3": stationary buffer 64->80 words, accumulator
// array 8->32 slots), independent of the real inference path. Deliberately
// exercises indices beyond the old ceilings -- mot=70 (only valid in the
// widened 80-word buffer) and slot=20 (only valid in the widened 32-slot
// accumulator array) -- to catch a decode width that silently stayed narrow
// despite the wider funct7/rd fields already being available at the ALU's
// ports (see core/mac4_copro/mac4_alu.sv).
static int mac4_v3_smoke_test(void) {
    const uint32_t word   = 0x0A090807u;  // lanes (uint8): 7, 8, 9, 10
    const uint32_t weight = 0x05FCFB03u;  // lanes (int8): 3, -5, -4, 5
    const int32_t expected = 7*3 + 8*(-5) + 9*(-4) + 10*5;  // 21-40-36+50 = -5

    mac4_reset_acc();
    mac4_load_stationary(word, 70);
    mac4_tiled(weight, 70, 20);

    const int32_t got = mac4_read_acc(20);
    int pass = (got == expected);

    mac4_reset_acc();
    const int32_t got_after_reset = mac4_read_acc(20);
    pass = pass && (got_after_reset == 0);

    printf("MAC4 v3 smoke test: %s (mot=70 slot=20, expected %d, got %d, post-reset got %d)\n",
           pass ? "PASS" : "FAIL", (int)expected, (int)got, (int)got_after_reset);
    return pass;
}

void readStimulus(
                  UDATA_T* inputBuffer,
                  Target_T* expectedOutputBuffer)
{
    envRead(ENV_SIZE_Y*ENV_SIZE_X*ENV_NB_OUTPUTS,
            ENV_SIZE_Y, ENV_SIZE_X,
            (DATA_T*) inputBuffer, //TODO
            OUTPUTS_SIZE[0], expectedOutputBuffer);
}

int processInput(        UDATA_T* inputBuffer,
                            Target_T* expectedOutputBuffer,
                            Target_T* predictedOutputBuffer,
			    UDATA_T* output_value)
{
    size_t nbPredictions = 0;
    size_t nbValidPredictions = 0;

    propagate(inputBuffer, predictedOutputBuffer, output_value);

    // assert(expectedOutputBuffer.size() == predictedOutputBuffer.size());
    for(size_t i = 0; i < OUTPUTS_SIZE[0]; i++) {
        if (expectedOutputBuffer[i] >= 0) {
            ++nbPredictions;

            if(predictedOutputBuffer[i] == expectedOutputBuffer[i]) {
                ++nbValidPredictions;
            }
        }
    }

    return (nbPredictions > 0)
        ? nbValidPredictions : 0;
}


int main(int argc, char* argv[]) {

    // const N2D2::Network network{};
    size_t instret, cycles;

#if ENV_DATA_UNSIGNED
    // aligned(4): conv1's accelerated path (convcellPropagate3) reads this
    // buffer directly via a single lw for even output columns -- an
    // unaligned base here would turn that into a genuinely misaligned
    // access, which hangs this core indefinitely (core/load_store_unit.sv).
    UDATA_T inputBuffer[ENV_SIZE_Y*ENV_SIZE_X*ENV_NB_OUTPUTS] __attribute__((aligned(4)));
#else
    std::vector<DATA_T> inputBuffer(network.inputSize());
#endif

    Target_T expectedOutputBuffer[OUTPUTS_SIZE[0]];
    Target_T predictedOutputBuffer[OUTPUTS_SIZE[0]];
    UDATA_T output_value;

    mac4_smoke_test();
    mac4_v2_smoke_test();
    mac4_v3_smoke_test();

    readStimulus(inputBuffer, expectedOutputBuffer);
    instret = -read_csr(minstret);
    cycles = -read_csr(mcycle);
    const int success = processInput(inputBuffer, 
                                                        expectedOutputBuffer, 
                                                        predictedOutputBuffer,
							&output_value);
    instret += read_csr(minstret);
    cycles += read_csr(mcycle);
    
    printf("Expected  = %d\n", expectedOutputBuffer[0]);
    printf("Predicted = %d\n", predictedOutputBuffer[0]);
    printf("Result : %d/1\n", success);
    printf("credence: %d\n", output_value);
    printf("image %s: %d instructions\n", stringify(MNIST_INPUT_IMAGE), (int)(instret));
    printf("image %s: %d cycles\n", stringify(MNIST_INPUT_IMAGE), (int)(cycles));

#ifdef OUTPUTFILE
    FILE *f = fopen("success_rate.txt", "w");
    if (f == NULL) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Could not create file:  success_rate.txt");
    }
    fprintf(f, "%f", successRate);
    fclose(f);
#endif
}
