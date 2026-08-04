#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "cpp_utils.h"
#include "env.h"
#include "Network.h"
#include "util.h"

// MAC4 custom instruction: rd = sum_{i=0..3}( uint8(rs1[8*i+:8]) * int8(rs2[8*i+:8]) )
// R-type encoding: opcode = custom-0 (0x0B), funct3 = 0, funct7 = 0.
static inline int32_t mac4(uint32_t packed_inputs, uint32_t packed_weights) {
    int32_t result;
    __asm__ volatile (
        ".insn r 0x0B, 0, 0, %0, %1, %2"
        : "=r"(result)
        : "r"(packed_inputs), "r"(packed_weights)
    );
    return result;
}

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
    UDATA_T inputBuffer[ENV_SIZE_Y*ENV_SIZE_X*ENV_NB_OUTPUTS];
#else
    std::vector<DATA_T> inputBuffer(network.inputSize());
#endif

    Target_T expectedOutputBuffer[OUTPUTS_SIZE[0]];
    Target_T predictedOutputBuffer[OUTPUTS_SIZE[0]];
    UDATA_T output_value;

    mac4_smoke_test();

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
