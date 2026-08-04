#ifndef N2D2_EXPORTC_MAC4_H
#define N2D2_EXPORTC_MAC4_H

#include <stdint.h>

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

// MAC4 v2: internal stationary input buffer + T=8 parallel 32-bit
// accumulators (see core/mac4_copro/include/mac4_instr_pkg.sv for the full
// bit-level spec). All 4 instructions below are register-only R-type under
// the same custom-0 opcode as mac4(), distinguished by funct3; `mot` (the
// stationary word index, 0..63) and `slot` (the accumulator index, 0..7)
// must be compile-time constants -- they are encoded directly into the
// instruction word (funct7 and/or rd), never carried in a GPR.
#define MAC4_NUM_ACC_SLOTS 8

// charger-stationnaire: stationary[mot] = word.
// funct3 = 1, funct7 = mot (immediate), rd unused, rs1 = word, rs2 unused.
static inline void mac4_load_stationary(uint32_t word, unsigned mot) {
    __asm__ volatile (
        ".insn r 0x0B, 1, %1, x0, %0, x0"
        :
        : "r"(word), "i"(mot)
    );
}

// MAC-tuilé: acc[slot] += sum_{i=0..3}( uint8(stationary[mot][8*i+:8]) * int8(weight[8*i+:8]) )
// funct3 = 2, funct7 = mot (immediate), rd = slot (encoded via register
// index -- MAC_TILED has no real destination, so `slot` must be a literal
// integer 0..7 so the preprocessor can select the x<slot> register name).
// rs1 = weight, rs2 unused.
#define mac4_tiled(weight, mot, slot)                                        \
    __asm__ volatile(".insn r 0x0B, 2, %1, x" #slot ", %0, x0"               \
                      :                                                      \
                      : "r"((uint32_t)(weight)), "i"((unsigned)(mot)))

// lire-accumulateur: returns acc[slot].
// funct3 = 3, funct7 = slot (immediate), rd = result, rs1/rs2 unused.
static inline int32_t mac4_read_acc(unsigned slot) {
    int32_t result;
    __asm__ volatile (
        ".insn r 0x0B, 3, %1, %0, x0, x0"
        : "=r"(result)
        : "i"(slot)
    );
    return result;
}

// réinitialiser-accumulateurs: acc[i] = 0 for all i in 0..MAC4_NUM_ACC_SLOTS-1.
// funct3 = 4, funct7/rd/rs1/rs2 unused.
static inline void mac4_reset_acc(void) {
    __asm__ volatile (".insn r 0x0B, 4, 0, x0, x0, x0");
}

#endif // N2D2_EXPORTC_MAC4_H
