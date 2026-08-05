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

// MAC4 v2, hardware widened ("v3", ticket "Coprocesseur MAC4 v3", map #2):
// the stationary buffer and accumulator array grew to 80 words / T=32 slots
// -- see core/mac4_copro/include/mac4_instr_pkg.sv and mac4_alu.sv for the
// full bit-level spec -- so `mot` (0..79) and `slot` (0..31) are now valid
// compile-time-constant arguments to the instructions below (still encoded
// directly into the instruction word, funct7 and/or rd, never a GPR).
//
// MAC4_NUM_ACC_SLOTS below deliberately still reads 8, not 32: it is
// NetworkPropagate.c's *current* per-tile channel count (T=8's original
// meaning), not the coprocessor's new total capacity -- conv1/conv2/fc1/fc2
// aren't yet restructured for weight-spatial-batching (that's tickets
// "conv2 : tuilage spatial poids-stationnaire (B=4)" and "conv1 : ..."), so
// their unmodified tiling math (tile_base += MAC4_NUM_ACC_SLOTS, the
// REPEAT()-unrolled per-tile loops, and the #error guards checking
// NB_OUTPUTS % MAC4_NUM_ACC_SLOTS) all still assume -- correctly, for now --
// that a tile is 8 channels. Bumping this to 32 without also rewriting that
// tiling math would silently process nonexistent channels 8..31 of every
// tile. Whichever of those two tickets lands first should raise this to 32
// (T x B = channel-tile(8) x spatial-batch(4)) as part of introducing the
// batched loop structure that actually needs the extra slots -- not before.
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
// integer 0..31 so the preprocessor can select the x<slot> register name;
// RISC-V register names already run to x31, so this needed no macro change
// when T grew from 8 to 32, only the ALU's decode width).
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

// réinitialiser-accumulateurs: acc[i] = 0 for all i in 0..T-1 -- all 32
// hardware slots, unconditionally (RESET_ACC has no per-slot addressing),
// regardless of how many of them MAC4_NUM_ACC_SLOTS above says current
// software actually uses.
// funct3 = 4, funct7/rd/rs1/rs2 unused.
static inline void mac4_reset_acc(void) {
    __asm__ volatile (".insn r 0x0B, 4, 0, x0, x0, x0");
}

#endif // N2D2_EXPORTC_MAC4_H
