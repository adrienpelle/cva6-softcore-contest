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
// MAC4_NUM_ACC_SLOTS below stays 8 -- and always will, not just until the
// weight-spatial-batching tickets land: it is the per-tile *channel* count
// (T=8's original meaning), used as-is by fc1/fc2 (permanently out of scope
// for batching per spec #34 -- no spatial axis to batch across) and, until
// each is individually restructured, by conv1/conv2 too. Their tiling math
// (tile_base += MAC4_NUM_ACC_SLOTS, REPEAT()-unrolled per-tile loops, and
// the #error guards checking NB_OUTPUTS % MAC4_NUM_ACC_SLOTS) all depend on
// this remaining 8; redefining it would silently break fc1/fc2 forever, not
// just conv1/conv2 temporarily. (An earlier version of this comment said a
// batching ticket should "raise this to 32" -- that was wrong, corrected
// while implementing "conv2 : tuilage spatial poids-stationnaire (B=4)".)
//
// The coprocessor's actual total capacity, and the batching factor conv2
// (and later conv1) multiply the channel-tile by, get their own constants
// instead -- see below.
#define MAC4_NUM_ACC_SLOTS 8

// Weight-spatial-batching (map #2 architecture decision, ticket "Décider :
// construire le tuilage spatial poids-stationnaire..."): a weight word,
// once loaded into a GPR, is reused across MAC_TILED calls to
// MAC4_SPATIAL_BATCH consecutive spatial output positions before being
// discarded, instead of one -- eliminating (B-1)/B of the redundant weight
// reloads convolutional weight sharing left on the table. Needs
// MAC4_NUM_ACC_SLOTS accumulator slots *per position in the batch* live at
// once (indexed position*MAC4_NUM_ACC_SLOTS+slot), which is exactly what
// the coprocessor was widened to T=32 for.
#define MAC4_SPATIAL_BATCH 4
#define MAC4_HW_ACC_SLOTS 32     // T in mac4_alu.sv -- the physical total.
#define MAC4_STATIONARY_WORDS 80 // StationaryWords in mac4_alu.sv.
#if (MAC4_NUM_ACC_SLOTS * MAC4_SPATIAL_BATCH) != MAC4_HW_ACC_SLOTS
#error "MAC4_NUM_ACC_SLOTS * MAC4_SPATIAL_BATCH no longer matches the coprocessor's actual T -- a batched layer's accumulator addressing (position*MAC4_NUM_ACC_SLOTS+slot) would exceed or underuse the real hardware range"
#endif

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
// index -- MAC_TILED has no real destination, so `slot` must reduce to a
// literal integer 0..31 so the preprocessor can select the x<slot> register
// name; RISC-V register names already run to x31, so this needed no macro
// change when T grew from 8 to 32, only the ALU's decode width).
// rs1 = weight, rs2 unused.
//
// MAC4_STRINGIZE expands its argument before stringizing it (the standard
// two-level `#x`-via-indirection idiom), unlike a bare `#slot` in
// mac4_tiled's own body, which C requires to stringize the *raw, unexpanded*
// argument text. That distinction only bites when `slot` is itself a macro
// call needing expansion -- e.g. weight-spatial-batching callers computing
// a per-position accumulator offset via utils.h's ADD(SLOT, N) -- a bare
// `#slot` would stringize "ADD(SLOT, N)" verbatim instead of its numeric
// result. Kept self-contained here rather than reusing utils.h's identical
// STR/STR_NX pair, since main.c calls mac4_tiled() too and doesn't include
// mnist/utils.h.
#define MAC4_STRINGIZE(x) MAC4_STRINGIZE_(x)
#define MAC4_STRINGIZE_(x) #x

#define mac4_tiled(weight, mot, slot)                                        \
    __asm__ volatile(".insn r 0x0B, 2, %1, x" MAC4_STRINGIZE(slot) ", %0, x0" \
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
