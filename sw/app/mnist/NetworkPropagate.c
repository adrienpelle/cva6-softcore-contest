#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifdef __riscv
#include "encoding.h"
#endif

#include "env.h"
#include "mem_info.h"
#include "mac4.h"
#include "utils.h"

#include "conv1.h"
#include "conv2.h"
#include "fc1.h"
#include "fc2.h"


static DATA_T mem[MEMORY_SIZE] __attribute__((aligned(4)));

// conv1 alignment fix (see convcellPropagate3): the env image shifted left
// by 2 bytes, rebuilt once per propagate() call. Only indices [0,
// ENV_MEM_CONT_SIZE - 2) are ever populated/read -- conv1's kernel never
// reaches the last 2 source bytes through this copy (verified against
// CONV1_CHANNELS_WIDTH/CONV1_OUTPUTS_WIDTH/CONV1_KERNEL_WIDTH).
static UDATA_T conv1_input_shifted[ENV_MEM_CONT_SIZE] __attribute__((aligned(4)));

static int clamp(int v, int lo, int hi) {
    if(v < lo) {
        return lo;
    }
    else if(v > hi) {
        return hi;
    }
    else {
        return v;
    }
}

static UDATA_T saturate(SUM_T value, uint32_t sat) {
    return clamp(value, (SUM_T)(0), ((SUM_T)(1) << sat) - 1);
}

static UDATA_T sat(SUM_T weightedSum, int output,
                                           ActivationFunction_T func,
                                           /* const Rescaling_T& __restrict rescaling */
                                           int shift)
{
    switch(func) {
        case Linear:
        case Saturation: {
            break;
        }
        case Rectifier: {
            if(weightedSum <= 0) weightedSum = 0;
            break;
        }
        default:
            printf("Unsupported activation function.\n");
            break;
    }

    return saturate(weightedSum>>shift, NB_BITS);
}

// Accelerated conv1 path (ticket #28): same tile-outer / sy-middle /
// output-channel-inner MAC4 v2 order as convcellPropagate2 below, but with
// two conv1-specific corrections on top:
//
// 1. Alignment: conv1 reads directly from the raw env image (ENV_MEM_STRIDE
//    = 1, single channel), so a kernel row's 4 bytes (KERNEL_WIDTH *
//    CONV1_NB_CHANNELS) start at `ix = ox * CONV1_STRIDE_X`, which is
//    4-byte aligned only for even ox (ix % 4 == 0) -- odd ox lands on ix % 4
//    == 2. conv1_input_shifted holds the same image shifted left by 2
//    bytes, rebuilt once per propagate() call; reading it at (ix - 2)
//    recovers the identical 4 bytes through an address that IS 4-byte
//    aligned. So every (oy, ox) picks one of two already-aligned sources --
//    never a genuinely misaligned lw, and 100% of kernel-row reads take the
//    fast path (vs. ~50% before this ticket, when only even ox did).
// 2. Codegen: that choice of source/base offset is made once per (oy, ox)
//    below (input_base/ix_base), not re-tested per sy or per output the way
//    the original per-call runtime alignment check was.
//
// CONV1_NB_OUTPUTS (16) is an exact multiple of T=8 (2 tiles, no partial
// tile) and KERNEL_WIDTH * NB_CHANNELS is exactly 4 bytes (one MAC4 group,
// no segment-word loop needed) -- both guarded by the #if checks below so a
// future shape change fails to compile instead of silently corrupting
// output.
#if (CONV1_KERNEL_WIDTH * CONV1_NB_CHANNELS) != 4
#error "conv1 kernel row is no longer exactly 4 bytes (one MAC4 group): convcellPropagate3's single mac4_load_stationary/mac4_tiled call no longer covers the whole row"
#endif
#if (CONV1_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS) != 0
#error "CONV1_NB_OUTPUTS is no longer an exact multiple of MAC4_NUM_ACC_SLOTS: convcellPropagate3's last tile would read/write out of bounds"
#endif

#define CONV1_MAC_TILE_SLOT(SLOT, TILE_BASE)                                \
    {                                                                       \
        const int wOffset = wOffsetSyTerm + CONV1_NB_CHANNELS               \
            * CONV1_KERNEL_WIDTH * CONV1_KERNEL_HEIGHT * ((TILE_BASE) + (SLOT)); \
        uint32_t weight_word;                                              \
        memcpy(&weight_word, weights + wOffset, sizeof(weight_word));      \
        mac4_tiled(weight_word, 0, SLOT);                                  \
    }

static void convcellPropagate3(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    int rescaling)
{
    for (int oy = 0; oy < CONV1_OUTPUTS_HEIGHT; ++oy) {
        const int iy = oy * CONV1_STRIDE_Y;

        for (int ox = 0; ox < CONV1_OUTPUTS_WIDTH; ++ox) {
            const int ix = ox * CONV1_STRIDE_X;

            int oOffset = CONV1_MEM_STRIDE * (ox + CONV1_OUTPUTS_WIDTH * oy);
            if (oOffset >= CONV1_MEM_CONT_SIZE) {
                oOffset += CONV1_MEM_WRAP_OFFSET - CONV1_MEM_CONT_OFFSET
                            - CONV1_MEM_CONT_SIZE;
            }

            // Alignment test hoisted here: once per (oy, ox), not once per
            // sy/output the way the original generic per-call path did it.
            const UDATA_T* __restrict input_base;
            int ix_base;
            if (ox & 1) {
                input_base = conv1_input_shifted;
                ix_base = ix - 2;
            } else {
                input_base = inputs;
                ix_base = ix;
            }

            for (int tile_base = 0; tile_base < CONV1_NB_OUTPUTS;
                    tile_base += MAC4_NUM_ACC_SLOTS) {
                mac4_reset_acc();

                for (int sy = 0; sy < CONV1_KERNEL_HEIGHT; ++sy) {
                    const int iOffset = ix_base
                        + CONV1_CHANNELS_WIDTH * (iy + sy);

                    uint32_t input_word;
                    // __builtin_assume_aligned: this address is 4-byte
                    // aligned at runtime for every (oy, ox) by construction
                    // (input_base/ix_base above), but GCC can't prove it
                    // statically through a runtime-branched base pointer
                    // plus a runtime sy-offset -- without the hint it falls
                    // back to a 9-instruction byte-packing sequence instead
                    // of the single lw this produces (ticket #31 on map #29,
                    // docs/research/conv1-codegen-fix.md: verified via
                    // disassembly and RTL sim, conv1 80305 -> 64602 cycles).
                    memcpy(&input_word,
                           __builtin_assume_aligned(input_base + iOffset, 4),
                           sizeof(input_word));
                    mac4_load_stationary(input_word, 0);

                    const int wOffsetSyTerm
                        = CONV1_NB_CHANNELS * CONV1_KERNEL_WIDTH * sy;

                    EVAL(REPEAT(MAC4_NUM_ACC_SLOTS, CONV1_MAC_TILE_SLOT, tile_base))
                }

                _Pragma("GCC unroll 8")
                for (int slot = 0; slot < MAC4_NUM_ACC_SLOTS; ++slot) {
                    const int output = tile_base + slot;
                    SUM_T weightedSum = biasses[output] + mac4_read_acc(slot);
                    outputs[oOffset + output]
                        = sat(weightedSum, output, CONV1_ACTIVATION, rescaling);
                }
            }
        }
    }
}

#undef CONV1_MAC_TILE_SLOT

// Accelerated conv2 path: weight-spatial-batching (map #2, ticket "conv2 :
// tuilage spatial poids-stationnaire (B=4)") on top of MAC4 v2's
// tile-outer/sy-middle/output-channel-inner order (see mac4.h /
// core/mac4_copro for LOAD_STATIONARY, MAC_TILED, READ_ACC, RESET_ACC):
// for each tile of T=8 output channels, reset the accumulators, then for
// each kernel row (sy), load *all* MAC4_SPATIAL_BATCH=4 spatial positions'
// input segments into disjoint sub-ranges of the stationary buffer, then
// for each of the tile's 8 channels, load that channel's weight word once
// and MAC-tile it against all 4 positions (4 fewer redundant weight `lw`s
// per (tile, sy, channel, segment-word) than MAC4 v2's per-position
// reload), then read back and saturate every position's every channel only
// after the sy sweep completes -- no partial sum ever goes back to memory
// between sy's, batches, or tiles.
//
// CONV2_OUTPUTS_WIDTH == MAC4_SPATIAL_BATCH == 4 (guarded below), so one
// full output row *is* one spatial batch -- the existing `oy` loop already
// sweeps batches, one per row, with no separate batch index needed. Each
// position's segment/accumulator addressing is offset by its `ox` within
// the row: stationary mot = ox*CONV2_SEGMENT_WORDS+w (0..79, exactly fills
// the 80-word buffer across 4 positions x 20 words), accumulator slot =
// ox*MAC4_NUM_ACC_SLOTS+channel-slot (0..31, exactly the widened T=32).
//
// This hardcodes CONV2_* rather than taking them as parameters:
// mac4_load_stationary()/mac4_tiled() encode their "mot" word index as a
// compile-time immediate, so the segment-word loop must be a genuine
// integer constant expression, not something that merely happens to fold to
// a constant after inlining. Valid only where conv2's actual shape holds:
// contiguous input segment (NB_CHANNELS == INPUT_MEM_STRIDE, no
// padding/partial window -- true for conv2, verified against
// CONV2_PADDING_X/Y and CONV2_OUTPUTS_*_NOPAD), a segment length that's an
// exact multiple of 4 bytes (KERNEL_WIDTH*NB_CHANNELS = 80 = 20 words, no
// scalar remainder), NB_OUTPUTS an exact multiple of T=8 (24 = 3 tiles, no
// partial tile), and OUTPUTS_WIDTH an exact multiple of B=4 (4, no partial
// batch -- unlike conv1, which needs a ragged tail batch since its
// OUTPUTS_WIDTH doesn't divide evenly by 4). conv1, fc1 and fc2 each got
// their own dedicated treatment for the ways they don't meet this: conv1
// (convcellPropagate3) needs a shifted input copy for alignment (ticket
// #28) and, once batched, a ragged tail batch (ticket "conv1 : tuilage
// spatial poids-stationnaire (B=4, avec reliquat)"); fc1/fc2
// (fccellPropagateUDATA_T/DATA_T, ticket #27) have no spatial axis at all
// to batch across -- permanently out of scope for weight-spatial-batching
// (spec #34) -- and keep MAC4 v2's per-tile order unbatched, with fc1's
// last tile running fewer accumulator slots and fc2's last 2 elements per
// channel still added in scalar C, same as before this ticket.
#define CONV2_SEGMENT_WORDS ((CONV2_KERNEL_WIDTH * CONV2_NB_CHANNELS) / 4)

// The _Pragma("GCC unroll N") literals below must equal CONV2_SEGMENT_WORDS,
// MAC4_NUM_ACC_SLOTS and MAC4_SPATIAL_BATCH exactly; CONV2_NB_OUTPUTS must
// divide evenly by MAC4_NUM_ACC_SLOTS (no partial last tile); the hardcoded
// position offsets in CONV2_MAC_TILE_SLOT (0/8/16/24) assume
// MAC4_NUM_ACC_SLOTS==8; and CONV2_OUTPUTS_WIDTH must equal
// MAC4_SPATIAL_BATCH exactly, since convcellPropagate2 treats one output
// row as one batch with no batch-index bookkeeping of its own. None of this
// is re-derivable by the compiler across a _Pragma's string argument or a
// hand-unrolled macro, so catch drift here instead of silently
// under-unrolling (reintroducing the "impossible constraint in asm" class
// of bug) or writing outputs[]/biasses[]/the stationary buffer out of
// bounds.
#if CONV2_SEGMENT_WORDS != 20
#error "CONV2_SEGMENT_WORDS changed: update the _Pragma(\"GCC unroll 20\") literals in convcellPropagate2 to match"
#endif
#if (CONV2_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS) != 0
#error "CONV2_NB_OUTPUTS is no longer an exact multiple of MAC4_NUM_ACC_SLOTS: convcellPropagate2's last tile would read/write out of bounds"
#endif
#if MAC4_NUM_ACC_SLOTS != 8
#error "MAC4_NUM_ACC_SLOTS changed: update the hardcoded position accumulator offsets (0/8/16/24) in CONV2_MAC_TILE_SLOT to match"
#endif
#if MAC4_SPATIAL_BATCH != 4
#error "MAC4_SPATIAL_BATCH changed: convcellPropagate2 hardcodes exactly 4 unrolled positions per batch (one full CONV2_OUTPUTS_WIDTH row) -- update the loop structure and CONV2_MAC_TILE_SLOT to match"
#endif
#if CONV2_OUTPUTS_WIDTH != MAC4_SPATIAL_BATCH
#error "CONV2_OUTPUTS_WIDTH no longer equals MAC4_SPATIAL_BATCH: convcellPropagate2 treats one full output row as exactly one spatial batch -- this no longer holds, the batching loop needs restructuring to span row boundaries"
#endif
#if (MAC4_SPATIAL_BATCH * CONV2_SEGMENT_WORDS) > MAC4_STATIONARY_WORDS
#error "conv2's batch no longer fits the coprocessor's stationary buffer (MAC4_SPATIAL_BATCH * CONV2_SEGMENT_WORDS words needed vs MAC4_STATIONARY_WORDS available)"
#endif

// mac4_tiled()'s accumulator-slot argument is spliced into a literal
// register name ("x" #slot in mac4.h, since MAC_TILED has no true
// destination register) -- a genuine preprocessor-time stringizing, not a
// GCC "i" asm-operand constraint like mac4_load_stationary()/mac4_read_acc()
// use. `ox*MAC4_NUM_ACC_SLOTS+slot` as a runtime-computed C expression would
// stringize to garbage text, not a register name, so each position's
// accumulator offset (0/8/16/24) has to be added at the preprocessor level
// instead, via utils.h's ADD() (SLOT is always 0..7 here, so ADD_0..ADD_7
// suffice -- no extension of ADD() itself was needed, only the INC() chain
// it's built on, already widened to 32 in ticket "Coprocesseur MAC4 v3").
#define CONV2_MAC_TILE_SLOT(SLOT, TILE_BASE)                                 \
    {                                                                        \
        const int wOffset = wOffsetSyTerm + CONV2_NB_CHANNELS               \
            * CONV2_KERNEL_WIDTH * CONV2_KERNEL_HEIGHT * ((TILE_BASE) + (SLOT)); \
        _Pragma("GCC unroll 20")                                            \
        for (int w = 0; w < CONV2_SEGMENT_WORDS; ++w) {                     \
            uint32_t weight_word;                                           \
            memcpy(&weight_word, weights + wOffset + 4 * w,                 \
                   sizeof(weight_word));                                    \
            mac4_tiled(weight_word, 0 * CONV2_SEGMENT_WORDS + w, SLOT);      \
            mac4_tiled(weight_word, 1 * CONV2_SEGMENT_WORDS + w, ADD(SLOT, 8)); \
            mac4_tiled(weight_word, 2 * CONV2_SEGMENT_WORDS + w, ADD(SLOT, 16)); \
            mac4_tiled(weight_word, 3 * CONV2_SEGMENT_WORDS + w, ADD(SLOT, 24)); \
        }                                                                   \
    }

static void convcellPropagate2(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    int rescaling)
{
    for (int oy = 0; oy < CONV2_OUTPUTS_HEIGHT; ++oy) {
        const int iy = oy * CONV2_STRIDE_Y;

        for (int tile_base = 0; tile_base < CONV2_NB_OUTPUTS;
                tile_base += MAC4_NUM_ACC_SLOTS) {
            mac4_reset_acc();

            for (int sy = 0; sy < CONV2_KERNEL_HEIGHT; ++sy) {
                // Load this row's 4 positions' input segments for this sy
                // before touching any weight -- each position's 20 words
                // land in its own disjoint sub-range of the stationary
                // buffer (mot = ox*CONV2_SEGMENT_WORDS+w).
                _Pragma("GCC unroll 4")
                for (int ox = 0; ox < CONV2_OUTPUTS_WIDTH; ++ox) {
                    const int ix = ox * CONV2_STRIDE_X;
                    int iOffset = CONV1_MEM_STRIDE
                        * (ix + CONV2_CHANNELS_WIDTH * (iy + sy));

                    if (iOffset >= CONV1_MEM_CONT_SIZE) {
                        iOffset += CONV1_MEM_WRAP_OFFSET - CONV1_MEM_CONT_OFFSET
                                    - CONV1_MEM_CONT_SIZE;
                    }

                    _Pragma("GCC unroll 20")
                    for (int w = 0; w < CONV2_SEGMENT_WORDS; ++w) {
                        uint32_t input_word;
                        memcpy(&input_word, inputs + iOffset + 4 * w,
                               sizeof(input_word));
                        mac4_load_stationary(input_word,
                                              ox * CONV2_SEGMENT_WORDS + w);
                    }
                }

                const int wOffsetSyTerm
                    = CONV2_NB_CHANNELS * CONV2_KERNEL_WIDTH * sy;

                // Each weight word is loaded once here and reused across
                // all 4 positions' MAC_TILED calls inside the macro -- the
                // actual weight-spatial-batching win.
                EVAL(REPEAT(MAC4_NUM_ACC_SLOTS, CONV2_MAC_TILE_SLOT, tile_base))
            }

            // Finalize (read back, bias, saturate, store) all 4 positions'
            // 8 channels for this tile before the next tile_base's
            // mac4_reset_acc() -- no partial sum ever leaves the
            // coprocessor between batches or tiles.
            _Pragma("GCC unroll 4")
            for (int ox = 0; ox < CONV2_OUTPUTS_WIDTH; ++ox) {
                const int oOffset = CONV2_MEM_STRIDE * (ox + CONV2_OUTPUTS_WIDTH * oy);
                _Pragma("GCC unroll 8")
                for (int slot = 0; slot < MAC4_NUM_ACC_SLOTS; ++slot) {
                    const int output = tile_base + slot;
                    SUM_T weightedSum = biasses[output]
                        + mac4_read_acc(ox * MAC4_NUM_ACC_SLOTS + slot);
                    outputs[oOffset + output]
                        = sat(weightedSum, output, CONV2_ACTIVATION, rescaling);
                }
            }
        }
    }
}

#undef CONV2_MAC_TILE_SLOT
#undef CONV2_SEGMENT_WORDS

// Accelerated fc1 path (ticket #27): same tile-outer / sy-middle /
// output-channel-inner MAC4 v2 order as convcellPropagate2/3, applied to a
// fully-connected layer (no spatial oy/ox loop -- OUTPUTS_HEIGHT/WIDTH are
// both 1 -- so "sy" is fc1's own iy loop over CHANNELS_HEIGHT). Each iy
// segment is NB_CHANNELS * CHANNELS_WIDTH = 96 bytes = 24 words exactly (no
// scalar element remainder, unlike fc2 below).
//
// FC1_NB_OUTPUTS (150) is NOT an exact multiple of T=8 (150 = 18*8 + 6), so
// unlike conv1/conv2 this needs a genuine remainder tile: 18 full tiles run
// through a runtime tile_base loop (REPEAT(8, ...), exactly like conv2),
// then one more explicit tail tile covers the last 6 output channels with
// REPEAT(6, ...) -- same instructions, just fewer accumulator slots
// addressed, no separate scalar queue needed for this kind (see spec #24,
// "Reliquat de canaux de sortie").
#if (FC1_NB_CHANNELS * FC1_CHANNELS_WIDTH) % 4 != 0
#error "fc1 segment length is no longer a multiple of 4: fccellPropagateUDATA_T's word-packed load/mac would leave a scalar remainder, like fc2's"
#endif
#define FC1_SEGMENT_WORDS ((FC1_NB_CHANNELS * FC1_CHANNELS_WIDTH) / 4)
#if FC1_SEGMENT_WORDS != 24
#error "FC1_SEGMENT_WORDS changed: update the _Pragma(\"GCC unroll 24\") literals in fccellPropagateUDATA_T to match"
#endif
#define FC1_FULL_TILES (FC1_NB_OUTPUTS / MAC4_NUM_ACC_SLOTS)
// A literal, not `(FC1_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS)`: REPEAT()/DEC() token-
// paste this against a digit suffix (DEC_6, ...), which only works for a
// literal decimal token, not an unevaluated preprocessor expression.
#define FC1_TAIL_SLOTS 6
#if FC1_TAIL_SLOTS != (FC1_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS)
#error "FC1_TAIL_SLOTS changed: update the literal here (and the _Pragma(\"GCC unroll 6\") literal in fccellPropagateUDATA_T) to match FC1_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS"
#endif

#define FC1_MAC_TILE_SLOT(SLOT, TILE_BASE)                                  \
    {                                                                       \
        const int wOffset = wOffsetIyTerm + FC1_NB_CHANNELS               \
            * FC1_CHANNELS_WIDTH * FC1_CHANNELS_HEIGHT * ((TILE_BASE) + (SLOT)); \
        _Pragma("GCC unroll 24")                                           \
        for (int w = 0; w < FC1_SEGMENT_WORDS; ++w) {                      \
            uint32_t weight_word;                                          \
            memcpy(&weight_word, weights + wOffset + 4 * w,                \
                   sizeof(weight_word));                                   \
            mac4_tiled(weight_word, w, SLOT);                              \
        }                                                                  \
    }

// Shared between the full-tile loop and the tail tile below: loads this
// tile's 4 iy segments into the stationary buffer and MAC-tiles them
// against SLOT_COUNT output channels' weights. `tile_base` and `inputs`/
// `weights` come from the enclosing scope (a local `tile_base` is in scope
// at both call sites).
#define FC1_TILE_BODY(SLOT_COUNT)                                           \
    for (int iy = 0; iy < FC1_CHANNELS_HEIGHT; ++iy) {                      \
        const int iOffset = FC1_NB_CHANNELS * FC1_CHANNELS_WIDTH * iy;      \
        _Pragma("GCC unroll 24")                                           \
        for (int w = 0; w < FC1_SEGMENT_WORDS; ++w) {                      \
            uint32_t input_word;                                           \
            memcpy(&input_word, inputs + iOffset + 4 * w,                  \
                   sizeof(input_word));                                    \
            mac4_load_stationary(input_word, w);                          \
        }                                                                   \
        const int wOffsetIyTerm = FC1_NB_CHANNELS * FC1_CHANNELS_WIDTH * iy; \
        EVAL(REPEAT(SLOT_COUNT, FC1_MAC_TILE_SLOT, tile_base))             \
    }

static void fccellPropagateUDATA_T(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    const int rescaling)
{
    for (int tile_base = 0; tile_base < FC1_FULL_TILES * MAC4_NUM_ACC_SLOTS;
            tile_base += MAC4_NUM_ACC_SLOTS) {
        mac4_reset_acc();

        FC1_TILE_BODY(MAC4_NUM_ACC_SLOTS)

        _Pragma("GCC unroll 8")
        for (int slot = 0; slot < MAC4_NUM_ACC_SLOTS; ++slot) {
            const int output = tile_base + slot;
            SUM_T weightedSum = biasses[output] + mac4_read_acc(slot);
            outputs[output] = sat(weightedSum, output, FC1_ACTIVATION, rescaling);
        }
    }

    {
        const int tile_base = FC1_FULL_TILES * MAC4_NUM_ACC_SLOTS;
        mac4_reset_acc();

        FC1_TILE_BODY(FC1_TAIL_SLOTS)

        _Pragma("GCC unroll 6")
        for (int slot = 0; slot < FC1_TAIL_SLOTS; ++slot) {
            const int output = tile_base + slot;
            SUM_T weightedSum = biasses[output] + mac4_read_acc(slot);
            outputs[output] = sat(weightedSum, output, FC1_ACTIVATION, rescaling);
        }
    }
}

#undef FC1_TILE_BODY
#undef FC1_MAC_TILE_SLOT

// Accelerated fc2 path (ticket #27): same tile-outer/output-inner order as
// fc1 above, but fc2 has a single segment per output channel (CHANNELS_
// HEIGHT == CHANNELS_WIDTH == 1, no iy loop needed) of NB_CHANNELS = 150
// bytes -- NOT a multiple of 4 (150 = 37*4 + 2). The 37 full words go
// through the same word-packed load/mac path as conv2/fc1; the last 2
// elements can't form a word, so they keep the exact scalar C fallback
// macsOnRange() used to fall through to (FC2_SCALAR_REMAINDER below), added
// onto the hardware accumulator after mac4_read_acc() -- this is the
// "reliquat scalaire" the ticket calls out as staying unchanged.
#if FC2_CHANNELS_HEIGHT != 1 || FC2_CHANNELS_WIDTH != 1
#error "fc2 shape changed: fccellPropagateDATA_T hardcodes a single NB_CHANNELS-long segment (CHANNELS_HEIGHT == CHANNELS_WIDTH == 1)"
#endif
#define FC2_SEGMENT_WORDS (FC2_NB_CHANNELS / 4)
#define FC2_SEGMENT_REMAINDER (FC2_NB_CHANNELS % 4)
#if FC2_SEGMENT_WORDS != 37
#error "FC2_SEGMENT_WORDS changed: update the _Pragma(\"GCC unroll 37\") literals in fccellPropagateDATA_T to match"
#endif
#if FC2_SEGMENT_REMAINDER != 2
#error "fc2 segment remainder is no longer 2 elements: update FC2_SCALAR_REMAINDER in fccellPropagateDATA_T to match FC2_NB_CHANNELS % 4"
#endif
#define FC2_FULL_TILES (FC2_NB_OUTPUTS / MAC4_NUM_ACC_SLOTS)
// A literal, not `(FC2_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS)` -- see FC1_TAIL_SLOTS
// above for why REPEAT()/DEC() need a literal decimal token here.
#define FC2_TAIL_SLOTS 2
#if FC2_TAIL_SLOTS != (FC2_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS)
#error "FC2_TAIL_SLOTS changed: update the literal here (and the _Pragma(\"GCC unroll 2\") literal in fccellPropagateDATA_T) to match FC2_NB_OUTPUTS % MAC4_NUM_ACC_SLOTS"
#endif

#define FC2_SCALAR_REMAINDER(WOFFSET)                                       \
    (inputs[FC2_SEGMENT_WORDS * 4]                                         \
        * weights[(WOFFSET) + FC2_SEGMENT_WORDS * 4]                       \
     + inputs[FC2_SEGMENT_WORDS * 4 + 1]                                   \
        * weights[(WOFFSET) + FC2_SEGMENT_WORDS * 4 + 1])

#define FC2_MAC_TILE_SLOT(SLOT, TILE_BASE)                                  \
    {                                                                       \
        const int wOffset = FC2_NB_CHANNELS * ((TILE_BASE) + (SLOT));      \
        _Pragma("GCC unroll 37")                                           \
        for (int w = 0; w < FC2_SEGMENT_WORDS; ++w) {                      \
            uint32_t weight_word;                                          \
            memcpy(&weight_word, weights + wOffset + 4 * w,                \
                   sizeof(weight_word));                                   \
            mac4_tiled(weight_word, w, SLOT);                              \
        }                                                                   \
    }

// Loads the single 150-element segment into the stationary buffer and
// MAC-tiles it against SLOT_COUNT output channels' weights. `tile_base`
// comes from the enclosing scope, as in FC1_TILE_BODY above.
#define FC2_TILE_BODY(SLOT_COUNT)                                           \
    _Pragma("GCC unroll 37")                                               \
    for (int w = 0; w < FC2_SEGMENT_WORDS; ++w) {                          \
        uint32_t input_word;                                               \
        memcpy(&input_word, inputs + 4 * w, sizeof(input_word));           \
        mac4_load_stationary(input_word, w);                              \
    }                                                                       \
    EVAL(REPEAT(SLOT_COUNT, FC2_MAC_TILE_SLOT, tile_base))

static void fccellPropagateDATA_T(
    const UDATA_T* __restrict inputs,
    DATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    const int rescaling)
{
    for (int tile_base = 0; tile_base < FC2_FULL_TILES * MAC4_NUM_ACC_SLOTS;
            tile_base += MAC4_NUM_ACC_SLOTS) {
        mac4_reset_acc();

        FC2_TILE_BODY(MAC4_NUM_ACC_SLOTS)

        _Pragma("GCC unroll 8")
        for (int slot = 0; slot < MAC4_NUM_ACC_SLOTS; ++slot) {
            const int output = tile_base + slot;
            const int wOffset = FC2_NB_CHANNELS * output;
            SUM_T weightedSum = biasses[output] + mac4_read_acc(slot)
                + FC2_SCALAR_REMAINDER(wOffset);
            outputs[output] = sat(weightedSum, output, FC2_ACTIVATION, rescaling);
        }
    }

    {
        const int tile_base = FC2_FULL_TILES * MAC4_NUM_ACC_SLOTS;
        mac4_reset_acc();

        FC2_TILE_BODY(FC2_TAIL_SLOTS)

        _Pragma("GCC unroll 2")
        for (int slot = 0; slot < FC2_TAIL_SLOTS; ++slot) {
            const int output = tile_base + slot;
            const int wOffset = FC2_NB_CHANNELS * output;
            SUM_T weightedSum = biasses[output] + mac4_read_acc(slot)
                + FC2_SCALAR_REMAINDER(wOffset);
            outputs[output] = sat(weightedSum, output, FC2_ACTIVATION, rescaling);
        }
    }
}

#undef FC2_TILE_BODY
#undef FC2_MAC_TILE_SLOT
#undef FC2_SCALAR_REMAINDER

static void maxPropagate1(
    const DATA_T* __restrict inputs,
    int32_t* __restrict outputs,
    DATA_T* output_value,
    int NB_CHANNELS,
    int INPUTS_HEIGHT, int INPUTS_WIDTH,
    // Memory mapping: outputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE)
{
    int iMaxInput = 0;
    DATA_T maxInput = SCHAR_MIN;

    for (int iy = 0; iy < INPUTS_HEIGHT; ++iy) {
        for (int ix = 0; ix < INPUTS_WIDTH; ++ix) {
            const int oPos = (ix + INPUTS_WIDTH * iy);
            int iOffset = INPUT_MEM_STRIDE * oPos;

            if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                            - INPUT_MEM_CONT_SIZE;
            }

            if (NB_CHANNELS > 1) {
                for (int ch = 0; ch < NB_CHANNELS; ++ch) {
                    if (inputs[iOffset + ch] > maxInput) {
                        iMaxInput = ch;
                        maxInput = inputs[iOffset + ch];
                    }
                }

                outputs[oPos] = (int32_t)(iMaxInput);
		*output_value = maxInput;
            }
            else {
                outputs[oPos] = (inputs[iOffset] > 0);
		*output_value = inputs[iOffset];
            }
        }
    }
}

void propagate(const UDATA_T* inputs, Target_T* outputs, UDATA_T* maxPropagate_val)
{
#ifdef SAVE_OUTPUTS
    FILE* env_stream = fopen("env_output.txt", "w");
    saveOutputs(ENV_NB_OUTPUTS, ENV_SIZE_Y, ENV_SIZE_X, ENV_MEM_CONT_OFFSET, ENV_MEM_CONT_SIZE, ENV_MEM_WRAP_OFFSET, ENV_MEM_WRAP_SIZE, ENV_MEM_STRIDE, inputs, env_stream, Network::Format::CHW);
    fclose(env_stream);
#endif
    // conv1 alignment fix (ticket #28): see convcellPropagate3.
    memcpy(conv1_input_shifted, inputs + 2, ENV_MEM_CONT_SIZE - 2);

    // conv1
    UDATA_T* conv1_output = (UDATA_T*) mem + CONV1_MEM_CONT_OFFSET;

    const unsigned long start_conv1 = read_csr(mcycle);

    convcellPropagate3(inputs, conv1_output, conv1_biases, conv1_weights, 8);

    printf("conv1: %lu cycles\n", read_csr(mcycle) - start_conv1);

#ifdef SAVE_OUTPUTS
    FILE* conv1_stream = fopen("conv1_output.txt", "w");
    saveOutputs(CONV1_NB_OUTPUTS, CONV1_OUTPUTS_HEIGHT, CONV1_OUTPUTS_WIDTH, CONV1_MEM_CONT_OFFSET, CONV1_MEM_CONT_SIZE, CONV1_MEM_WRAP_OFFSET, CONV1_MEM_WRAP_SIZE, CONV1_MEM_STRIDE, conv1_output , conv1_stream, Network::Format::CHW);
    fclose(conv1_stream);
#endif




    // conv2
    UDATA_T* conv2_output = (UDATA_T*) mem + CONV2_MEM_CONT_OFFSET;

    const unsigned long start_conv2 = read_csr(mcycle);

    convcellPropagate2(conv1_output , conv2_output, conv2_biases, conv2_weights, 8);

    printf("conv2: %lu cycles\n", read_csr(mcycle) - start_conv2);

#ifdef SAVE_OUTPUTS
    FILE* conv2_stream = fopen("conv2_output.txt", "w");
    saveOutputs(CONV2_NB_OUTPUTS, CONV2_OUTPUTS_HEIGHT, CONV2_OUTPUTS_WIDTH, CONV2_MEM_CONT_OFFSET, CONV2_MEM_CONT_SIZE, CONV2_MEM_WRAP_OFFSET, CONV2_MEM_WRAP_SIZE, CONV2_MEM_STRIDE, conv2_output , conv2_stream, Network::Format::CHW);
    fclose(conv2_stream);
#endif




    // fc1
    UDATA_T* fc1_output = (UDATA_T*) mem + FC1_MEM_CONT_OFFSET;

    const unsigned long start_fc1 = read_csr(mcycle);

    fccellPropagateUDATA_T(conv2_output, fc1_output, fc1_biases, fc1_weights, 8);

    printf("fc1: %lu cycles\n", read_csr(mcycle) - start_fc1);

#ifdef SAVE_OUTPUTS
    FILE* fc1_stream = fopen("fc1_output.txt", "w");
    saveOutputs(FC1_NB_OUTPUTS, FC1_OUTPUTS_HEIGHT, FC1_OUTPUTS_WIDTH, FC1_MEM_CONT_OFFSET, FC1_MEM_CONT_SIZE, FC1_MEM_WRAP_OFFSET, FC1_MEM_WRAP_SIZE, FC1_MEM_STRIDE, fc1_output , fc1_stream, Network::Format::CHW);
    fclose(fc1_stream);
#endif




    // fc2
    DATA_T* fc2_output = (DATA_T*) mem + FC2_MEM_CONT_OFFSET;

    const unsigned long start_fc2 = read_csr(mcycle);

    fccellPropagateDATA_T(fc1_output, fc2_output, fc2_biases, fc2_weights, 11);

    printf("fc2: %lu cycles\n", read_csr(mcycle) - start_fc2);

#ifdef SAVE_OUTPUTS
    FILE* fc2_stream = fopen("fc2_output.txt", "w");
    saveOutputs(FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE, fc2_output , fc2_stream, Network::Format::CHW);
    fclose(fc2_stream);
#endif

    maxPropagate1(fc2_output, outputs, maxPropagate_val, FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE);

#ifdef SAVE_OUTPUTS
    FILE* max_stream = fopen("max_output.txt", "w");
    saveOutputs(FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE, outputs, max_stream, Network::Format::CHW);
    fclose(max_stream);
#endif

}

/*template<>
float Network::backpropagate(const DATA_T* input, const std::int32_t* labels){
   const float loss = 0.0f;
   return loss;
 }

int Network::gradientCheck(){
   return(0);
}*/


