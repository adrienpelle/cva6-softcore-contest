# Weight-stationary spatial batching: a tradeoff model (not a build)

Research for GitHub issue #32 (map #2, spec #24), opened off the back of
issue #30's cycle profile (`docs/research/mac4-v2-cycle-profile.md`, branch
`research/mac4-v2-cycle-profile`) and issue #31's conv1 codegen fix
(`docs/research/conv1-codegen-fix.md`, branch `research/conv1-codegen-fix`,
merged as `2cb8107` on `cv32a6_contest_25_26`). This ticket is scoped to
**modeling** a tradeoff, not building it: no new RTL, no new C code, no
simulation. Everything below is derived from the current committed source
(`sw/app/mnist/NetworkPropagate.c`, `conv1.h`, `conv2.h`, `mac4.h`) and the
RTL that defines the hardware's actual limits
(`core/mac4_copro/include/mac4_instr_pkg.sv`, `core/mac4_copro/mac4_alu.sv`),
all read at commit `2cb8107` on branch `research/weight-spatial-batching-tradeoff`
(branched from `cv32a6_contest_25_26` at that exact commit; nothing pushed,
nothing touched on `cv32a6_contest_25_26` or GitHub).

## 0. The redundancy this ticket targets

`convcellPropagate2`'s weight offset (`NetworkPropagate.c:233-234`):

```c
const int wOffset = wOffsetSyTerm + CONV2_NB_CHANNELS
    * CONV2_KERNEL_WIDTH * CONV2_KERNEL_HEIGHT * ((TILE_BASE) + (SLOT));
```

and `convcellPropagate3`'s (conv1, `NetworkPropagate.c:100-101`):

```c
const int wOffset = wOffsetSyTerm + CONV1_NB_CHANNELS
    * CONV1_KERNEL_WIDTH * CONV1_KERNEL_HEIGHT * ((TILE_BASE) + (SLOT));
```

are both functions of `sy`/`tile_base`/`slot` only (`wOffsetSyTerm` itself is
`NB_CHANNELS * KERNEL_WIDTH * sy`, also independent of `oy`/`ox`) — **never**
of the spatial output position. Both `oy`/`ox` loops (`NetworkPropagate.c:251-254`
conv2, `:114-117` conv1) sit *outside* the `tile_base`/`sy` loops that
actually touch weight memory, so every one of conv2's 16 and conv1's 121
spatial positions reloads the exact same weight words from memory via `lw`
— the classic convolutional-weight-sharing redundancy, present because
weight sharing was never exploited on the *spatial* axis; MAC4 v2 (tickets
#25-28) only exploits it on the *tile* axis (reusing an input segment across
a tile's 8 output channels within one `(oy,ox,sy)`).

This is a different axis from the "70%+ input-reload redundancy" issue #30
already characterized (`docs/research/mac4-v2-cycle-profile.md` §6): that
finding is about the *same input segment* being reloaded once per tile
because `tile_base` sits outside `sy`. This ticket's finding is about the
*same weight word* being reloaded once per spatial position because `oy`/`ox`
sit outside `tile_base`/`sy`. The two are orthogonal and (as shown in §5)
interact when restructured.

fc1/fc2 have **no** spatial axis at all — `FC1_OUTPUTS_HEIGHT == FC1_OUTPUTS_WIDTH == 1`,
`FC2_CHANNELS_HEIGHT == FC2_CHANNELS_WIDTH == 1` (confirmed by the `#if`
guards at `NetworkPropagate.c:410-412` and the comment at `:300-305`) — a
fully-connected layer computes its single output position once; there is no
`(oy,ox)` loop to batch across. This entire redundancy axis, and everything
below, applies only to conv1 and conv2.

## 1. Baseline (B=1) numbers, from `docs/research/mac4-v2-cycle-profile.md`

| layer | spatial positions | tiles (NB_OUTPUTS/T) | sy trip count | segment words/position | dynamic weight `lw` = dynamic `MAC_TILED` |
|---|---|---|---|---|---|
| conv2 | `CONV2_OUTPUTS_HEIGHT * CONV2_OUTPUTS_WIDTH` = 4×4 = **16** | 24/8 = **3** | `CONV2_KERNEL_HEIGHT` = **5** | `CONV2_KERNEL_WIDTH*CONV2_NB_CHANNELS/4` = 5×16/4 = **20** | **38400** |
| conv1 | `CONV1_OUTPUTS_HEIGHT * CONV1_OUTPUTS_WIDTH` = 11×11 = **121** | 16/8 = **2** | `CONV1_KERNEL_HEIGHT` = **4** | `CONV1_KERNEL_WIDTH*CONV1_NB_CHANNELS/4` = 4×1/4 = **1** | **7744** |

(`CONV2_OUTPUTS_HEIGHT`/`WIDTH` confirmed = 4 each in `conv2.h:11-12` — the
ticket's own premise's "16 spatial positions" checks out. `CONV1_OUTPUTS_HEIGHT`/`WIDTH`
= 11 each, `conv1.h:11-12` — conv1 has **121**, not a power of two, which
matters a great deal below.)

Both counts satisfy `weight_loads = tiles × spatial_positions × sy × T × segment_words`
(conv2: 3×16×5×8×20=38400; conv1: 2×121×4×8×1=7744), which is exactly the
"redundant across spatial position" structure §0 describes: every one of the
`spatial_positions` factor's 16 (or 121) repetitions reloads the identical
`tiles × sy × T × segment_words` set of weight words.

Also from that doc (§6): today's *tile*-axis input-reload redundancy is
66.7% for conv2 (3 tiles → `1 - 1/3`) and 50.0% for conv1 (2 tiles →
`1 - 1/2`) — cited here because §5 below shows this figure is **unchanged**
by spatial batching.

## 2. Loop restructuring, worked through concretely

### 2.0 What "weight-stationary" can mean here

The CV-X-IF interface is register-only (established constraint, no `x_mem`):
`mac4_tiled(weight, mot, slot)` (`mac4.h:42-45`) takes the weight as a
**register operand** (`rs1`), reloaded fresh every call — there is no
hardware weight-side stationary buffer symmetric to the input-side one
(`mac4_load_stationary`). "Weight-stationary" here therefore does not mean
a new hardware buffer; it means a **software/register-level** reuse: load a
weight word into a GPR once, then issue several `mac4_tiled()` calls against
different `stationary[mot]` segments (i.e. different spatial positions)
*before* that register is discarded — eliminating the `lw` that today
precedes every one of those calls, while the `mac4_tiled` calls themselves
still happen (they are genuine per-position compute, not redundant — see
§2.3).

For that register reuse to work, the loop order must put weight-touching
code (`slot`, `w`) **outside** the position loop, and — critically — **all B
positions' input segments for the current `sy` must already be resident in
the stationary buffer** before that weight/slot/w sweep begins, because the
position loop is walked once per `(slot, w)` pair, not once per `sy`. This
is the mechanism behind the stationary-buffer-growth finding in §4.

### 2.1 B=2, conv2 — worked pseudocode

```c
for (batch = 0; batch < 16; batch += 2) {              // 8 batches
    int p0 = batch, p1 = batch + 1;                     // raster-order spatial indices

    for (tile_base = 0; tile_base < 24; tile_base += 8) {  // 3 tiles, unchanged
        mac4_reset_acc();          // must clear 16 live slots this round -- see 2.4

        for (sy = 0; sy < 5; ++sy) {
            // -- load BOTH positions' 20-word segments before touching any weight --
            for (int w = 0; w < 20; ++w)
                mac4_load_stationary(input_word(p0, sy, w), /*mot=*/ 0*20 + w);
            for (int w = 0; w < 20; ++w)
                mac4_load_stationary(input_word(p1, sy, w), /*mot=*/ 1*20 + w);

            // -- now sweep weights once per (slot,w); reuse the loaded register across p0,p1 --
            for (slot = 0; slot < 8; ++slot) {
                for (int w = 0; w < 20; ++w) {
                    uint32_t weight_word = weights[wOffset(tile_base, sy, slot, w)];  // 1 lw
                    mac4_tiled(weight_word, /*mot=*/0*20 + w, /*acc=*/0*8 + slot);    // p0
                    mac4_tiled(weight_word, /*mot=*/1*20 + w, /*acc=*/1*8 + slot);    // p1
                }
            }
        }

        // finalize both positions' 8 channels for this tile
        for (int pos = 0; pos < 2; ++pos)
            for (slot = 0; slot < 8; ++slot) {
                int output = tile_base + slot;
                SUM_T weightedSum = biasses[output] + mac4_read_acc(pos*8 + slot);
                outputs[oOffset(pos == 0 ? p0 : p1) + output] = sat(weightedSum, ...);
            }
    }
}
```

Per `(tile_base, sy)`: 8×20 = 160 `mac4_tiled` calls execute (same as today),
but only **160 `lw`** for weights (not 320 — the p0/p1 calls share one
loaded register). `mot` spans 0..39 (2×20 — fits comfortably under the
64-word stationary buffer, see §4). Accumulator addresses span 0..15 (2×8).

### 2.2 B=4, conv2 — worked pseudocode (same skeleton, quadrupled)

```c
for (batch = 0; batch < 16; batch += 4) {               // 4 batches
    for (tile_base = 0; tile_base < 24; tile_base += 8) {
        mac4_reset_acc();          // 32 live slots this round

        for (sy = 0; sy < 5; ++sy) {
            for (int p = 0; p < 4; ++p)
                for (int w = 0; w < 20; ++w)
                    mac4_load_stationary(input_word(batch+p, sy, w), p*20 + w);  // mot 0..79

            for (slot = 0; slot < 8; ++slot) {
                for (int w = 0; w < 20; ++w) {
                    uint32_t weight_word = weights[wOffset(tile_base, sy, slot, w)];  // 1 lw
                    for (int p = 0; p < 4; ++p)
                        mac4_tiled(weight_word, p*20 + w, p*8 + slot);                // 4 calls, 1 lw
                }
            }
        }

        for (int p = 0; p < 4; ++p)
            for (slot = 0; slot < 8; ++slot) finalize(batch+p, tile_base, slot);
    }
}
```

`mot` now spans **0..79** — this exceeds the 64-word stationary buffer that
exists in the RTL today (`StationaryWords = 64`, `mac4_alu.sv:62`; addressed
by `funct7_i[5:0]`, `mac4_alu.sv:86,89` — hard 6-bit/64-slot ceiling as
currently wired). B=4 conv2 batching is **not representable on the current
coprocessor** without a stationary-buffer change; see §4 for exactly how
much and what kind of change.

### 2.3 Why `MAC_TILED` count does *not* shrink with B

A point worth making explicit because it's easy to get wrong: today
`lw`:`MAC_TILED` is exactly 1:1 (38400:38400 for conv2, per
`mac4-v2-cycle-profile.md` §2(c)). Batching breaks that 1:1 ratio, but **only
on the `lw` side**. Total `MAC_TILED` executions =
`tiles × spatial_positions × sy × T × segment_words`, which counts genuine
multiply-accumulate work — every spatial position still needs its own
convolution sum, so it still needs its own `MAC_TILED` call per weight word,
regardless of B. That product is **38400 for conv2 at every B** (16
positions × 3 tiles × 5 sy × 8 slot × 20 w, independent of how positions are
grouped into batches). Only the `lw` feeding those calls drops, because B of
them now share one register load. This is why §5's savings are correctly
described as "redundant *load* elimination," not "redundant *compute*
elimination" — there is no compute redundancy to remove here; conv2's
`MAC_TILED` was never redundant on any axis (issue #30 already established
this for the tile axis, and it holds equally for the spatial axis).

### 2.4 conv1's B=2 case — same accumulator story, very different buffer story

```c
for (batch = 0; batch < 121; batch += 2) {              // 60 full + 1 remainder of 1 (121 is odd)
    int B_eff = min(2, 121 - batch);
    for (tile_base = 0; tile_base < 16; tile_base += 8) {   // 2 tiles
        mac4_reset_acc();                                    // 2*8=16 live slots
        for (sy = 0; sy < 4; ++sy) {
            for (int p = 0; p < B_eff; ++p)
                mac4_load_stationary(input_word(batch+p, sy), p);   // mot 0..1 -- 1 word/position!
            for (slot = 0; slot < 8; ++slot) {
                uint32_t weight_word = weights[wOffset(tile_base, sy, slot)];  // 1 lw
                for (int p = 0; p < B_eff; ++p)
                    mac4_tiled(weight_word, p, p*8 + slot);
            }
        }
        for (int p = 0; p < B_eff; ++p)
            for (slot = 0; slot < 8; ++slot) finalize(batch+p, tile_base, slot);
    }
}
```

Same accumulator requirement (`T × B` = 16) as conv2's B=2 case — this is
driven purely by the fixed `T=8` hardware and is layer-agnostic. But the
stationary buffer only needs **`B` words** (conv1's segment is 1 word, not
20) — negligible at any candidate B. conv1's story really does differ from
conv2's, but on the *buffer* axis, not the *accumulator* axis (§4, §6).

conv1 also surfaces a structural wrinkle conv2 doesn't have: **121 is odd**
and shares no factor with any power-of-two B, so every even batch size
leaves a ragged remainder batch (`ceil(121/B)` batches, not `121/B`) — see
§5's conv1 table for the resulting (small) loss versus the naive `/B`
estimate. (A natural alternative batch size not in the ticket's requested
set would be `B=11`, one full output row — `121 = 11×11` divides evenly —
but that reintroduces conv1's `ox`-parity alignment complication
(`NetworkPropagate.c:126-136`, ticket #28) inside a single batch, since even
and odd `ox` read from two different base arrays; out of scope to resolve
here, just flagged as the real complication a B=11 design would need to
handle that B=2/4/8/16 (which straddle row boundaries anyway once B>1) also
already have in smaller doses.)

## 3. Accumulator requirement: why `T × B`, not `(NB_OUTPUTS/T) × T × B`

This is the ticket's central question, worked through explicitly:

**Claim: accumulator slots required = `T × B` = `8B`, and this does NOT grow
with the number of tiles (`NB_OUTPUTS/T`).**

Reasoning: within one `(batch, tile_base)` iteration, the `sy` loop
accumulates partial sums for that tile's 8 channels × B positions = 8B
distinct accumulator slots. Because `RESET_ACC` clears *all* T accumulators
globally with no per-slot reset (`mac4_alu.sv:170-171`, `for i<T`), those 8B
slots must stay live and untouched from the batch's first `RESET_ACC`
through its last `READ_ACC` — i.e. for the duration of one tile's `sy`
sweep. But — and this is the crux — **each tile is fully resolved (read,
biased, saturated, stored) before the next `tile_base` begins**, exactly as
`convcellPropagate2` does today (`NetworkPropagate.c:258-292`: `RESET_ACC` →
`sy` sweep → `READ_ACC`+epilogue, then loop back to a fresh `RESET_ACC` for
the next `tile_base`). Nothing in the batching restructuring changes that:
§2.1/§2.2's pseudocode finalizes all B positions' 8 channels for `tile_base`
immediately after its `sy` loop, then starts the next `tile_base` with a
fresh `mac4_reset_acc()`. So accumulator lifetime never needs to span more
than one tile's worth of channels (8) times one batch's worth of positions
(B) — **not** all `NB_OUTPUTS/T` tiles at once. The alternative — holding
accumulators live across the *entire* `tile_base` sweep too, so that input
segments could *also* be reused across tiles (not just across positions) —
was considered and rejected below (§3.1) precisely because it inflates the
requirement to `NB_OUTPUTS × B`, a much worse trade.

Confirmed for B=2 (16 slots) and B=4 (32 slots) in §2.1/§2.2's pseudocode
(`mac4_reset_acc()`/finalize appear once per `(batch, tile_base)`, never
spanning multiple tiles); the pattern generalizes directly: **`accumulator
slots = 8 × B`** for every B, for both conv1 and conv2 (layer-independent,
since it only depends on `T` and `B`).

### 3.1 Rejected alternative: also reusing input across tiles (`NB_OUTPUTS × B`)

A tempting extension: if the accumulators for *all* `NB_OUTPUTS/T` tiles
(not just one) stayed live simultaneously across the whole `sy` sweep, the
loop could reorder to `sy` outermost and `tile_base` innermost *within* a
fixed batch — letting the same B positions' input segments serve every tile
too, not just every slot within one tile. This would need
`NB_OUTPUTS × B` accumulator slots (conv2: 24×B; conv1: 16×B) instead of
`T × B` (conv2/conv1 both: 8×B) — a further `NB_OUTPUTS/T` = 3x (conv2) or
2x (conv1) multiplier on top of everything in §4/§6's tables. Given that
even the modest `T×B` requirement already exceeds current hardware at B=2
(§4), the `NB_OUTPUTS×B` variant is strictly worse on the axis that's
already the binding constraint, for no additional weight-load benefit (§2.3
already shows the `lw`-elimination mechanism only needs positions to share a
weight register, not tiles) — it's flagged here as considered and rejected,
not part of the primary model below.

### 3.2 An alternative that avoids accumulator growth entirely — at a cost

A separate, real alternative *does* avoid growing T: keep the **total**
accumulator budget fixed at 8 and split it as `(T/B) channels × B
positions` instead of `T channels × 1 position` — i.e. shrink the per-tile
channel width as B grows, rather than widening the accumulator array. This
requires zero accumulator hardware change. But `tiles_total` then becomes
`NB_OUTPUTS × B / T` (B× more, smaller, tiles), and because `tile_base`
still sits outside `sy`, each of those B× more tiles independently redoes
the *entire* B-position segment load for its own `sy` sweep — the
`LOAD_STATIONARY` count, which §5 shows is otherwise B-invariant, now scales
by **B** instead. For conv2: weight-load+`MAC_TILED` drops by the same
38400×(1−1/B) as the primary model (§2.3's mechanism is unaffected by which
accumulator-splitting choice is made), but stationary-load traffic (14400
baseline, per `mac4-v2-cycle-profile.md` §2(a)) grows to roughly 14400×B.
Net effect on conv2's total dynamic instructions: B=2 → −38400×(1/2) +
14400×(2−1) = −19200+14400 = **−4800 net worse than the primary model's
−19200**, but still better than baseline; B=4 → −38400×(3/4)+14400×3 =
−28800+43200 = **+14400, net WORSE than baseline** — the crossover into
negative returns happens between B=2 and B=4 for conv2 under this variant.
This is a strictly worse curve than the primary (`T×B`-accumulator) model
for every B>1 on total instructions, but it's the only version buildable
with **zero** accumulator-array RTL change — worth naming as the "no
accumulator growth" point on the tradeoff surface even though the rest of
this document uses the primary model.

## 4. Stationary-buffer sizing and the input-reload-redundancy question

**Does input reload get better, worse, or stay the same?** Per §2.0's
mechanism: the position loop is walked once per `(slot, w)` pair (not once
per `sy`), so every position's segment must already be loaded before that
sweep starts — but each segment is still loaded **exactly once per
`(position, sy, tile_base)`**, same as today. Total `LOAD_STATIONARY`
dynamic count is therefore **unchanged by B**: `tiles × spatial_positions ×
sy × segment_words` = 3×16×5×20=4800 for conv2, 2×121×4×1=968 for conv1, at
every B (verified directly against §2.1/§2.2's pseudocode: the load loop
runs once per position per `sy`, regardless of how positions are grouped
into batches). The *tile*-axis redundancy fraction from issue #30 (66.7%
conv2, 50.0% conv1) is likewise unchanged — `tile_base` is still outside
`sy`, so a fixed batch's segments still get reloaded once per tile,
independent of B. **Neither better nor worse; the input-reload picture is
exactly what §0 characterized before this ticket.**

What *does* change: **residency**. Because all B positions' segments must
be simultaneously resident (not just loaded once each in sequence — see
§2.0), the stationary buffer's *peak occupancy* grows from `segment_words`
to `B × segment_words`:

| B | conv2 buffer need (20×B) | fits in 64-word HW buffer? | conv1 buffer need (1×B) | fits? |
|---|---|---|---|---|
| 1 | 20 | yes (current) | 1 | yes (current) |
| 2 | 40 | yes | 2 | yes |
| 4 | **80** | **no** — exceeds 64 | 4 | yes |
| 8 | 160 | no | 8 | yes |
| 16 | 320 | no | 16 | yes |

(3 is the largest B that fits conv2 under the current 64-word buffer: 3×20=60.)

conv2's 20-word segment makes it the buffer-bound layer; conv1's 1-word
segment means its buffer requirement is trivial at every candidate B — a
genuinely different story per layer, as the ticket predicted.

### 4.1 What it would take to widen the buffer/addressing, precisely

Read directly from `core/mac4_copro/mac4_alu.sv` and
`core/mac4_copro/include/mac4_instr_pkg.sv` (not hypothesized):

- **`mot` (stationary word index)**: `assign mot_idx = funct7_i[5:0];`
  (`mac4_alu.sv:89`) — 6 of `funct7`'s 7 bits are wired, giving the current
  64-word ceiling. `funct7[6]` is unused (the decode table treats all of
  `funct7` as "don't care" for LOAD_STATIONARY/MAC_TILED,
  `mac4_instr_pkg.sv:74-84`), so widening to `funct7_i[6:0]` (128 words) is
  a pure ALU change — no new instruction encoding, no assembler/macro
  change (`mac4_load_stationary()`'s `"i"(mot)` operand already carries the
  full 7-bit field). That covers conv2 up to B=6 (120≤127) and conv1 up to
  B=127 — but **not** conv2's B=8 (160) or B=16 (320), which need the
  physical `stationary_q[]` array *and* the addressing field genuinely
  widened beyond 7 bits (a real encoding change, since `funct7` is
  RISC-V-format-limited to 7 bits under the current R-type scheme).
- **accumulator `slot`**: MAC_TILED's slot comes from `rd_i[2:0]`
  (`mac4_alu.sv:90`, only 3 of the 5 bits RISC-V's `rd` field provides are
  wired); READ_ACC's comes from `funct7_i[2:0]` (`mac4_alu.sv:91`, only 3 of
  funct7's 7 bits). `mac4_tiled()`'s macro already emits any literal
  0-31 as a register name (`"x" #slot`, `mac4.h:43`) with **no encoding
  change needed** to go to `rd_i[4:0]` (32 slots) — this exactly covers
  B=4's `T×B`=32 requirement (§3) at zero encoding cost, only an ALU
  widening + a bigger `acc_q[T]`/`acc_n[T]` array. READ_ACC similarly has 4
  more funct7 bits already available (`funct7_i[5:0]` reaches 64 with zero
  encoding change). **B=8 (64 slots) and B=16 (128 slots) exceed
  MAC_TILED's 5-bit `rd`-field ceiling of 31 outright** — register names
  only run to `x31` in the assembler, so this needs a genuine new encoding
  (e.g. stealing bits from `funct7`, legal since MAC_TILED's decode table
  entry already treats `funct7` as don't-care, but it's a real ISA/toolchain
  change, not just wider RTL wires).
- **`RESET_ACC`** needs no addressing change at any B — it already loops
  `for (int unsigned i = 0; i < T; i++)` (`mac4_alu.sv:171`), so it scales
  automatically with whatever `T` becomes; only its per-call gate cost
  (more flip-flops cleared per cycle) grows.

Net: **B=2 and B=4** sit in a "cheap to widen" zone for the accumulator
*encoding* (ALU-only, no ISA change) but B=4 already needs the buffer
*depth* widened past 64 for conv2. **B=8 and B=16** need a genuine
instruction-encoding change for the accumulator slot field, on top of a much
larger stationary buffer, regardless of layer.

## 5. Weight-load reduction — full table

Formula (verified against §2.1/§2.2/§2.4's pseudocode, not assumed):
`weight_loads(B) = tiles × ceil(spatial_positions / B) × sy × T × segment_words`
— i.e. exactly the baseline formula with `spatial_positions` replaced by
`ceil(spatial_positions / B)` batches. For conv2 (16 positions, divides
evenly by every candidate B) this is precisely `38400 / B`. For conv1 (121
positions, not divisible by any candidate B) the `ceil()` matters and the
reduction falls slightly short of a clean `/B`.

### conv2 (16 positions, divides evenly)

| B | batches | weight `lw` (dynamic) | reduction vs. B=1 | % of conv2's 96072-instr total |
|---|---|---|---|---|
| 1 (baseline) | 16 | 38400 | — | 40.0% |
| 2 | 8 | 19200 | −19200 (−50.0%) | 20.0% |
| 4 | 4 | 9600 | −28800 (−75.0%) | 10.0% |
| 8 | 2 | 4800 | −33600 (−87.5%) | 5.0% |
| 16 | 1 | 2400 | −36000 (−93.75%) | 2.5% |

(B=16's 2400 matches the theoretical floor exactly:
`CONV2_WEIGHTS_SIZE/4` = 24×5×5×16/4 = **2400** — every distinct weight word
loaded exactly once across the whole layer, confirming the model.)

### conv1 (121 positions, does not divide evenly by any candidate B)

| B | batches (`ceil(121/B)`) | weight `lw` (dynamic) | naive `/B` estimate | overhead from ragged batching |
|---|---|---|---|---|
| 1 (baseline) | 121 | 7744 | 7744 | — |
| 2 | 61 | 3904 | 3872 | +32 (+0.8%) |
| 4 | 31 | 1984 | 1936 | +48 (+2.5%) |
| 8 | 16 | 1024 | 968 | +56 (+5.8%) |
| 16 | 8 | 512 | 484 | +28 (+5.8%) |

(Floor at B=121, one giant batch: `CONV1_WEIGHTS_SIZE/4` = 16×4×4×1/4 =
**64** — matches exactly, confirming the model at the extreme.)

Because `MAC_TILED` count is B-invariant (§2.3), these `lw` reductions are
also the *total* dynamic-instruction reductions for the weight-touching part
of each layer — nothing else in the weight-load-then-MAC_TILED pair changes.

## 6. Consolidated tradeoff table

Primary model (§2-§4; "no accumulator growth" alternative is §3.2, not
repeated here). All accumulator/buffer figures are **hardware requirements**,
not something built by this ticket.

### conv2

| B | accumulator slots (`8B`) | fits current T=8 HW? | stationary buffer words (`20B`) | fits current 64-word HW? | weight-`lw` dynamic count | reduction | input-reload redundancy fraction |
|---|---|---|---|---|---|---|---|
| 1 | 8 | yes (current) | 20 | yes | 38400 | baseline | 66.7% (tile axis, unchanged by B) |
| 2 | 16 | no — 2x, but within MAC_TILED's 32-slot (`x0`-`x31`) encoding ceiling (ALU-only widen) | 40 | yes | 19200 | −50.0% | 66.7% (unchanged) |
| 4 | 32 | no — 4x, exactly at MAC_TILED's 32-slot (`x0`-`x31`) encoding ceiling (ALU-only widen) | 80 | **no** — needs buffer depth + `mot`-field widened past 64 | 9600 | −75.0% | 66.7% (unchanged) |
| 8 | 64 | no — exceeds MAC_TILED's 5-bit `rd` field (32 values, `x0`-`x31`, max); needs new encoding | 160 | no — needs buffer depth + `mot`-field widened well past 128 | 4800 | −87.5% | 66.7% (unchanged) |
| 16 | 128 | no — exceeds MAC_TILED's 5-bit `rd` field; needs new encoding | 320 | no — same, larger | 2400 (= theoretical floor) | −93.75% | 66.7% (unchanged) |

### conv1

| B | accumulator slots (`8B`) | fits current T=8 HW? | stationary buffer words (`1B`) | fits current 64-word HW? | weight-`lw` dynamic count | reduction vs. baseline | input-reload redundancy fraction |
|---|---|---|---|---|---|---|---|
| 1 | 8 | yes (current) | 1 | yes | 7744 | baseline | 50.0% (tile axis, unchanged by B) |
| 2 | 16 | no — same 2x ceiling story as conv2 (32-slot `x0`-`x31` limit) | 2 | yes | 3904 | −49.6% | 50.0% (unchanged) |
| 4 | 32 | no — same 4x ceiling story as conv2 (32-slot `x0`-`x31` limit) | 4 | yes | 1984 | −74.4% | 50.0% (unchanged) |
| 8 | 64 | no — same new-encoding story as conv2 | 8 | yes | 1024 | −86.8% | 50.0% (unchanged) |
| 16 | 128 | no — same new-encoding story as conv2 | 16 | yes | 512 | −93.4% | 50.0% (unchanged) |

The two layers share an identical accumulator story (driven purely by fixed
`T=8` hardware, layer-independent) but diverge sharply on the buffer axis:
conv2's 20-word segment makes it buffer-bound starting at B=4; conv1's
1-word segment never comes close to the 64-word ceiling at any candidate B
— conv1's only real friction is the 121-position ragged-batch overhead
(§5), not buffer capacity.

## 7. What this does not include

Per scope, this is not evaluated: fc1/fc2 (§0, no spatial axis to batch),
whether to actually build any of this, which B (if any) is "worth it", or
what the loop-control/epilogue instruction-count delta would be beyond the
`MAC_TILED`/`lw`/`LOAD_STATIONARY` buckets already modeled (a full
static-disassembly-style accounting the way `mac4-v2-cycle-profile.md` did
for the *existing* code would require actually building a candidate — out
of scope here by the ticket's own "decisions, not deliverables" framing).
The one exception made is §3.2's "no accumulator growth" variant, included
because it's a real, zero-hardware-cost point on the same tradeoff surface
that a reader evaluating "is this worth building" would otherwise be missing
context for — not a recommendation for it over the primary model.

## 8. Branch / GitHub state

All of the above lives at `docs/research/weight-spatial-batching-tradeoff.md`
on branch `research/weight-spatial-batching-tradeoff`, branched from
`cv32a6_contest_25_26` at commit `2cb8107` (tip of that branch at the time
of this research — includes tickets #25-28's MAC4 v2 work and #31's conv1
codegen fix). Nothing was pushed to any remote, `cv32a6_contest_25_26` was
not modified, and no GitHub comment/issue/PR action was taken — this
document is the deliverable, for a human to relay.
