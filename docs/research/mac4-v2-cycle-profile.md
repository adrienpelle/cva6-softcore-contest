# MAC4 v2 static cycle profile (post-#25/#26/#27/#28)

Research for GitHub issue #30 (part of map #2, spec #24). Question: now that
all four layers of `propagate()` use the MAC4 v2 tile-outer/sy-middle/
output-channel-inner order (tickets #25-28, all closed, commit `14fc348`),
where do cycles go, and does that change the map's fog candidates
("go past T=8 accumulator slots" vs "unroll the remaining runtime loops")?

**Method**: same static-disassembly approach as issue #18's post-MAC4-v1
profile (see its resolution comment for the precedent), not empirical
`mcycle` instrumentation — issue #18 already found that instrumenting inside
a hot inlined path perturbs GCC's inlining/unrolling decisions and inflates
measured cycles 2.6x-4.6x, making the measurement useless. All addresses and
counts below are read directly out of `sw/app/mnist.riscv`, built from the
current committed source on branch `cv32a6_contest_25_26` (no source files
modified) via `sg docker -c "docker run --rm -v $(pwd):/workspace -w
/workspace sw-docker:vfft bash -c 'make benchmark APP=mnist'"`, then
disassembled with `riscv64-unknown-elf-objdump -d` (both with and without
`--no-show-raw-insn`, the raw-hex form needed to identify the custom `.insn`
opcodes, which objdump prints as `.insn 4, 0x...` rather than a mnemonic).
Real measured cycle totals (conv1 80305, conv2 197614, fc1 90480, fc2 3326;
`propagate()` overall 537576) are the ones already recorded in the map/ticket
history from RTL sim, used here only as a sanity cross-check, not
re-measured.

## 0. Everything is one inlined function; the loop structure survives, unrolling doesn't (fully)

```
$ riscv64-unknown-elf-objdump -t sw/app/mnist.riscv | grep -iE 'propagate|convcell|fccell'
80001700 g     F .text	00003218 .hidden propagate
```

`convcellPropagate2`/`convcellPropagate3`/`fccellPropagateUDATA_T`/
`fccellPropagateDATA_T` do **not** appear as separate symbols — all four are
folded into a single 0x3218-byte (3206-instruction) `propagate()`, confirming
the `static` + single-call-site + `-O3 -funroll-all-loops` inlining
prediction.

What's *not* fully true of the v1-era profile (issue #18): back then,
`macsOnRange()`'s hot loop was itself flattened into straight-line unrolled
code with no backward branches. In v2, `-funroll-all-loops` unrolls the
*compile-time-constant-trip-count inner* loops (the segment-word loop
`_Pragma("GCC unroll N")` and the 8-slot/6-slot/2-slot epilogue loops), but
the **outer runtime loops — `sy`/`iy` and `tile_base`, plus conv1/conv2's
spatial `oy`/`ox` — remain genuine backward-branch loops** in the binary,
confirmed by scanning every branch/jump in `propagate()` for a target address
lower than its own address:

```
80001850 bne s8,a4,800017e0   <- conv1 sy loop   (CONV1_KERNEL_HEIGHT=4)
800019a4 bne t5,t6,800017d4   <- conv1 tile_base loop (16/8=2 tiles)
800019b4 bne a7,t2,80001798   <- conv1 ox loop (11)
800019c0 bne t0,a1,8000178c   <- conv1 oy loop (11)
800020d8 bne s1,s9,80001abc   <- conv2 sy loop   (CONV2_KERNEL_HEIGHT=5)
80002230 bne s6,a5,80001aa4   <- conv2 tile_base loop (24/8=3 tiles)
80002250 bne a4,a1,80001a7c   <- conv2 ox loop (4)
8000226c bne a2,a4,80001a64   <- conv2 oy loop (4)
800029ac bne a6,t6,800022e0   <- fc1 full-tile iy loop (FC1_CHANNELS_HEIGHT=4)
80002afc bne t5,a2,800022d0   <- fc1 full-tile tile_base loop (18 tiles)
80003060 bne t5,a5,80002b18   <- fc1 tail-tile iy loop (the 1 remainder tile has no tile_base loop)
```

fc2 has **no** backward branches of its own at all: `FC2_FULL_TILES =
FC2_NB_OUTPUTS/8 = 10/8 = 1` and `FC2_TAIL_SLOTS = 2` are both trivial
(trip count 1), fc2 has no `iy` loop (`FC2_CHANNELS_HEIGHT == 1`), so its
entire body — both tiles — is straight-line unrolled code, bounded by
`propagate+0x2a6c` (`0x8000316c`, its `RESET_ACC`) through the printf call at
`0x80004768`. (The other backward `j`/`bge` pairs found near
`0x80004838-0x80004918` belong to the also-inlined `maxPropagate1()`
argmax loop, not fc2's MAC path — it never issues any MAC4/v2 instruction, so
it's out of scope here.)

One correction to the ticket's premise worth flagging: **`CONV2_KERNEL_HEIGHT`
is 5, not 4** — `sw/app/mnist/conv2.h:18` says `#define CONV2_KERNEL_HEIGHT
5` (`CONV2_KERNEL_WIDTH` is also 5, giving the `5*16/4=20`-word segment the
source's own `#if CONV2_SEGMENT_WORDS != 20` check already asserts). Using 5
matters for every conv2 dynamic count below; using 4 (as originally assumed)
underestimates conv2 by 20%.

## 1. Identifying the custom instructions in the disassembly

Per `core/mac4_copro/include/mac4_instr_pkg.sv:9-16,66-97` and
`sw/app/mnist/mac4.h`, all five MAC4/v2 ops are R-type under opcode
`0001011` (custom-0, `0x0B`), distinguished by `funct3` (bits `[14:12]`):
`0`=MAC4 (v1, register-only), `1`=LOAD_STATIONARY, `2`=MAC_TILED,
`3`=READ_ACC, `4`=RESET_ACC. Objdump renders them as raw
`.insn 4, 0x...` (unknown-opcode) lines, so they were pulled out of
`propagate()`'s disassembly programmatically by decoding
`(word & 0x7f) == 0x0b` and `funct3 = (word >> 12) & 0x7`:

| funct3 | op | static count in `propagate()` |
|---|---|---|
| 0 | MAC4 (v1) | **0** — confirms v1's plain register MAC4 is gone; everything now goes through v2 |
| 1 | LOAD_STATIONARY | 143 |
| 2 | MAC_TILED | 874 |
| 3 | READ_ACC | 40 |
| 4 | RESET_ACC | 6 |

**These static counts are a perfect, closed-form match to the source's loop
structure**, which is itself strong confirmation the disassembly is being
read correctly. Each of the 4 layers/tiles contributes one static site per
category (since the runtime loops replay the same code, they don't multiply
the *static* count, only the *dynamic* one):

- RESET_ACC (6): one per {conv1, conv2, fc1-full, fc1-tail, fc2-full,
  fc2-tail} reset site → 1+1+1+1+1+1 = 6. ✓
- READ_ACC (40): 8 slots × {conv1, conv2, fc1-full, fc2-full} (32) + 6 slots
  (fc1-tail) + 2 slots (fc2-tail) = 32+6+2 = 40. ✓
- LOAD_STATIONARY (143): conv1 1 word + conv2 20 words + fc1-full 24 +
  fc1-tail 24 + fc2-full 37 + fc2-tail 37 = 1+20+24+24+37+37 = 143. ✓
- MAC_TILED (874): conv1 1×8=8 + conv2 20×8=160 + fc1-full 24×8=192 +
  fc1-tail 24×6=144 + fc2-full 37×8=296 + fc2-tail 37×2=74 =
  8+160+192+144+296+74 = 874. ✓

## 2. Dynamic (weighted-by-trip-count) instruction counts per layer

Trip counts derived from the headers (`conv1.h`/`conv2.h`/`fc1.h`/`fc2.h`)
and `MAC4_NUM_ACC_SLOTS=8`:

| layer | oy×ox | tiles | sy/iy | sy-body execs | tile-body execs |
|---|---|---|---|---|---|
| conv1 | 11×11=121 | 2 (16/8) | 4 | 121×2×4 = **968** | 121×2 = **242** |
| conv2 | 4×4=16 | 3 (24/8) | 5 | 16×3×5 = **240** | 16×3 = **48** |
| fc1 full | — | 18 | 4 | 18×4 = **72** | **18** |
| fc1 tail | — | 1 (6 left) | 4 | 1×4 = **4** | **1** |
| fc2 full | — | 1 (8 of 10) | — (no iy) | **1** | **1** |
| fc2 tail | — | 1 (2 left) | — | **1** | **1** |

Multiplying the static per-body instruction patterns (below, each cited from
a disassembly excerpt) by these trip counts gives per-layer dynamic
instruction counts. Five buckets, matching the ticket's requested categories:

**(a) Stationary-segment loading** — `LOAD_STATIONARY` + whatever feeds it.
The feed instructions differ *by layer*, which is itself a finding:

- conv1 (`0x800017e0`-`0x80001850`): every word is packed via **4× `lbu` + 4×
  `sb` (spill to stack) + 1× `lw` (reload as word)**, i.e. 9 instructions to
  assemble one 4-byte `input_word`, before the 1 `LOAD_STATIONARY`. Example:
  ```
  80001800: lbu s11,0(a2) / lbu s10,1(a2) / lbu s9,2(a2) / lbu t3,3(a2)
             sb s11,60(sp) / sb s10,61(sp) / sb s9,62(sp) / sb t3,63(sp)
             lw s11,60(sp)
  80001804: .insn 4, 0x000d900b     <- LOAD_STATIONARY
  ```
  This is despite ticket #28's shifted-buffer trick guaranteeing every
  `(oy,ox)` picks an address that genuinely is 4-byte aligned at runtime —
  GCC still can't *prove* that statically (the base pointer is chosen
  per-`(oy,ox)` between two different arrays via a runtime branch, and the
  offset includes a runtime `sy` term), so it defensively byte-packs instead
  of emitting a single `lw`. Net: **9 overhead instructions per stationary
  load, for a segment that is only 1 word long** — the byte-packing tax is
  9x the size of the data it's fetching.
- conv2 (`0x80001acc`-`0x80001ad0`): **1× `add` (base + runtime offset) + 1×
  `lw`**, then `LOAD_STATIONARY`, repeated per word (20/sy-iteration):
  ```
  80001acc: add s10,s5,a4 / lw s10,0(s10) / .insn 4,0x000d100b
  ```
- fc1/fc2 (`0x800022e0` onward, `0x8000316c` onward): **1× `lw`** (immediate
  offset off a precomputed base pointer, no separate `add`), then
  `LOAD_STATIONARY`:
  ```
  800022e0: lw ra,352(t6) / .insn 4,0x0000900b
  ```

  | layer | static pattern per word | dynamic LOAD_STATIONARY | dynamic feed overhead | total |
  |---|---|---|---|---|
  | conv1 | 4 lbu+4 sb+1 lw (9) + insn | 968 | 8712 | 9680 |
  | conv2 | 1 add+1 lw (2) + insn | 4800 | 9600 | 14400 |
  | fc1 | 1 lw + insn | 1824 | 1824 | 3648 |
  | fc2 | 1 lw + insn | 74 | 74 | 148 |

**(b) Weight loading** — always exactly **1 `lw` per `MAC_TILED`**, no extra
`add` (weight base pointer + compile-time-constant immediate offset,
e.g. `lw a4,-2048(a5)` right before `.insn 4,0x0007200b` at
`0x80001bcc`-`0x80001bd0`). This is a hard 1:1 ratio — every `MAC_TILED`
reads a genuinely distinct weight word, so there is no redundancy to remove
here regardless of tiling.

**(c) MAC_TILED execution** itself (from §1's per-body counts × trip counts):

| layer | dynamic weight `lw` = dynamic MAC_TILED |
|---|---|
| conv1 | 968 × 8 = **7744** |
| conv2 | 240 × 160 = **38400** |
| fc1 | 72×192 + 4×144 = 13824+576 = **14400** |
| fc2 | 1×296 + 1×74 = **370** |

**(d) RESET_ACC/READ_ACC + bias-add/saturate epilogue.** The epilogue for
each output slot is a fixed 9-instruction static pattern (confirmed at e.g.
`0x8000185c`-`0x80001878` and `0x800020e0`-`0x80002100`):
`READ_ACC → lw (bias) → add → not → srai → and → srai (branchless
max(x,0)) → bge (compare vs 255) → sb`, with an extra `li ...,255`
executed only on the branch-taken half:
```
80001854: .insn 4,0x3d0b        <- READ_ACC
80001858: lw s9,0(t1)           <- bias
8000185c: add t3,s10,s9         <- weightedSum = bias + acc
80001860: not s11,t3 / srai a4,s11,0x1f / and a2,t3,a4 / srai s10,a2,0x8   <- clamp-to-0 (branchless)
80001870: bge a0,s10,80001878   <- clamp-to-255 (branchy)
80001878: sb s10,576(a6)        <- store
```
RESET_ACC (dynamic = tile-body execs) + READ_ACC (dynamic = tile-body
execs × slot count) + the 8-instruction non-branch epilogue per READ_ACC:

| layer | RESET_ACC | READ_ACC | epilogue overhead (×8) | total |
|---|---|---|---|---|
| conv1 | 242 | 1936 | 15488 | 17666 |
| conv2 | 48 | 384 | 3072 | 3504 |
| fc1 | 19 | 150 | 1200 | 1369 |
| fc2 | 2 | 10 | 140* | 152 |

\* fc2's epilogue is 14, not 8, instructions/slot: on top of the base 8 it
adds the `FC2_SCALAR_REMAINDER` 2-element tail (2× `lbu` + 2× `mul` + 2×
`add`) — see §3.

**(e) Loop control** (branch + induction-variable instructions for
`sy`/`iy`/`tile_base`/`oy`/`ox`, e.g. conv2's sy-loop closing sequence
`addi s1,s1,1 / addi a3,a3,176 / addi a5,a5,80 / bne s1,s9,80001abc` at
`0x800020cc`-`0x800020d8`): roughly 5379 (conv1, includes the `ox&1`
alignment branch, §3), 1368 (conv2), 300 (fc1), ~10 (fc2, no real loops).
This bucket is the least precisely counted (estimated per-iteration
overhead × trip count, not exhaustively enumerated instruction-by-
instruction like (a)-(d)), but it's also small enough in every layer except
conv1 that the imprecision doesn't move the overall picture.

## 3. Layer-specific items the ticket asked about

**conv1's `ox & 1` alignment branch** (`0x8000179c`-`0x800017ac`):
```
8000179c: andi t1,a7,1
800017a0: beqz t1,800017b0
800017a4: j 800048cc          <- odd-ox path: conv1_input_shifted base
800017b0: ...                  <- even-ox path: inputs base
```
Executed once per `(oy,ox)` (121 times), not per `sy`/tile_base — ticket #28
hoisted it out of the inner loops as documented in
`NetworkPropagate.c:126-136`. It's 2-3 instructions per occurrence, folded
into the loop-control bucket above; small on its own, but it's the reason
conv1 needs *any* branch here at all where conv2/fc1/fc2 don't.

**fc1's partial last tile**: already reflected structurally above as the
"fc1 tail" row throughout — 1 explicit tile of 6 slots (`FC1_TAIL_SLOTS=6`,
`_Pragma("GCC unroll 6")`) instead of a 7th full-8 runtime tile_base
iteration, per `NetworkPropagate.c:373-385`.

**fc2's 2-element scalar remainder**: `FC2_SEGMENT_REMAINDER = 2`
(150 = 37×4 + 2), added after `mac4_read_acc()` via plain C multiply-add,
confirmed at `0x80004084`-`0x800040a0`:
```
80004084: lbu s0,884(s2)     / lbu t3,885(s2)     <- 2 leftover input bytes
8000408c: li a6,-35          / li s5,-26          <- weight VALUES as compile-time immediates
80004094: mul s11,s0,a6      / mul a5,t3,s5        <- because tile_base/slot are compile-time
                                                       constants here, wOffset is too, so GCC
                                                       constant-folds straight from the `const`
                                                       weights array instead of loading it
```
Interesting side-detail: because fc2's tiling has only 2 tiles total (both
with compile-time-constant `tile_base`), the compiler resolves the remainder
weight *values* at compile time and bakes them in as `li` immediates rather
than loading them — the scalar-remainder path pays 0 weight-`lw` per element
(vs. the MAC_TILED path's 1 `lw`/word), at the cost of 1 `mul` per element
that a MAC_TILED-covered word wouldn't need.

## 4. Per-layer category breakdown (dynamic instruction share)

Summing (a)-(e) per layer:

| layer | stationary load | weight load | MAC_TILED | RESET/READ+epilogue | loop control | **layer total** |
|---|---|---|---|---|---|---|
| conv1 | 9680 (20.1%) | 7744 (16.1%) | 7744 (16.1%) | 17666 (36.6%) | 5379 (11.2%) | **48213** |
| conv2 | 14400 (15.0%) | 38400 (40.0%) | 38400 (40.0%) | 3504 (3.6%) | 1368 (1.4%) | **96072** |
| fc1 | 3648 (10.7%) | 14400 (42.2%) | 14400 (42.2%) | 1369 (4.0%) | 300 (0.9%) | **34117** |
| fc2 | 148 (14.1%) | 370 (35.2%) | 370 (35.2%) | 152 (14.5%) | 10 (1.0%) | **1050** |
| **all** | 27876 | 60914 | 60914 | 22691 | 7057 | **179452** |

Network-wide layer share: conv1 26.9%, conv2 53.5%, fc1 19.0%, fc2 0.6%.

## 5. Cross-check against real RTL cycle measurements

| layer | real cycles | real share | static-model share | delta |
|---|---|---|---|---|
| conv1 | 80305 | 21.6% | 26.9% | +5.3pp |
| conv2 | 197614 | 53.2% | 53.5% | +0.3pp |
| fc1 | 90480 | 24.3% | 19.0% | -5.3pp |
| fc2 | 3326 | 0.9% | 0.6% | -0.3pp |

conv2's share matches almost exactly (this is also the layer with by far the
largest, most regular inner loop, so it's the best-conditioned case for a
static count). conv1 and fc1 are off by ~5 points in opposite directions —
plausibly because conv1's real cost includes pipeline effects a flat
instruction count can't see (load-use stalls: nearly every `lbu`/`lw` in
conv1's byte-packing chain directly feeds the next instruction; branch
mispredict/refill on 121 short `(oy,ox)` iterations), while fc1's very
regular, larger loop body (192 unrolled `MAC_TILED`s per iteration) is
comparatively pipeline-friendly and this model may be crediting it with too
much epilogue/loop-control weight relative to conv1. Implied average CPI
(371725 real cycles / 179452 modeled instructions ≈ 2.07) is unsurprising
for an in-order core with a load feeding almost every custom instruction —
consistent with issue #18's own framing of "CPI≈1" as an optimistic
lower bound, not a claim of accuracy. This is read as *directional*
confirmation (conv2 dominates, fc2 is noise, conv1/fc1 are the second tier),
not a precision claim, per the ticket's own stated bar.

## 6. Redundant per-tile stationary reload

The ticket specifically asks: since MAC4 v2's tile-outer/sy-middle order
reloads the *same* input segment once per tile rather than once total across
a layer's tiles (unavoidable with T=8 fixed accumulator slots: a tile must
fully finish its `RESET_ACC`→sy-sweep→`READ_ACC` before the same 8 physical
accumulators can be reused for the next tile, so the sy-sweep — and its
stationary loads — must replay), what fraction of stationary-load traffic is
this redundancy?

| layer | tiles | redundant fraction (= 1 − 1/tiles) | dynamic LOAD_STATIONARY | redundant loads | + redundant feed overhead | total waste |
|---|---|---|---|---|---|---|
| conv1 | 2 | 50.0% | 968 | 484 | 484×9=4356 | 4840 |
| conv2 | 3 | 66.7% | 4800 | 3200 | 3200×2=6400 | 9600 |
| fc1 | 19 (18+1 tail) | 94.7% | 1824 | 1728 | 1728×1=1728 | 3456 |
| fc2 | 2 (1+1 tail) | 50.0% | 74 | 37 | 37×1=37 | 74 |
| **all** | | | **7666** | **5449 (71.1%)** | | **17970** |

71.1% of all dynamic `LOAD_STATIONARY` executions network-wide are
re-fetching a segment already loaded for an earlier tile at the same
`(oy,ox,sy)` / `(tile,iy)` position — but `LOAD_STATIONARY`-plus-feed is
itself only 15.5% of total dynamic instructions (27876/179452), so this
redundancy, even eliminated completely, bounds out at **≤10.0% of any single
layer's instruction count** (conv2: 9600/96072=10.0%; fc1: 3456/34117=10.1%;
conv1: 4840/48213=10.0%; fc2: 74/1050=7.0%) and **≤10.0% of the network
total** (17970/179452).

## 7. What "go past T=8" and "unrolling the remaining loops" would each buy

**Increasing T** only ever removes the *redundant* portion of §6 (it doesn't
touch weight loading or MAC_TILED at all — every output channel needs its
own weight word and its own accumulator read regardless of tiling, so
`RESET_ACC` count and the *redundant-load* count shrink with fewer tiles,
but `READ_ACC`/epilogue count does not, since every channel is still read
exactly once). Concretely:
- conv1: `T=16` exactly covers `CONV1_NB_OUTPUTS=16` in one tile → fully
  eliminates conv1's redundancy (saves the full 4840, ~10% of conv1).
- conv2: needs `T=24` to collapse to 1 tile (3x current accumulator count)
  → saves ~9600, ~10% of conv2.
- fc1: `FC1_NB_OUTPUTS=150` — even `T=32` (4x) only cuts 19 tiles to 5,
  recovering ~2700 of fc1's 3456 redundant instructions (~7.9% of fc1);
  fully eliminating it needs `T=150`, clearly impractical in hardware.
- fc2: already only 2 tiles: `T=10` would fully cover it, saving 74
  instructions — negligible in absolute terms (fc2 is 0.6% of the network).

Rolled up: even an aggressive, hardware-expensive T bump (T=16 for conv1,
T=24 for conv2, T=32 for fc1) recovers on the order of **7-8% of total
network-wide dynamic instructions** for a 2-4x larger accumulator array and
more complex tile bookkeeping. It **does not touch** the 60914+60914 =
121828 weight-load+MAC_TILED instructions that are 67.9% of the network
total, because that 1:1 ratio is structural (every weight word is read
exactly once, used exactly once, regardless of T).

**Unrolling the remaining runtime loops** (`sy`/`iy`/`tile_base`/`oy`/`ox`)
removes loop-control instructions, bucket (e): 7057 total, **3.9% of the
network-wide instruction count**, concentrated almost entirely in conv1
(5379, 11.2% of conv1) because conv1 is the only layer paying the
loop-control tax across 121 spatial positions rather than a handful of
tile/iy iterations. Fully unrolling conv1's `oy`/`ox` loops (121
combinations) would multiply conv1's ~500-instruction-per-(oy,ox) static
body by 121x for a return capped at 11.2% of conv1 (≈3.0% of the network) —
a bad code-size/return trade even before counting the icache pressure a
~60000-instruction unrolled conv1 body would add.

## 8. Conclusion

Neither of the two fog candidates named in the ticket is where the
remaining cycles are. In the two layers that dominate the network (conv2
53.5%, fc1 19.0% — 72.5% combined), 80-84% of dynamic instruction traffic
is the **structural 1:1 weight-`lw` : `MAC_TILED` pair**, which is invariant
to both T and loop unrolling — every weight word is read from memory exactly
once per (output channel, segment word), and the current register-only
CV-X-IF interface has no way to keep a *weight* stationary the way v2
already keeps *inputs* stationary. That's the real remaining floor, and
moving it needs a different kind of change than either fog candidate: e.g. a
second, weight-side stationary/streaming buffer (symmetric to what v2 did
for inputs), or a wider `MAC_TILED` that consumes more than one weight word
per issued instruction — not more accumulator slots and not more unrolling.

The two named candidates aren't zero-value, but they're each capped well
under 10% of the network total (§7), for real hardware/code-size cost, and
conv1 — the one layer where they'd help most — has a bigger, cheaper,
non-architectural win sitting right next to them: **90% of conv1's
stationary-load instructions (8712 of 9680) are pure byte-packing overhead**
(4×`lbu`+4×`sb`+1×`lw` to assemble one word `lw` should produce directly),
because GCC can't statically prove the runtime-selected, ticket-#28-aligned
address is 4-byte aligned even though it provably always is at runtime. That
alone is 4.9% of the *entire network's* dynamic instruction count
(8712/179452) recoverable by a codegen fix (alignment assertion /
restructured pointer type / hand-written `lw`), not an RTL or loop-structure
change, and it's larger than everything "go past T=8" could win back from
conv1's actual redundant reloads (4840).

**Recommendation**: if a new ticket comes out of this, prioritize (1) the
conv1 byte-packing codegen fix (cheap, ~4.9% network-wide, no hardware
change) over both named candidates; if capacity remains, (2) a symmetric
weight-side stationary/streaming mechanism is the only lever that touches
the dominant 67.9% weight-load+MAC_TILED floor in conv2/fc1, which neither
"go past T=8" nor "unroll remaining loops" reaches at all; (3) T-scaling and
loop-unrolling are both real but secondary, each worth low-single-digit to
~8% at best and only in the layers where tiles are few (conv1, fc2) or
where code-size is cheap to pay (neither is true for fc1's 150-output
tiling).
