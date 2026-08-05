# Tile-axis stationary-reload redundancy vs. CNN_C2's single-accumulator streaming model

Research for GitHub issue #41 (child of map #38), opened off the back of
issue #30's cycle profile (`docs/research/mac4-v2-cycle-profile.md`, branch
`research/mac4-v2-cycle-profile`), issue #32's spatial-batching tradeoff
(`docs/research/weight-spatial-batching-tradeoff.md`, branch
`research/weight-spatial-batching-tradeoff`), and issue #39's CNN_C2
architecture comparison (`docs/research/cnn-c2-architecture-comparison.md`,
branch `research/cnn-c2-architecture-comparison`). All of this repo's
sources were read at commit `a3bb582` on branch `cv32a6_contest_25_26`
(this document's branch, `research/stationary-reload-redundancy-vs-streaming`,
was cut from that exact commit); the comparison fork's sources were read via
`git -C /home/a21pelle/Claude/WorkSpace/cva6-softcore-contest-rsp show
84162172:<path>` (never checked out, never `cd`-ed into) at commit
`84162172c2c82424e903e43557b2b4a743873cb6` on branch
`cv32a6_contest_25_26_CNN_C2`. Nothing is pushed, `cv32a6_contest_25_26` was
not touched, and no GitHub issue/comment action was taken.

## 0. What changed under this question since #30/#32 were written

#30 and #32 were both written against commit `14fc348` (MAC4 v2, B=1, no
spatial batching, conv1's stationary load still byte-packed). Three things
have since landed on `cv32a6_contest_25_26` and are already reflected in the
`a3bb582` source read for this document:

- `fbebbef` widened the coprocessor to `T=32` accumulator slots / 80-word
  stationary buffer (`core/mac4_copro/mac4_alu.sv:72-73`).
- `90c0ff2`/`a3bb582` built weight-spatial-batching (B=4) for conv2 and
  conv1 respectively — exactly the design #32 modeled in its §2.1/§2.2
  pseudocode, now real code (`sw/app/mnist/NetworkPropagate.c:403-470` conv2,
  `:151-295` conv1).
- A separate, already-merged alignment fix (commit `022e31c`, referenced in
  `NetworkPropagate.c:210-217`'s comment: *"this address is 4-byte aligned
  at runtime for every position by construction... but GCC can't prove it
  statically (ticket #31)"*) replaced conv1's 9-instruction byte-packing
  load feed (`4×lbu+4×sb+1×lw`, #30 §2(a)) with
  `__builtin_assume_aligned(...)` + a single `memcpy`-as-load, which compiles
  down to the same `add`+`lw` pattern conv2 already used. This is confirmed
  directly in the current binary's disassembly (§2.2 below) and matters a
  lot for question 2: #30's conv1 redundant-load cost estimate (4840
  instructions, dominated by 9×-overhead byte-packing) is now stale by
  roughly 3x.

None of this changes #32's central structural finding — accumulator slots
required = `T×B` = 32 (§3 of that doc) — which this document's §3 revisits
from a different angle.

## 1. CNN_C2's actual accumulator model

**Confirmed: exactly one hardware accumulator register, not an array,
cleared by a single instruction that also drives the input-buffer's
read/write mode and rewinds its read pointer.**

`core/cvxif_example/mac/mac_cfu.sv:29-30` (all line numbers below are from
`84162172`, not this repo):

```systemverilog
logic[31:0] accu_q = 32'd0;
logic[31:0] accu_n = 32'd0;
```

A scalar 32-bit register, `mac4_alu.sv`'s equivalent of `acc_q[T]` collapsed
to `T=1`. Its update (`mac_cfu.sv:94,103`):

```systemverilog
accu_n = acc_q ? mac_res : accu_q;          // acc_q here is the *enable* signal from mac_controller, not the register
...
accu_q <= reset_acc ? 32'd0 : accu_n;
```

`reset_acc` is `mac_controller`'s `acc_clr_o` (`mac_cfu.sv:59`), driven by
`mac_controller.sv:76-79`:

```systemverilog
if (opcode_i==MAC_CTRL) begin
    buffer_addr_n = 5'd0; // Inputs buffer: go back to first element
    if (valid_o == 1) begin
        acc_clr_n = 1; // Clear accu
        buffer_we_n = operand_a_i == 32'd0 ? 1 : 0; // Select inputs buffer mode (R/W)
```

So a single `MAC_CTRL` instruction (`mac_instr.h`'s `initInputBuffer()` and
`readAccu()` are both thin wrappers around it, `mac_instr.h:194-221`) does
three things at once: clears the one accumulator, rewinds the input
buffer's read pointer to element 0, and picks write-mode (fill from
registers, `operand_a_i==0`, `initInputBuffer()`) or read-mode (replay from
the buffer, `operand_a_i!=0`, `readAccu()`). This is confirmed by the actual
software driver, `sw/app/mnist/NetworkPropagate.c:100-136,205`
(`convcellPropagate1`, `84162172`):

```c
for (int oy...) {
    for (int ox...) {
        initInputBuffer();                    // once per spatial position
        for (int output = 0; output < NB_OUTPUTS; ++output) {   // every channel
            for (int sy...) {
                ... macsOnRangeCustom(inputs+iOffset, weights+wOffset, &weightedSum,
                                       KERNEL_WIDTH*NB_CHANNELS, output);
            }
            weightedSum = readAccu(weightedSum);   // read AND reset AND rewind, before next channel
            outputs[...] = sat(weightedSum, ...);
        }
    }
}
```

`macsOnRangeCustom` (`mac_instr.h:223-268`) is the mechanism that makes this
work with only one accumulator: it branches on `output==0`. For the first
channel, it issues `MAC8_EXEC`/`MAC4_EXEC` in **write mode**
(`buffer_we_q=1`, set by the position's `initInputBuffer()` call) — each
call both computes that channel's partial product *and* writes the operand
into `inputs_buffer.sv`'s SRAM at the auto-incrementing address
(`mac_controller.sv:61,63`: `buffer_addr_n = buffer_addr_q + 1` per call,
`+2` for `MAC8_EXEC` in read mode since it fetches two packed words per
call, `inputs_buffer.sv:20-22`). For every subsequent channel (`output!=0`),
`macsOnRangeCustom` issues the *same* `MAC8_EXEC` sequence in **read mode**
(`mac_instr.h:250-263`, `mac8_buffer_unroll*`), with the input operand
literally `x0` — the ALU pulls its input entirely from the resident buffer
(`inputs_buffer.sv:18`: `output_o = {unsigned_buffer[addr_i+1],
unsigned_buffer[addr_i]}`), re-walked from address 0 (rewound by the
*previous* channel's `readAccu()`) at zero memory-load cost. **The entire
per-position input window is loaded from memory exactly once (during
channel 0), then replayed via on-chip SRAM reads for every remaining
channel of the layer — not just for one 8-channel tile.**

`inputs_buffer.sv:12` sizes this SRAM to exactly 100 words (`logic [31:0]
unsigned_buffer[99:0]`), addressed by a 7-bit index. Cross-checked against
their `conv2.h` (`CONV2_KERNEL_HEIGHT=5`, `CONV2_KERNEL_WIDTH=5`,
`CONV2_NB_CHANNELS=16` — identical shape to this repo's conv2, both forks
share the same N2D2-exported network): `5×5×16 = 400` bytes = **exactly 100
4-byte words** — the buffer is sized to hold *one spatial position's entire
conv2 kernel receptive field*, not one kernel row. This is the concrete
number that makes §3's synthesis below possible to size.

**Why didn't #39 flag this?** Re-reading `docs/research/cnn-c2-architecture-comparison.md`
confirms #39's brief was to catalogue *levers* worth pursuing (8-lane MAC,
input-buffer auto-increment, shared conv function) as candidate wins for
map #38 — it correctly identified the input-side stationary buffer and its
auto-incrementing address as "the main probable lever," but treated
accumulator *count* as an implementation detail of that same buffer
mechanism rather than an independent axis to interrogate on its own. It
wasn't wrong, just scoped one level higher than this ticket: #39 asked "what
makes CNN_C2 fast," not "what makes CNN_C2's accumulator array 1 slot
instead of our 32" — the latter is what unlocks §3's actual tradeoff.

## 2. Real cycle cost of today's tile-axis reload redundancy (commit `a3bb582`)

### 2.1 The dynamic instruction counts are unchanged by B (already established)

`docs/research/weight-spatial-batching-tradeoff.md` §4 already proved,
purely from the B=4 pseudocode's structure (not from disassembly), that
total dynamic `LOAD_STATIONARY` count is **B-invariant**: every position's
segment is still loaded exactly once per `(position, sy, tile)`, regardless
of how positions are grouped into batches. That means #30's original B=1
formula still holds exactly at B=4:

| layer | tiles | positions | sy | segment words | dynamic `LOAD_STATIONARY` | redundant fraction (tile axis) | redundant loads |
|---|---|---|---|---|---|---|---|
| conv2 | 3 | 16 | 5 | 20 | 3×16×5×20 = **4800** | 1 − 1/3 = **66.7%** | 3200 |
| conv1 | 2 | 121 | 4 | 1 | 2×121×4×1 = **968** | 1 − 1/2 = **50.0%** | 484 |

(`CONV2_NB_OUTPUTS/MAC4_NUM_ACC_SLOTS = 24/8 = 3` tiles,
`CONV1_NB_OUTPUTS/MAC4_NUM_ACC_SLOTS = 16/8 = 2` tiles — both from
`sw/app/mnist/conv1.h`/`conv2.h`, both unchanged since #30.) This is
confirmed structurally correct for the actual `a3bb582` code by inspection
of `NetworkPropagate.c`: conv2's `sy` loop (`:417-450`) sits inside the
`tile_base` loop (`:413-468`), which sits inside the `oy` loop (`:410-469`)
— a batch's 4 positions' segments are loaded fresh for every `(oy,
tile_base, sy)` combination, i.e. once per tile, exactly as before batching.
Same for conv1's `batch_start`/`tile_base`/`sy` nesting (`:162-238`).

### 2.2 Per-load feed overhead, re-measured on the current binary

Rebuilt `sw/app/mnist.riscv` clean (`rm` all `.o`/`.riscv`/`.mem`/`.bin`
artifacts, then `make benchmark APP=mnist` inside the `sw-docker:vfft`
container, per this repo's own build-environment convention) and
disassembled it (`riscv-none-elf-objdump -d`). Every layer's boundary is
unambiguous — each layer's `printf("<layer>: %lu cycles\n", ...)` call
(`NetworkPropagate.c:740,758,776,794`) is a `jal` to the same `<printf>`
symbol, giving 4 clean split points in the disassembly at
`0x80002220`/`0x80003940`/`0x80004814`/`0x80005e18`.

Static custom-instruction counts per region (funct3-decoded, same method as
#30 §1) confirm conv2/fc1/fc2 exactly match their macro-expansion-derived
predictions (high confidence): conv2 `LOAD_STATIONARY`=80, `MAC_TILED`=640,
`RESET_ACC`=1, `READ_ACC`=32 (matches `ox`-unroll(4)×`w`-unroll(20) for
loads, `slot`-unroll(8)×`w`-unroll(20)×4-position macro for MAC_TILED,
exactly as `CONV2_MAC_TILE_SLOT`, `NetworkPropagate.c:387-401`, predicts).
conv1's static counts (`LOAD_STATIONARY`=8, `MAC_TILED`=64) came out ~1.6x
higher than a naive macro-expansion count predicts (5 and 40) — plausibly
`-funroll-all-loops` partially unrolling conv1's `batch_start`/`sy` loops
differently than conv2's, or code-layout duplication; not fully resolved
here (would need more disassembly archaeology than this ticket's
"static/analytical, not new RTL sim" budget justifies) and, importantly,
**doesn't affect the dynamic-count argument in §2.1**, which is a semantic
property of the algorithm (every position's segment must be loaded once per
tile, full stop) independent of how GCC happens to lay out the static code.

What the disassembly *does* newly confirm (this is the actual new finding
of this section): conv1's stationary-load feed is now the same 2-instruction
`add`+`lw` pattern conv2 already used, not #30's 9-instruction byte-packing
chain. Example, any `LOAD_STATIONARY` site in the current binary:

```
80001990: add  t6,t4,t5
80001994: lw   s1,0(t6)
80001998: .insn 4,0x4900b        <- LOAD_STATIONARY
```

vs. #30's pre-alignment-fix conv1 pattern (`4×lbu+4×sb+1×lw`, 9
instructions). This is the alignment-fix ticket (#31, commit `022e31c`)
paying off exactly where #30 predicted it would (§8 of that doc: *"90% of
conv1's stationary-load instructions... pure byte-packing overhead"*).

### 2.3 Updated redundant-load "waste" (instructions)

Per-load waste = 1 `LOAD_STATIONARY` + 2 feed instructions = 3, for *both*
layers now (conv2 unchanged from #30; conv1 down from 10):

| layer | redundant loads | waste/load | total redundant-load instructions | #30's equivalent figure (pre-fix) |
|---|---|---|---|---|
| conv2 | 3200 | 3 | **9600** | 9600 (unchanged — conv2's codegen didn't change) |
| conv1 | 484 | 3 | **1452** | 4840 (3.3x lower now) |

### 2.4 Converting to cycles

No new RTL simulation was run for this (permitted by the ticket; the
existing `sim.log` in this repo's working tree — produced by an earlier,
unrelated session's `make sim` run at this same commit — has the top-level
`Result: 1/1 / credence: 82 / 140084 instructions / 247849 cycles` triplet
that matches the ticket's own cited baseline, but its UART capture does not
include `propagate()`'s per-layer `printf` lines, for reasons not
investigated here — not blocking, since #30's own precedent already
establishes that CPI-scaling from a known real/modeled-instruction ratio is
an accepted way to convert a static count to a cycle estimate).

Two CPI anchors are available, both defensible:

- **Current overall measured CPI**: `247849 / 140084 = 1.769` (today's real
  total, all 4 layers, at `a3bb582`).
- **#30's per-layer CPI** (pre-batching, pre-alignment-fix baseline, but
  each layer's *load-then-custom-instruction* dependency pattern this
  specific redundant-load waste is made of is structurally the same kind of
  chain now): conv2 = `197614/96072 = 2.057`, conv1 = `80305/48213 = 1.665`.

| layer | redundant-load waste (instr) | cycle estimate @ layer CPI | cycle estimate @ overall CPI (1.769) |
|---|---|---|---|
| conv2 | 9600 | 9600×2.057 ≈ **19,750** | 9600×1.769 ≈ **16,980** |
| conv1 | 1452 | 1452×1.665 ≈ **2,420** | 1452×1.769 ≈ **2,570** |
| **combined** | 11,052 | **≈ 22,200** | **≈ 19,550** |

So today's tile-axis reload redundancy costs roughly **19,500–22,200
cycles**, i.e. **7.9%–9.0% of `propagate()`'s current 247,849-cycle total**
— a real but bounded slice, consistent with #30's original finding that
this specific redundancy "bounds out at ≤10.0% of any single layer's
instruction count" even before this ticket's B=4/alignment-fix updates. The
range is presented as an estimate, not a measurement — the ticket's own
"static/analytical count is fine" allowance is being used at face value
here, same as #30/#32 did.

## 3. Is single-accumulator streaming compatible with B=4 weight-spatial-batching?

### 3.1 The two models, as literally written, in pseudocode

**CNN_C2 (from `84162172`'s actual `convcellPropagate1`, §1 above,
compressed):**

```c
for (position) {                          // oy, ox — one at a time, no batching
    initInputBuffer();                    // clears the ONE accumulator, rewinds buffer, write mode
    for (channel = 0; channel < NB_OUTPUTS; ++channel) {   // ALL channels, no tiling
        for (sy)
          for (sx)
            macsOnRangeCustom(...);        // channel 0: writes+computes; channel>0: reads buffer, computes
        outputs[position][channel] = readAccu(...);  // reads, clears, rewinds -- ready for next channel
    }
}
```

One accumulator suffices because channels are processed **fully
sequentially to completion** — channel `c`'s entire `sy×sx` reduction
finishes, gets read out, and the accumulator is zeroed *before* channel
`c+1`'s first product is ever computed. Nothing is ever interleaved.

**Our B=4 weight-spatial-batching (`NetworkPropagate.c:403-470`,
compressed, matching #32 §2.2's pseudocode exactly):**

```c
for (tile_base ...) {                      // channels 0..7, 8..15, 16..23 -- 3 tiles
    mac4_reset_acc();                       // clears ALL 32 accumulator slots
    for (sy ...) {
        for (ox = 0..3) load_stationary(...);       // this sy-row, all 4 positions
        for (slot = 0..7) {                          // 8 channels IN THIS TILE
            weight = weights[...];                    // load ONCE
            for (ox = 0..3) mac_tiled(weight, ox, ox*8+slot);   // reuse across 4 positions
        }
    }
    for (ox=0..3) for (slot=0..7) outputs[...] = read_acc(ox*8+slot);  // finalize this tile's 8 channels x 4 positions
}
```

32 accumulators are live simultaneously because **channels within a tile are
interleaved across the `sy` sweep** — channel `slot`'s partial sum gets a
contribution at `sy=0`, then again at `sy=1`, ..., and isn't finalized until
*all 8 channels* of the tile have received all 5 `sy` contributions.

### 3.2 Where the tension is real

These two loop orders genuinely clash exactly where the ticket predicted:
CNN_C2 wants **weight varying fastest, against a fixed input**, with
*channel* as the axis that advances to full completion before anything else
moves (position stays fixed for the whole channel sweep). Our batching wants
**position varying fastest, against a fixed weight**, with *channel* only
advancing far enough to exhaust one tile (8 channels) before the position
batch itself advances. You cannot literally splice CNN_C2's
`initInputBuffer()`-once-per-position / `readAccu()`-once-per-channel idiom
directly into `convcellPropagate2`'s inner loop without one of them giving
way — if channel is the interleaved axis (ours), you need live-accumulator
capacity per channel; if channel is the sequential-to-completion axis
(CNN_C2's), you only need live-accumulator capacity per *position*.

### 3.3 But the underlying trick — resident input, replayed for free, is layer-order-agnostic — and it does combine

The mechanism that lets CNN_C2 get away with only 1 accumulator isn't
really "streaming" or "no batching" — it's **holding the entire per-position
kernel receptive field resident** (not just one `sy` row) **and finishing
one channel completely before starting the next**, so no accumulator ever
needs to outlive more than one channel's reduction. Both of those properties
are independent of whether *position* is batched. Concretely, our own
`mac4_tiled(weight, mot, slot)` already supports "load once, replay for
free across many calls" — that's exactly what today's tile body already
does across 8 channels × 4 positions (32-way reuse of one `LOAD_STATIONARY`)
before a fresh load is needed. The only reason today's reuse tops out at one
*tile* (8 channels) instead of the *whole layer* (`NB_OUTPUTS` channels) is
that the stationary buffer only holds one `sy` row (20 words for conv2,
`MAC4_STATIONARY_WORDS=80` sized for `B×20`, `mac4.h:53`) — not the full
5-row kernel window.

So a genuine synthesis exists — **widen the stationary buffer to hold the
full per-position-batch kernel window (all `sy`, all segment words, for all
B positions), keep B=4 weight-spatial-batching's "one weight load, B
`mac_tiled` calls" trick, but process channels fully sequentially
(CNN_C2-style) instead of interleaved-per-tile**:

```c
for (position_batch of 4) {
    // load the WHOLE kernel window (all 5 sy rows, all 20 words/row) for all 4 positions -- 400 words
    for (w = 0; w < KERNEL_HEIGHT*SEGMENT_WORDS; ++w)
        for (p = 0; p < 4; ++p)
            load_stationary(batch[p]'s w-th window word, mot = p*400_words_per_position... /* p*100 + w */);

    for (channel = 0; channel < CONV2_NB_OUTPUTS; ++channel) {   // NO tiling -- sweep every channel
        mac4_reset_acc();                    // only 4 slots needed now, not 32
        for (w = 0; w < KERNEL_HEIGHT*SEGMENT_WORDS; ++w) {
            weight = weights[channel][w];     // load ONCE
            for (p = 0; p < 4; ++p)
                mac_tiled(weight, p*100 + w, p);   // reuse across 4 positions, same trick as today
        }
        for (p = 0; p < 4; ++p)
            outputs[batch[p]][channel] = read_acc(p);   // finalize, ready for next channel
    }
}
```

This needs **B=4 accumulator slots, not `T×B`=32** — a 8x reduction versus
today's hardware — and it **structurally eliminates the tile-axis
redundancy this whole ticket is about**: input is loaded once per
position-batch, full stop, replayed for free across all 24 conv2 channels
(no more "once per tile" reload, because there is no more tile). The cost:
the stationary buffer must hold `KERNEL_HEIGHT × SEGMENT_WORDS × B` words —
for conv2, `5 × 20 × 4 = 400` words, versus today's 80. (For a single,
unbatched position, that's `5×20=100` words — which is *exactly* the size
CNN_C2's own `inputs_buffer.sv:12` uses, `100`, because their conv2 has the
identical `5×5×16` kernel shape, confirmed against their `conv2.h`. That
match is a strong sanity check that this synthesis's buffer-sizing math is
right, not a coincidence.)

**Answer to the ticket's three-way framing**: not directly compatible as
literally written today (§3.2's clash is real), and not mutually exclusive
either — closer to **"one subsumes the other, at a real hardware cost that
wasn't on #32's table."** CNN_C2's "finish one channel to completion, one
accumulator" idea, applied at B=4 position-batch granularity instead of
CNN_C2's own B=1, absorbs weight-spatial-batching's core trick (one weight
load serving B positions) intact, needs only `B=4` accumulators (down from
`T×B=32`), and *also* eliminates the tile-axis reload redundancy quantified
in §2 — it doesn't just coexist with weight-spatial-batching, it dissolves
both the accumulator-count dilemma #32 wrestled with (§3.1 of that doc:
`T×B` vs. the rejected `NB_OUTPUTS×B`) and this ticket's reload-redundancy
question in one restructuring. The price is the stationary buffer growing
5x (80→400 words for conv2 at B=4) and needing `mot` addressing up to 399 —
**9 bits**, beyond even the widened 7-bit `funct7_i[6:0]` field
(`mac4_alu.sv:110`, 0..127) that MAC4 v3 already stretched to. Per #32
§4.1's own framework, this lands past the "ALU-only, no ISA change" zone
that B=2/B=4's *accumulator* widening enjoyed — it needs a genuine new
buffer-addressing encoding, in the same broad category #32 already flagged
as necessary for B≥8's *buffer depth* (though for a different reason: there
it was more positions, here it's a bigger per-position window).

## 4. Compatibility with lever 1 (8-lane MAC widening)

CNN_C2 itself is direct existence proof that an 8-lane MAC datapath and a
single/few-accumulator streaming model are not in tension — they already
coexist in one shipped design. Their `mac.sv:3` parameterizes
`VEC_WIDTH` (instantiated at 8 lanes for `MAC8_EXEC`, `mac_cfu.sv`'s
`VEC_WIDTH` parameter default), and `mac_controller.sv:61` auto-increments
the buffer address by 1 (write mode) or 2 (read mode, since `MAC8_EXEC`
consumes 8 bytes = 2 packed 4-byte buffer words per call) — the same single
`accu_q` register from §1 accumulates 8-lane products exactly as it does
4-lane ones. Lever 1 (widening `core/mac4_copro/mac4_alu.sv`'s dot-product
datapath from 4 to 8 lanes) and the accumulator-count/buffer-topology
question in §3 are genuinely **orthogonal axes**: lane width changes how
many bytes one `MAC_TILED`-equivalent instruction consumes per call (halving
the count of weight-`lw`+`MAC_TILED` pairs, the dominant 67.9%-of-network
cost #30 §8 identified as the real remaining floor), while accumulator count
and stationary-buffer topology govern how many channels/positions can be
live or resident at once. Neither forecloses the other.

One second-order interaction worth naming: if 8-lane widening also widens
the stationary buffer's *word width* to 8 bytes (not just the MAC datapath),
the §3.3 synthesis's buffer-word count would roughly halve for the same
byte volume (conv2's full-window B=4 requirement drops from 400 four-byte
words to roughly 200 eight-byte words) — easing, but not eliminating, the
9-bit addressing pressure identified there (`ceil(log2(200))=8` bits, still
past the current 7-bit `funct7_i[6:0]` field). Whether the stationary
buffer's word width should track the MAC datapath's lane width at all is
itself an open design choice not resolved by this ticket — flagged here as
a concrete follow-on question for whichever session scopes lever 1's actual
RTL change, not answered here.

## 5. Summary — explicitly NOT a recommendation

Per this repo's established convention (every prior research ticket in this
chain — #30, #32, #39 — ends this way, and #41's own text says the choice
belongs to a later, human-directed session): the table below is a summary
of findings, not a proposal to build anything.

| # | Question | Finding |
|---|---|---|
| 1 | CNN_C2's accumulator model | Confirmed: exactly **one** hardware accumulator (`accu_q`, `mac_cfu.sv:29`), reset+read+buffer-rewind by a single `MAC_CTRL` instruction issued once per output channel (`readAccu()`) and once per spatial position for the write-mode variant (`initInputBuffer()`). Not multiple. #39 didn't flag this because its brief was cataloguing levers (buffer auto-increment, 8-lane MAC, shared conv function), not interrogating accumulator count as its own axis. |
| 2 | Real cycle cost of today's redundant reloads (`a3bb582`) | Dynamic redundant-load instruction count: conv2 = 9600, conv1 = 1452 (down 3.3x from #30's pre-alignment-fix 4840, since conv1's feed overhead dropped from 9 to 2 instructions/load). Estimated cycle cost (CPI-scaled, not directly measured): **≈19,500–22,200 cycles combined, ≈7.9%–9.0% of the current 247,849-cycle `propagate()` total.** |
| 3 | Single-accumulator streaming vs. B=4 weight-spatial-batching | Not directly compatible as literally written (genuine loop-order clash: CNN_C2 wants channel-sequential-to-completion with position fixed; our batching wants position-parallel-via-register-reuse with channel interleaved per tile). Not mutually exclusive either: a synthesis exists — full per-position-batch kernel window resident + channel-sequential processing + B=4's weight-reuse trick intact — needing only **B=4 accumulators (not T×B=32)** and structurally eliminating the tile-axis redundancy, at the cost of a **5x larger stationary buffer** (400 words for conv2 vs. today's 80) needing 9-bit `mot` addressing, beyond the current 7-bit field. |
| 4 | Compatibility with lever 1 (8-lane MAC) | Orthogonal axis, not in tension with either accumulator model — CNN_C2 itself already ships 8-lane MAC + 1-accumulator streaming as one coexisting design. A secondary open question (not resolved here): whether widening the MAC datapath should also widen the stationary buffer's word width, which would roughly halve §3's buffer-word requirement but not eliminate its addressing-width pressure. |

## 6. Branch / GitHub state

Everything above lives at
`docs/research/stationary-reload-redundancy-vs-streaming.md` on branch
`research/stationary-reload-redundancy-vs-streaming`, branched from
`cv32a6_contest_25_26` at commit `a3bb582`. Nothing was pushed to any
remote, `cv32a6_contest_25_26` was not modified, and no GitHub issue/comment
action was taken — this document is the deliverable, for a human (or the
coordinating session) to relay. The comparison fork
(`/home/a21pelle/Claude/WorkSpace/cva6-softcore-contest-rsp`) was read only
via `git show`, never checked out or modified. `sw/app/mnist.riscv` and its
build siblings (`mnist.bin`/`.coe`/`.mem`, `libcva6.a`) were rebuilt clean
in this repo's working tree as part of §2.2's disassembly — they are
untracked build artifacts (same as they were before this session), not
committed.
