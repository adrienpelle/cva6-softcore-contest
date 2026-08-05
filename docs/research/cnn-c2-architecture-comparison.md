# CNN_C2 architecture comparison: how cva6-softcore-contest-rsp beats MAC4 v2

Research for GitHub issue #39 (map #38, "Explorer l'approche CNN_C2 de
cva6-softcore-contest-rsp : au-delà de MAC4 v2 pour mnist"). This ticket is
pure reading: no RTL, no C code, no simulation in this repo. Everything below
is derived from the `cva6-softcore-contest-rsp` fork at commit
`84162172c2c82424e903e43557b2b4a743873cb6`, branch
`cv32a6_contest_25_26_CNN_C2`, read via `git -C
/home/a21pelle/Claude/WorkSpace/cva6-softcore-contest-rsp show/diff`
(that repo's working tree was never touched — no `cd`, no checkout). Our own
side of the comparison is the committed state of this repo at `a3bb582`
(`core/mac4_copro/mac4_alu.sv`, `sw/app/mnist/mac4.h`), branched here as
`research/cnn-c2-architecture-comparison` off `cv32a6_contest_25_26` at that
same commit; nothing pushed, nothing touched on `cv32a6_contest_25_26` or
GitHub.

Numbers being compared: our clean 247849 cycles / 140084 instructions
(env0003, Result:1/1, credence 82, CPI≈1.77) vs. their reported 221387
cycles / 146080 instructions (same test, same result, same credence,
CPI≈1.52).

## 0. Files read in full

- `core/cvxif_example/mac/mac_cfu.sv` (169 lines) — top-level coprocessor module.
- `core/cvxif_example/mac/mac_controller.sv` (92 lines) — control/decode FSM.
- `core/cvxif_example/mac/mac.sv` (37 lines) — MAC datapath.
- `core/cvxif_example/mac/inputs_buffer.sv` (29 lines) — input buffer.
- `core/cvxif_example/include/cvxif_instr_pkg.sv` (diff vs. their base branch) — decode table.
- `sw/app/mnist/mac_instr.h` (266 lines) — software instruction wrappers.
- `sw/app/mnist/NetworkPropagate.c` (diff vs. their base branch, 27 lines).
- `sw/app/mnist/main.c` (full file, on `cv32a6_contest_25_26_CNN_C2`) — measurement harness.
- `core/cvxif_example/cvxif_example_coprocessor.sv` (diff + relevant excerpt) — top-level wiring.
- Full diffstat: `git -C cva6-softcore-contest-rsp diff cv32a6_contest_25_26 cv32a6_contest_25_26_CNN_C2 --stat`.

## 1. Execution model: single-cycle throughput, not a multi-cycle FSM

`mac_controller.sv` is **not** a multi-cycle wait-state FSM. It has no
counter, no `state_q`/`state_n` enum, and — critically — `valid_o = valid_i`
(`mac_controller.sv:56`): the controller passes issue-validity straight
through combinationally, one instruction accepted per cycle whenever
`issue_ready_i` is asserted. All its outputs (`buffer_addr_n`, `acc_e_n`,
`acc_clr_n`, `buffer_we_n`, `exec_o`) are pure combinational functions of
`opcode_i`/`operand_a_i`/current CSR state (`mac_controller.sv:58-90`); the
only registered state is a handful of 1-bit/7-bit "CSRs" (`buffer_addr_q`,
`buffer_we_q`, `acc_e_q`, `acc_clr_q`, `mac_controller.sv:24-44`) that update
every clock edge regardless of stalling. So structurally this is
**steady-state 1 instruction/cycle**, same throughput class as our
`mac4_alu.sv`.

The one real difference from our fully-combinational `mac4_alu.sv` (whose
dot product is built entirely out of `always_comb` blocks, e.g.
`mac4_alu.sv:116-141`) is that `mac.sv` registers the partial products:

```systemverilog
always_ff @(posedge clk_i or negedge rst_ni) begin : regs
    ...
    res[i] <= $signed({1'b0, inputs[i*8 +: 8]}) * $signed(weights[i*8 +: 8]);
    ...
end
always_comb begin
    mac4_result = sum(res[]);
    mac_result_o = mac4_result + accu;
end
```
(`mac.sv:15-28`)

`res[]` is clocked, so `mac_result_o` at cycle N reflects the product of the
operands presented at cycle N-1, summed with `accu_q` as of cycle N. This is
a genuine 1-stage pipeline register inside the multiply, not a stall: as
long as instructions issue back-to-back (which CV-X-IF does for accepted,
non-stalling ops), the pipeline fills and drains at 1 instruction/cycle,
just delayed by one cycle end-to-end. It behaves like a load-use-style
1-cycle latency, not an N-cycle busy-wait. **Answer: 1-cycle throughput,
1-cycle pipeline latency — comparable in kind to our single-cycle pipelined
`mac4_alu.sv`, not a slower multi-cycle design.**

What one instruction accomplishes depends on opcode and buffer mode (see §2
and §3): a `MAC4_EXEC`/`MAC8_EXEC` instruction does either (a) stage one
32-bit (4×int8) input word into the hardware buffer *while also* computing
a real 4-lane MAC against it (fill/write mode), or (b) a genuine 8-lane MAC
reading two pre-staged buffer words against two register-supplied weight
words (read/exec mode). Never a whole convolution window in one instruction
— always a handful of MAC lanes, same granularity class as our `mac4_tiled`.

## 2. Work per instruction vs. CPI: where the ~1.52 vs ~1.77 gap actually comes from

**Certain, from the code:** their compute instruction is 2x wider than ours.
`mac.sv` is instantiated with `VEC_WIDTH=8` (`mac_cfu.sv:19,73-75`,
`.VEC_WIDTH(VEC_WIDTH)` with the module's own default `VEC_WIDTH=8` at
`mac_cfu.sv` parameter list). In "exec"/read mode (`fill_buffer_q==0`),
`inputs = buffer_output_q` (64 bits = 8 packed int8 from the stationary
buffer) and `weights = {registers_i[0], registers_i[1]}` (64 bits = 8
packed int8 from two GPRs, `mac_cfu.sv:92-93`) — a genuine **8-wide**
int8×int8 MAC per `MAC8_EXEC`. Our `mac4_tiled` does 4 lanes per
instruction (uint8×int8, per the memory index and confirmed by
`mac4_alu.sv:134-141`'s 4-element `input_lane`/`weight_lane` arrays). So
each of their execute-mode instructions is worth 2x one of ours.

`MAC4_EXEC` (opcode `0x2`, decode `register_read={rd:0,rs2:1,rs1:0}` at
`cvxif_instr_pkg.sv` new table, entry 2) only supplies `rs2`; `rs1` reads as
whatever the assembly passes (`mac_instr.h`'s `mac4_buffer_unroll4` passes
literal `x0`), so `weights = {registers_i[0]=0, registers_i[1]=w1}` and only
the lower 4 lanes are non-zero — functionally a 4-wide MAC riding the same
8-wide datapath, i.e. width-equivalent to our own instruction, used as a
fallback for iteration counts not divisible by 8.

**Also certain, and the more interesting finding:** the "fill" pass isn't
pure overhead. In write/fill mode (`fill_buffer_q==1`, `mac_cfu.sv:90-91`),
`inputs = {32'd0, registers_i[0]}` and `weights = {32'd0, registers_i[1]}` —
these are the actual `(input, weight)` operands passed by
`mac4_unroll4`/`mac4_unroll8` in `mac_instr.h:62-64,73-90`, e.g.
`MAC8_EXEC ", %[res1], %[in1], %[w1]"` → `rs1=in1` (real input word),
`rs2=w1` (real weight-0 word). So the very same instruction that stages a
32-bit input word into `inputs_buffer` (`buffer_we_q=1` write path,
`inputs_buffer.sv:19` `unsigned_buffer[addr_q_i] <= input_i`) **also**
computes a real 4-lane MAC of that input against output-channel-0's weight
and accumulates it (`acc_e_n=1` unconditionally whenever
`opcode_i==MAC4_EXEC||MAC8_EXEC` and valid, `mac_controller.sv:60-70` — no
`fill_buffer_q` gating on `acc_e_n`). This is confirmed by
`NetworkPropagate.c`'s call structure (§4): `initInputBuffer()` is called
once per spatial position, `macsOnRangeCustom(..., output)` is then called
with `output==0` first (fill path) and `output==1..NB_OUTPUTS-1` after
(read path), and `weightedSum = readAccu(weightedSum)` reads back the
accumulator once per output channel including channel 0. So **no
instruction is wasted purely on data movement** — the fill pass does
double duty (stage + compute channel 0), and only channels 1..7 pay for
genuine "extra" instructions, and those are 8-wide.

**Reasoned inference (not directly provable from static RTL/C alone,
flagged as such):** given they issue *more* total instructions (146080 vs.
140084) yet finish in *fewer* cycles (221387 vs. 247849), the CPI gap
(≈1.52 vs. ≈1.77) most plausibly comes from **fewer address-generation
instructions interleaved between MACs**, not from doing more raw compute
per instruction (2x width alone would cut MAC-instruction count, not raise
total instruction count). Their `mac_controller.sv` auto-increments
`buffer_addr_q` in hardware every accepted `MAC4_EXEC`/`MAC8_EXEC`
(`mac_controller.sv:63-67`) — the software driving it (`mac_instr.h`'s
`mac8_buffer_unroll*` functions) never computes a buffer address; it just
loads two weight words per call and issues the op. Our `mac4_tiled`/
`mac4_load_stationary` calling convention (per the memory index's map #2/#3
work) still requires explicit `lw`s and pointer arithmetic around each MAC
call in `NetworkPropagate.c`. In an in-order 5-stage pipe, a `lw` that
feeds the very next MAC-issuing instruction's source register is exactly
the RAW-hazard shape that stalls; removing it from the instruction stream
(by pushing addressing into the coprocessor's own address counter) removes
that stall even though the *total instruction count* rises elsewhere (extra
`MAC_CTRL`/`readAccu()` calls — one pair per spatial position × output
channel, `NetworkPropagate.c` diff hunks at `initInputBuffer()`/
`readAccu()` call sites, §4). This explains "more instructions, fewer
cycles, lower CPI" self-consistently, but confirming it precisely would
need an objdump/waveform trace, which is out of scope here (no simulation
per the ticket's own scoping) — **flagged as inference, not verified.**

## 3. `inputs_buffer.sv`: a genuine stationary buffer, input-side, reused across output channels

Structurally it is a plain array (`logic [31:0] unsigned_buffer[99:0]`,
`inputs_buffer.sv:11`), not a FIFO: it is addressed randomly by
`addr_q_i`/`addr_i` (write/read pointers driven by `mac_controller.sv`, not
by an internal head/tail counter that consumes-on-read), and read access
returns two adjacent words packed as 64 bits (`{unsigned_buffer[addr_i+1],
unsigned_buffer[addr_i]}`, `inputs_buffer.sv:24`) — a value can be read
many times without being popped, exactly the "load once, reuse" semantics
of a stationary buffer.

Its role, confirmed by the `NetworkPropagate.c` call pattern (§2, §4): one
spatial position's input receptive field (`KERNEL_WIDTH * NB_CHANNELS` or
`NB_CHANNELS` elements, up to 100 words per `inputs_buffer.sv:11`'s
`[99:0]` sizing) is staged into the buffer **once** (during the `output==0`
fill pass), then re-read for every remaining output channel at that same
spatial position (`output==1..NB_OUTPUTS-1`, exec/read mode, weights change
per call but inputs come straight from the buffer). This is the *dual* of
our own weight-stationary reasoning in
`docs/research/weight-spatial-batching-tradeoff.md` §2.0: we found no
hardware-side stationary buffer on the weight operand (weights are always a
fresh register load, "weight-stationary" there had to mean
software/register reuse); CNN_C2 instead put the stationary buffer on the
**input** side and amortizes it across the **output-channel** axis, which
is the natural reuse axis for a fixed receptive field with per-channel
weights — the CNN structure itself (same input window, many output-channel
weight sets) maps directly onto "stage input once, stream weights through."
**Answer: yes, analogous to a stationary buffer, not a FIFO; reused across
`NB_OUTPUTS` output-channel calls per spatial position, not consumed once.**

## 4. Why only 27 lines of `NetworkPropagate.c` changed: one shared layer function, not four

Tracing the call sites in `NetworkPropagate.c` (via
`git -C cva6-softcore-contest-rsp show cv32a6_contest_25_26_CNN_C2:sw/app/mnist/NetworkPropagate.c`):

- `convcellPropagate1(...)` is called **twice** from `propagate()`: once for
  conv1 (`inputs=inputs, ..., weights=conv1_weights`, line ~438) and once
  for conv2 (`inputs=conv1_output, ..., weights=conv2_weights`, line
  ~468) — **the same generic convolution function serves both conv layers**,
  parameterized by the ~20 trailing template-like arguments (channel counts,
  strides, kernel size, memory offsets), not two hand-specialized loop nests.
- `fccellPropagateUDATA_T(...)` is called once, for fc1 (line ~501).
- `fccellPropagateDATA_T(...)` is called once, for fc2 (line ~532) — this
  function's body is **untouched** by the diff except two blank-line
  removals; it still calls the old scalar `macsOnRange` (lines 351, 368 in
  the post-diff file). **fc2 is not accelerated at all.**

So the 27 changed lines are: add `initInputBuffer()` once per `ox`/output
loop entry in `convcellPropagate1` and once per `och` loop entry in
`fccellPropagateUDATA_T`; rename the two `macsOnRange(...)` call sites in
each to `macsOnRangeCustom(..., output)`/`macsOnRangeCustom(..., och)`
(adding the output-channel index as a new parameter so the callee can
decide fill-vs-exec mode); and add one `weightedSum = readAccu(weightedSum)`
per output channel, right before saturation. Because `convcellPropagate1`
is shared, this single ~15-line edit accelerates **both conv1 and conv2
simultaneously** — three of the network's four compute layers (conv1,
conv2, fc1) go through the same small patch; only fc2 was left on the old
path. **This is the real explanation for "27 lines" vs. our multi-hundred
line per-layer restructuring: their base architecture already had one
generic conv function and one generic (unsigned) FC function shared across
layers, so the acceleration is a single change applied at a shared choke
point, not new per-layer loop restructuring like our B=4 weight-spatial
batching (which had to be hand-applied separately to conv1 and conv2
because our `NetworkPropagate.c` has separate, already-diverged loop
bodies per layer by the time MAC4 v2/tiling was added).** No iteration
logic was pushed into hardware beyond buffer-address auto-increment (§2);
the `oy`/`ox`/`sy`/`sx` spatial loops are untouched software loops exactly
as before — only the innermost per-output-channel MAC call changed.

## 5. Constraint compliance: register-only CV-X-IF confirmed, no other pipeline files touched

No memory-interface signal (`x_mem_req`/`x_mem_resp`/`mem_req`/`mem_resp`/
similar) appears anywhere in `mac_cfu.sv`, `mac_controller.sv`, or
`inputs_buffer.sv` — every port on all three modules is a plain
data/control signal (`clk_i`, `rst_ni`, `registers_i`, `opcode_i`, buffer
addresses, `accu`, etc.); `inputs_buffer.sv` is filled exclusively from
`registers_i` via `mac_cfu.sv`'s muxes (`mac_cfu.sv:90-93`), never from a
memory port. `cvxif_example_coprocessor.sv`'s CV-X-IF signal set is also
unchanged in kind — `compressed_req/resp`, `issue_req/resp`, `register`
(`cvxif_example_coprocessor.sv:38-48`) — with `mac_cfu_i` instantiated in
place of the old `copro_alu_i`, driven by `registers_i`/`opcode_i` and
producing `result_o`/`valid_o`/`we_o`, no new interface class added
(diff only: `+13/-13`, all within the coprocessor's internal
instantiation, `mac_cfu`-shaped rather than `copro_alu`-shaped, plus the
one added `issue_ready_i` port). This matches the map's premise exactly.

Diffstat (`git -C cva6-softcore-contest-rsp diff cv32a6_contest_25_26
cv32a6_contest_25_26_CNN_C2 --stat`) confirms the touched-file set is
exactly what the map claims and nothing more: `Makefile` (sim tooling),
`core/Flist.cva6` (file list), `core/cvxif_example/copro_alu.sv` (deleted),
`core/cvxif_example/cvxif_example_coprocessor.sv` (rewiring),
`core/cvxif_example/include/cvxif_instr_pkg.sv` (decode table),
`core/cvxif_example/mac/{inputs_buffer,mac,mac_cfu,mac_controller}.sv`
(new), `core/include/cv32a6_im_contest_config_pkg.sv` (the `CvxifEn`
toggle, `+1/-1`), `sw/app/mnist/NetworkPropagate.c` (+27), `sw/app/mnist/
mac_instr.h` (new), `wave.do` (sim tooling). **No CVA6 core pipeline file
(frontend, issue stage, ID stage, commit stage, LSU, etc.) outside
`core/cvxif_example/` and the one-line config toggle is touched.** **Answer:
confirmed on both counts — genuinely register-only, no other pipeline files
touched.**

## 6. Measurement methodology: their reported numbers are clean, no smoke-test overhead

`sw/app/mnist/main.c` on `cv32a6_contest_25_26_CNN_C2` times exactly:

```c
readStimulus(inputBuffer, expectedOutputBuffer);
instret = -read_csr(minstret);
cycles = -read_csr(mcycle);
const int success = processInput(inputBuffer, expectedOutputBuffer,
                                  predictedOutputBuffer, &output_value);
instret += read_csr(minstret);
cycles += read_csr(mcycle);
```

`processInput()` calls only `propagate(inputBuffer, predictedOutputBuffer,
output_value)` plus a trivial prediction-accuracy loop over
`OUTPUTS_SIZE[0]` (1 element for this network) — no coprocessor
self-test, no `mac4_smoke_test()`-equivalent call anywhere in `main.c`, and
`NetworkPropagate.c`'s `propagate()` body (traced via the call sites in
§4) contains only the four layer-propagation calls plus `saveOutputs()`/
`maxPropagate1()` bookkeeping already present pre-diff — nothing new was
added inside the timed window beyond the per-layer `initInputBuffer()`/
`readAccu()` calls that are themselves part of the real compute (§1-§3),
not instrumentation. **Answer: their reported 221387 cycles / 146080
instructions is a clean measurement, directly comparable in methodology to
our own clean 247849/140084 — no smoke-test stripping needed, and none
appears to have been present to strip. The raw comparison holds.**

## Summary

| | Ours (mac4 v2, `a3bb582`) | CNN_C2 (`8416217`) |
|---|---|---|
| Cycles (env0003, clean) | 247849 | 221387 |
| Instructions | 140084 | 146080 |
| CPI | ≈1.77 | ≈1.52 |
| MAC width/instr (exec mode) | 4 lanes (uint8×int8) | 8 lanes (int8×int8), 4-lane fallback |
| Compute pipeline | fully combinational (`mac4_alu.sv`) | 1 registered product stage (`mac.sv`), still 1 instr/cycle throughput |
| Stationary buffer axis | none on hardware side (weight reload every call); input side has our own separate stationary-buffer instructions | input-side buffer, reused across `NB_OUTPUTS` output channels per spatial position, hardware-managed address counter |
| Layers accelerated | conv1 (B=4 batched), conv2 (tiled), separately hand-restructured | conv1 + conv2 (one shared `convcellPropagate1`), fc1; fc2 unaccelerated |
| Software change footprint | multi-hundred-line per-layer restructuring | 27 lines, because the base function was already shared across conv1/conv2 |
| CV-X-IF register-only | yes | yes, confirmed |
| Other core files touched | none (besides config toggle) | none (besides config toggle) |
| Measurement methodology | clean (smoke tests stripped) | clean (none present) |

The headline takeaway: CNN_C2's cycle-count win is **not** explained by a
fundamentally different execution model (still 1 instruction/cycle
throughput, still register-only CV-X-IF, still a handful of MAC lanes per
instruction) — it comes from (a) a 2x-wider MAC datapath (8 lanes vs. our
4), (b) a hardware-managed input-stationary buffer that removes
address-generation instructions from the hot loop (likely the main CPI
driver, though this specific causal claim is inference, not verified by
waveform), and (c) a software architecture where one generic conv function
already covered both conv layers, so the acceleration patch needed to touch
only 27 lines instead of being re-derived per layer.
