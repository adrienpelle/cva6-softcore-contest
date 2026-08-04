# CV-X-IF issue/register/commit/result protocol in this CVA6 fork

Research for GitHub issue #5 (child of #1, "wayfinder" map). Goal: understand how
CVA6's CV-X-IF integration handles a coprocessor instruction that takes multiple
cycles, and whether flush/speculation can hit an in-flight coprocessor
instruction, before designing the mnist accelerator's internal pipeline and
hazard handling.

Method: read the actual RTL in this repo. No secondary sources were used — there
is no CV-X-IF write-up under `docs/` in this repo (checked `docs/03_cva6_design`,
`docs/04_cv32a6_design`; neither mentions CV-X-IF). All line numbers refer to the
files as they exist on branch `cv32a6_contest_25_26` at the time of writing
(commit `2b256e6`).

## 0. Config baseline: is CV-X-IF even on?

`core/include/cv32a6_im_contest_config_pkg.sv:21` — `localparam CVA6ConfigCvxifEn = 0;`,
wired into the config struct at line 101 (`CvxifEn: bit'(CVA6ConfigCvxifEn)`).

**CV-X-IF is currently disabled in the contest config.** Enabling it (`CvxifEn: 1`)
is a prerequisite for this effort and is presumably in scope of a different,
sibling ticket — flagging it here since every finding below is about the
*mechanism* that exists in the RTL, gated behind a parameter that isn't
currently flipped on.

## 1. Multi-cycle coprocessor instruction: the actual handshake

### 1.1 Issue interface (`cvxif_issue_register_commit_if_driver.sv`)

`core/cvxif_issue_register_commit_if_driver.sv:41-54`:
```
// X_ISSUE_REGISTER_SPLIT = 0 : Issue and register transactions are synchrone
assign register_valid_o  = issue_valid_o;
...
always_comb begin
  issue_valid_o       = valid_i && ~flush_i;
  issue_req_o.instr   = x_off_instr_i;
  issue_req_o.hartid  = hart_id_i;
  issue_req_o.id      = x_trans_id_i;
  register_o.rs       = register_i;
  register_o.rs_valid = rs_valid_i;
end
```
Issue and register (operand) transactions are always sent together, in the same
cycle (`X_ISSUE_REGISTER_SPLIT = 0`), i.e. the coprocessor gets the opcode and
operand values simultaneously — it never has to ask for register values later.
`issue_req_o.id` is set to `x_trans_id_i`, which is driven from
`issue_instr_i[0].trans_id` (`core/issue_read_operands.sv:236`) — i.e. the
**scoreboard slot index** of the instruction, not a coprocessor-chosen tag.

**Commit is fired unconditionally, in the same cycle as issue acceptance**, not
gated on the coprocessor producing a result:
`core/cvxif_issue_register_commit_if_driver.sv:56-64`:
```
/* WARNING */
// Always commit since speculation in execute in not possible : TODO to be verified

// Always do commit transaction with issue
// If instruction goes to execute then it is not speculative
assign commit_valid_o       = issue_valid_o && issue_ready_i;
assign commit_o.hartid      = issue_req_o.hartid;
assign commit_o.id          = issue_req_o.id;
assign commit_o.commit_kill = 1'b0;
```
`commit_kill` is hardwired to `0`. The CV-X-IF `x_commit` channel's whole
purpose in the spec is to let the core tell the coprocessor "this instruction
was spoken for too early / is being killed, don't bother" — this driver never
uses that (see §2). Note the comment is itself a "TODO to be verified" left by
the original CVA6/OpenHW authors — this is not settled/reviewed behavior, it's
an assumption baked into the driver.

### 1.2 Issue-side acceptance and stall (`issue_read_operands.sv`)

`core/issue_read_operands.sv:250-255`:
```
assign cvxif_req_allowed = (issue_instr_i[0].fu == CVXIF) && !stall_waw[0];
assign cvxif_instruction_valid = !issue_instr_i[0].ex.valid && issue_instr_valid_i[0] && cvxif_req_allowed;
assign x_transaction_accepted_o = x_issue_valid_o && x_issue_ready_i && x_issue_resp_i.accept;
assign x_transaction_rejected = x_issue_valid_o && x_issue_ready_i && ~x_issue_resp_i.accept;
assign x_issue_writeback_o = x_issue_resp_i.writeback;
assign x_id_o = x_issue_req_o.id;
```
`core/issue_read_operands.sv:964-967`:
```
issue_ack_o = issue_ack;
// Do not acknoledge the issued instruction if transaction is not completed.
if (issue_instr_i[0].fu == CVXIF && !(x_transaction_accepted_o || x_transaction_rejected)) begin
  issue_ack_o[0] = issue_instr_i[0].ex.valid && issue_instr_valid_i[0];
end
```
**Important nuance**: `issue_ack_o[0]` is only held low while the *issue
handshake itself* hasn't resolved (i.e. while waiting for `x_issue_ready_i` /
`x_issue_resp_i.accept`), not while waiting for the coprocessor's actual
*result*. The comment at `core/issue_read_operands.sv:286-288` says:

```
// CVXIF is always ready to try a new transaction on 1st issue port
// If a transaction is already pending then we stall until the transaction is done.(issue_ack_o[0] = 0)
// Since we can not have two CVXIF instruction on 1st issue port, CVXIF is always ready for the pending instruction.
```

but nothing in the FU-busy logic backs up "we stall until the transaction is
done" in the sense of *waiting for the result*. The `fus_busy` struct
(`core/issue_read_operands.sv:136` declares a `cvxif` bit) is only ever driven
for issue port 1 (`fus_busy[1].cvxif = 1'b1` at line 321, to forbid CVXIF on
the superscalar second port) — **`fus_busy[0].cvxif` is never set**, so
`fu_busy[0]` for a CVXIF op is always `0`. WAW hazard checking
(`stall_waw`, `core/issue_read_operands.sv:910-941`) only blocks issuing a new
instruction that targets the *same destination register* as a still-issued
(uncommitted) entry; it does not treat "CVXIF FU busy executing a previous op"
as a structural hazard at all.

**Conclusion: nothing in this core throttles back-to-back CVXIF issues based on
the coprocessor still being busy computing a previous result.** As long as
`x_issue_ready_i` stays asserted, the core will issue instruction N+1 to the
coprocessor on the very next eligible cycle after instruction N was accepted,
even if instruction N's `result_valid_i` hasn't fired yet. Throttling / limiting
the number of outstanding transactions is entirely the coprocessor's own
responsibility via `x_issue_ready_i` (standard CV-X-IF issue-side backpressure).
Multiple in-flight transactions must be told apart by `id` (=scoreboard trans
id, `TRANS_ID_BITS = clog2(NrScoreboardEntries)`, see §3) since `result_i.id`
in the result channel is how the FU disambiguates which physical/architectural
register a returning result belongs to (§1.3).

### 1.3 Result interface and writeback (`cvxif_fu.sv`)

`core/cvxif_fu.sv:53-63`:
```
assign result_ready_o = 1'b1;
assign x_ready_o = 1'b1; // Readyness of cvxif_fu is determined in issue stage by CVXIF issue interface
// Result signals
assign x_valid_o = x_illegal_i || result_valid_i;
assign x_result_o = result_i.data;
assign x_trans_id_o = x_illegal_i ? x_trans_id_i : result_i.id;
assign x_we_o = result_i.we;
assign x_rd_o = result_i.rd;
```
`cvxif_fu` is a pure combinational pass-through with **no internal state and no
timeout/latency logic of its own** — it does not track "instruction is in
flight." The multi-cycle latency of a coprocessor operation is handled entirely
outside this module, on the coprocessor side: the FU simply waits, combinationally
each cycle, for `result_valid_i` to go high with a matching `result_i.id`, and
forwards it that same cycle to the writeback bus. `result_ready_o` is hard-wired
to `1` — **the core can never backpressure the result channel**; the coprocessor
must present a result only when it actually has one, and CVA6 always drains it
immediately.

This result is wired straight into the global writeback bus:
`core/cva6.sv:788-791`:
```
assign trans_id_ex_id[X_WB] = x_trans_id_ex_id;
assign wbdata_ex_id[X_WB]   = x_result_ex_id;
assign ex_ex_ex_id[X_WB]    = x_exception_ex_id;
assign wt_valid_ex_id[X_WB] = x_valid_ex_id;
```
(`X_WB` = write-back port index 4, `core/include/ariane_pkg.sv:207-208`
— shared/mutually-exclusive with `ACC_WB`.)

And into the scoreboard: `core/scoreboard.sv:195-224`, gated only by
`mem_q[trans_id_i[i]].issued` (see §2 for why this matters):
```
for (int unsigned i = 0; i < CVA6Cfg.NrWbPorts; i++) begin
  if (wt_valid_i[i] && mem_q[trans_id_i[i]].issued) begin
    ...
    mem_n[trans_id_i[i]].sbe.result = wbdata_i[i];
    if (mem_n[trans_id_i[i]].sbe.fu == ariane_pkg::CVXIF) begin
      if (x_we_i) mem_n[trans_id_i[i]].sbe.rd = x_rd_i;
      else mem_n[trans_id_i[i]].sbe.rd = 5'b0;
    end
    if (ex_i[i].valid) mem_n[trans_id_i[i]].sbe.ex = ex_i[i];
    ...
  end
end
```
Only once this writeback fires does `mem_n[trans_id].sbe.valid` become `1`, which
is the sole gate that lets `commit_stage.sv` retire the instruction (see
`core/commit_stage.sv:156`, `375` — both check `commit_instr_i[0].valid`).
**So: however many cycles the coprocessor takes, the instruction simply sits at
its scoreboard slot, un-committable, until `result_valid_i` fires for its
`id`.** There is no timeout. The core will stall commit (and everything behind
it in program order, since `NrCommitPorts = 1` and commit is strictly
in-order) indefinitely if the coprocessor never asserts `result_valid_i`.

### Summary answer to Q1

- Issue and operand-register delivery happen together, one cycle, no split
  (`X_ISSUE_REGISTER_SPLIT = 0`).
- Commit-channel handshake fires immediately upon issue acceptance — it is
  **not** gated on the result being ready; per the CV-X-IF spec this "early
  commit" is legal for a non-speculative core and is what lets the coprocessor
  run for many cycles independently.
- The instruction's `id` (issue interface `id` field) equals its **CVA6
  scoreboard transaction id**, not a coprocessor-invented tag.
- Multi-cycle latency is realized purely by delaying `result_valid_i` — nothing
  in `cvxif_fu.sv`/`issue_read_operands.sv` imposes or expects any particular
  latency; the interface is fully latency-agnostic on the result side.
- Nothing structurally prevents the core issuing a *second* CVXIF instruction
  to the coprocessor before the first one's result has come back (see §1.2)
  — this is a real hazard the coprocessor design must consider unless it
  deasserts `x_issue_ready_i` while busy.

## 2. Flush / speculation while a CV-X-IF instruction is in flight

### 2.1 No flush signal reaches the coprocessor once issued

`cvxif_fu.sv` (§1.3) has **no `flush_i` port at all** — confirmed by its full
port list, `core/cvxif_fu.sv:20-51`. The only place `flush_i` appears in the
CV-X-IF driver chain is in `cvxif_issue_register_commit_if_driver.sv:48`
(`issue_valid_o = valid_i && ~flush_i;`), which only suppresses issuing a *new*
transaction on the cycle flush is asserted — it cannot retroactively cancel an
already-accepted transaction. And as shown in §1.1, `commit_o.commit_kill` is
hardwired to `1'b0`, so the one CV-X-IF-standard mechanism designed for exactly
this ("tell the coprocessor the result of transaction X is no longer wanted")
is never used by this driver.

**Conclusion: there is no way, via the CV-X-IF interface as wired in this
core, to tell the coprocessor "abandon what you're computing."** Once issued
and accepted, the coprocessor must eventually assert `result_valid_i` for that
`id`, or the core stalls forever (§1.3).

### 2.2 Can a flush actually happen while a CVXIF instruction is in flight?

Two different flush signals exist, generated in `core/controller.sv`:

- `flush_unissued_instr_o` — asserted alone on a branch mispredict
  (`core/controller.sv:108-113`, `resolved_branch_i.is_mispredict`). Wired to
  `issue_stage.flush_unissued_instr_i` (`core/cva6.sv:825`), which only affects
  instructions **not yet issued** into the scoreboard
  (`core/scoreboard.sv:171`: `if (decoded_instr_valid_i[i] && decoded_instr_ack_o[i] && !flush_unissued_instr_i)`).
  It does **not** touch already-issued scoreboard entries, so a CVXIF
  instruction already dispatched to the coprocessor is unaffected by a branch
  misprediction.

- `flush_id_o` (full scoreboard flush) — asserted together with
  `flush_unissued_instr_o` on: `fence`/`fence.i`/`sfence.vma`/`hfence.*`
  (`core/controller.sv:118-200`), CSR/accelerator side effects and AMO flush
  (`core/controller.sv:206-218`), and **exceptions / `eret` / debug entry**
  (`core/controller.sv:224-231`, `if (ex_valid_i || eret_i || ...)`). Wired to
  `issue_stage.flush_i` → `scoreboard.flush_i` (`core/cva6.sv:826`,
  `core/issue_stage.sv:207`). When this fires, `core/scoreboard.sv:254-262`
  unconditionally clears **every** scoreboard entry's `issued` bit, regardless
  of whether that entry's result has arrived:
  ```
  if (flush_i) begin
    for (int unsigned i = 0; i < CVA6Cfg.NR_SB_ENTRIES; i++) begin
      mem_n[i].issued       = 1'b0;
      mem_n[i].cancelled    = 1'b0;
      mem_n[i].sbe.valid    = 1'b0;
      mem_n[i].sbe.ex.valid = 1'b0;
    end
  end
  ```

Whether this full flush can actually catch an **in-flight, not-yet-returned**
CVXIF instruction depends on whether the triggering event (exception/CSR/fence)
can be recognized while a CVXIF instruction is still executing further back in
program order. Tracing exception recognition:

- `commit_stage.sv:156` (commit-ack logic) and `commit_stage.sv:374-399`
  (exception/interrupt logic, which is also where a pending interrupt merged
  in via `csr_exception_i` is recognized) are both gated on
  `commit_instr_i[0].valid`. This is the scoreboard's head-of-queue entry: it
  only becomes valid once its writeback has landed (§1.3). So a CVXIF
  instruction *itself*, sitting at the commit head, blocks all exception/
  interrupt recognition until its own result arrives — the core can't flush
  because of it before it completes.
- But `NrScoreboardEntries = 4` (`core/include/cv32a6_im_contest_config_pkg.sv:50`,
  `core/include/build_config_pkg.sv:80`) means the scoreboard holds several
  in-flight instructions at once, issued ahead of commit. It is entirely
  possible for an **older** instruction (already at/near the commit head) to
  fault/trap or for a fence/CSR side-effect instruction ahead of the CVXIF op
  to reach commit, while a **younger** CVXIF instruction has already been
  issued to and accepted by the coprocessor and is still waiting on
  `result_valid_i`. In that scenario `flush_id_o` fires based on the *older*
  instruction, and per `core/scoreboard.sv:254-262` above, it clears the
  younger CVXIF instruction's scoreboard slot (`issued = 0`) too — while the
  coprocessor keeps computing, oblivious, with no way to be told.

  When the coprocessor's stale result eventually arrives (`result_valid_i` +
  matching `id`), the writeback gate `mem_q[trans_id_i[i]].issued`
  (`core/scoreboard.sv:198`) is what decides its fate:
  - If that scoreboard slot has not yet been reused by a new instruction, the
    stale write is a no-op (the check requires `.issued`, which was cleared).
  - **If a new instruction has since been issued into that same slot** (slots
    are reused: `issue_pointer_n` resets to `0` on flush,
    `core/scoreboard.sv:279`, and `TRANS_ID_BITS = clog2(4) = 2` gives only 4
    distinct ids to cycle through, `core/include/build_config_pkg.sv:81`),
    `mem_q[trans_id_i[i]].issued` will be `1` again for the *new*
    instruction, and the stale coprocessor result will be written into the
    wrong (unrelated) instruction's destination register / exception field.

  This is a real, RTL-traceable hazard window, not a hypothetical — it is the
  direct consequence of (a) commit-channel firing without waiting for the
  result (§1.1, itself flagged "TODO to be verified" by the original authors)
  combined with (b) no kill signal ever reaching the coprocessor (§2.1) and
  (c) small, reused transaction-id space. It has not been confirmed against a
  simulation/waveform in this pass — only traced statically through the RTL —
  so treat it as "very likely exploitable" rather than "proven," but it is
  consistent with the driver's own "TODO to be verified" comment.

### Summary answer to Q2

- **No**, the interface as wired in this core does **not** give the
  coprocessor any flush/kill signal for an in-flight transaction.
  `commit_o.commit_kill` exists in the protocol (`core/include/cvxif_pkg.sv:49-52`)
  but is hardwired `0` (`core/cvxif_issue_register_commit_if_driver.sv:64`);
  `cvxif_fu.sv` has no `flush_i` input at all.
- **Yes**, a full-pipeline flush (exception, `eret`, debug entry, fence,
  `sfence.vma`, CSR/AMO side effects) can occur while a CVXIF instruction is
  still executing in the coprocessor, whenever the flush is triggered by an
  *older* instruction than the CVXIF op. A branch misprediction alone
  (`flush_unissued_instr_o` only) cannot — it doesn't touch already-issued
  scoreboard entries.
- Practical implication for the accelerator design: **the coprocessor must
  always complete and return a result for every issued transaction — there is
  no abort path** — and the design should treat "stale result written to a
  reused/wrong destination after an intervening flush" as a hazard to actively
  avoid (e.g. by keeping internal latency short/bounded, or by having the
  accelerator's own logic double check hart/id plausibility — CVA6 itself
  provides no help here).

## 3. Is this config in-order / single-issue on this path?

`core/include/cv32a6_im_contest_config_pkg.sv:78-79`:
```
SuperscalarEn: bit'(0),
NrCommitPorts: unsigned'(1),
```
Derived fields in `core/include/build_config_pkg.sv`:
- `cfg.NrIssuePorts = unsigned'(CVA6Cfg.SuperscalarEn ? 2 : 1);` (line 50) → **1**
- `cfg.SpeculativeSb = CVA6Cfg.SuperscalarEn;` (line 51) → **0** (no speculative
  scoreboard behavior; the `bmiss`/cancel path in `scoreboard.sv:230-236` is
  dead code for this config)
- `cfg.NR_SB_ENTRIES = CVA6Cfg.NrScoreboardEntries;` (line 80) → **4**
  (`CVA6ConfigNrScoreboardEntries = 4`, `cv32a6_im_contest_config_pkg.sv:50`)
- `cfg.TRANS_ID_BITS = $clog2(CVA6Cfg.NrScoreboardEntries);` (line 81) → **2**
- `cfg.NrWbPorts = (CVA6Cfg.CvxifEn || EnableAccelerator) ? 5 : 4;` (line 22)
  → **5** once `CvxifEn` is turned on (adds the shared `X_WB`/`ACC_WB` port,
  index 4, `core/include/ariane_pkg.sv:207-208`)

So: **single issue port, single commit port, no superscalar dispatch, no
speculative scoreboard.** Commit is strictly in program order
(`commit_pointer_q` advances by exactly `commit_ack_i[0]` each cycle,
`core/scoreboard.sv:269-275`; the whole `NrCommitPorts == 2` branch is unused
here).

What is **not** strictly in-order: *issue*. The scoreboard is a small
(4-entry) reorder-ish structure that lets instructions be issued to their FUs
out of program order relative to each other's *completion* — a fixed-latency
ALU op issued after a CVXIF op can complete and even (per the writeback gate
in `core/scoreboard.sv:198`, which only checks `.issued`, not program order)
write back before the CVXIF op's result arrives. But **issue itself is
in-order** (`decoded_instr_i` is a straight FIFO fed from decode, one slot per
cycle since `NrIssuePorts = 1`), and **commit is strictly in-order** and
blocks on scoreboard-entry validity — so from the coprocessor's perspective:

- Instructions are offered to it on the issue interface in program order
  (single issue port; there is no reordering of *which* instruction reaches
  the CVXIF issue check first).
- Only one CVXIF instruction is issue-checked against WAW at a time
  (`stall_waw[0]`), but as shown in §1.2 nothing prevents a *second* CVXIF
  instruction being issued before the first's result returns, i.e. up to
  several CVXIF ops can be simultaneously in flight in the coprocessor
  (bounded by `NR_SB_ENTRIES = 4` and by however many the coprocessor's own
  `x_issue_ready_i` allows).
- Results can be consumed in any order relative to issue order (the
  scoreboard writeback path is keyed purely by `trans_id`/`id`, not FIFO
  order) — but commit still only retires in program order, so out-of-order
  *completion* by the coprocessor is safe as long as ids are correct; it just
  means an early-finishing younger instruction's result sits parked in its
  scoreboard slot until older instructions ahead of it commit.

### Summary answer to Q3

Yes, for this contest config: **single issue port, single commit port,
in-order issue, strictly in-order commit, no speculative scoreboard**
(`NrIssuePorts=1`, `NrCommitPorts=1`, `SuperscalarEn=0`, `SpeculativeSb=0`).
This means:
- No reordering to worry about *among instructions offered to the
  coprocessor* — they arrive at the issue interface in program order, one at
  a time.
- But it is **not safe to assume only one CVXIF instruction is outstanding at
  a time** — the core does not enforce that (§1.2). If the accelerator's
  internal pipeline can only usefully hold one operation, it must say so
  itself via `x_issue_ready_i` (deassert while busy); if it can pipeline
  multiple operations, it must track each by the `id` field (= scoreboard
  trans id, range `0..3`, `TRANS_ID_BITS=2`) since results can be returned
  out of issue order.
- Hazard handling against flush/speculation cannot be delegated to the core —
  see §2: the core provides no kill signal, so the coprocessor design should
  assume every issued instruction will run to completion, and should be aware
  that (per the identified but unconfirmed hazard in §2.2) a stale result
  returned after an intervening flush could, in principle, be mis-attributed
  by the core's scoreboard to a different, later instruction reusing the same
  4-entry id space. Minimizing/bounding internal latency reduces the size of
  that window.

## Files read (primary sources)

- `core/cvxif_issue_register_commit_if_driver.sv`
- `core/cvxif_fu.sv`
- `core/issue_stage.sv`
- `core/issue_read_operands.sv`
- `core/scoreboard.sv`
- `core/ex_stage.sv`
- `core/cva6.sv`
- `core/controller.sv`
- `core/commit_stage.sv`
- `core/include/cvxif_pkg.sv`
- `core/include/cv32a6_im_contest_config_pkg.sv`
- `core/include/build_config_pkg.sv`
- `core/include/config_pkg.sv`
- `core/include/ariane_pkg.sv`
- `core/cvxif_example/cvxif_example_coprocessor.sv` (reference example only;
  it is a single-cycle-ish ALU and does not demonstrate multi-cycle latency
  handling, so it was not used as a design template here)

No secondary sources (official CV-X-IF spec doc, etc.) were consulted — none
exist under `docs/` in this repo for CV-X-IF.
