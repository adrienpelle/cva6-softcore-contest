# CV-X-IF `x_mem` wiring feasibility on this CVA6 fork

Research for GitHub issue #19 (child of #17, "beyond MAC4" map, itself the
successor to the closed wayfinder map issue #1). Question: what would it take
to wire the CV-X-IF memory extension (`x_mem_req_t`/`x_mem_resp_t`) so a
coprocessor can issue its own memory accesses instead of being fed operands
through the 2-port GPR issue/register interface, and is that compatible with
this fork's existing issue/commit protocol and the parent map's stated
constraint that "budget for invasiveness" stays on the CV-X-IF coprocessor
side, not the CV32A6 core (pipeline/front-end/multi-issue/LSU) itself.

Method: read the actual RTL in this repo. All line numbers refer to the files
as they exist on branch `cv32a6_contest_25_26` at commit `f02aa15` (the tip at
the time of writing). One external source was consulted for the CV-X-IF
memory-channel protocol semantics themselves, since `docs/` in this repo has
no CV-X-IF write-up (confirmed by prior research,
`docs/research/cvxif-issue-commit-protocol.md`): the upstream
[openhwgroup/core-v-xif](https://github.com/openhwgroup/core-v-xif) spec
(`docs/source/x_ext.rst`), quoted where used and clearly marked as spec text,
not repo RTL.

## 0. Scope caveat carried over from prior tickets

`core/include/cv32a6_im_contest_config_pkg.sv:21` still has
`localparam CVA6ConfigCvxifEn = 0;` — CV-X-IF is disabled in the contest
config used for `sw/app/mnist`. Everything below characterizes the mechanism
that exists (or doesn't) in the RTL, not a currently-active data path in the
contest build.

Separately and importantly for §2: `core/include/cv64a6_imafdc_sv39_config_pkg.sv:21`
and `core/include/cv64a6_imafdc_sv39_wb_config_pkg.sv:21` both set
`CVA6ConfigCvxifEn = 1`. Those two configs are the ones exercised by
`.github/workflows/ci.yml:16` (`target: [cv64a6_imafdc_sv39, cv64a6_imafdc_sv39_wb]`).
So CV-X-IF's issue/register/commit/result path is not just "shared but
dormant" code — it is **live and CI-tested** on those configs, even though it
is inert on `cv32a6_im_contest`. Any change to the CV-X-IF files touched below
is a change to something CI actually runs.

## 1. Interface gap

### 1.1 The `x_mem_*` structs exist, but are effectively dead in two ways

`core/include/cvxif_pkg.sv:54-68` defines the memory-channel structs:
```
typedef struct packed {
  logic [X_ID_WIDTH-1:0]  id;
  logic [31:0]            addr;
  logic [1:0]             mode;
  logic                   we;
  logic [1:0]             size;
  logic [X_MEM_WIDTH-1:0] wdata;
  logic                   last;
  logic                   spec;
} x_mem_req_t;

typedef struct packed {
  logic       exc;
  logic [5:0] exccode;
} x_mem_resp_t;

typedef struct packed {
  logic [X_ID_WIDTH-1:0]  id;
  logic [X_MEM_WIDTH-1:0] rdata;
  logic                   err;
} x_mem_result_t;
```
with `X_MEM_WIDTH = 64` (`cvxif_pkg.sv:17`) — note this is **twice XLEN** for
this 32-bit core (`X_DATAWIDTH = riscv::XLEN` = 32, `cvxif_pkg.sv:14,18-19`),
i.e. even this dead definition already models a wider-than-register memory
beat, which is directionally what "more input-data reuse" wants.

`cvxif_pkg.sv:85-108` also defines a `cvxif_req_t`/`cvxif_resp_t` pair that
*includes* `x_mem_valid`/`x_mem_req`/`x_mem_ready`/`x_mem_resp`/
`x_mem_result_valid`/`x_mem_result` fields (lines 92-95, 104-105).

**But these are not the types actually wired anywhere.** The real
`cvxif_req_t`/`cvxif_resp_t` used at every RTL instantiation site are built by
a *different* pair of macros, `` `CVXIF_REQ_T `` / `` `CVXIF_RESP_T ``,
defined in `core/include/cvxif_types.svh:51-71`:
```
`define CVXIF_REQ_T(Cfg, x_compressed_req_t, x_issue_req_t, x_register_req_t, x_commit_t) struct packed { \
    logic              compressed_valid; \
    x_compressed_req_t compressed_req; \
    logic              issue_valid; \
    x_issue_req_t      issue_req; \
    logic              register_valid; \
    x_register_t       register; \
    logic              commit_valid; \
    x_commit_t         commit; \
    logic              result_ready; \
}
`define CVXIF_RESP_T(Cfg, x_compressed_resp_t, x_issue_resp_t, x_result_t) struct packed { \
    logic               compressed_ready; \
    x_compressed_resp_t compressed_resp; \
    logic               issue_ready; \
    x_issue_resp_t      issue_resp; \
    logic               register_ready; \
    logic               result_valid; \
    x_result_t          result; \
}
```
**Zero mem fields.** These macros are what actually gets bound to the
`cvxif_req_t`/`cvxif_resp_t` parameter types at both instantiation points in
this repo: `core/cva6.sv:301,303` and `corev_apu/src/ariane.sv:37-38`
(`` `CVXIF_REQ_T(CVA6Cfg, ...) ``/`` `CVXIF_RESP_T(CVA6Cfg, ...) ``). The
`cvxif_pkg::cvxif_req_t`/`cvxif_resp_t` struct with `x_mem_*` fields
(`cvxif_pkg.sv:85-108`) is referenced exactly once in the whole repository,
in `verif/tb/uvmt/cva6_tb_wrapper.sv:50-51` (a UVM testbench wrapper, part of
the OpenHW verification environment, not the `make sim`/`make benchmark` path
this project actually builds and simulates through — confirmed by
`docs/research/cvxif-build-path.md` §1.2, which traces the real sim flow
through `corev_apu/tb/ariane_testharness.sv`, unrelated to `verif/tb/uvmt`).

So: **the x_mem struct definitions in `cvxif_pkg.sv` are not just unwired,
they are shadowed by a second, independent type-generation path
(`cvxif_types.svh`) that never had mem fields added to it in the first
place.** Wiring x_mem is not "flip a field on that's already plumbed through
to the port" — it requires extending the `CVXIF_REQ_T`/`CVXIF_RESP_T` macros
themselves (or introducing new ports alongside them), which is a change to
the type definitions consumed by both `core/cva6.sv` and
`corev_apu/src/ariane.sv`, i.e. by both entry points into this core.

### 1.2 `cvxif_fu.sv` has no memory port at all

`core/cvxif_fu.sv`'s full port list (`cvxif_fu.sv:20-51`) has exactly three
groups of signals: the issue-side scalar signals (`x_valid_i`,
`x_trans_id_i`, `x_illegal_i`, `x_off_instr_i` in; `x_ready_o`,
`x_trans_id_o`, `x_exception_o`, `x_result_o`, `x_valid_o`, `x_we_o`,
`x_rd_o` out), and the result-channel pair (`result_valid_i`, `result_i`,
`result_ready_o`). **No `x_mem_valid`/`x_mem_ready`/`x_mem_req`/`x_mem_resp`/
`x_mem_result` port exists.** The module body (`cvxif_fu.sv:55-71`) is a pure
combinational pass-through with no internal state:
```
assign result_ready_o = 1'b1;
assign x_ready_o = 1'b1;
assign x_valid_o = x_illegal_i || result_valid_i;
assign x_result_o = result_i.data;
assign x_trans_id_o = x_illegal_i ? x_trans_id_i : result_i.id;
assign x_we_o = result_i.we;
assign x_rd_o = result_i.rd;
```
There is no logic anywhere in this module (or in `cva6.sv`'s
`gen_cvxif_input_assignement`/`gen_cvxif_output_assignement` blocks,
`core/cva6.sv:767-800`, which populate every field of `cvxif_req`/read every
field of `cvxif_resp_i` that the macro-generated struct actually has) that
touches a memory channel, because — per §1.1 — the struct doesn't have one.

### 1.3 The LSU has no spare port and no coprocessor-facing interface

`core/load_store_unit.sv` accepts exactly **one** functional-unit input per
cycle: `fu_data_i` (`load_store_unit.sv:45`, single `fu_data_t`, not an
array) gated by a single `lsu_valid_i`/`lsu_ready_o` handshake
(`load_store_unit.sv:46-49`), and it internally decides load-vs-store from
`lsu_ctrl.fu` (`load_store_unit.sv:525-550`). Results come back on exactly
one load port (`load_trans_id_o`/`load_result_o`/`load_valid_o`/
`load_exception_o`, `load_store_unit.sv:52-58`) and one store port
(`store_trans_id_o`/.../`store_exception_o`, `load_store_unit.sv:61-67`).
There is no third "coprocessor" request/response port on this module's
boundary at all — a coprocessor cannot address the LSU without either
multiplexing onto the existing single `fu_data_i` slot (stealing core-load/
store issue bandwidth) or adding a wholly new port group to this module.

Downstream, the D$ request array is a hardwired 3-entry vector,
`dcache_req_ports_i`/`dcache_req_ports_o` are declared `[2:0]`
(`load_store_unit.sv:138,140`) and each of the 3 slots is already claimed:
port `[0]` goes to the MMU/PTW (`load_store_unit.sv:309-310`), port `[1]` to
the load unit (`load_store_unit.sv:480-481`), port `[2]` to the store unit
(`load_store_unit.sv:439-441`). **There is no 4th, spare port at the LSU
boundary for the contest's cache subsystem as configured today** — see §1.4
for how a 4th requester slot *does* exist one level up, in `cva6.sv`, but is
currently claimed by a different, mutually-exclusive interface.

### 1.4 A memory-capable coprocessor interface already exists in this codebase — for the *other*, mutually-exclusive accelerator port, not CV-X-IF

This is the most load-bearing finding for the effort estimate in §2.
CVA6 has a second, older, generic accelerator interface (`ACC`, from ETH
Zurich, predating CV-X-IF) implemented by `core/acc_dispatcher.sv`, gated by
`config_pkg`'s `EnableAccelerator` field. `core/cva6.sv:802-804`:
```
if (CVA6Cfg.CvxifEn && CVA6Cfg.EnableAccelerator) begin : gen_err_xif_and_acc
  $error("X-interface and accelerator port cannot be enabled at the same time.");
end
```
**`CvxifEn` and `EnableAccelerator` are mutually exclusive on this core.**
`acc_dispatcher.sv` *does* get two dedicated D$ request ports, arbitrated one
level up in `cva6.sv` on top of the LSU's own 3: `core/cva6.sv:629-634`
declares `dcache_req_i_t [1:0] dcache_req_ports_acc_cache;` alongside the
LSU's `[2:0] dcache_req_ports_ex_cache`, and `core/cva6.sv:1283-1318` merges
them into a 4-wide `dcache_req_to_cache`/`dcache_req_from_cache` bus that
actually reaches the cache subsystem (`NumPorts=4` on
`core/cache_subsystem/cva6_hpdcache_subsystem.sv:28`, matching
`core/cva6.sv:1296-1297`'s comment "Acc dispatcher and store buffer share a
dcache request port"). `acc_dispatcher.sv:99-100` exposes these as
`acc_dcache_req_ports_o`/`acc_dcache_req_ports_i`.

So the *arbitration/port-count* half of "give a coprocessor its own memory
port" is precedented in this exact codebase, just not for CV-X-IF. But two
things temper how much that precedent actually saves:

1. **Even this precedent doesn't wire real traffic.**
   `acc_dispatcher.sv:458`: `assign acc_dcache_req_ports_o = '0;` — the ACC
   interface's own memory-request output is tied to zero in this file. The
   port exists at the cva6.sv/cache-subsystem boundary, but nothing inside
   `acc_dispatcher.sv` ever drives a real `dcache_req_i_t` onto it. There is
   no worked example anywhere in this repo of a coprocessor actually issuing
   a load/store through these ports.
2. **What *is* implemented in `acc_dispatcher.sv` is the hazard-tracking
   scaffolding around such accesses**, not the accesses themselves: a
   non-speculative gate that only lets an ACC instruction fire once it's
   reached the top of the scoreboard (`insn_pending_q`/`insn_ready_q`,
   `acc_dispatcher.sv:195-223`), and four saturating counters tracking
   speculative-vs-dispatched loads and stores in flight
   (`acc_dispatcher.sv:361-449`) used to stall the *core's own* scalar
   loads/stores against them (`acc_dispatcher.sv:133-151`,
   `stall_issue`/`ACC_OP_LOAD`/`ACC_OP_STORE` cases) and to hold off
   `eret`/fence-class commits until posted accelerator stores drain
   (`acc_dispatcher.sv:347-355`, `wait_acc_store_q`/`ctrl_halt_o`). That
   scaffolding is what a real x_mem implementation would need to reproduce
   for CV-X-IF — see §2 and §4.

### 1.5 Existing register-only coprocessor pattern, for contrast

`core/mac4_copro/mac4_coprocessor.sv` (which "[r]eplaces
`core/cvxif_example/cvxif_example_coprocessor.sv` wholesale", per its own
header comment, lines 1-5) and the template it replaced both only ever touch
`cvxif_req_i.issue_req`, `.issue_valid`, `.register`, `.register_valid`, and
produce `cvxif_resp_o.result*`/`.issue_ready`/`.issue_resp`/
`.compressed_ready`/`.compressed_resp` (`mac4_coprocessor.sv:60-132`) —
exactly the fields the `` `CVXIF_REQ_T ``/`` `CVXIF_RESP_T `` macros define
(§1.1) and nothing else. Both are single always_comb/small-FSM modules with
no memory-request state machine of any kind: all of their "memory access" is
implicitly done for them, upstream, by the core's own register file read
(`register_i`/`rs_valid_i` in
`core/cvxif_issue_register_commit_if_driver.sv:38-39`, fed by the regular
GPR read ports) before the operands ever reach the coprocessor. This is the
"in, compute, out" pattern the whole coprocessor perimeter (issue driver,
`cvxif_fu.sv`, the coprocessor module itself) is built around; nothing in
that perimeter currently models a coprocessor that needs multiple cycles
*before* it can even start computing because it's still waiting on its own
memory request.

### Summary answer to Q1

To route a coprocessor-initiated memory request from the coprocessor through
`cvxif_fu.sv` into the LSU and back, the following would need to be added,
none of which exist today:
- New `x_mem_req`/`x_mem_resp`/`x_mem_result` fields on the *actual* wire
  type, i.e. changes to `` `CVXIF_REQ_T ``/`` `CVXIF_RESP_T `` in
  `cvxif_types.svh` (shared by both `core/cva6.sv` and
  `corev_apu/src/ariane.sv`) — the existing `cvxif_pkg.sv` struct fields
  cannot just be "turned on," they're not in the type that's actually
  instantiated.
- New ports and logic on `cvxif_fu.sv` to carry those fields between the
  `cvxif_req_i`/`cvxif_resp_o` bus and... something. `cvxif_fu.sv` is
  presently purely combinational and stateless; a coprocessor mem request
  that must round-trip through the LSU over multiple cycles either needs
  `cvxif_fu.sv` to grow state, or (more likely, following the `acc_dispatcher`
  precedent) the memory-request plumbing bypasses `cvxif_fu.sv` entirely and
  goes coprocessor → new dedicated D$ port(s), mirroring
  `acc_dispatcher.sv`'s pattern rather than routing through the existing
  "issue → register → result" scalar path at all.
- A new D$ requester port. `load_store_unit.sv`'s own 3 ports (`[2:0]`) are
  fully claimed (MMU/PTW, load, store — §1.3); the one existing precedent for
  a coprocessor-owned 4th port (`acc_dispatcher.sv`'s two `acc_dcache_req_ports`)
  is wired up one level higher, in `cva6.sv`'s arbitration
  (`core/cva6.sv:1283-1318`), and is currently occupied by, and mutually
  exclusive with, CV-X-IF (`core/cva6.sv:802-804`).
- Hazard-tracking scaffolding equivalent to `acc_dispatcher.sv`'s
  non-speculative gate and load/store-pending counters (§1.4 point 2, detailed
  further in §4), since nothing on the CV-X-IF side of this core currently
  tracks "is this in-flight coprocessor operation still safe to have side
  effects for."

## 2. Effort/risk estimate

### 2.1 Files that would need to change

Based on §1, at minimum:
- `core/include/cvxif_types.svh` — extend `` `CVXIF_REQ_T ``/`` `CVXIF_RESP_T ``
  macros (or add a parallel x_mem-specific struct/port group) to actually
  carry mem request/response/result fields. **Shared by both configs that use
  CV-X-IF** (§0) — this is not a coprocessor-perimeter-only file.
- `core/cvxif_pkg.sv` — if kept as the source of the mem struct field
  *types* (`x_mem_req_t` etc. already exist there, §1.1), at minimum needs to
  stop being dead code and get actually referenced by `cvxif_types.svh`.
- `core/cvxif_fu.sv` — new ports/logic to carry mem-channel signals between
  the top-level bus and whatever handles them (itself, or pass-through to a
  new module).
- `core/cva6.sv` — the `` `CVXIF_REQ_T ``/`` `CVXIF_RESP_T `` type
  instantiation (`cva6.sv:301,303`), the `gen_cvxif_input_assignement`/
  `gen_cvxif_output_assignement` blocks (`cva6.sv:767-800`) that would need
  new mem-field assignments, and — per the `acc_dispatcher` precedent — new
  D$ port declarations/arbitration analogous to `cva6.sv:629-634` and
  `cva6.sv:1283-1318` if a dedicated coprocessor D$ port is the chosen design
  (almost certainly required, given §1.3's finding that the LSU's existing 3
  ports are all claimed).
- `corev_apu/src/ariane.sv` — same `` `CVXIF_REQ_T ``/`` `CVXIF_RESP_T ``
  instantiation exists here too (`ariane.sv:37-38`); this is the top-level
  wrapper used by the FPGA/synthesis flow (per
  `docs/research/cvxif-build-path.md` §1.3), so it needs the same update to
  stay consistent, or the two entry points diverge on the CV-X-IF type shape.
- `core/load_store_unit.sv` and/or `core/cache_subsystem/cva6_hpdcache_subsystem.sv`
  — only if the design routes coprocessor memory traffic *through* the LSU's
  existing 3 D$ ports rather than adding new ones at the `cva6.sv` level; even
  the "add new ports at cva6.sv level, mirroring acc_dispatcher" option still
  touches `cva6_hpdcache_subsystem.sv`'s `NumPorts` parameterization
  (`cva6_hpdcache_subsystem.sv:28`, currently fixed at 4 and already fully
  allocated between LSU (3) and ACC (1 pair sharing 1 arbitrated slot, per
  `cva6.sv:1295-1297`) — CV-X-IF would need this bumped to 5, or would need to
  fight ACC for its existing slot, which conflicts with §1.4's mutual-exclusion
  finding: since `CvxifEn` and `EnableAccelerator` are mutually exclusive
  today, x_mem *could* legitimately repurpose the ACC path's port
  infrastructure while `EnableAccelerator=0`, but that means writing new logic
  gated on `CvxifEn` into what is currently `EnableAccelerator`-only wiring in
  `cva6.sv`, i.e. touching shared arbitration code either way.
- A new coprocessor-side module (e.g. `core/mac4_copro/` growing a memory
  request state machine, or a new sibling module) — this part *is*
  legitimately coprocessor-perimeter-only.
- Non-speculative issue gating / outstanding-request tracking (§1.4 point 2,
  §4) — where this logic lives depends on the design, but the only existing
  precedent (`acc_dispatcher.sv`) implements it as part of the
  core-side dispatcher module, not inside the coprocessor.

**Rough count: 5-7 core/-tree files touched** (`cvxif_types.svh`,
`cvxif_pkg.sv`, `cvxif_fu.sv`, `cva6.sv`, `ariane.sv`, plus LSU/cache-subsystem
port-count changes, plus a new/extended coprocessor module), before any
verification-environment updates (`verif/tb/uvmt/cva6_tb_wrapper.sv` already
references the old `cvxif_pkg::cvxif_req_t`/`resp_t` shape and would need
reconciling too, though as established in §1.1 it isn't part of this
project's actual sim path).

### 2.2 Isolated to the coprocessor perimeter, or shared-core-touching?

**Not isolated.** Three separate findings all point the same direction:
- `cvxif_types.svh`'s macros are consumed identically by both `cva6.sv` and
  `ariane.sv` (§1.1, §2.1) — there is exactly one copy of the CV-X-IF wire
  type in this repo, shared by every config that turns `CvxifEn` on,
  including the CI-tested `cv64a6_imafdc_sv39`/`cv64a6_imafdc_sv39_wb`
  (§0). Any field added there is visible to, and must at minimum
  default/tie off cleanly for, those configs' existing (register-only)
  coprocessors and CI tests (`verif/tests/testlist_cvxif.yaml`).
- The one in-repo precedent for a coprocessor-owned D$ port
  (`acc_dispatcher.sv`) required new port declarations and arbitration logic
  inside `cva6.sv` itself (§1.4), not just inside the accelerator module —
  because the D$ requester count is a structural property of the cache
  subsystem instantiation (`NumPorts` parameter,
  `cva6_hpdcache_subsystem.sv:28`), which is shared, single-instance
  infrastructure, not something a coprocessor can privately extend.
- `load_store_unit.sv`'s 3 D$ ports are already fully claimed (§1.3); adding
  a 4th CV-X-IF-owned port either goes through `load_store_unit.sv` itself
  (definitely shared/core code) or bypasses it by extending `cva6.sv`'s own
  arbitration the way `acc_dispatcher.sv` does (still core-level, still
  shared, and still collides with the `CvxifEn`/`EnableAccelerator`
  mutual-exclusion logic at `cva6.sv:802-804`).

**This conflicts directly with a constraint the parent map issue (#17) states
explicitly**: "Out of scope: Modifications profondes du core CV32A6
lui-même (pipeline/front-end/multi-issue/LSU) — le budget d'invasivité de cet
effort se limite au coprocesseur CV-X-IF (potentiellement étendu à x_mem),
pas au cœur RISC-V." Per the findings above, a real x_mem implementation
cannot stay within "CV-X-IF coprocessor, potentially extended to x_mem"
without also touching `cva6.sv`'s D$ arbitration and/or
`load_store_unit.sv`/`cva6_hpdcache_subsystem.sv`'s port count — i.e. exactly
the LSU/pipeline-adjacent core structures issue #17 flags as off-limits. This
should be surfaced as a scope conflict, not silently absorbed.

### 2.3 Risk to `cv64a6_imafdc_sv39*` CI configs

Concretely at risk if `cvxif_types.svh`/`cvxif_pkg.sv`/`cvxif_fu.sv` are
touched: those files are compiled and simulated for
`cv64a6_imafdc_sv39`/`cv64a6_imafdc_sv39_wb` in `.github/workflows/ci.yml:16`
(§0), using `core/cvxif_example/cvxif_example_coprocessor.sv` as their
coprocessor (the register-only template `mac4_copro` replaced only "on the
build/config path used for `sw/app/mnist`", per `mac4_coprocessor.sv:4-5` —
i.e. `cvxif_example` is presumably still what those CI configs instantiate,
though this was not independently re-verified by grepping every config's
coprocessor binding in this pass). Any change to the shared macro/struct
shape must keep those configs' existing register-only path bit-identical, or
CI breaks. This is the same shared-constant risk the parent ticket already
flagged for `NrRgprPorts`/`NR_RGPR_PORTS` (issue #15, referenced from #17) —
`x_mem` wiring adds a second, larger instance of the same category of risk
(shared type macro, not just a shared scalar constant).

### Summary answer to Q2

Not cheap, not isolated. Minimum ~5-7 files across the CV-X-IF plumbing and
`cva6.sv`'s cache-port arbitration, touching code shared with the CI-tested
`cv64a6_imafdc_sv39*` configs, and — per §1.3/§1.4 — very likely requiring
either new `load_store_unit.sv` ports or new `cva6.sv`-level D$ arbitration
(competing with or repurposing the existing, currently-unused-for-CV-X-IF
`acc_dispatcher` port pair). Both of those land squarely inside the
"LSU/pipeline" category issue #17 explicitly excludes from this effort's
scope.

## 3. Throughput/latency

### 3.1 Port count and contention

As established in §1.3, this config's LSU has exactly one load port and one
store port, each serviced by exactly one D$ request slot
(`dcache_req_ports_i/o[1]` for loads, `[2]` for stores,
`load_store_unit.sv:439-441,480-481`), fed by a single `fu_data_i` issue slot
per cycle (`load_store_unit.sv:45-49`). **There is no spare/idle port for a
coprocessor to use without either sharing the existing load/store ports
(direct contention with the core's own memory traffic) or getting a new,
dedicated port** — which, per §1.4/§2.2, means extending `cva6.sv`'s
D$-requester arbitration (currently 4-wide, 3 for the LSU + 1 shared/muxed
slot for the mutually-exclusive ACC interface,
`cva6.sv:1283-1318`/`cva6_hpdcache_subsystem.sv:28`).

### 3.2 Outstanding-request depth in this config

`core/include/cv32a6_im_contest_config_pkg.sv:52-54`:
```
localparam CVA6ConfigNrLoadPipeRegs = 1;
localparam CVA6ConfigNrStorePipeRegs = 0;
localparam CVA6ConfigNrLoadBufEntries = 1;
```
`CVA6ConfigNrLoadBufEntries = 1` means **this config's load unit supports
exactly one outstanding load at a time**, even though `load_unit.sv`'s own
header comment (lines 16-19) says the module was extended to "support
multiple outstanding load operations to the data cache" — that capability
exists in the RTL but is configured down to 1 for `cv32a6_im_contest`.
`NrLoadPipeRegs=1`/`NrStorePipeRegs=0` (`load_store_unit.sv:492-510`) add one
extra pipeline register stage on the load-result return path and none on the
store side. Concretely: the core's own load/store traffic in this config is
**already serialized to one in-flight memory op at a time** — there is no
slack capacity being left idle that a coprocessor could opportunistically use
without the core noticing contention.

### 3.3 Load latency (qualitative — exact cycle count not independently
confirmed for the actual configured cache)

`docs/03_cva6_design/ex_stage.md:61-172` (an in-repo CVA6 design doc,
general/architecture-level, not config-specific) describes a multi-cycle,
pipelined load path: address generation (an addition) in one stage, then a
virtually-indexed/physically-tagged cache lookup where "an additional bank of
registers delays the [address-translation] answer... an additional cycle"
(`ex_stage.md:251-253`), i.e. at minimum a handful of cycles from address
generation to data return even on a cache hit, before counting any store-buffer
aliasing stall (`ex_stage.md:86-101`) or D$ miss.

Caveat on this citation: `cv32a6_im_contest` is configured with
`CVA6ConfigDcacheType = HPDCACHE_WT` (`cv32a6_im_contest_config_pkg.sv:66`),
i.e. the **HPDCache** (`core/cache_subsystem/cva6_hpdcache_subsystem.sv`),
not the older `std_cache_subsystem.sv` this design doc was originally written
against. The HPDCache's own RTL is vendored as a git submodule
(`core/cache_subsystem/hpdcache`, per `.gitmodules`) that **is not checked
out in this worktree** (empty directory) — so its exact MSHR count / hit
latency in cycles could not be read directly in this pass and is not cited
here as a hard number. The structural facts in §3.1/§3.2 (port count,
`NrLoadBufEntries=1`) come from files that *are* present and apply regardless
of which cache backend is plugged in, since they're properties of
`load_store_unit.sv`/the config package, not of the HPDCache internals.

### Summary answer to Q3

The realistic throughput ceiling for an autonomous coprocessor sharing this
LSU is low and contended: at most one outstanding memory transaction total
in this config (`NrLoadBufEntries=1`), no idle port to use without either
adding new D$-requester infrastructure (§1.4/§2) or directly competing with
the core's own loads/stores for the existing load/store ports, and a
multi-cycle (address-gen + VIPT-translation + cache-access) pipeline even on
a hit. This config is tuned for a single scalar core with no memory-level
parallelism to spare; a coprocessor wanting wide, low-latency, high-reuse
memory bandwidth would be fighting a config that currently provides the
opposite of that, structurally, for the core's own use, let alone a second
requester.

## 4. Compatibility with existing issue/commit constraints

### 4.1 Recap of the established protocol facts (re-verified at current HEAD)

`core/cvxif_issue_register_commit_if_driver.sv:56-64`, unchanged since the
prior ticket's research: `commit_o.commit_kill = 1'b0;` is hardwired — the
core **never** sends a live kill signal to the coprocessor once an
instruction has been issued to it. `core/include/build_config_pkg.sv:81`:
`cfg.TRANS_ID_BITS = $clog2(CVA6Cfg.NrScoreboardEntries);`, with
`CVA6ConfigNrScoreboardEntries = 4` (`cv32a6_im_contest_config_pkg.sv:50`) →
`TRANS_ID_BITS = 2` → up to 4 distinct in-flight CV-X-IF transaction ids, as
stated in the ticket. Both facts still hold at commit `f02aa15`.

### 4.2 What the CV-X-IF spec requires for `spec`/`commit_kill` on the memory
channel

Quoting the upstream CV-X-IF spec (`openhwgroup/core-v-xif`,
`docs/source/x_ext.rst`), which defines the `spec` field this repo's
`x_mem_req_t.spec` (`cvxif_pkg.sv:62`) is named after:

> "A coprocessor shall never *initiate* memory request transactions for
> instructions that have already been killed at least a clk cycle earlier."
>
> "A coprocessor shall never initiate a speculative memory request
> transaction(s) on cycles after a cycle in which it receives
> `commit_kill = 1`."
>
> "If a memory request transaction or memory result transaction is already in
> progress at the time that the processor signals `commit_kill = 1`, then
> these transaction(s) will complete as normal."

The whole safety model for a memory-capable coprocessor issuing *speculative*
accesses (i.e. before its instruction is known non-speculative/committed) is
built around the coprocessor being able to observe `commit_kill` transition
to `1` and stop initiating new speculative requests from that point on.

### 4.3 Why that safety model doesn't transfer to this integration

This fork's `commit_kill` is permanently `0` (§4.1) — the one signal the spec's
speculative-memory-access safety rule depends on never fires, ever, on this
core. That is not, by itself, automatically unsafe: if the core genuinely
never abandoned an issued-but-not-yet-completed CV-X-IF instruction, "no kill
signal ever needed" would be consistent. But the prior research ticket
(`docs/research/cvxif-issue-commit-protocol.md` §2.2) already traced, in this
same RTL, that this assumption is false: a full-pipeline flush
(`flush_id_o`, triggered by an *older* instruction's exception/`eret`/fence/
CSR side effect) clears **every** scoreboard entry's `.issued` bit
unconditionally (`core/scoreboard.sv:257,259`, both lines unchanged at
`f02aa15`), including a *younger* CV-X-IF instruction that is still
outstanding at the coprocessor — with no signal of any kind reaching the
coprocessor to tell it this happened. That finding was written for the
register-only case, where the consequence of a late/stray result is (at
worst) a wrong GPR writeback into a reused scoreboard slot, gated by the
`mem_q[trans_id].issued` check in `scoreboard.sv:198` (per the prior ticket).

**For x_mem this is a strictly worse hazard class.** A register-result
writeback that lands on the wrong (reused) destination is a data-corruption
bug, bounded to one GPR. A coprocessor that has *already initiated or is
midway through a memory transaction* when its instruction gets silently
abandoned has no bounded blast radius in the same way: per the spec text in
§4.2, a coprocessor is only permitted to keep an in-flight memory transaction
running past a kill because it will be told about the kill and the
*transaction itself* (not just the eventual register write) is understood by
both sides to be in a defined, already-in-flight state. On this core, the
coprocessor is never told at all — so a design that (for throughput,
per §3) wants to keep issuing further speculative sub-transactions for an
instruction it still believes is live has no way to know when to stop, and a
stray coprocessor-initiated **store** (unlike a stray register write) would
have a real, memory-visible side effect the core's own scoreboard-slot-reuse
guard does nothing to contain.

### 4.4 Ordering vs. the core's own loads/stores

Independent of the kill-signal gap: this core's memory-ordering guarantees
("two loads to the same address... return in issue order," "a store followed
by a load to the same address can only be satisfied if the store has already
been committed," `docs/03_cva6_design/ex_stage.md:92-101`) are built entirely
inside `load_store_unit.sv`'s own load-unit/store-unit/store-buffer
interlocks (page-offset comparison against the store buffer,
`ex_stage.md:150-164`). A coprocessor issuing memory requests through a
*separate* port (§1.3/§1.4/§3.1 — the only architecturally plausible option
given the LSU's 3 ports are all claimed) sits entirely outside that
interlock unless new logic re-establishes it. The one in-repo precedent for
this (`acc_dispatcher.sv`) does exactly that: it stalls the *core's own*
scalar loads/stores against outstanding accelerator loads/stores via
speculative/dispatched pending counters (`acc_dispatcher.sv:133-151,
361-449`) and holds barriers until accelerator stores drain
(`acc_dispatcher.sv:347-355`). None of that machinery exists for CV-X-IF
today, and — per §4.3 — even `acc_dispatcher.sv`'s non-speculative gate
(`insn_pending_q`/`insn_ready_q`, only dispatching once "no longer
speculative") is a *different*, arguably more conservative strategy than
what CV-X-IF's own `commit_kill`+`spec` protocol assumes: `acc_dispatcher`
avoids the kill problem by simply never issuing a memory access until the
instruction is already known non-speculative, i.e. it never actually
exercises the CV-X-IF `spec=1`/`commit_kill` mechanism from §4.2 at all. That
is very likely the only safe strategy on this particular core (issue only
once the instruction cannot be flushed anymore), but it also removes any
timing benefit from prefetching data speculatively — a design point worth
being explicit about in any x_mem spec that follows from this research.

### Summary answer to Q4

**No, not compatible as the protocol is currently wired, without additional
coprocessor/core-side hazard tracking that does not exist today.** Adding
asynchronous, multi-cycle coprocessor-initiated memory accesses does not
merely inherit the already-known "stale result after a silent flush" hazard
from the register-only case (§4.3, citing
`docs/research/cvxif-issue-commit-protocol.md` §2.2) — it makes the
consequence of that hazard worse (a real, unbounded memory side effect vs. a
bounded, slot-reuse-guarded GPR write) and it additionally introduces a new
ordering hazard against the core's own loads/stores (§4.4) that
`load_store_unit.sv`'s existing page-offset/store-buffer interlocks do
nothing to cover for a second, independent requester port. The one
functioning precedent in this codebase (`acc_dispatcher.sv`) sidesteps the
worst of this by never issuing speculatively in the first place — but
implementing that same discipline for CV-X-IF means writing new
non-speculative gating logic that, per §2, does not fit inside "CV-X-IF
coprocessor only."

## Overall conclusion

x_mem wiring on this CVA6 fork is **not cheap and not isolated**. The struct
definitions already sitting in `cvxif_pkg.sv` are a red herring — they are
shadowed by a completely separate, mem-field-free type-generation path
(`cvxif_types.svh`) that is what's actually instantiated in `cva6.sv` and
`ariane.sv`, so "the types exist, just wire them up" understates the work by
a full macro-rewrite. The LSU offers no spare port and, in this config, no
memory-level parallelism to share (`NrLoadBufEntries=1`) even if it did.
The one working precedent for a coprocessor-owned memory port in this
codebase (`acc_dispatcher.sv`) required new arbitration inside `cva6.sv`
itself and a from-scratch non-speculative gating scheme — and even so never
wired real request traffic — which is strong evidence that a genuine x_mem
implementation will need to touch `cva6.sv`'s D$ arbitration and very likely
`load_store_unit.sv`/`cva6_hpdcache_subsystem.sv`'s port count, directly
conflicting with parent issue #17's stated constraint that this effort's
invasiveness budget stops at the CV-X-IF coprocessor boundary. On the
protocol side, this fork's permanently-`0` `commit_kill` combined with the
scoreboard's unconditional full-flush-on-older-instruction behavior
(re-confirmed at `f02aa15`) means a spec-faithful speculative x_mem
implementation is not safe here; the only defensible strategy is to issue
coprocessor memory accesses exclusively once non-speculative (mirroring
`acc_dispatcher.sv`'s own choice), which forfeits most of the latency-hiding
benefit x_mem would otherwise offer.

## Files read (primary sources)

- `core/include/cvxif_pkg.sv`
- `core/include/cvxif_types.svh`
- `core/cvxif_fu.sv`
- `core/cvxif_issue_register_commit_if_driver.sv`
- `core/cva6.sv`
- `corev_apu/src/ariane.sv`
- `core/load_store_unit.sv`
- `core/load_unit.sv`
- `core/acc_dispatcher.sv`
- `core/cache_subsystem/cva6_hpdcache_subsystem.sv`
- `core/cache_subsystem/std_cache_subsystem.sv`
- `core/mac4_copro/mac4_coprocessor.sv`
- `core/cvxif_example/` (template `mac4_copro` replaced, referenced for
  contrast per its own header comment)
- `core/scoreboard.sv`
- `core/include/cv32a6_im_contest_config_pkg.sv`
- `core/include/cv64a6_imafdc_sv39_config_pkg.sv`
- `core/include/cv64a6_imafdc_sv39_wb_config_pkg.sv`
- `core/include/build_config_pkg.sv`
- `core/include/ariane_pkg.sv`
- `.github/workflows/ci.yml`
- `verif/tb/uvmt/cva6_tb_wrapper.sv`
- `docs/03_cva6_design/ex_stage.md` (in-repo design doc, general/
  architecture-level — flagged where it may predate the HPDCache backend
  actually configured for `cv32a6_im_contest`)
- `docs/research/cvxif-issue-commit-protocol.md` (this project's own prior
  research, re-verified against current HEAD rather than taken on faith)
- `docs/research/cvxif-build-path.md` (this project's own prior research,
  used to establish which testbench/build path is the real one)

External source consulted (spec text, not repo RTL, clearly marked at each
use above): [openhwgroup/core-v-xif](https://github.com/openhwgroup/core-v-xif),
`docs/source/x_ext.rst` (memory-channel field/handshake semantics, §4.2).

Not consulted / not available in this pass: the HPDCache submodule RTL
(`core/cache_subsystem/hpdcache`) is not checked out in this worktree
(confirmed via `.gitmodules` + empty directory listing), so its internal
MSHR count and hit-latency-in-cycles could not be cited directly; §3.3 is
qualified accordingly.
