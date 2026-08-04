# CV-X-IF build/simulation path for `sw/app/mnist`

Research for GitHub issue #3 (child of wayfinder map issue #1).
Method: primary-source only, every claim traced to a file/line in this checkout
(branch `research/cvxif-build-path`, based on `cv32a6_contest_25_26` @ `2b256e6`).

## 0. Important scope caveat found while researching

`README.md:88-92` and `README.md:15` state the **2025-2026 contest is about accelerating
the `fft` app**, not `mnist`:

> "**mnist** & **coremark** are usied [sic] to make sure nothing is broken. But they are
> not the app we are interested in speeding up." — `README.md:90`
>
> "**The "fft" app is the only one we are interested in for the contest.**" — `README.md:92`

The root `Makefile` itself defaults `APP` to `mnist` (`Makefile:62`, `APP ?= mnist`), and
everything below about the build/sim mechanics applies identically to `mnist`, `fft`, or
any other app under `sw/app` — the app selection is just a `make` variable. But since the
ticket specifically asks about `mnist`, it's worth flagging explicitly that the repo's own
README treats `mnist` as a regression-check app, not the contest's acceleration target.
This doesn't change the answer to the three questions (the CV-X-IF plumbing is
app-agnostic), but it is a fact a human should confirm against the actual issue #3/#1
text before scoping work around `mnist` specifically.

## 1. Build/simulation path actually used for `sw/app/mnist`

### 1.1 Software build (produces the `.mem` image simulation reads)

- `sw/app/Makefile` builds all apps listed in `bmarks` (`coremark`, `helloworld`,
  `helloworld_printf`, `mnist`, `fft` — `sw/app/Makefile:47-52`).
- Compiler flags are fixed at `-march=rv32im_zicsr -mabi=ilp32` (`sw/app/Makefile:84,102`)
  — **no custom extension letter is declared**, and
  linking uses `sw/bsp/config/link.ld` via `RISCV_LDFLAGS` (`sw/app/Makefile:118`).
- Per-app rule chain: `%.riscv → %.hex/%.bin → %.mem → %.coe`
  (`sw/app/Makefile:126-169`), with `%.mem: %.bin` invoking
  `$(utils_dir)/bin2mem.py` (`sw/app/Makefile:164-165`).
- The root `Makefile` has a `benchmark` target that drives this:
  `cd sw/app && make $(APP).mem && make $(APP).coe` (`Makefile:357-358`).

### 1.2 Hardware build + simulation (QuestaSim path — the one documented in `README.md`)

- Root `Makefile:111`: `target ?= cv32a6_im_contest`. This is exported as `TARGET_CFG`
  if not already set (`Makefile:112-114`).
- `core/Flist.cva6:62-63` compiles, in order:
  `core/include/config_pkg.sv` then `core/include/${TARGET_CFG}_config_pkg.sv` — i.e.
  with the default `target`, this resolves to
  `core/include/cv32a6_im_contest_config_pkg.sv`.
- `Makefile:237` independently also lists
  `tbs := core/include/$(target)_config_pkg.sv corev_apu/tb/jtag_pkg.sv
  corev_apu/tb/ariane_tb.sv corev_apu/tb/ariane_testharness.sv` — same config file, same
  `target` variable.
- `Makefile:315` `build: $(library) $(library)/.build-srcs $(library)/.build-tb` compiles
  the RTL + testbench with QuestaSim (`vlog`/`vcom`/`vopt`), pulling in
  `core/Flist.cva6`, the `src` list (includes `corev_apu/src/ariane.sv`,
  `corev_apu/tb/ariane_testharness.sv`, etc. — `Makefile:166-207`), and the `tbs` list.
- `Makefile:344-348` — the actual simulation target:
  ```
  sim: build
      vsim... ${top_level}_optimized +permissive-off +binary_mem=$(APP_PATH)/$(APP).mem | tee sim.log
  ```
  where `APP_PATH := $(root-dir)/sw/app` (`Makefile:75`) and `APP ?= mnist`
  (`Makefile:62`), so `make sim APP=mnist` (or the default `make sim`) looks for
  `sw/app/mnist.mem`.
- **Important gap**: `sim` depends only on `build` (hardware), **not** on the software
  `.mem` file (`Makefile:344`, `sim: build`). Nothing in the `sim` target's prerequisite
  chain invokes `sw/app/Makefile`. `README.md:250-252` claims `make sim APP=...`
  "Compiles the software application to be run on CVA6 with RISCV tool chain," but this
  is **not what the Makefile does** — the user must separately run
  `make benchmark APP=mnist` (`Makefile:357-358`) or `cd sw/app && make mnist` before
  `make sim APP=mnist`, or the simulation will `$readmemh` a stale/missing `.mem` file.
- The `.mem` file is loaded at simulation time by
  `corev_apu/tb/ariane_tb.sv:159-162`:
  ```
  $value$plusargs("binary_mem=%s", binary_mem);
  $readmemh(binary_mem, dut.i_sram.gen_cut[0].i_tc_sram_wrapper.i_ram.Mem_DP);
  ```
- This whole flow matches the documented usage in `README.md:237-263` ("Simulation get
  started" — `make sim APP=fft` example, applies identically to `APP=mnist`).

### 1.3 FPGA / synthesis path reuses the same config selection

- `Makefile:669` `cva6_fpga: $(ariane_pkg) $(util) $(src) $(fpga_src) $(uart_src)
  $(src_flist) ...` and `Makefile:629-630` pass `TARGET_CFG=$(TARGET_CFG)` and
  `HPDCACHE_TARGET_CFG=$(HPDCACHE_TARGET_CFG)` through — the same `TARGET_CFG` exported
  at `Makefile:112-114`. So the FPGA/synthesis flow (`README.md:66-76`, `make cva6_fpga`)
  is gated by the same config package as the simulation flow.

### 1.4 CI (`.gitlab-ci.yml`, `.github/workflows/ci.yml`) is unrelated to mnist/contest flow

- `.github/workflows/ci.yml:16,26` runs `make run-${testcase}-verilator target=...` only
  for `cv64a6_imafdc_sv39` and `cv64a6_imafdc_sv39_wb` (64-bit targets) — no `cv32a6_*`
  target, no `mnist`/`fft` app, no Verilator run of the contest config at all.
- `.gitlab-ci.yml:49` sets `DV_TARGET: cv32a6_embedded` as the default for the inherited
  upstream ThalesGroup DV pipeline (riscv-tests, compliance, arch-tests, ASIC synth
  jobs at lines 202-344) — this is the **generic upstream CVA6 verification pipeline**,
  unrelated to `sw/app/mnist` or the `make sim APP=` flow. It never references `mnist`,
  `fft`, or `cv32a6_im_contest`. It also depends on an external `setup-ci` repo
  (`.gitlab-ci.yml:24-26`) not present in this checkout, so most of it can't even be
  traced further here.
- **Conclusion**: there is no CI job in this repo that builds/simulates `mnist` (or any
  contest app) end-to-end. The only path that does this is the root `Makefile`'s
  `sim`/`benchmark` targets described in 1.1–1.2, run manually per `README.md`.

## 2. Which config package is actually used, and is `CvxifEn` really 0 on that path?

**Confirmed: `core/include/cv32a6_im_contest_config_pkg.sv` is the config package used**
for `mnist` (and for the FPGA build), via the default `target ?= cv32a6_im_contest`
(`Makefile:111`) → `TARGET_CFG` (`Makefile:112-114`) → `core/include/${TARGET_CFG}
_config_pkg.sv` (`core/Flist.cva6:63`, `Makefile:237`).

In that file:
- `core/include/cv32a6_im_contest_config_pkg.sv:21`: `localparam CVA6ConfigCvxifEn = 0;`
- `core/include/cv32a6_im_contest_config_pkg.sv:101`:
  `CvxifEn: bit'(CVA6ConfigCvxifEn),` inside the `cva6_cfg` struct literal (of type
  `config_pkg::cva6_user_cfg_t`, per `core/include/config_pkg.sv:219`).

So yes — `CVA6ConfigCvxifEn = 0` in `cv32a6_im_contest_config_pkg.sv` is exactly the
value that reaches the RTL for the `mnist` build/sim path. This is **not** a stale or
unused config; it is the one instantiated by `make sim APP=mnist` today.

### Other config packages, for reference

`grep`-ing every `core/include/*_config_pkg.sv` for `CVA6ConfigCvxifEn`:

| Config package | `CvxifEn` |
|---|---|
| `cv32a6_im_contest_config_pkg.sv` (**used for mnist**) | 0 |
| `cv32a6_embedded_config_pkg.sv` | 1 |
| `cv32a6_ima_sv32_fpga_config_pkg.sv` | 0 |
| `cv32a6_imac_sv0_config_pkg.sv` | 0 |
| `cv32a6_imac_sv32_config_pkg.sv` | 0 |
| `cv32a6_imafc_sv32_config_pkg.sv` | 0 |
| `cv64a6_imafdc_sv39_config_pkg.sv` | 1 |
| `cv64a6_imafdc_sv39_wb_config_pkg.sv` | 1 |
| `cv64a6_imafdc_sv39_hpdcache_config_pkg.sv` | 1 |
| `cv64a6_imafdc_sv39_hpdcache_wb_config_pkg.sv` | 1 |
| `cv64a6_imafdch_sv39_config_pkg.sv` | 1 |
| `cv64a6_imafdch_sv39_wb_config_pkg.sv` | 1 |
| `cv64a6_imafdcv_sv39_config_pkg.sv` | 0 |
| `cv64a6_imafdc_sv39_openpiton_config_pkg.sv` | 0 |
| `cv64a6_imadfcv_sv39_polara_config_pkg.sv` | 0 |

**Caveat on `cv32a6_embedded_config_pkg.sv` as a "reference" for enabling CV-X-IF**:
it is structurally **out of sync with the current `core/include/config_pkg.sv` schema**.
It builds a raw `config_pkg::cva6_cfg_t` literal directly
(`core/include/cv32a6_embedded_config_pkg.sv:73`, `localparam config_pkg::cva6_cfg_t
cva6_cfg = '{...}`), whereas every other config package (including
`cv32a6_im_contest_config_pkg.sv`) builds a `config_pkg::cva6_user_cfg_t`
(`core/include/config_pkg.sv:219`) and lets `build_config_pkg::build_config()`
(`core/include/build_config_pkg.sv`) derive the full `cva6_cfg_t`
(`core/include/config_pkg.sv:377`) with all its fields. `cv32a6_embedded_config_pkg.sv`'s
struct literal is missing dozens of fields present in the current `cva6_cfg_t` (e.g.
`XLEN`, `VLEN`, `PLEN`, `MemTidWidth`, `IcacheByteSize`/`DcacheByteSize`,
`UseSharedTlb`/`SharedTlbDepth`, `DebugEn`, `TvalEn`, `PMPNapotEn`,
`X_*` CV-X-IF port-width fields, etc.), with no `default:` key in the literal. This
strongly suggests the file predates a `config_pkg.sv` refactor and is not guaranteed to
elaborate as-is in the current tree. **This was not compiled/verified in this
investigation** (no simulator run performed) — flagging as an open risk, not a confirmed
build failure. Point is: don't copy `cv32a6_embedded_config_pkg.sv`'s pattern verbatim;
follow `cv32a6_im_contest_config_pkg.sv`'s existing `cva6_user_cfg_t` + `build_config()`
pattern and just flip `CVA6ConfigCvxifEn` (and add a `CvxifEn: bit'(CVA6ConfigCvxifEn)`
line, which is already there at line 101).

## 3. What else is needed beyond flipping `CvxifEn` to 1

Tracing how `CvxifEn` propagates through the RTL shows the wiring is **entirely
generate-block-gated by the config package's `CvxifEn` field** — no separate Makefile
define, testbench parameter, or simulation plusarg is required to activate the
interface and the example coprocessor once the flag is set. Concretely:

1. **`core/include/build_config_pkg.sv`**: `cfg.CvxifEn = CVA6Cfg.CvxifEn;` propagates
   the user-config bit into the built `cva6_cfg_t`, and
   `int unsigned NrWbPorts = (CVA6Cfg.CvxifEn || EnableAccelerator) ? 5 : 4;` widens the
   writeback-port count when CV-X-IF is on — this recomputation happens automatically,
   no manual edit needed.
2. **`core/ex_stage.sv:597`**: `if (CVA6Cfg.CvxifEn) begin : gen_cvxif` instantiates
   `cvxif_fu` (the CVA6-side adapter converting the internal FU interface to the
   `cvxif_req`/`cvxif_resp` X-interface channels); otherwise a `gen_no_cvxif` stub drives
   the outputs to `'0`.
3. **`core/cva6.sv:776-802`**: gates the `cvxif_req`/`x_*` signal wiring
   (`gen_cvxif_output_assignement`) and asserts
   `if (CVA6Cfg.CvxifEn && CVA6Cfg.EnableAccelerator) $error(...)` — CV-X-IF and the
   accelerator port (`RVV`) are mutually exclusive, irrelevant here since `RVV=0` in the
   contest config.
4. **`corev_apu/src/ariane.sv:111-141`** (the `ariane` top-level wrapper instantiated by
   `corev_apu/tb/ariane_testharness.sv:555`) — **this is the key auto-instantiation
   point**:
   ```
   if (CVA6Cfg.CvxifEn) begin : gen_example_coprocessor
     cvxif_example_coprocessor #(...) i_cvxif_coprocessor (
       .clk_i, .rst_ni,
       .cvxif_req_i ( cvxif_req ),
       .cvxif_resp_o( cvxif_resp )
     );
   end else begin
     always_comb begin
       cvxif_resp = '0;
       cvxif_resp.compressed_ready = 1'b1;
       cvxif_resp.issue_ready = 1'b1;
       cvxif_resp.register_ready = 1'b1;
     end
   end
   ```
   With the contest config's `CvxifEn=0` today, this generate-block instead just
   auto-acks a no-op response. Flipping `CvxifEn=1` in
   `cv32a6_im_contest_config_pkg.sv` will, with **no other RTL wiring change**, cause
   `ariane.sv` to instantiate `core/cvxif_example/cvxif_example_coprocessor.sv` — the
   stock OpenHW example coprocessor — tightly coupled register-only, exactly matching
   the "touch the core as little as possible" goal in the ticket description.
5. **All CV-X-IF source files are already compiled unconditionally** — they're just not
   instantiated when the flag is 0. `core/Flist.cva6:75-83` lists
   `cvxif_compressed_if_driver.sv`, `cvxif_issue_register_commit_if_driver.sv`,
   `cvxif_example/include/cvxif_instr_pkg.sv`, `cvxif_fu.sv`,
   `cvxif_example/cvxif_example_coprocessor.sv`, `cvxif_example/instr_decoder.sv`,
   `cvxif_example/compressed_instr_decoder.sv`, `cvxif_example/copro_alu.sv`
   unconditionally, and `Makefile`'s `copro_src` variable
   (`Makefile:129-130`) mirrors this — so **no Flist/Makefile edits are needed** to
   compile a real coprocessor either, as long as replacement files live under
   `core/cvxif_example/` (or the Flist/Makefile `copro_src` var is updated to point at
   new files, if the real design lives elsewhere).
6. **Decoder-side trigger is already generic and requires no change**:
   `core/decoder.sv:1459-1467` — when `CVA6Cfg.CvxifEn` is set, *any* otherwise-illegal
   instruction (`is_illegal_i || illegal_instr`) is redirected to
   `instruction_o.fu = CVXIF; instruction_o.op = ariane_pkg::OFFLOAD;` and sent to the
   X-interface. The actual accept/reject and opcode decode is entirely the
   coprocessor's job (`core/cvxif_example/instr_decoder.sv` matches against
   `core/cvxif_example/include/cvxif_instr_pkg.sv`'s `CoproInstr` table, which currently
   defines a small enumerated set of custom-3-opcode (`riscv_pkg.sv:259`,
   `OpcodeCustom3 = 7'b11_110_11`) instructions:
   `ILLEGAL/NOP/ADD/DOUBLE_RS1/.../ADD_RS3_R`). To add real custom instructions for
   `mnist`, the encoding table in a `cvxif_instr_pkg.sv`-equivalent file needs new
   entries, and the coprocessor's `instr_decoder.sv`/ALU need the accelerate logic. No
   CVA6 core file needs new opcode-space handling for this — `decoder.sv` already routes
   *all* illegal instructions to CV-X-IF unconditionally.

### Things confirmed NOT required (no evidence of them in this repo)

- **No linker script change needed** for a register-only coprocessor: `sw/bsp/config
  /link.ld` is unrelated to CV-X-IF register interfaces (custom instructions operate on
  the GPR file passed through the X-interface register ports, not memory-mapped I/O);
  nothing in `link.ld` was found to reference CVXIF/coprocessor regions (not directly
  inspected line-by-line here, but no MMIO/coprocessor-address symbols showed up in any
  CVXIF-related grep across the repo).
- **No testbench parameter/define needed**: `corev_apu/tb/ariane_testharness.sv:30`
  defaults `CVA6Cfg` to `cva6_config_pkg::cva6_cfg` — i.e. whatever config package was
  compiled as `core/include/${TARGET_CFG}_config_pkg.sv` (since that file always
  `package cva6_config_pkg;` and exports `cva6_cfg`). There's a single point of control:
  the `target`/`TARGET_CFG` Makefile variable selects the config file, and that file's
  `CvxifEn` bit alone drives every generate block above.
- **No simulation plusarg / `defines` flag needed**: `Makefile:63` `defines ?=
  WT_DCACHE` (compile-time Verilog `` `define ``) is unrelated to CV-X-IF; grepping
  the whole repo found no `` `ifdef CVXIF `` / `` `define CVXIF_EN `` style guard
  anywhere — everything is a runtime SystemVerilog `generate if` off the config
  struct, not a preprocessor define.

### Things that ARE required but are outside the RTL/build system (software toolchain)

- **Toolchain support for emitting the custom encoding**: `sw/app/Makefile:91-92` fixes
  `-march=rv32im_zicsr` — no custom extension is declared to GCC. No `.insn` directive,
  custom-opcode intrinsic, or inline-asm usage of any kind was found anywhere under
  `sw/` (checked `sw/bsp/hal/encoding.h` and all of `sw/` for `.insn`/`__builtin` — only
  hits are unrelated CSR/`wfi` inline asm in `sw/bsp/hal/encoding.h` and
  `sw/bsp/hal/syscalls.c:146`). So `sw/app/mnist`'s C code will need `asm volatile
  (".insn ...")` (or equivalent) to emit custom instructions from C, since the
  `riscv-none-elf-gcc` toolchain used here (per `README.md`'s Docker-based flow,
  `sw/app/Makefile:56-60`) has no built-in mnemonics for opcode-space
  `OpcodeCustom0`..`OpcodeCustom3` (`core/include/riscv_pkg.sv:231,239,251,259`) beyond
  what `.insn` raw-encoding syntax provides. This is unavoidable toolchain-level work,
  not a repo config change.

## Summary answers

1. **Build/sim path**: `make sim APP=mnist` (root `Makefile:344-348`, documented in
   `README.md:241-263`) → `build` compiles RTL/TB with QuestaSim using
   `core/Flist.cva6` + `target=cv32a6_im_contest` (`Makefile:111`, default) →
   `vsim ... +binary_mem=sw/app/mnist.mem`, which must be pre-built separately via
   `make benchmark APP=mnist` or `cd sw/app && make mnist` (the `sim` target does *not*
   build the software itself, contrary to what `README.md:250-252` implies). CI in this
   repo (`.gitlab-ci.yml`, `.github/workflows/ci.yml`) does not exercise this
   mnist/contest path at all — it's a separate, generic upstream CVA6 DV pipeline.
   Also worth flagging: `README.md` itself says the 2025-26 contest's actual target app
   is `fft`, not `mnist`.

2. **Config package actually used**: `core/include/cv32a6_im_contest_config_pkg.sv`,
   selected via `target ?= cv32a6_im_contest` (`Makefile:111`) →
   `core/include/${TARGET_CFG}_config_pkg.sv` (`core/Flist.cva6:63`, `Makefile:237`).
   Its `CVA6ConfigCvxifEn = 0` (line 21) is indeed the live value on the mnist path —
   not stale, not shadowed by another config.

3. **Beyond flipping the bit**: essentially nothing else in the RTL/build system.
   `corev_apu/src/ariane.sv:111-141` already auto-instantiates
   `core/cvxif_example/cvxif_example_coprocessor.sv` under a `CVA6Cfg.CvxifEn`
   generate-if, and all CV-X-IF sources are already unconditionally compiled
   (`core/Flist.cva6:75-83`, `Makefile` `copro_src`). What *is* needed: (a) real
   coprocessor RTL + a real custom-instruction encoding table replacing/extending
   `core/cvxif_example/include/cvxif_instr_pkg.sv`'s `CoproInstr` (the decoder already
   routes all illegal instructions to CV-X-IF unconditionally, `core/decoder.sv:1459-
   1467`), and (b) a way to emit those encodings from `sw/app/mnist`'s C code (e.g.
   `.insn` inline asm), since the fixed `-march=rv32im_zicsr` toolchain flags
   (`sw/app/Makefile:91-92`) declare no custom extension and no `.insn` usage exists
   anywhere in `sw/` today.
