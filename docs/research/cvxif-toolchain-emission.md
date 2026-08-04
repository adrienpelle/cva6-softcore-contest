# Research: emitting custom RISC-V instructions from this repo's toolchain, and cvxif_example's opcode footprint

Ticket: GitHub issue [#4](https://github.com/adrienpelle/cva6-softcore-contest/issues/4) ("Mécanisme d'émission d'instructions custom depuis le C dans ce toolchain"), child of #1 (wayfinder map). Context: a future custom RISC-V extension + register-only tightly-coupled CV-X-IF coprocessor to accelerate `sw/app/mnist`.

Method: primary sources only — files in this repo, plus base RISC-V ISA knowledge only where needed to confirm the custom-opcode-space encoding rules. Every claim below is cited to a file path (and line numbers where useful).

---

## 1. How can custom instructions be emitted from C in this repo's toolchain?

### 1.1 Toolchain identity

The bare-metal toolchain actually built/used for `sw/app` is defined in `util/gcc-toolchain-builder/config/gcc-13.1.0-baremetal.sh`:

- Target triple: `riscv-none-elf` (`util/gcc-toolchain-builder/config/gcc-13.1.0-baremetal.sh:24`, `export TARGET=riscv-none-elf`).
- GCC: `releases/gcc-13.1.0` (`gcc-13.1.0-baremetal.sh:44`, built from `https://github.com/gcc-mirror/gcc.git`), configured `--enable-languages=c` only (`gcc-13.1.0-baremetal.sh:48`) — no C++.
- Binutils: tag `binutils-2_40` (i.e. GNU binutils 2.40) (`gcc-13.1.0-baremetal.sh:34`).
- libc: newlib `newlib-4.3.0` (`gcc-13.1.0-baremetal.sh:56`).
- This is invoked by `util/gcc-toolchain-builder/get-toolchain.sh:72` (`CONFIG_NAME="gcc-13.1.0-baremetal"` is the default config) and `util/gcc-toolchain-builder/build-toolchain.sh`.

Note: `Dockerfile:59-62` shows the toolchain-build step is currently **commented out** in the Docker image build (`#RUN cd gcc-toolchain-builder && ...`), so the toolchain is expected to be built/installed separately (e.g. by the developer, following `util/gcc-toolchain-builder/README.md`), not baked into the container as shipped. The rest of the Dockerfile only installs `openocd` and generic build prerequisites (`Dockerfile:29-36`).

`sw/app/Makefile:56` confirms the resulting compiler is invoked with prefix `riscv-none-elf-` at build time: `RISCV_PREFIX ?= riscv-none-elf-`, `RISCV_GCC ?= $(RISCV_PREFIX)gcc`. The `mnist` app is compiled with this same toolchain via the shared `bmarks` rule (`sw/app/Makefile:45-50` lists `mnist` alongside `coremark`, `helloworld`, `fft`).

Compiler flags used for all `sw/app` binaries (including `mnist`): `-march=rv32im_zicsr -mabi=ilp32 ... -O3 -mcmodel=medany -static` (`sw/app/Makefile:84-97` and `:99-115`). So the target ISA string is `rv32im_zicsr` — base integer (I) + multiply (M) + Zicsr, no C (compressed), no F/D.

A second, independently-configurable toolchain path exists for the UVM verification testbench: `verif/bsp/Makefile:1-4` defaults to `RISCV_EXE_PREFIX ?= $(RISCV)/bin/riscv32-unknown-elf-` (a different default triple name than `sw/app`'s `riscv-none-elf-`), overridable via `CV_SW_TOOLCHAIN`/`RISCV`. `verif/sim/Makefile:239` also invokes `riscv-none-elf-nm` directly, so in practice the same `riscv-none-elf` GCC/binutils toolchain built by `util/gcc-toolchain-builder` is expected to serve both `sw/app` and `verif/`.

### 1.2 Mechanisms available for hand-encoded custom instructions

The repo does not document or use any GCC custom-instruction **intrinsic/builtin** anywhere (no `__builtin_riscv_*`, no custom pragma, nothing found in `sw/` or elsewhere). No evidence of that mechanism in this repo.

What the repo *does* use, in two different forms, both resting on GNU **binutils' `.insn` GAS pseudo-op** (a binutils/GAS feature, not GCC-specific, so it is orthogonal to the GCC 13.1.0 version and works with binutils 2.40 as configured here):

1. **`.insn r` pseudo-op with the `CUSTOM_3` predefined symbol**, used in GAS macros for the corev-dv random-instruction-generator environment:
   - `verif/env/corev-dv/user_extension/x_extn_user_define.h:17` — `.insn r CUSTOM_3, 0x1, 0x0, \rd, \rs1, \rs2` (wrapped in GAS macro `cus_add`)
   - and similarly at lines 22, 27, 32, 37, 42, 47 for `cus_nop`, `cus_add_rs3`, `cus_u_add`, `cus_s_add`, `cus_add_multi`, `cus_exc`.
   - `CUSTOM_0`/`CUSTOM_1`/`CUSTOM_2`/`CUSTOM_3` are RISC-V-port GAS predefined symbols for the four custom-opcode base-opcode encodings, usable directly as the first operand of `.insn r`/`.insn i`/etc. This is a binutils feature (present in binutils 2.40, which is what this repo's config builds), confirmed in-repo purely by usage — not independently verified against binutils release notes since that's outside the repo.
   - This is pure **assembly** (`.h` file full of GAS `.macro`/`.insn` used from `.S` files), not C. No `.c` file in the repo uses `.insn` or `CUSTOM_*`.

2. **Raw `.word` hex/binary encoding**, used by the standalone CV-X-IF regression assembly tests:
   - `verif/tests/custom/cv_xif/cvxif_macros.h:13-19` defines macros like `CUS_ADD(rs1,rs2,rd)` → `.word 0b0000000##rs2####rs1##001##rd##1111011` — i.e. hand-built 32-bit binary literals assembled with GAS's `.word` directive and token-pasting, no `.insn` used at all here.
   - Invoked from `.S` test files, e.g. `verif/tests/custom/cv_xif/cvxif_add_nop.S:28` (`CUS_ADD(01010,01010,01010,01011);`). Again pure assembly, not C.

**No inline-asm (`__asm__ volatile(...)`) invocation of a custom opcode was found anywhere in the repo.** The only two `asm volatile`/`__asm__` occurrences in `sw/` are in `sw/bsp/hal/encoding.h` and `sw/bsp/hal/syscalls.c` (CSR/syscall access, unrelated to custom opcodes — grep of `sw/` for `.insn`, `asm volatile`, `__asm__`, `cvxif`, `custom0/1/2/3` confirms no custom-opcode inline asm in application code).

**Conclusion for Q1**: The toolchain (GCC 13.1.0 / binutils 2.40, target `riscv-none-elf`) supports emitting hand-encoded custom instructions via GNU `as`'s `.insn` pseudo-op (with `CUSTOM_0..CUSTOM_3` symbols) or raw `.word` encoding — both are assembler-level mechanisms usable from inline asm in C (`__asm__ volatile(".insn r CUSTOM_3, ...")`) or from standalone `.S` files. The repo currently only exercises these from `.S` files (verification test / corev-dv contexts), never from C, and never in `sw/app`. No compiler intrinsic/builtin path exists in this repo. For the mnist coprocessor work, `__asm__ volatile(".insn r CUSTOM_0, funct3, funct7, rd, rs1, rs2" : "=r"(rd) : "r"(rs1), "r"(rs2))` (or `.insn r` with an explicit numeric opcode instead of `CUSTOM_0`) is the pattern to follow — it is the same primitive already relied on elsewhere in the repo, just not yet ported into C/`sw/`.

---

## 2. Does `core/cvxif_example/` have a software-side invocation example anywhere in the repo?

**In `sw/`: no.** `grep -rli "cvxif|cv-x-if|cv_x_if|custom0|custom-0|custom_0" sw/` returns nothing. No C or assembly file under `sw/` references cvxif or issues any of `cvxif_instr_pkg.sv`'s opcodes.

**Elsewhere in the repo: yes**, but only in the verification environment, not application software:

- `verif/tests/custom/cv_xif/*.S` — nine hand-written assembly regression tests (`cvxif_add_nop.S`, `cvxif_exc.S`, `cvxif_illegal.S`, `cvxif_issexc.S`, `cvxif_multi.S`, `cvxif_nopexc.S`, `cvxif_rs3.S`, `cvxif_s_mode.S`, `cvxif_u_mode.S`) plus the shared macro header `cvxif_macros.h`, which directly issue the example coprocessor's custom-3 encoded opcodes (e.g. `CUS_ADD`, `CUS_NOP`, `CUS_ADD_RS3`, `CUS_EXC` — see `verif/tests/custom/cv_xif/cvxif_macros.h:13-19` and usage in `verif/tests/custom/cv_xif/cvxif_add_nop.S:28-33`). These are assembly-only bare-metal test programs, compiled/linked outside `sw/app`'s build path.
- `verif/env/corev-dv/user_extension/x_extn_user_define.h` and `verif/env/corev-dv/custom/cvxif_custom_instr.sv` — a constrained-random instruction generator ("corev-dv") extension that knows how to randomly generate the same custom-3 opcodes for coverage-driven simulation, using `.insn r CUSTOM_3, ...` macros (see Q1 above).
- A full UVM testbench presence for CV-X-IF: `verif/env/uvme/cvxif_vseq/uvme_cvxif_base_vseq.sv`, `verif/env/uvme/cvxif_vseq/uvme_cvxif_vseq.sv`, `verif/env/uvme/cov/uvme_cvxif_covg.sv` (coverage model), and `verif/env/uvme/cvxif_vseq/custom_instruction.rst`.

**Important caveat found while tracing this**: in the contest's actual build configuration, CV-X-IF is **disabled**. `core/include/cv32a6_im_contest_config_pkg.sv:21` sets `localparam CVA6ConfigCvxifEn = 0;`, and `Makefile:111` shows `target ?= cv32a6_im_contest` is this repo's default build target. So `cvxif_example_coprocessor` (instantiated conditionally-generated logic in `corev_apu/src/ariane.sv:112-128`) is not wired into the contest's default core build; only `core/include/cv32a6_embedded_config_pkg.sv:20` and `cv32a6_embedded_config_pkg_deprecated.sv:20` set `CVA6ConfigCvxifEn = 1`. A new extension will need to flip this flag for whichever config it targets (this needs to be part of the implementation plan, not just software emission).

**Conclusion for Q2**: `cvxif_example`'s only software-side invocation examples in this repo are in the RTL verification stack (`verif/tests/custom/cv_xif/*.S`, `verif/env/corev-dv/...`, `verif/env/uvme/cvxif_vseq/...`), never in `sw/`. There is no C example anywhere, and no example wired into the bare-metal `sw/app` build. Also, the example coprocessor is not enabled in the contest's default core configuration (`CvxifEn=0` for `cv32a6_im_contest`), so it isn't "live" in the shipped hardware config today.

---

## 3. Constraints on custom opcode space and instruction format for a new extension

### 3.1 Base-opcode map, as encoded in this repo's own decoder

`core/include/riscv_pkg.sv:229-259` defines the full RV32/64G base opcode map used by CVA6's decoder, including the four custom slots:

```
localparam OpcodeCustom0 = 7'b00_010_11;   // riscv_pkg.sv:231  -> 0b0001011
localparam OpcodeCustom1 = 7'b01_010_11;   // riscv_pkg.sv:239  -> 0b0101011
localparam OpcodeCustom2 = 7'b10_110_11;   // riscv_pkg.sv:251  -> 0b1011011
localparam OpcodeCustom3 = 7'b11_110_11;   // riscv_pkg.sv:259  -> 0b1111011
```

These four 7-bit values match the RISC-V unprivileged ISA spec's reserved "custom-0 / custom-1 / custom-2(-rv128) / custom-3(-rv128)" base-opcode slots (opcode field `inst[6:2]` combined with the fixed `inst[1:0]=11` quadrant-3 marker for 32-bit instructions) — base ISA knowledge, not repo-specific, used only to confirm the repo's constants line up with the standard.

`core/decoder.sv`'s main `case (instr.rtype.opcode)` statement (`core/decoder.sv:191` onward) **never matches on `OpcodeCustom0`, `OpcodeCustom1`, `OpcodeCustom2`, or `OpcodeCustom3`** (confirmed by grep — none of the four appear as case labels anywhere in `core/decoder.sv`). Any instruction with one of those opcodes therefore falls through to `default: illegal_instr = 1'b1;` (`core/decoder.sv:1456`). Then, if `CVA6Cfg.CvxifEn` is set, `core/decoder.sv:1459-1471` catches *any* illegal instruction (`is_illegal_i || illegal_instr`) — not just the custom-opcode ones — and routes it to the `CVXIF` functional unit as an `OFFLOAD` op, decoding `rs1`/`rs2`/`rd` (and `rs3` for the R4-type-shaped Madd/Msub/Nmadd/Nmsub opcodes) generically off the raw instruction bits (`core/decoder.sv:1462-1469`).

Implication: **all four custom slots (custom-0, custom-1, custom-2, custom-3) are entirely free** in this core — none is claimed by the base decoder, so all become CVXIF-offload candidates once `CvxifEn=1`.

### 3.2 What `cvxif_instr_pkg.sv` actually claims

`core/cvxif_example/include/cvxif_instr_pkg.sv` populates a 10-entry `CoproInstr` table (`cvxif_instr_pkg.sv:56-137`). Its opcode usage is **not confined to the official custom space**:

- 6 of the 10 entries use opcode `1111011` = **custom-3** (`OpcodeCustom3`) — the `NOP`, `ADD`, `DOUBLE_RS1`, `DOUBLE_RS2`, `ADD_MULTI`, `ADD_RS3_R` entries (`cvxif_instr_pkg.sv:60,68,76,84,92,100`, each commented `// custom3 opcode`). These are all **R-type** shaped (funct7/rs2/rs1/funct3/rd/opcode), with `ADD_RS3_R` additionally packing an `rs3` field into what would normally be part of `funct7` (an R4-type-like reuse of custom-3's bit space, not the standard RISC-V R4-type opcode).
- The other 4 entries (`MADD_RS3_R4`, `MSUB_RS3_R4`, `NMSUB_RS3_R4`, `NMADD_RS3_R4`, `cvxif_instr_pkg.sv:108,116,124,132`) instead reuse the **standard R4-type floating-point fused-multiply-add opcodes** — `1000011` (MADD), `1000111` (MSUB), `1001011` (NMSUB), `1001111` (NMADD) — which correspond exactly to `riscv_pkg.sv`'s `OpcodeMadd`/`OpcodeMsub`/`OpcodeNmsub`/`OpcodeNmadd` (`riscv_pkg.sv:245-248`). These are **not** part of the custom-opcode space at all; they are the base ISA's reserved F/D-extension R4-type opcodes. They only end up illegal/offloadable here because this core's configs disable F/D (`CVA6ConfigRVF = 0`, e.g. `core/include/cv32a6_im_contest_config_pkg.sv:15`), so the base decoder's `OpcodeMadd/Msub/Nmsub/Nmadd` case (`core/decoder.sv:1134` onward) presumably gates on RVF being enabled and otherwise falls through to illegal — meaning cvxif_example is deliberately squatting on FP-extension opcode space that is only "free" as long as F/D is never added to this core config. This is a real collision risk called out for a new extension to be aware of, not a safe pattern to imitate for an original design.

`instr_decoder.sv` (`core/cvxif_example/instr_decoder.sv:44`) matches offloaded instructions purely by `(CoproInstr[i].mask & issue_req_i.instr) == CoproInstr[i].instr` — i.e. the whole scheme (opcode/funct3/funct7 fields, R-type vs R4-type-shaped) is a data table, not hardwired logic, so a brand-new extension can define an entirely independent `CoproInstr`-shaped table using unclaimed opcode bits without touching this file's structure.

### 3.3 Resulting constraints for a new extension

- **Free custom slots, unused by `cvxif_example`**: custom-0 (`0001011`), custom-1 (`0101011`), and custom-2 (`1011011`) are completely untouched by `cvxif_instr_pkg.sv` and by the base decoder — safe to claim entirely for a new extension.
- **Custom-3 (`1111011`) is partially used** by 6 of `cvxif_example`'s 10 entries. If `cvxif_example` remains present/enabled alongside a new extension, a new design must avoid custom-3's specific `funct7`/`funct3` sub-encodings listed in `cvxif_instr_pkg.sv:56-104` (or replace `cvxif_example` outright, which is straightforward since it's disabled by default — `CvxifEn=0` in the contest config, `core/include/cv32a6_im_contest_config_pkg.sv:21` — and not compiled into any `sw/app` build).
- **Do not imitate the Madd/Msub/Nmsub/Nmadd opcode reuse** (`1000011`/`1000111`/`1001011`/`1001111`) for a new, from-scratch extension: that is base-ISA-reserved FP opcode space, only incidentally free because RVF is off in this contest config, and is a fragile choice to build new instructions on.
- **Instruction format**: because CV-X-IF offload in this core is triggered purely by "decoded as illegal, `CvxifEn` set" (`core/decoder.sv:1459-1471`) and the coprocessor-side decode is a generic mask/match table (`instr_decoder.sv:44`), a new extension is free to choose any bit layout within a chosen custom-opcode's `inst[31:7]` — R-type (`funct7|rs2|rs1|funct3|rd|opcode`) is the natural default matching `riscv_pkg.sv`'s `rtype_t` (`core/include/riscv_pkg.sv`, `rtype_t` struct used generically for `instr.rtype.opcode` throughout `decoder.sv`), and is what `cvxif_example` itself mostly uses. An R4-type layout (extra `rs3` field, as CVA6's `r4type_t` struct and `cvxif_example`'s `ADD_RS3_R`/`MADD_RS3_R4` entries do) is also directly supported by the existing decode path (`core/decoder.sv:1466-1469` explicitly special-cases `RS3` immediate-select for the Madd/Msub/Nmadd/Nmsub-shaped opcodes) — relevant if the mnist coprocessor needs 3 source registers per instruction. Since the ticket wants a **register-only** tightly-coupled coprocessor, plain R-type (2 source regs, 1 dest) under an unused custom-0/1/2 opcode is the simplest, most collision-free choice; R4-type under a custom opcode (not the borrowed FP opcodes) remains available if a 3-operand instruction is needed.

---

## Summary

1. **Toolchain**: GCC 13.1.0 (C-only) + binutils 2.40 + newlib 4.3.0, target triple `riscv-none-elf`, built via `util/gcc-toolchain-builder` (commented out of the Docker image itself — `Dockerfile:59-62` — so it must be built/installed separately). `sw/app` compiles for `rv32im_zicsr`/`ilp32` (`sw/app/Makefile:56,84-97`). The mechanism to emit hand-encoded custom instructions is GNU `as`'s `.insn` pseudo-op (with predefined `CUSTOM_0..CUSTOM_3` symbols, as used in `verif/env/corev-dv/user_extension/x_extn_user_define.h`) or raw `.word` binary encoding (`verif/tests/custom/cv_xif/cvxif_macros.h`) — both are assembler-level, usable from `.S` files or from `__asm__ volatile(...)` in C. No compiler intrinsic/builtin exists in this repo, and no C file anywhere uses either mechanism yet.
2. **No software-side example exists for `cvxif_example` in `sw/`** — only in the RTL verification stack (`verif/tests/custom/cv_xif/*.S`, `verif/env/corev-dv/...`, `verif/env/uvme/cvxif_vseq/...`), all assembly, none of it wired into `sw/app`'s build. Also, `cvxif_example` is disabled in the contest's default core config (`CvxifEn=0` in `core/include/cv32a6_im_contest_config_pkg.sv:21`).
3. **Opcode space**: custom-0/1/2 (`0001011`/`0101011`/`1011011`) are completely unclaimed by both the base decoder and `cvxif_example`, and are the safest choice for a new extension. Custom-3 (`1111011`) is partly used by `cvxif_example`'s R-type/R4-type-ish entries and should be avoided or coordinated around unless `cvxif_example` is replaced outright. `cvxif_example` additionally squats on the standard FP R4-type opcodes (Madd/Msub/Nmsub/Nmadd), which is fragile and should not be copied. Both R-type and R4-type instruction layouts are already supported end-to-end by the decoder/offload path; for a register-only coprocessor, plain R-type under a free custom-0/1/2 opcode is the cleanest, non-colliding choice.
