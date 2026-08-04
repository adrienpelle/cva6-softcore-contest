# Research: are `sw/app/mnist`'s `macsOnRange()` buffers actually 4-byte aligned?

Ticket: GitHub issue #11 (child of wayfinder map issue #1) — add a register-only
CV-X-IF coprocessor instruction that reads 4 contiguous int8 bytes as one 32-bit
word (`lw`) before handing them to hardware for a 4-lane int8 SIMD MAC, to speed
up `macsOnRange()` in `sw/app/mnist/NetworkPropagate.c`.

This note answers, from a real build of `sw/app/mnist` and from the actual
`core/` RTL, whether that `lw`-based operand fetch is safe.

## Method

`sw/app/mnist` was built twice, from primary sources, and the two builds were
cross-checked against each other:

1. **Host toolchain** — `riscv-none-elf-gcc` 15.2.0 (xPack GNU RISC-V Embedded
   GCC), found at
   `/home/a21pelle/.local/xPacks/riscv-none-elf-gcc/xpack-riscv-none-elf-gcc-15.2.0-1/bin`,
   invoked exactly as `sw/app/Makefile` does:
   `make mnist.riscv` from `sw/app/`. GCC 15's default C dialect turns an
   existing pre-existing implicit-int-to-pointer-conversion warning in
   `sw/bsp/config/fpga_platform_config.h`/`sw/bsp/drivers/uart/uart.c`
   (`UART_TypeDef *g_uart_0.hw_reg = FPGA_UART_0_BASE` where
   `FPGA_UART_0_BASE` is an `int`-typed macro) into a hard error; this was
   worked around only by passing `-Wno-int-conversion` on the command line
   (`make mnist.riscv RISCV_GCC="riscv-none-elf-gcc -Wno-int-conversion"`), no
   source files were modified. The rest of the build used the Makefile's
   stock flags verbatim (`-march=rv32im_zicsr -mabi=ilp32 -O3
   -mcmodel=medany -funroll-all-loops ...`).
2. **Project's own Docker image** — `sw-docker:vfft` (already built/available
   in this environment, `riscv-none-elf-gcc` (GCC) 13.1.0 from
   `/util/gcc-toolchain-builder/riscv_toolchain`), run as:
   `docker run --rm -v <repo>/sw:/workdir sw-docker:vfft bash -c 'cd
   /workdir/app && make clean && make mnist.riscv'`. This build needed **no**
   flag workaround (GCC 13 still just warns on the int-conversion). Both
   builds produced a valid, linked `mnist.riscv` (`ELF 32-bit LSB executable,
   UCB RISC-V, soft-float ABI ... statically linked`).

Both binaries were inspected with `riscv-none-elf-nm`, `riscv-none-elf-objdump
-d`, and `riscv-none-elf-readelf -S`. The two builds (different GCC major
versions, different symbol-ordering heuristics) agree on every alignment
conclusion below, which is good independent confirmation that what's reported
is not a fluke of one specific compiler version.

## Q1 — Are the buffers actually 4-byte aligned in the built binary?

**The array *bases* are 4-byte aligned in both builds — but the pointers
`macsOnRange()` is actually called with are frequently *not* 4-byte aligned,
because of how `inputs + iOffset` is computed in `convcellPropagate1()` /
`fccellPropagateUDATA_T()`. Base-array alignment is necessary but not
sufficient.**

### Base addresses (host build, GCC 15.2.0 xPack)

```
80013f3c r conv1_biases
80013f7c r conv1_weights
8001407c r conv2_biases
800140dc r fc1_biases
80014334 r fc2_biases
8001435c r fc2_weights
80014938 r fc1_weights
80022a38 r conv2_weights
80026550 b mem
```
(`riscv-none-elf-nm -n mnist.riscv`, from `sw/app/mnist.riscv` built as above)

### Base addresses (Docker build, project's `sw-docker:vfft`, GCC 13.1.0)

```
8000f144 r conv1_biases
8000f184 r conv1_weights
8000f284 r conv2_biases
8000f2e4 r fc1_biases
8000f53c r fc2_biases
8000f564 r fc2_weights
8000fb40 r fc1_weights
8001dc40 r conv2_weights
800214ec b mem
```

Every one of these 9 addresses is a multiple of 4 in both builds (verified
with `n % 4` in Python for each address — all `0`). `mem` is also a multiple
of 16 in both builds. The stack-resident `inputBuffer` in `main()` (the
top-level `inputs` buffer for the whole network, `UDATA_T
inputBuffer[ENV_SIZE_Y*ENV_SIZE_X*ENV_NB_OUTPUTS]`) is likewise aligned: the
disassembly of `main` (`riscv-none-elf-objdump -d mnist.riscv`) shows
`addi sp,sp,-624` then `addi a0,sp,16` as the address handed to
`processInput`/`propagate`; 624 is a multiple of 16, and the RV32 psABI
guarantees `sp` is 16-byte aligned at a function's entry, so `sp+16` is
16-byte (hence 4-byte) aligned.

### But: `macsOnRange()`'s actual call-site offsets are not all 4-aligned

`macsOnRange()` is never called with a bare array base — it's called as
`macsOnRange(inputs + iOffset, weights + wOffset, ...)`
(`sw/app/mnist/NetworkPropagate.c:169-172` in `convcellPropagate1`, and
`:265-268`/`:282-285` in `fccellPropagateUDATA_T`), where
`iOffset = INPUT_MEM_STRIDE * iPos` and `iPos` depends on the current output
pixel/row being computed. Whether `inputs + iOffset` is 4-aligned therefore
depends on `INPUT_MEM_STRIDE` (== the layer's input channel count), not on
whether the underlying array is 4-aligned.

Walking the three layers that call `macsOnRange` in `propagate()`
(`NetworkPropagate.c:437`, `:467`, `:500`, `:530`-ish for fc2), using the
`INPUT_MEM_STRIDE` values each call site actually passes and the shape
constants in `conv1.h`/`conv2.h`/`fc1.h`/`fc2.h`/`mem_info.h`:

* **conv1** (`inputs` = the raw image buffer): `INPUT_MEM_STRIDE =
  ENV_MEM_STRIDE = 1` (`mem_info.h:10`), matching `CONV1_NB_CHANNELS = 1`
  (`conv1.h:10`, single grayscale channel — the fast-path condition
  `NB_CHANNELS == INPUT_MEM_STRIDE` is `1 == 1`, true). With
  `CONV1_STRIDE_X = 2`, `CONV1_PADDING_X = 0` (`conv1.h:21,23`) and
  `CONV1_CHANNELS_WIDTH = 24` (`conv1.h:15`, a multiple of 4):
  `iOffset = iPos = ix + 24*(row terms) = 2*ox + 24*(...)`. The `24*(...)`
  term is always `≡ 0 (mod 4)`, so `iOffset mod 4 == (2*ox) mod 4`, which is
  `0` for even `ox` and `2` for odd `ox`. `CONV1_OUTPUTS_WIDTH = 11`
  (`conv1.h:11`), so **5 of the 11 output-column starting addresses per row
  (`ox` = 1,3,5,7,9) are misaligned by exactly 2 bytes** relative to a
  4-byte boundary — i.e., almost half of conv1's `macsOnRange` calls (each
  reading exactly `KERNEL_WIDTH*NB_CHANNELS = 4` int8 bytes, the same width
  the coprocessor design wants to `lw`) would issue a misaligned 4-byte
  access.
* **conv2** (`inputs` = `conv1_output`, i.e. `mem[CONV1_MEM_CONT_OFFSET..]`):
  `INPUT_MEM_STRIDE = CONV1_MEM_STRIDE = 16` (`mem_info.h:18`, matches
  `CONV2_NB_CHANNELS = 16`, `conv2.h:10`). `iOffset = 16 * iPos` is always a
  multiple of 16 — **always 4-aligned**.
* **fc1** (`inputs` = `conv2_output`): `INPUT_MEM_STRIDE = CONV2_MEM_STRIDE =
  24` (`mem_info.h:26`, matches `FC1_NB_CHANNELS = 24`, `fc1.h:10`).
  `FC1_CHANNELS_WIDTH = 4` (`fc1.h:13`), so `iPos = 4*iy` and
  `iOffset = 24*4*iy = 96*iy` — **always 4-aligned**.
* **fc2** (`inputs` = `fc1_output`): `FC2_CHANNELS_WIDTH = FC2_CHANNELS_HEIGHT
  = 1` (`fc2.h:13-14`, already a flat 150-vector), so the `iy` loop in
  `fccellPropagateUDATA_T` runs once with `iPos = 0`, hence `iOffset = 0`
  always — **trivially 4-aligned**.

So, in the current network topology, **only conv1 has a real intra-array
misalignment problem**, and it's structural: the raw image is single-channel
(`CONV1_NB_CHANNELS = 1`) with a horizontal stride of 2 pixels
(`CONV1_STRIDE_X = 2`), so consecutive `macsOnRange` windows only line up on
a 4-byte boundary every other output column. This is not something that any
amount of `__attribute__((aligned(4)))` on `conv1_weights`/`inputBuffer`/`mem`
can fix by itself, because the base is already aligned — the misalignment is
introduced by the `+ iOffset` pointer arithmetic at the call site, which
walks the buffer 2 bytes at a time.

## Q2 — Is `MEMORY_ALIGNMENT 1` in `mem_info.h` just an informational leftover?

**Yes — it is dead code with zero effect on the build.**
`grep -rn "MEMORY_ALIGNMENT" sw/app/mnist/` finds exactly one hit: the
`#define MEMORY_ALIGNMENT 1` in `mem_info.h:8` itself. It is never
referenced by any `.c` file, never used in an `alignas`/`__attribute__`,
never used in a `#pragma`. It is purely N2D2-codegen metadata (the file
header says `// N2D2 auto-generated file.`) describing byte alignment in
N2D2's own export model, and it has no bearing on how GCC actually lays out
`mem[]` or any of the weight/bias arrays.

What *does* determine `mem[]`'s actual alignment turned out to be
`-O3`-driven autovectorization, not anything in the source or in
`mem_info.h`. Compiling `NetworkPropagate.c` to assembly
(`riscv-none-elf-gcc ... -O3 -S`) shows an explicit compiler-emitted
directive immediately before the symbol:
```
	.align	2
	.type	mem, @object
	.size	mem, 2160
mem:
```
Recompiling the same file with `-fno-tree-vectorize -fno-tree-slp-vectorize`
added to the `-O3` flags makes that `.align 2` directive disappear entirely
(confirmed by direct diff of the generated `.s`); the same happens simply at
`-O1` (where GCC's tree-vectorizer isn't enabled by default). This is GCC's
IPA "increase alignment for vectorization" pass acting on a `static
int8_t mem[2160]` because at `-O3` it is used inside loops GCC judges
vectorizable — it is an optimizer side effect, not a guarantee tied to any
attribute or to `mem_info.h`. `conv1_weights`/`conv2_weights`/etc.
(`static const WDATA_T ...[]`, `WDATA_T` = `int8_t`) get *no* explicit
`.align` directive at all in the generated assembly; their observed 4-byte
alignment is a layout coincidence: each weights array is placed immediately
after its layer's `..._biases` array, and `BDATA_T` (the bias element type)
is `int32_t` (`typedefs.h:107`, since `NB_BITS == 8` selects
`typedef int32_t SUM_T; typedef SUM_T BDATA_T;`), so each biases array is
naturally 4-byte aligned by ordinary C object-alignment rules and its byte
size (`4 * NB_OUTPUTS`) is always a multiple of 4 — which pushes the
following weights array's start back onto a 4-byte boundary too, purely
because of what happens to sit right before it in `.rodata`. Nothing in the
source declares or requires this ordering.

**Conclusion: today's 4-byte alignment of the *_weights arrays and of `mem`
is emergent and fragile** — a consequence of (a) bias-array natural
alignment plus declaration order, and (b) `-O3` autovectorization heuristics
that could change between GCC versions, optimization levels, or with
unrelated code changes that make GCC no longer judge the array
"vectorizable." `MEMORY_ALIGNMENT 1` correctly documents this fragility (it
literally says "1-byte alignment guaranteed") even though it isn't wired
into the build.

## Q3 — Does the CV32A6 contest config support misaligned accesses?

**No — neither in hardware nor via software emulation. A misaligned `lw`/`sw`
is a fatal, non-recoverable event on this target, not merely an
expensive one.**

* `core/include/cv32a6_im_contest_config_pkg.sv` (the config used for this
  contest, `localparam CVA6ConfigXlen = 32;` at line 13, `MmuPresent = 0`
  at line 68) has **no** `MisalignedAccessEn`-style field. A repo-wide
  `grep -rn -iE "misalign"` across `core/` turns up only the RTL modules
  that *detect and trap* misalignment (`core/load_store_unit.sv`,
  `core/load_unit.sv`, `core/store_unit.sv`, `core/branch_unit.sv`, MMU/PTW
  files, `core/csr_regfile.sv`) plus the cause-code constants in
  `core/include/riscv_pkg.sv:333,335` (`LD_ADDR_MISALIGNED = 4`,
  `ST_ADDR_MISALIGNED = 6`). There is no config-gated alternate path.
* `core/load_store_unit.sv:598-639` (`data_misaligned_detection`) is purely
  combinational and unconditional for this XLEN=32 config: for word-sized
  ops (`LW, LWU, SW, ...`, line 622-630) it sets
  `data_misaligned = 1'b1` whenever `lsu_ctrl.vaddr[1:0] != 2'b00` — no
  parameter checked, no partial/two-beat access performed. When
  `data_misaligned` is set, `misaligned_exception.cause` is set to
  `riscv::LD_ADDR_MISALIGNED` (loads) or `riscv::ST_ADDR_MISALIGNED`
  (stores) (lines 642-669) and this is delivered as a precise trap — there
  is no hardware fallback that splits the access into two aligned beats.
* On the software side, `sw/bsp/hal/handlers.S`'s `sw_irq_handler` only
  special-cases `mcause` values `EXCEPTION_ILLEGAL_INSN` (2),
  `EXCEPTION_ECALL_M` (11), and `EXCEPTION_BREAKPOINT` (3)
  (`handlers.S:53-60`). `LD_ADDR_MISALIGNED` (4) and `ST_ADDR_MISALIGNED`
  (6) match none of these and fall into `handle_unknown`
  (`handlers.S:77-82`), which prints `"unknown exception handler
  entered\n"` and jumps to `end_handler_ret` **without incrementing
  `mepc`**. `mret` therefore returns to the exact same faulting
  instruction, which faults again immediately — an infinite trap loop that
  spams UART forever rather than recovering. (There's also a completely
  unused `__no_irq_handler` in the same file that would `jal ra, puts` +
  loop forever without even a `mret`, i.e. total hang — whichever handler
  ends up wired to `mtvec`, neither path resumes execution.)

**So: a misaligned `lw` from the proposed coprocessor operand-fetch path
does not cost extra cycles on this core — it is unrecoverable.** It raises
`LD_ADDR_MISALIGNED`, the installed handler doesn't advance `mepc`, and the
core spins forever re-taking the same trap. This makes the alignment
question load-bearing for correctness, not just performance, for any design
that does a plain `lw` of the 4 int8 operands.

## Q4 — Simplest fix

Two separate problems need two separate fixes; padding/aligning the arrays
alone only solves one of them.

1. **Array-base alignment (weights, biases, `mem`, `inputBuffer`) — solved
   cheaply and is safe to make explicit.** The N2D2-generated headers
   (`conv1.h`, `conv2.h`, `fc1.h`, `fc2.h`) all follow the same
   textual pattern:
   ```c
   static const BDATA_T conv1_biases[CONV1_NB_OUTPUTS] = { ... };
   static const WDATA_T conv1_weights[CONV1_WEIGHTS_SIZE] = { ... };
   ```
   Adding `__attribute__((aligned(4)))` right after `static const` on each
   `WDATA_T`/`UDATA_T` array declaration (`conv1_weights`, `conv2_weights`,
   `fc1_weights`, `fc2_weights`, and — for symmetry/future-proofing —
   `conv1_biases`/etc. even though those are already naturally 4-byte typed)
   is a one-line-per-array change that is mechanically trivial to keep
   applying if/when N2D2 regenerates these files (a single sed/template
   tweak in whatever generates the `static const WDATA_T ..._weights[]`
   line). The same attribute on `static DATA_T mem[MEMORY_SIZE];` in
   `NetworkPropagate.c:13` removes the current dependence on `-O3`
   autovectorization ever deciding to align it — this is the more important
   one of the two, since it stops the alignment from being an accidental
   side effect that a future compiler-flag or GCC-version change could take
   away silently (as demonstrated above: dropping to `-O1` or adding
   `-fno-tree-vectorize` already makes the current alignment vanish). This
   requires no `MEMORY_SIZE`/layout changes and doesn't touch
   `mem_info.h`.
2. **Intra-buffer offset alignment for conv1 — the attribute above does not
   fix this.** Because `CONV1_NB_CHANNELS == 1` and `CONV1_STRIDE_X == 2`,
   `macsOnRange`'s `inputs + iOffset` walks the (aligned) `inputBuffer` 2
   bytes at a time, so roughly half of conv1's MAC windows start 2 bytes
   off a word boundary regardless of how well-aligned `inputBuffer` itself
   is. Realistic options, in order of simplicity:
   * **Scope the custom instruction to conv2/fc1/fc2 only**, where the
     per-call offsets are provably always multiples of 4 (shown in Q1),
     and leave conv1 (11×11×16×16 = 30,976 MACs, vs. much larger conv2/fc1
     layers) on the existing scalar `macsOnRange` C loop. This needs zero
     buffer changes and is the lowest-risk option, at the cost of not
     accelerating conv1.
   * **Pad the raw image to a 4-channel-equivalent stride** (e.g. store
     each grayscale pixel replicated/padded into a 4-byte slot, or
     interleave 4 adjacent pixels into a lane-friendly layout before
     conv1 runs) so `CONV1_STRIDE_X`-driven offsets become multiples of 4.
     This changes `ENV_MEM_STRIDE`/`CONV1_MEM_STRIDE` semantics and touches
     the N2D2 memory-mapping constants in `mem_info.h`, so it's a bigger,
     more invasive change than it looks at first.
   * **Make the coprocessor operand fetch tolerant of the 2-byte case**
     (e.g. two `lh`s instead of one `lw` when the address is only
     2-aligned, falling back to scalar `macsOnRange` only for the fully
     unaligned case) — more hardware/decode complexity, but keeps conv1
     accelerated without touching the memory layout at all.

   Given the ticket's scope (register-only CV-X-IF coprocessor doing a
   straight `lw`-then-MAC), the **recommended minimal fix is (1) plus
   scoping the instruction to conv2/fc1/fc2 (bullet 1 of point 2)**: add
   `__attribute__((aligned(4)))` to the generated weight arrays and to
   `mem[]`, and don't attempt to accelerate conv1's `macsOnRange` calls with
   the new instruction, since conv1's misalignment is structural
   (single-channel input, stride-2 window) rather than a layout artifact
   fixable by array alignment.

## Evidence index

* Build commands and full compiler flags: `sw/app/Makefile:84-118`,
  reproduced verbatim in both the host and Docker builds described above.
* Symbol addresses: `riscv-none-elf-nm -n mnist.riscv` on both the
  host-toolchain and `sw-docker:vfft`-toolchain builds of
  `sw/app/mnist.riscv` (addresses listed above).
* Section alignment: `riscv-none-elf-readelf -S mnist.riscv` —
  `.rodata` (`Al=8`), `.data`/`.sdata` (`Al=8`), `.sbss`/`.bss` (`Al=4`).
* Stack alignment of `inputBuffer`: `riscv-none-elf-objdump -d mnist.riscv`,
  disassembly of `<main>` (`80000120 <main>: addi sp,sp,-624` /
  `addi a0,sp,16`).
* `mem[]` alignment origin: `riscv-none-elf-gcc -O3 -S
  sw/app/mnist/NetworkPropagate.c` vs. the same command with
  `-fno-tree-vectorize -fno-tree-slp-vectorize` added, and vs. `-O1` —
  `.align 2` before `mem:` present only in the plain `-O3` output.
* `macsOnRange` call sites and offset arithmetic:
  `sw/app/mnist/NetworkPropagate.c:31-39` (`macsOnRange` itself),
  `:67-208` (`convcellPropagate1`, calls at `:169-172`, `:194-198`),
  `:210-292` (`fccellPropagateUDATA_T`, calls at `:265-268`, `:282-285`),
  `:423-` (`propagate`, the actual `INPUT_MEM_STRIDE` arguments per layer
  at `:437-441` conv1, `:467-474` conv2, `:500-507` fc1).
  `mem_info.h:9-48` for the `*_MEM_STRIDE`/`*_MEM_CONT_OFFSET` constants,
  `conv1.h`/`conv2.h`/`fc1.h`/`fc2.h` for `NB_CHANNELS`/`STRIDE_X`/
  `CHANNELS_WIDTH` per layer.
* Hardware misalignment trap: `core/load_store_unit.sv:598-669`
  (`data_misaligned_detection`), `core/include/riscv_pkg.sv:329-335`
  (cause codes), `core/include/cv32a6_im_contest_config_pkg.sv` (full file
  — no misalignment-related config field) and
  `core/include/config_pkg.sv` (`cva6_user_cfg_t` struct definition,
  likewise no such field).
* Trap-handler behavior on an unhandled cause:
  `sw/bsp/hal/handlers.S:33-82` (`sw_irq_handler`/`handle_unknown`).
* `MEMORY_ALIGNMENT` unused check: `grep -rn "MEMORY_ALIGNMENT"
  sw/app/mnist/` → single hit, `mem_info.h:8`.
* Linker script (referenced for section placement, not itself the source
  of the 4-byte alignment): `sw/bsp/config/link.ld`, used via
  `RISCV_LDFLAGS=-L./ -lcva6 -static -nostartfiles -T
  $(src_dir)/../bsp/config/link.ld` in `sw/app/Makefile:118`.
