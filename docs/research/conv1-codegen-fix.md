# conv1 stationary-word byte-packing: codegen fix

Research for GitHub issue #31 (map #2, spec #24), opened off the back of
issue #30's cycle profile (`docs/research/mac4-v2-cycle-profile.md`, branch
`research/mac4-v2-cycle-profile`). That profile found: **90% of conv1's
stationary-load instructions (8712 of 9680, 4.9% of the entire network's
dynamic instruction count) are pure byte-packing overhead** — `4x lbu + 4x
sb (spill to stack) + 1x lw` to assemble one 4-byte `input_word` in
`convcellPropagate3` (`sw/app/mnist/NetworkPropagate.c`), where a direct
`lw` should suffice, because ticket #28 already guarantees the access is
4-byte-aligned at runtime for every `(oy, ox)` — GCC just can't *prove* it
statically. This ticket's job: find and verify the minimal-diff codegen fix.

**Method**: built via `sg docker -c "docker run --rm -v $(pwd):/workspace -w
/workspace sw-docker:vfft bash -c 'make benchmark APP=mnist'"` (same
`sw-docker:vfft` image, same stock flags — `-march=rv32im_zicsr -O3
-fno-tree-loop-distribute-patterns -funroll-all-loops -falign-jumps=4
-falign-functions=16` — as every prior ticket, nothing added), disassembled
with `riscv64-unknown-elf-objdump -d`/`-t` (local toolchain,
`/opt/riscv/bin`), correctness/cycles confirmed by
`source ~/questa_setup.sh && make sim APP=mnist batch-mode=1`. All work done
on a throwaway branch `research/conv1-codegen-fix`, off `cv32a6_contest_25_26`
at commit `14fc348` (tickets #25-28 already merged); nothing pushed, nothing
touched on `cv32a6_contest_25_26`.

## 0. Reproducing the baseline

`propagate()` is still one fully-inlined 0x3218-byte function (no separate
`convcellPropagate3` symbol — confirmed via `objdump -t`). The conv1
stationary-word load site, `NetworkPropagate.c:146-149`
(`memcpy(&input_word, input_base + iOffset, sizeof(input_word));
mac4_load_stationary(input_word, 0);`), disassembles as:

```
800017e0: lbu s11,0(a2)      800017f0: sb s11,60(sp)     80001800: lw s11,60(sp)
800017e4: lbu s10,1(a2)      800017f4: sb s10,61(sp)     80001804: .insn 4, 0x000d900b   <- LOAD_STATIONARY
800017e8: lbu s9,2(a2)       800017f8: sb s9,62(sp)
800017ec: lbu t3,3(a2)       800017fc: sb t3,63(sp)
```

`0x000d900b`: opcode bits `[6:0] = 0x0b` (custom-0), `funct3 = (word>>12)&7
= 1` → `LOAD_STATIONARY`, matching `sw/app/mnist/mac4.h`'s
`.insn r 0x0B, 1, %1, x0, %0, x0` encoding and confirming this is the exact
site the ticket names — 9 instructions to assemble a word that ticket #28
proves is already aligned, before the 1 real `LOAD_STATIONARY`.

## 1. Candidates tried

**Candidate 1 — `__builtin_assume_aligned` on the `memcpy` source pointer**
(the ticket's first suggestion) was tried first and worked immediately, so
candidates 2 (value-range hint via `__builtin_unreachable`) and 3
(`may_alias`-typed pointer-cast dereference) were never needed — there was
no remaining problem for them to solve. Candidate 4 (heavier restructuring)
likewise wasn't needed; its cost is not reported since it wasn't built.

### The fix

```c
uint32_t input_word;
memcpy(&input_word,
       __builtin_assume_aligned(input_base + iOffset, 4),
       sizeof(input_word));
mac4_load_stationary(input_word, 0);
```

A 2-line diff against the committed `NetworkPropagate.c` (only the `memcpy`
call site changes — nothing else in the file, no new helper types, no
restructuring):

```diff
                     uint32_t input_word;
-                    memcpy(&input_word, input_base + iOffset,
+                    memcpy(&input_word,
+                           __builtin_assume_aligned(input_base + iOffset, 4),
                            sizeof(input_word));
                     mac4_load_stationary(input_word, 0);
```

Committed as `957dd99` on `research/conv1-codegen-fix`.

**Why this was the missing piece**: GCC can prove alignment through a
`memcpy` source pointer only when it can see, at the call site, that the
pointer's value is provably a multiple of the target alignment — normal
routes are a `__attribute__((aligned(N)))`-typed base plus a
compile-time-constant offset, or a value-range/divisibility fact it derived
itself. Here the base pointer (`input_base`) is chosen by a *runtime*
branch between two different `aligned(4)` arrays (`inputs` vs.
`conv1_input_shifted`), and the offset (`iOffset = ix_base +
CONV1_CHANNELS_WIDTH * (iy + sy)`) mixes a runtime `ix_base` with a partly
runtime `sy` term — even though a human can trace through ticket #28's
invariants and show the sum is always a multiple of 4, GCC's alignment
analysis doesn't propagate divisibility facts through a runtime-selected
base pointer plus a multi-term runtime offset that far. `__builtin_assume_aligned`
sidesteps needing GCC to derive that itself: it's a direct assertion attached
to the exact pointer value handed to `memcpy`, which the alignment-sensitive
lowering (deciding whether a `memcpy` of a compile-time-constant size can
become a single load/store instead of a byte copy) trusts without needing a
proof chain.

### Disassembly after the fix

Same site, rebuilt with only the 2-line diff above:

```
800017fc: lw ra,0(a4)              <- single lw, direct 4-byte load
80001804: .insn 4, 0x000d900b      <- LOAD_STATIONARY (unchanged encoding)
```

9 instructions (4 `lbu` + 4 `sb` + 1 `lw`) collapsed to 1 (`lw`) — an 8
instruction/occurrence reduction, confirmed by a whole-function scan: static
`lbu` count in `propagate()` drops from 195 (baseline) to 4 (fixed build);
all 4 remaining `lbu`s are fc2's unrelated 2-element scalar-remainder tail
(`FC2_SCALAR_REMAINDER`, `NetworkPropagate.c:419-423` — a pre-existing,
documented, different code path, not conv1's stationary load), confirmed by
address (`0x80004148`/`0x800047b0`, inside `fccellPropagateDATA_T`'s inlined
body, nowhere near conv1's `0x800017xx` region) and by count (`150 = 37*4 +
2` → exactly 2 leftover elements × 2 tile bodies = 4 `lbu`, matching the
cycle-profile doc's own accounting of that path). **Zero** `lbu`/`sb`
byte-packing instructions remain anywhere in conv1's inlined body.

### Instruction-count/code-size delta

Decoding every custom-0 (`0x0B`) instruction in `propagate()` by `funct3`
(same method as the cycle-profile doc's §1):

| funct3 | op | baseline static count | fixed static count |
|---|---|---|---|
| 1 | LOAD_STATIONARY | 143 | 146 |
| 2 | MAC_TILED | 874 | 898 |
| 3 | READ_ACC | 40 | 40 |
| 4 | RESET_ACC | 6 | 6 |

LOAD_STATIONARY/MAC_TILED both grew by a small, exactly-consistent amount
(+3 and +24 = +3×8 slots): removing 8 instructions from conv1's stationary
load site shrank that loop's per-iteration body enough that GCC's
`-funroll-all-loops` cost heuristic now fully unrolls conv1's `sy` loop
(trip count 4) into 4 static copies instead of leaving it a 1-static-site
backward-branch loop — a side effect of the fix, not a regression: it also
deletes that loop's `bne`/induction-variable overhead entirely for conv1,
on top of the per-word saving.

Whole-binary `.text` size (`riscv64-unknown-elf-size sw/app/mnist.riscv`):

| | baseline | fixed | delta |
|---|---|---|---|
| `.text` | 143692 bytes | 143884 bytes | **+192 bytes (+48 instructions)** |
| `propagate()` size | 0x3218 (12824 B / 3206 instr) | 0x32e0 (13024 B / 3256 instr) | +200 B / +50 instr |

So the *static* code grows slightly (full `sy`-loop unrolling for conv1
adds more copies of a now-much-smaller body than the loop overhead it
removes costs), entirely confined to `propagate()` — every other function in
the binary is byte-identical. The *dynamic* win is what matters here and is
much larger: each of conv1's 968 dynamic stationary-word loads
(121 `(oy,ox)` × 2 tiles × 4 `sy`, per the cycle-profile doc) drops from 10
executed instructions (9 byte-packing + `LOAD_STATIONARY`) to 2 (`lw` +
`LOAD_STATIONARY`), i.e. removes 8×968 = 7744 of the profiled 8712
byte-packing instructions outright (the remaining ~968 were never
byte-packing to begin with — they're the `LOAD_STATIONARY`s themselves,
which the profile already counted separately), plus removes conv1's `sy`-loop
branch/induction overhead for the now fully-unrolled loop.

## 2. RTL verification

`source ~/questa_setup.sh && make sim APP=mnist batch-mode=1` (Questa/UVM
sim; note this worktree needed `git submodule update --init --recursive`
first — `core/cache_subsystem/hpdcache` and the other RTL submodules weren't
checked out by default in the isolated worktree). Full run, ~19.5 minutes:

```
[UART]: MAC4 smoke test: PASS (expected -18, got -18)
[UART]: MAC4 v2 smoke test: PASS (expected acc[0]=-44 acc[1]=10, got acc[0]=-44 acc[1]=10, post-reset acc[0]=0 acc[1]=0)
[UART]: conv1: 64602 cycles
[UART]: conv2: 197593 cycles
[UART]: fc1: 90439 cycles
[UART]: fc2: 3329 cycles
[UART]: Expected  = 4
[UART]: Predicted = 4
[UART]: Result : 1/1
[UART]: credence: 82
[UART]: image env0003: 238345 instructions
[UART]: image env0003: 521742 cycles
```

Both smoke tests PASS, `Result : 1/1` (correct prediction), confirming the
fix changes nothing about correctness — expected, since
`__builtin_assume_aligned` only asserts a fact that was already true at
runtime (ticket #28's invariant); it adds no new behavior, just removes a
defensive fallback the compiler no longer needs.

**conv1 cycles: 80305 (baseline, per issue #30's profile) → 64602 (fixed) —
a 15703-cycle / 19.6% reduction**, larger than the instruction-count
prediction alone would suggest (removing ~7744 of ~238345 total dynamic
instructions is ~3.2% of the *network's* instructions, and conv1 specifically
loses a much larger fraction of *its own* instruction count — the profile
doc's bucket (a) conv1 total was 9680 out of a conv1 total that's a small
slice of 238345). The extra cycle win beyond raw instruction count is
consistent with `sb`-then-immediate-`lw` (store-then-load) sequences
being exactly the kind of pattern an in-order-ish pipeline pays extra
stall cycles for beyond their raw instruction count (RAW hazard through
memory, not just through registers) — removing them wins more per
instruction than an average instruction removal would. conv2/fc1/fc2 are
unchanged (197593/90439/3329 — within noise of the profile's baseline
197614/90480/3326, as expected since this fix touches only conv1's source).

## 3. Conclusion / disposition

`__builtin_assume_aligned` alone is a clean, minimal (2-line), verified fix:
removes exactly the targeted byte-packing overhead (confirmed by disassembly
and by an exhaustive `lbu` scan of the whole function), costs +192 bytes /
+48 static instructions of `.text` (all in `propagate()`, from GCC choosing
to fully unroll conv1's now-cheaper `sy` loop rather than any inherent cost
of the fix itself), and is RTL-verified correct (`Result: 1/1`, both smoke
tests PASS) with a real 19.6% conv1 cycle reduction. No heavier
restructuring (candidate 4) was needed or evaluated.

This looks ready to fold into a future implementation ticket essentially
as-is — it's a 2-line, self-contained, already-RTL-verified change with no
loose ends. The one thing worth double-checking in that implementation
ticket rather than assuming from this research pass: this was verified
against the single `env0003` test image (the same one every prior mnist
ticket in this map has used) — if the implementation ticket's acceptance
bar includes a broader image set, it should re-run against that set, though
there's no mechanism by which this change could be image-dependent (it
changes zero runtime behavior/values, only how one already-aligned load is
encoded).
