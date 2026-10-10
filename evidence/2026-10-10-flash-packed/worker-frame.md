# Expert worker frame removal

The current worker source keeps the existing `expert_split_worker` entry and six arguments. It expands the owned control and dispatch functions into that entry, uses private packed helpers, and passes logical CTA rank/count through an 8-byte shared context. The native source generator embeds the two GNU awk stages in `recipe.rs`; no additional runtime source file is required. GNU awk is needed when generating a new worker artifact.

## Frame cause

Separate CUDA 11.4 assembly and linking of the compiled root API source produces 252 registers and a 504-byte linked stack. The entry has a zero-byte frame, but its deepest reachable call chain is:

| Function | Frame bytes | Spill stores/loads |
|---|---:|---:|
| expert_split_worker | 0 | 0/0 |
| split_expert_layer | 312 | 284/284 |
| split_mv | 24 | 0/0 |
| packed_matvec | 8 | 0/0 |
| packed_h_13_8_* | 160 | 160/160 |
| Total | 504 | |

The caller retains descriptor and routing state across nested calls. The dynamic packed dispatcher also makes wide helper frames reachable for a batch-1 IQ request. Whole-program assembly makes the stack larger and is not the retained fix.

## Source changes

Private helpers stage inputs without per-column prefetch register arrays. They decode each 32-value weight group once, retain its packed int16 operands in shared memory, and consume that group across all active columns. Per-thread accumulators also stay in shared memory. IQ4_NL's 16-entry codebook is initialized once per helper invocation. The fused IQ2/IQ3 implementations remain the signed-stream helpers from 5817876.

The public helper bodies and ABI remain unchanged. Private worker helpers use XMAD for the worker's kinds 0/2. The entry requires 256 threads, matching the existing resident launch. The dynamic shared contract remains 40960 bytes.

Shared offsets are relative to dynamic `scratch`. Intervals are half-open:

| Storage | Byte interval at maximum capacity |
|---|---|
| Packed inputs and scales | [0, 11264) |
| IQ4 codebook | [11264, 11328) |
| Original source-column indices | [12288, 12320) |
| Per-thread accumulators | [16384, 24576) |
| Per-thread decoded packed weights | [24576, 40960) |

The private rank/count context uses 8 bytes of static shared memory. Thread 0 stores the uniform values, and every thread crosses a CTA barrier before the typed call. Private G/H calls have eight arguments, and private fused calls have seven, avoiding the outgoing local argument area. This private ABI is confined to the generated worker module.

## CPU compilation receipts

The release library build succeeds on archy. Calling the public `recipe::expert_worker_ptx()` from that library and assembling its emitted source with `ptxas -c -arch=sm_52 -O1`, then linking with `nvlink --kernels-used expert_split_worker`, produces:

| Resource | Before | After |
|---|---:|---:|
| Linked registers | 252 | 195 |
| Linked stack bytes | 504 | 0 |
| Static shared bytes | 16 | 80 |
| Global bytes | 1087632 | 1087632 |

All 105 private helpers report zero frame and spill bytes. The entry reports zero frame and spill bytes. The linked SASS has zero LDL/STL instructions. The layout/mask/row-partition CPU check reports 8 layouts, 3216 source positions, and 2785280 row positions with zero mismatches. This check covers addressing and partition geometry, not complete GPU arithmetic.

Artifacts are under `/home/nate/codex/cx-fl-kern-build/worker-frame/` on archy. `shared-worker.ptx` and `shared-worker.cubin` come from the compiled public API, not a replacement test entry.

## Rebuild

The sole Makefile's `packed-worker-helpers` target regenerates the private helper section from the existing CUDA source. It preserves the hand-owned public PTX section. Build the release library, emit `recipe::expert_worker_ptx()`, and assemble/link using the worker's existing flags. The embedded expansion stages use the current control source each time; the worker entry is not a frozen copy of an older root.

No GPU execution, independent context, model attachment, or benchmark ran for this candidate. GPU arithmetic, masks, throughput, and final logits require cx-flash's resident-owner reload. The instruction ratios in the signed-stream handoff remain optimized inner-loop counts, not linked-worker performance. PR #1097 remains draft, and nothing is merged.
