# Dense and expert forward checkpoint

## Source and interface

This checkpoint starts from root 39527ec, including the new fused packed library and integrated expert-forward source. It reconciles source checkpoint 56249db with root's full fencing, report field order, and worker compilation before weight loading. Apply this current-root commit by itself. The holder retains a single target/head `ExpertExecution` and the original GGUF payloads. Ordinary NativeTape buffers hold descriptors for external weighted nodes, with no second upload of their matrices.

Biasless matrix Contraction nodes and fused ExpertOut nodes share one ordered job stream. Every participating die uses one 256-thread persistent grid per forward invocation, across all position chunks and jobs. Dense workers prepare the source once, compute their resident row slice over all 16 SMs, and write disjoint rows directly into the main output before publishing completion. The main waits for every response before the consumer runs. Interleaved attention views use a checked output-row map; they do not require separate small matvecs. F32/BF16 matrices use direct typed arithmetic, and quantized matrices use the canonical fourteen-input helper and measured 256-thread tuples. Dynamic shared storage is 40960 bytes.

The topology is physical main die 4 and dies 0/1/3/5/7, with physical 2/6 fenced before any context or allocation. Die 0 retains the UUID-bound usable cap. The plan spills only explicitly recorded zero-pick target expert bundles, excluding layer 47 and all MTP experts. Missing observations remain resident. The input reference is read-only; live router counters start at zero and write a separate CSV. `ExpertExecution::finish` collects device status and operation observations without downloading or writing the router counters. The CLI exports once after the timed request, or on `/report`; public callers can use `Placed::export_router_picks()`.

The actual native caller compilation found and fixed the old expert caller's undefined LLVM shared-storage reference and absent PTX call declaration. Packed functions use their own external `scratch[]`; the legacy explicit shared pointer is unused and receives zero. The dispatcher declaration precedes generated callers. The device linker names the real `recipe_model_forward`/`recipe_model_load` entries.

## Evidence

- `plan.rs` runs the actual lowered target/head model on CPU, context 128 and five-position windows, using root's saved post-context free-byte budgets and reference-router seed.
- `placement.md` prints storage, reserve, RAM spill, dense read bytes, and reference selected expert bytes. These are a deterministic plan using saved budgets. Actual startup recomputes free VRAM.
- `dense-slices.tsv` records 3914 resident row slices. Planning verifies that every output channel of every external dense job has exactly one die owner, including all attention permutations.
- `plan.exit` is 0: zero unplaced bytes and 6727680000 pinned RAM bytes.
- The [original caller receipt](https://github.com/nm-z/recipe-dev/blob/a0664f03ad3d4f8ab3f6d2fa84046b3a8bbf4098/evidence/2026-10-10-flash-expert-forward/dense/caller.log) and `caller.exit` record an actual NativeModelIr to LLVM to PTX to device-linked cubin compilation of a dense native caller. The focused fixture checks that external weights use an eight-byte descriptor plus existing buffer slack, and that this module has no incompatible single-token entry. That receipt uses compile-only assembly with the installed CUDA 11.4 toolkit and creates no CUDA context or model allocation. The imported inline fixture is removed from this integration source; root now compiles the actual target/head before weight upload.
- `assembly.log`/`assembly.exit` record complete composed worker assembly and device link. The retained worker resource report is in `assembly.log`; the dynamic shared allocation is 40960 bytes.
- `artifacts.txt` records source and artifact hashes and retained worker, dense, dispatcher, and native forward symbols.

No GPU invocation or real dispatch/combine overhead follows from these receipts. Root owns the combined restart and GPU lock. The next model run must verify actual DenseRowSplit and ExpertSplit observations, status for every die/phase/CTA, generation rate, and final logits. `/report` now exports actual selected expert counts and read bytes. MTP refresh changes that omit projections must preserve the shared worker job range and main call order; a module reload cannot change this graph/allocation layout.
