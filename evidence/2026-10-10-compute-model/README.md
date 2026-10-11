# Per-device compute prediction

Refs #979. `predict-compute.mjs` updates the existing evaluator from
`/home/nate/codex/flashnext/probes/compute-model/predict-compute.mjs`.

The evaluator reads `precision node` and `probe phase` records from the same
capture. Each phase joins its device and node index to that device's precision
record. Preliminary graph records without placement fields are skipped. The
input JSONL corpus and split-region dimension estimates are removed.
Rows and terms come from `local_rows` and `local_terms`. Mixed-format rows come
from captured stored weight counts divided by the captured local term count.
SASS lookup uses the captured `run_key`; a mixed-format node uses its segment
decoder names. The fit still alternates training and held-out nodes per format,
per device, and per trace.

Captures must include `device`, `stored_formats`, `stored_bytes`, `shard_rows`,
`shard_terms`, `local_rows`, `local_terms`, `run_key`, and `packed`. Local counts
must agree with the recorded shard runs. Missing fields, conflicting captures,
and unmatched phase indices fail the run. Historical captures without those
fields cannot establish acceptance.

## GPU acceptance pending

The CEO owns GPU execution on Archy. Capture Flash Next 4-block and Qwen-27B on
`nv0.nv4`, with generation budget 2, `RECIPE_TRACE=1`, and the current
`recipe.rs` and phase-instrumented `amd-nv-cpu.ll`. Preserve both precision
records and phase clocks in each trace. Phase indices must match the zero-based
precision node indices. Preserve SASS dumps of the cubins named by those same
captures under `/home/nate/codex/cx-979-evidence/gpu/`.

After those captures exist, run this command from the PR checkout:

```bash
M60_SM_CLOCK_MHZ=1174 node evidence/2026-10-10-compute-model/predict-compute.mjs /home/nate/codex/cx-979-evidence/gpu/fn4-nv04-phase-budget2.trace /home/nate/codex/cx-979-evidence/gpu/q27-nv04-phase-budget2.trace -- /home/nate/codex/cx-979-evidence/gpu/*.sass.txt
```

Expected result: exit 0, a finite held-out median absolute percentage error of
25% or less, nonzero supported held-out nodes, and zero unmatched phase nodes.
The report prints the supported node count and per-format training and test
counts. Review those counts with the error result.

The evaluator also accepts `.cubin` files after `--`. Set `CUOBJDUMP` to an
installed disassembler that supports `sm_52`. Benji's CUDA 13.4 `cuobjdump`
invocation rejects that architecture. Saved `.sass.txt` dumps let the same
evaluator run on a CPU machine without disassembling again.

## CPU verification on Benji

Build with `CARGO_BUILD_JOBS=3` and
`CARGO_TARGET_DIR=/home/nate/work/target-cx-recipe979`, using the release profile.
Delete that target directory after the PR is merged or closed. Evidence
directories contain logs and small outputs.

The evaluator's syntax check and historical-trace rejection run on Benji.
The original Flash Next capture fails with `no captured per-device compute
nodes` before SASS processing. CPU execution does not establish the held-out
GPU error result.
