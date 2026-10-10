# Dense split size policy

This checkpoint follows the CEO's size-policy correction in issue 1092, comment 6102648385, and its CPU proposal in comment 6102767239. The earlier deeper batching prototype is not shipped.

Dense-on-main stays the default baseline. Set `RECIPE_DENSE_SPLIT=1` to enable large-matrix row splitting. `RECIPE_DENSE_SPLIT_MIN_BYTES` defaults to 8000000; only mapped biasless Contraction jobs with combined packed/raw matrix bytes strictly above this threshold become external jobs. QKV's already-ready mapped planes share one source and one transport point. Set the minimum to zero only when explicitly reproducing the former all-matrix split policy.

The actual target goes from 629 external dense points to 134, leaving 495 dense jobs inline on main. Its 48 expert points remain, for 182 external target points per decoded token. The MTP graph has seven large split calls, including four uses of the same EH projection; state-only refresh omits its terminal vocabulary job. Shared expert gate/up/down tensors are below 8 MB in this file, so the literal threshold leaves them on main. Hyper down/up matrices are 3481600 bytes each.

The current fixed node order has no adjacent independent pair among the 134 selected target jobs. QKV is already a multi-plane job. No speculative execution of a dependent projection is introduced. The CPU lowered graph has a syntactic dense dependency depth of 387 across all 629 jobs; a plain ready-list batch did not establish the earlier 100-150 estimate. The size filter itself reaches that range.

The proposal was posted before changing production source. Its node decisions and placement use real GGUF metadata, context 128, five-position windows, the current 128 MiB execution reserve, and saved post-context free bytes. Main's assigned packed/raw weight bytes are 7034930432, of which 5264409600 are resident experts. Main reserve is 1415837066; pinned RAM spill is 7331302400 with zero unplaced bytes. These are CPU plans, not a runtime rate or allocation receipt.

This policy changes placement. Existing row-split weights cannot be converted into full main-resident small matrices through module reload alone. Root owns consolidation or the next startup. Before a full load, the exact frozen executable must pass the CEO's two-layer real-shape main-die-4 plus one-worker end-to-end gate: 16-token prefill, two decode steps, at least one worker-resident MoE layer, and finite logits. This track launches no independent GPU check or full load.

Baseline and policy release/CPU receipts are recorded here. No 30 ms/token Contraction result, 60/80 tok/s result, or final-logit acceptance is claimed.

Codex session: 01a125be-b8d6-74f1-b49a-96c89cb9263a.
