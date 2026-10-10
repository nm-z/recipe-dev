# Delivery order

The caller fixes are isolated in 65b7591b355421076dbf2e7044b610c252d19029, directly on 39527ec41429a138fc00d38962df95af61cc8a5e. That first commit contains no dense split and preserves eager compilation, reply-zero/token-ID controls, and draft-seconds reporting.

This second commit adds the remaining dense checkpoint on top of the caller-only commit. Its runtime source files are byte-identical to the previously pushed db70a8ded2dc5f0f4d0315dc575976ad65066f31 checkpoint, including the restored layer-only owner keys and selected_experts/weight_bytes trace columns. The committed build, caller, worker, placement, and owner-correction receipts apply to that source. The placement remains zero-unplaced with 6727680000 pinned RAM bytes.

Apply the caller-only commit first for the immediate restart. Add this second commit when root is ready to change the graph and resident allocation layout. Do not also apply a0664f03, 56249db, or db70a8d to this sequence; their source is already represented here.

GPU arithmetic, router dispatch/combine overhead, whole-model rate, and final logits remain acceptance gates owned by the integrated run.

Codex session: 01a125be-b8d6-74f1-b49a-96c89cb9263a.
