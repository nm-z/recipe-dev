# Owner-key correction

The CEO correction restores layer-only owner-map keys. This MTP checkpoint uses block 48; target experts use blocks 0 through 47. Owner-map construction, stored keys, lookup, and descriptor binding all use the layer index. Matrix bindings continue to carry their existing model identity.

This follow-up changes those five owner-key sites. Dense row-split dispatch, output-row mapping, placement, pinned RAM spill, and the `selected_experts`/`weight_bytes` trace columns remain in the dense checkpoint. The worker protocol and GPU descriptor layout do not change.

The current root load can continue. This source correction requires no GPU reset, reattachment, or interruption. Its CPU library/CLI check receipt is recorded separately from the earlier checkpoint's build and hardware evidence.

Codex session: 01a125be-b8d6-74f1-b49a-96c89cb9263a.
