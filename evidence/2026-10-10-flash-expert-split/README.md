# Flash-Next expert split

This source track supplies placement, packed per-die loading, and callable transport to cx-flash's holder and forward pass on issue #1092. The branch starts from minimal after merged #1093. The forward pass starts from PR #942 in cx-flash's integration branch.

## Placement

The inventory uses revision `766911a6b7369840a91dbcd95f9f997acaab6cd6` of the three UD-Q2_K_XL shards. `Gguf::tensor_headers` reads the actual headers without validating or reading unfinished payloads. The target has 1224 tensors; the requested MTP header has 34. Packed expert matrices account for 46084915200 bytes. Target nonexpert tensors plus the complete MTP head, including its own Q8 output, account for 6987059712 bytes. Token and PLE lookup tables remain mapped in RAM.

[placement.md](placement.md) contains the live free-VRAM table and the physical-capacity upper bound. Both respect the existing conservative die-0 usable cap. The main-die reserve is 256 MiB KV plus 256 MiB working storage; each expert die reserves 256 MiB working storage. Neither table is a resident fit. Under the upper bound, 7686374400 expert bytes remain unplaced. The loader rejects this plan before GPU allocation.

`placement.rs` contains the CPU checks used on Archy. It reads the target headers and `mtp-header.partial` at `/mnt/sentry-nfs/flash-next-q2`, and the measured VRAM CSV at `/home/nate/codex/cx-fl-split-build/vram.csv`. It checks byte conservation, all 24576 expert bundles under a separate roomy fixture budget, same-die matrix placement, alignment, selected-slot mapping, invalid IDs, and forbidden main dies. The roomy fixture is not an Archy capacity claim.

## Holder interface

Retain `ExpertSplitWeights` across requests and module replacement. `load` copies exact packed GGUF slices without dequantization. It checks the six permitted UUIDs, changed free VRAM, download control files, source descriptors, and bidirectional CUDA peer access. Failed creation drops new allocations.

- `weight_address(mtp,name,expert)` returns physical die, pointer, packed bytes, and GGML type.
- `owner_address(layer)` returns the resident main-die u32 expert-owner table.
- `response_table_address()` returns resident `{payload:u64,sequence_address:u64}` records for final combine.
- `channel(index)` returns the request packet, response packet, and worker route pointer. Channel order is physical dies 0,1,3,4,5; main is die 2.
- Route IDs use native count-plus-ID records, with stride `top_k+1`, at route+0. Compact FP32 coefficients use stride `top_k` at route+512. Capacity is five positions and top_k at most 16.
- Packets reserve five 10240-row FP32 positions. The sequence address follows the allocated payload. Use `packet_rows=4096` for 2560-row target expert streams, or `10240` for the wider stream.
- Reserve a checked sequence range once before queuing a token. Layer l uses base+l+1.

The root `barrier.ptx` is the owned runtime source. Append `p2p_functions()` followed by `expert_split_functions()` to the holder's persistent module. Evidence files are not runtime dependencies.

## Callable transport

All pointer parameters below are u64; dimensions and sequences are u32; deadlines are u64 GPU globaltimer nanoseconds.

- `split_send(hidden,selected,coefficients,owners,request,records,routing,slots,remote_sequence,sequence,deadline,die,experts,top_k,rows,positions,packet_rows) -> u32`. Main sends compact routes and complete active hidden columns, then calls the merged P2P publication. Main completion slots come from the channel's response packet. Inactive positions still publish their empty route.
- `split_local_combine(records,coefficients,expert_outputs,partial,top_k,rows,positions) -> u32`. Expert CTAs write disjoint `[position,compact_selected_slot,row]` planes. After the caller's grid barrier, this reduces the planes on the worker. Zero-count records produce zero. Each expert can use one CTA and read packed weights once for every active position through the packed/MTP helpers.
- `split_return(partial,response,slots,remote_sequence,sequence,deadline,rows,positions,packet_rows) -> u32`. After the worker's grid barrier, copy the entire batch and publish its response. Worker completion slots come from the request packet.
- `split_combine(responses,output,sequence,deadline,rows,workers,positions,packet_rows) -> u32`. Main waits for every local response flag and sums in fixed die order. Preserve the caller's grid barrier before the next operation consumes output.

Queue one persistent grid per die per token, with at most one resident CTA per SM. All participating CTAs call transport collectively. The holder owns grid completion between matrix production, worker reduction, response copying, and subsequent main operations. No machine action or peer atomics are added between layers.

## Evidence boundaries

The library and relocatable owned PTX assemble. `split-link.log` records the current relocatable object: send has an 8-byte stack frame and four spill bytes; main combine has a 32-byte stack frame and 28 spill bytes. These are out-of-line ABI costs. The final composed persistent module needs its own register/spill check and real-forward timing.

The first source snapshot, d1768e9, had a sequential scalar accumulation fixture. Its linked entries assembled without stack/spills, at 31/27 registers. The GPU job ran only on the assigned die-1/die-2 UUIDs under the shared die1 lock. It exited 101 with a timeout on its first token. No measured token or passing payload result exists. `timeout/` preserves the exact generated PTX, hashes, preflight, compiler receipt, CSV header, exit code, and error. The saved Rust and entry sources describe that failed snapshot. The scalar accumulation helper is superseded in the current runtime source by `split_local_combine`; the saved fixture is not a second runtime path.

Following the subsequent #1092 instruction, no more isolated GPU fixtures are queued. Further measurements belong in cx-flash's real forward pass. Resident full-weight loading, real-router dispatch/combine overhead, whole-model token rate, and final-logit agreement remain unverified. Die 7 stays unused pending clarification of the direct continuation instruction and the newer issue placement decision. Die 6 remains fenced.

PR #1095 is a draft handoff, not a readiness declaration. It must not merge from this track.
