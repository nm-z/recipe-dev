cx-fl-p2p: **8,000,000-byte threshold proposal, before production implementation**, calculated with the actual target/head binding and current integrated source dda97818. This is a CPU-only policy simulation; no model load or GPU job was launched.

| Model | Original dense jobs | Proposed split jobs / unbatched round trips | Main-inline dense jobs | Main-inline matrix bytes per invocation |
|---|---:|---:|---:|---:|
| Target | 629 | 134 | 495 | 1181230080 |
| MTP | 20 | 7 | 13 | 34718720 |

The target therefore drops from **629 to 134 dense split points before any batching**, within the earlier 100-150 range. Including its 48 expert points gives **182 external job points per decoded target token**. The threshold selects the large QKV/gate/output matrices, the large PLE key projection, and the vocabulary head. Actual shared gate/up/down matrices are about 1.1-1.3 MB each and stay main-inline under this literal rule. Hyper down/up are 3,481,600 bytes each and stay main-inline. QKV is one mapped Contraction job, so its combined ready planes use their aggregate bytes.

The MTP head has seven split calls, including four calls to the same 13,926,400-byte `nextn.eh_proj` tensor. Its call/read totals therefore differ from unique resident bytes. State-only refresh omits its final vocabulary split job.

**Full target+MTP main-die placement:**

| Main die 4 quantity | Bytes |
|---|---:|
| Assigned packed/raw weight bytes, including experts | 7034930432 |
| Resident expert bytes within that assignment | 5264409600 |
| Non-expert assignment, including main's large row slices and alignment | 1770520832 |
| KV, request, module/scratch, and native-storage reserve | 1415837066 |
| Total assigned plus reserve | 8450767498 |
| Saved post-context free budget | 8450998272 |
| Remaining after reserves | 230774 |

**Zero unplaced bytes. Total explicitly cold pinned-RAM spill: 7,331,302,400 bytes.** This uses the current 128 MiB per-die execution reserve and saved free-byte budgets; live startup must recompute free bytes. The dense-on-main target+MTP baseline previously reported main reserve 1,430,938,826 bytes and 7,346,252,800 pinned bytes. These are placement figures, not measured speedups.

The policy changes which full matrices must reside on main. It cannot be enabled by a module-only reload into the old six-way split weight layout unless the holder also reassembles those main weights. I am preserving the discarded batch prototype separately and have not altered root's holder. The size-policy implementation follows this report, and the CEO's exact-binary two-layer multi-die gate remains required before any full load.

Raw per-node choices and CPU receipt: `/home/nate/codex/cx-fl-dense-batching-build/threshold-{target,mtp}.tsv`, `threshold-plan.log`, `threshold-plan.exit` (0). Code is only in the scratch proposal build at this stage.

Codex session: 01a125be-b8d6-74f1-b49a-96c89cb9263a.
