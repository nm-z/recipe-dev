The remaining sweeps finish with exit 0. No new NVRM/Xid appears in any sweep interval. All three address-dependent patterns match every checked 32-bit word. Power and application clocks remain unchanged.

| Physical die | Usable limit bytes | Checked payload bytes | Reserved/unallocatable within limit | Result |
|---:|---:|---:|---:|---|
| 0 | 6186598400 | 6112411648 | 74186752 | PASS |
| 1 | 8524136448 | 8449949696 | 74186752 | PASS, kernel-track receipt |
| 2 | 5798205440 | 0 | Not measured | CUDA context creation fails 214 |
| 3 | 7988903936 | 7914713088 | 74190848 | PASS |
| 4 | 8524136448 | 8449949696 | 74186752 | PASS |
| 5 | 8524136448 | 8449949696 | 74186752 | PASS |
| 7 | 8524136448 | 8449949696 | 74186752 | PASS |

Die 6 is absent from the allowed UUID table and every invocation. Die 0's excluded band is 2337538048 bytes. The die-2 probe never reaches allocation. The stopped applications remain stopped; no GPU model or sweep remains from root.

Root source/receipts: `/home/nate/codex/flash-1092-evidence/capped-memory-sweep`; the remaining-die receipts are in its `remaining` directory. Die 1's independent receipt is `/home/nate/codex/cx-fl-kern-build/memory-pattern-results.csv`. The three patterns use seeds aaaaaaaa, 55555555, and 12345678 with an address-dependent mix. These are memory checks, not matvec/model rates.

The next integrated startup still needs the die-2 recovery/fence decision, the full dense/expert/MTP dispatch source, and its final zero-unplaced table. The cold spill ranking is ready in `reference-router-only/spill-frequency.tsv`. No IPC export or live driver-call capture is used.
