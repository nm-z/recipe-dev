# Source-exact 381b10ee misaligned access

On 2026-10-10, the renewed assignment requests the one-layer real-shape memcheck from CEO comment 6100908729. This run uses only physical die 1, UUID GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28, under `/home/nate/codex/flash-1092-die1.lock`.

The production worker module is assembled from the five source fragments used by `ExpertExecution::worker_source()` at 381b10ee666db7b8ff843e42a19da177610f4c32: peer functions, expert transport, MTP functions, packed helpers, and forward glue. Line annotations add debug information without changing instructions. Real layer-0 IQ2_XS gate/up matrices are 640x2560; IQ4_NL down is 2560x640. The fixture launches the original `expert_split_worker` with production 192-byte descriptors, 16 CTAs, 256 threads, and 40960 dynamic shared bytes.

## Fault

The original run at 15:28:06-15:28:08 PDT exits 86:

```text
Invalid __global__ read of size 4 bytes
at 0x10 in .../worker.ptx:15:p2p_wait
by thread (0,0,0) in block (14,0,0)
Address 0xf00000001 is misaligned
ERROR SUMMARY: 257 errors
3840 errors were skipped (error buffer overflow)
```

Kernel: `expert_split_worker`. Device function: `p2p_wait`. The sanitizer's reported PTX line is 15; the flag read in the original source is line 17, `ld.volatile.global.u32 value,[pointer]`. Disassembly confirms PC 0x10 is `LDG.E.CV R0,[R4]`. Line information associates that scheduled instruction with the earlier parameter line.

The fault occurs at the first internal completion barrier. `split_grid` stores outgoing parameters once, calls `p2p_publish`, and then reuses those parameter objects for `p2p_wait`. The publication call overwrites their backing storage; the wait receives 0xf00000001 as its flag address.

## Fix and verification

The independently published fix is 3927904e7decb92c12c2bc1410c145b25f248f2d. It checks publication's result, opens a fresh parameter scope, and restores the wait address, sequence, and deadline from preserved registers. The repaired module differs from the original only in `split_grid`.

| Source | Real experts | Columns | Memcheck errors | CTA statuses | Response sequence | Finite returned words |
|---|---:|---:|---:|---|---:|---:|
| Original 381b10ee | 2 | 2 | 257 recorded, 3840 skipped | Failed | 0 | 0 |
| 381b10ee + restored wait arguments | 2 | 2 | 0 | 16/16 pass | 1 | 8192/8192 |
| Same repaired module, prefill window | 3 | 5 | 0 | 16/16 pass | 1 | 20480/20480 |

The repaired runs complete at 15:30:30 and 15:31:15-15:31:16 PDT. The five-column case exercises compact capacities 8/4 with unequal selected-position masks. Both wrappers exit 0. Die 1 returns to 0 MiB after each run; every context and the shared lock is released. The bounded kernel journals contain no new NVRM/Xid entries. No hardware setting, reset, other-die context, model attachment, or full load occurs.

These receipts prove the reproduced memory/protocol fault is corrected in this fixture. They do not prove stock arithmetic agreement or full-model correctness. The fix already exists in later integration roots; a current-source failure after that fix needs separate localization. The new zero-stack worker is outside this source-exact reproduction.

Artifacts on archy: `/home/nate/codex/cx-fl-kern-build/xid13-381/`, including fixture.c, worker.ptx, worker-lines.ptx, fixed-worker.ptx, fixed-worker-lines.ptx, wait.sass, and original/fixed/batch5 logs and state files.

```text
worker.ptx SHA256 8598f304bf11d417005724fc179e919db52c8ea0251ba4f9327bd6090207d272
fixed-worker.ptx SHA256 ea049f61c70c383c42937bf663b13b0ab38bb96848cfa572ba85f6250c02add0
worker-lines.cubin SHA256 752f9f1ac68afb045a3cfdc15fa0806b432cff959582a11685a825f82e3111a2
fixed-worker.cubin SHA256 ad6396acb7d5e18161ee33ae1799403341238b02f0d34afba5954c4c6fa36b38
```
