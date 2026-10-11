# Resident TopK and weighted-return candidates

Targets: TopK <=10 us/call and complete ExpertOut <=60 us/call on #1092. The main module keeps the full 48-layer model and original Q8_0 MTP head resident. Performance and final-logit acceptance remain open.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| V8 main cubin | `6b05a3c9ff8aed1972b051c1d3e5b47ac4306a1998f88f308671b9ac4ad72c61` |
| V8 retained main PTX | `2e4c7d89533788b43142489d5f9344c49194cb509b524a05f0841292050cb861` |
| Coalesced worker cubin | `0c3419b499679a9b52e857f585159978d4d3a96754facd0a4f80fe6cfd3e83e0` |
| Coalesced worker PTX | `c9b4e1e6b45b91181114c88af9a2f49b5bbb9f3dd3b8f48b6f680743af8e2451` |
| Owner's warm report | `60f6270eb03fe7e4730968c7b3ac7b6a3b0ffc5ad3ec5190e9fd204aa7219148` |

Original paths on Archy:

- `/home/nate/.cache/recipe/native/recipe-native-73e3eb8b479cb016/{artifact.cubin,.recipe-native-639662-0.ptx}`
- `/home/nate/codex/flash-logit-stores/expert-worker.{cubin,ptx}`
- `/home/nate/codex/flash-1092-evidence/resident-v8-warm-plain-report.tsv`

Local source and evidence copies are under `/home/nate/codex/fl-topk/`. Large generated PTX stays outside this repository.

## Baseline

Request 12, `main`, `archy:nv4`, `begin >= 65`, and `end - begin == 1`:

| Operation | Records | Device-clock seconds | Mean us/call |
|---|---:|---:|---:|
| TopK | 768 | 0.267227136 | 347.952 |
| ExpertOut | 768 | 1.004150784 | 1307.488 |

These are 48 operations across 16 decode windows. The owner's generation rate is 3.4752725454 tok/s. Operation intervals overlap and are not an additive token-wall decomposition. ExpertOut includes dispatch, expert arithmetic, peer waits, and combine. Timer locations and complete wait scope stay unchanged.

## TopK contract

`flash_router_top10` uses staged FP32 scores in the existing CTA tile. CTA widths 256 and 512 perform ten warp/CTA maximum reductions. Equal scores choose the lower expert ID. The ten winners are sorted by expert ID before coefficient summation. The existing FP32 exponential polynomial, range limit, addition order, division, zero exclusion, and count-plus-IDs representation are preserved. Any nonfinite score or other CTA width returns to the original scalar body.

Shared bytes remain within `[0,4140)`: scores `[0,2048)`, reduction scratch and final weights `[2048,4096)`, and selected count/IDs `[4096,4140)`. Every lane retains its final weights before scratch is overwritten. CTA barriers protect shared staging, reductions, sorting, and scratch reuse. The existing operation grid barrier remains inside the timing interval.

`splice-router.sh` requires a caller-supplied source hash and exactly 48 recognized TopK functions. It retains each original function signature, source staging, copy-out, timer, grid barrier, and nonfinite fallback. Source comparison confirms that the remaining canonical text is identical after excluding the 48 TopK bodies, the added prototype, and the new helper.

## Weighted-return contract

`flash_weighted_return` consumes local records `[position][count + 10 IDs]`, coefficients `[position][10]`, and expert planes `[position][10][2560]`. Active slots use the existing ascending compact-slot order and one `fma.rn.f32` each. It writes the existing peer payload `[position][4096]` directly and zeroes rows 2560 through 4095.

Publication retains the existing `p2p_publish_tagged` call with tag `sequence * 64 + 63`, original sequence, deadline, completion slots, and peer sequence address. Every payload writer participates. The compact partial buffer remains allocated under the frozen ABI, but the fused path does not read or write it. The intermediate local grid stage with tag `sequence * 64 + 62` is removed. Invalid record counts report 212; publication failure reports 114. A missing writer fails through the publication deadline instead of the removed intermediate stage.

`splice-weighted-return.sh` replaces only the recognized `worker_reduce` region of `split_expert_layer`, adds the helper prototype, and appends the helper. It retains the operation and worker clock boundaries. Apply it independently to the compatible main and worker PTX; no entry signature, allocation layout, packed kernel, or coefficient table changes.

## Pending validation

`check-router.c` compares the scalar algorithm and a modeled 256/512-lane reduction across ties, signed zeros, finite extremes, underflow, and randomized finite inputs. It also checks nonfinite fallback and compares compact-combine/return against direct weighted return for active counts 0 through 10 and one through five positions. This CPU model check does not execute PTX or prove GPU publication.

Build and check through the repository Makefile on Benji or Archy only. When Benji is available, wrap each command in one CEO build slot, use nice 19 and three jobs, and write artifacts only under the worker's owned target directory. Archy builds and Benji SSH are currently paused by the owners.

```bash
flock --close /home/nate/work/.build-slot-$((RANDOM%4+1)).lock nice -n 19 env CARGO_BUILD_JOBS=3 CARGO_TARGET_DIR=/home/nate/work/target-fl-topk make -j3 router-check ROUTER_CHECK=/home/nate/work/target-fl-topk/check-router
```

Assembly uses `make native-module NATIVE_PTX=... NATIVE_CUBIN=...`. Main entries are the default; worker builds set `NATIVE_ENTRIES=expert_split_worker`. Use the approved sm_52 assembler and linker. No CUDA C or LLVM kernel generation is used.

Resident validation goes through `/home/nate/codex/flash-queue/{in,out}`. Coordinate main replacement with cx-flash's module transaction; reload workers through the supported queue control. Review reload logs, exact module hashes, per-op records, IDs, and final logits. The original final-logit comparison currently fails, so no accepted correctness or speedup is claimed.
