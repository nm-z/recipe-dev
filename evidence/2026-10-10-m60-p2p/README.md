# M60 P2P flag barrier

This standalone #1092 evidence runs only on Archy's dies 1 and 2. It provides an owned PTX artifact, one Makefile, a Rust CUDA harness, and a two-layer Q4_K handoff demo. Recipe's weight holder and forward pass remain owned by cx-flash.

## Build and run

Use nvcc 12.9 to emit sm_52 PTX, set `.version 7.4`, and assemble with `/opt/cuda-11.4/bin/ptxas` for driver 470. The Rust harness links the CUDA driver and an existing CUDA 11 runtime for the `cudaMemcpyPeer` comparison. Override `CUDART` in Make if that runtime is elsewhere.

```sh
make all
flock /home/nate/codex/flash-1092-die1.lock make run
```

Before running, read the latest #1092 comments and coordinate the measurement window. cx-flash applies power caps. `make run` selects the two full UUIDs; the harness checks each UUID before creating that device's context. GPU 6 is fenced. This program never selects it or changes caps, clocks, or other processes.

## Transport contract

Each receiving die owns its packet allocation. Both CUDA contexts enable peer access. A packet contains 4,096 32-bit payload words, a 32-bit sequence, 15 padding words, and a completion word. Its size is 16,452 bytes; sequence and completion offsets are 16,384 and 16,448 bytes.

One 256-thread CTA runs per die per token. For each hop, the sender writes 16 KB through the peer mapping. Every payload-writing thread executes `__threadfence_system()`, then the CTA synchronizes, then thread 0 writes the receiver's sequence. The receiver spins on its own local sequence with `ld.volatile.global.u32`. Payload reads and peer writes also use volatile PTX instructions. There are no peer atomics. Sequence values are unique across tokens; callers must recreate or reset the buffers before the 32-bit sequence range wraps.

The relay alternates directions. The consumer checks all 4,096 words at every hop and increments the vector for the following hop, so stale or early publication produces errors. Odd hop counts finish with a completion acknowledgement from the last consumer. Both kernels have a device deadline. The negative case deliberately omits the first flag and requires both dies to time out.

The `layer` entry packs FP32 activations to the Q8_1-equivalent int16 register layout and calls the #1091 Q4_K XMAD helper inside the persistent CTA. Die A publishes its packed-layer result as a 16 KB FP32 vector. Die B waits, computes its layer, and acknowledges completion. Both layer launches occur before either final machine synchronization. The demo allocates and uploads two weight matrices once and reuses them for all measured tokens.

The demo uses synthetic, valid packed Q4_K 4096x4096 matrices, 9,437,184 bytes per die. An independent scalar Rust calculation checks the same Q4_K/Q8_1 algebra. This exercises the handoff and arithmetic contract. A single persistent CTA does not establish full-die matvec bandwidth, Flash-Next inference, final logits, or a whole-model decode rate.

## Measurement

`results.txt` records aggregate machine intervals from before both launches through both final synchronizations. Report copies and preparation occur outside that interval. For 200 measured tokens, the harness reports mean per-token and per-hop time, token p50/p99, and die A's `%globaltimer` interval. Twenty warmup tokens precede each series. All payload checks are included in the device execution.

The required gate is mean machine end-to-end time below 10 us per hop for both 7 and 100 hops. Device timing is a local interval on die A; no timestamps from separate GPUs are subtracted. The same-pair `cudaMemcpyPeer` baseline copies 4 KB and 16 KB and synchronizes the destination context after each hop. The previously reported 6.2 us / 4 KB is historical comparison data, not this run's result.

The packed demo has two warmup tokens and eight measured tokens. Its timing includes the device arithmetic checks and final acknowledgement. Build diagnostics, selected-device state, exit statuses, and artifact hashes accompany the raw measurements.

## Integration

`publish`, `wait_word`, `Packet`, and the `layer` entry show the reusable flag contract. Keep the receiver's flag local, preserve every writer's system fence, and publish only after all writers have completed. Production multi-CTA layers need an on-die completion scheme before one CTA publishes a sequence. This one-CTA demo does not provide that completion scheme or change Recipe's runtime scheduling.

The owned `packed.cuh` helper comes from `8118be62cb9ac7189439b0e7b38663d764914a10`, the #1091 evidence branch. It preserves packed weights and uses signed 16-bit XMAD with integer accumulation, then applies floating-point block scales. cx-flash can integrate this contract alongside the full packed-kernel track, expert split, and MTP from PR #942.

## Measured result

The final run on October 10, 2026, from 10:59:13 to 10:59:16 UTC exits 0 and passes both requested gates. The assigned dies have 113 W caps applied by cx-flash. Full raw measurements are in `results.txt`; `initial/` retains the first run that failed the gate. The initial source and PTX are preserved in commit `bd50443f`.

| 16 KB path | Hops per token | Machine us per token | Machine us per hop | Die A device us per hop |
|---|---:|---:|---:|---:|
| Flag relay | 7 | 64.359 | 9.194 | 7.501 |
| Flag relay | 100 | 693.454 | 6.935 | 6.806 |
| cudaMemcpyPeer and destination sync | 7 | 82.153 | 11.736 | |
| cudaMemcpyPeer and destination sync | 100 | 1152.409 | 11.524 | |

The flag relay's p99 token times are 66.010 us for seven hops and 702.378 us for 100 hops. One hop, including two launches and final completion, costs 24.122 us. The gate applies to the requested seven-hop and 100-hop cases. The four-KB copy comparison measures 9.773 and 9.713 us per hop, respectively; it does not reproduce the historical 6.2 us under this synchronization boundary.

The vector relay uses 128-bit volatile loads and peer stores and retains each received vector in registers for its next publication. Disassembly contains `LDG.E.CV.128`, `STG.E.WT.128`, and system fences, with no atomics. It uses 46 registers and 1,048 shared bytes without spills. Delayed publication passes; missing publication reaches both device deadlines.

The repeated packed two-layer demo averages 3,074.133 us per token. All eight measured tokens pass the independent algebra check. Maximum absolute errors are 9.5e-7 on die A and 1.2e-7 on die B. Weight allocations remain resident across all ten demo tokens, including warmups. The demo establishes no Flash-Next token rate or final-logit result.
