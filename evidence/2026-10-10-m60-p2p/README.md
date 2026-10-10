# M60 P2P runtime barrier

Part of #1092. Recipe owns `barrier.ptx` at the repository root and the `PeerPacket` allocation API in `recipe.rs`. The root Makefile assembles this hand-owned PTX with ptxas 11.4 for sm_52 and builds the Rust measurement harness. CUDA C++ sources and generation rules have been removed. The original PTX came from nvcc 12.9; subsequent changes are authored directly in PTX, with `.version 7.4` for driver 470.

## Build and run

Run on archy from the repository root:

```sh
make all
flock /home/nate/codex/flash-1092-die1.lock make run
```

Read the latest #1092 comments before each GPU job. cx-flash owns caps and clocks. The harness selects and verifies the full UUIDs of dies 1/2 before creating CUDA contexts. It does not stop services or select die 6.

## Runtime contract

`recipe::PeerPacket::new(name)` allocates a resident receiving packet and one zeroed 128-byte completion slot per SM. Enable both directions with `packet.enable_sender(&peer)`. Retain these allocations alongside the weight holder across tokens. The payload is 16 KB; the sequence is at byte 16384. `reserve_sequences(hops)` returns a base, and hop h uses base+h. Both participants reserve the same range. Zero is empty; overflow returns an error. Complete the previous token on both dies before reusing or destroying its packet.

`recipe::p2p_functions()` returns owned PTX function definitions. Append them after the forward kernel's PTX header. They introduce no additional kernel launch:

```text
p2p_wait(address: u64, want: u32, deadline_ns: u64) -> u32
p2p_publish(slots: u64, remote_sequence: u64, sequence: u32, deadline_ns: u64) -> u32
```

`p2p_wait` polls a local volatile flag and returns 1 on receipt or 0 on deadline. A receiving CTA's thread 0 calls it, then shares the result through CTA shared memory and a CTA barrier. Treat timeout as a failed token; never consume the payload after timeout.

All payload-writing threads call `p2p_publish` after their peer stores. Every thread executes `membar.sys`, then the CTA synchronizes. Thread 0 writes its own local completion slot. CTA 0's first warp polls all slots in parallel. Its thread 0 writes the peer flag only after every slot equals the current sequence. There are no peer atomics. Each CTA returns its completion result through shared memory. Other CTAs can reach their next local wait while CTA 0 finishes the publication. A publication timeout on CTA 0 and a receipt timeout on its peer fail the token.

The receiving acquire orders reads of the local packet with `membar.gl`; every payload writer retains its system fence. The coordinator polls local slots with volatile loads and performs its system fence before the peer flag store.

All CTAs must remain resident. Recipe rejects devices with more than 32 SMs for this coordinator. Use a one-dimensional grid and block. The grid uses one CTA per SM, 256 threads each; the caller must also verify occupancy permits at least one such CTA per SM. Do not launch an oversubscribed grid or another persistent grid concurrently. Each CTA must finish its own output and peer stores before posting its slot. Publication covers every CTA, including CTAs that produce no part of a particular packet. Sequences distinguish hops and tokens without clearing slots between them.

`PeerPacket::activate()` selects Recipe's context for resident allocations. `context_address()` borrows that context for existing CUDA driver integration; the caller must not destroy it. cx-flash owns wiring these functions into the PR #942 forward pass. This PR supplies and exercises the runtime primitive directly through the public Recipe packet API.

## Multi-CTA packed layer measurement

`p2p_layers` runs one persistent CTA on each of all 16 SMs per die. Each token queues both die launches once and performs 100 alternating layer computations and peer handoffs without a machine action between layers. The two dense synthetic Q4_K matrices use FP16 block scales of 2^-16 to keep the repeated chain bounded and remain packed and resident across all iterations. Each layer has 4096 input columns and either 2560 or 10240 output rows; every row is computed and checked against independent scalar Q4_K/Q8_1 algebra at every hop. Each CTA copies only rows it computed, fences, and posts its own sequence slot. The handoff carries the first 4096 rows, padding the 2560-row result with zeros. This is a transport and packed-layer demonstration, not a complete model definition.

The harness records actual `%smid` and CTA IDs, verifies participation of all 16 SMs, queries occupancy, and checks all local completion slots after each token. Reference preparation, initial activation upload, and result downloads are outside the timed interval; all device arithmetic checks are inside. The machine interval includes both launches and both final synchronizations. Each packed series uses five warmups and 20 measured tokens. Weights upload exactly once per die for each shape.

`p2p_grid_relay` isolates the same all-SM completion and 16 KB transport over 100 hops, with 200 measured tokens after five warmups. A separate 5 us delay on CTA 7 checks that publication waits for that CTA. Omitting CTA 7's first slot must reach a device deadline on both dies without publishing the first peer sequence.

The original one-CTA vector relay, same-pair cudaMemcpyPeer plus destination synchronization, and two-layer packed demo remain regression measurements. The original <10 us end-to-end gate applies to the requested 7/100-hop transport series. Packed-layer time includes weight reads and arithmetic and is reported separately from transport.

## Evidence

`multi/` contains the review revision's build diagnostics, selected-device state, aggregate machine and device measurements, exit status, and source/artifact hashes. `results.txt` retains the preceding one-CTA measurement. `initial/` retains the first failing measurement; commit bd50443f preserves its source. Historical hashes refer to those historical artifacts. Current hashes are in `multi/artifact-sha256.txt`.

No whole-model token rate or final-logit agreement is established by these probes. Those measurements belong to cx-flash's resident forward-pass integration.
