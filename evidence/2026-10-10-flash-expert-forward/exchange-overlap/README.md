# Expert dispatch overlap

Based on Archy's integration source 0807217f. The existing main entry, worker entry, 192-byte job descriptors, sequence allocation, and resident buffers are retained. Only hand-written PTX changes expert execution: phase 1 gathers and publishes the request; phase 2 executes the main-die expert share and combines all returns before the first output consumer. The synchronous path uses phase 0.

The native caller defers completion only across independent nodes. It completes before a consumer, another external job, a recurrent body, or the final node. Fingerprint or individual expert observations retain synchronous execution. Existing operation clocks add dispatch and completion durations; intervening shared-expert work keeps its own clocks.

The attention result feeds the residual, FFN hyper normalization, expert input, router, and TopK. Dispatch cannot overlap that same position's attention tail without changing inputs. The approved overlap is the nine independent shared-expert nodes between each routed output and its first consumer. All 48 target layers and the head have this window under the current 8 MB placement policy.

CPU/source checks on Archy:

- `CUDA_VISIBLE_DEVICES= CARGO_BUILD_JOBS=3 nice -n 5 make build` exits 0. An accidentally duplicated waiting build is terminated; the retained completion and final build receipt are from the successful build.
- Full target and Q8_0 MTP metadata lowering uses context 128, five-position windows, and five target delta slots. Existing value, context, reset, cast, packed scratch, route, and clock offsets and allocation sizes compare identically before and after.
- Target: 2616 nodes, 703392320 context bytes, and 25047124 value bytes. MTP: 93 nodes, 928884 context bytes, and 6267212 value bytes.
- Emitted source has one dispatch/completion pair per routed layer: 48 target pairs and one head pair. The run logs' general `dispatch_calls` string count also includes unrelated operations, and `complete_calls` counts repeated pointer names; these raw counts are not exchange measurements.

The full source probe and comparison artifacts are at Archy `/home/nate/codex/cx-fl-exchanges-build/`. No GPU context, module reload, independent model load, or settings change runs in this check. Native module assembly, resident arithmetic, exchange cost per token, and final-logit agreement remain holder gates. The current resident baseline is queued as `20261011T0007-cx-fl-p2p-baseline.job` through `/home/nate/codex/flash-queue`.

Codex session: 01a1283c-6941-70c0-939c-3adaa44c98ed.
