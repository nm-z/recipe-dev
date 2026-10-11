# IQ2 unsigned dictionary candidate

This candidate changes only the six typed fused IQ2_XS gate/up bodies: capacities 1/2, 256/512 threads, and the shipped 8/16 row-lane combinations. The current generic dispatcher, selectors, Q-format bodies, IQ3/IQ4 bodies, preparation, all module globals, and caller ABI remain byte-identical.

Each CTA caches existing `iq2xs_grid` (4096 bytes) and `ksigns64` (1024 bytes) in existing dynamic shared storage at `[16384,21504)`. Capacity-2 input and scales occupy `[0,14080)`, so they do not overlap. The 40960-byte requirement is unchanged. Each dictionary load uses an 8-byte vector, matching the original global table alignment.

The inner loop reconstructs four signed int16 pairs from the unsigned eight-value grid and sign bytes using PRMT, XOR, and integer addition. Every magnitude is positive, so adding each negative lane's one-bit correction cannot carry between the 16-bit halves. Original `mad.wide.s16`, integer sums, activation scales, group scales, FP32 accumulation, reductions, and fused SiLU remain unchanged. Forty-eight reads from the 1 MiB signed dictionary are replaced across the six bodies. No new codebook or device allocation is added.

The candidate PTX SHA-256 is `5d24bb655b9a8bdafc24418774a8dcb47e8f6968e57059ccc31b5b804cc45c8b`. The source base is current integration `0807217f`. Full candidate source and splice tools are under `archy:/home/nate/codex/cx-fl-kern-build/iq2-unsigned/`. The patch must remain independently cherry-pickable from the diagnostic module work and fl-xmad's selector/Q body work.

The CPU checker compares every reconstructed pair to the canonical GGML grid/sign algebra and the shipped signed dictionary. Its final run passes 262144 new unsigned-pair checks, 262144 existing IQ2 pair checks, 131072 IQ3 pair checks, 16777216 IQ3 index checks, and 1015808 IQ3 scale checks, with zero mismatches. CPU proof uses benji through the Makefile, with CUDA hidden, nice 5, three jobs, `CARGO_TARGET_DIR=/home/nate/work/cx-shared-target`, and scratch files in its own directory. It reads reference table bytes with memcpy to avoid type aliasing in the scratch byte-table include.

Final checker source SHA-256: `d2d67056f6d3dcc8aa3b677809dbbff141413a31d53109ad220fbff70942af74`. Executable: `4048866fc81d2a6d82be1416c4b7f0d0bc29b5aa353d35c6189b17034fe96d7f`. Complete log, exit, header, and byte-table receipts are at `benji:/home/nate/work/cx-fl-kern-pair-check/`. Function comparison confirms that only the six intended typed IQ2 fused bodies differ from the original PTX; every other function and all original globals compare exactly.

No IQ2 candidate assembly, GPU run, bandwidth, model-rate, or final-logit acceptance is claimed. Further archy builds are paused by cx-flash after SSH starvation; the CPU source remains ready for permitted assembly and holder-only measurement. The required 130.77 GB/s per shape and 20 us inline average remain open.
