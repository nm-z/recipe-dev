# Fused IQ signed-stream candidate

This source update replaces the 12 typed `packed_gate_up_17_*` and `packed_gate_up_18_*` bodies. All other shipped PTX functions, dispatchers, declarations, and table initializers are byte-identical to parent 063f8cb9503b5ce098c3f302102e6e789c5f6eb3. The existing integration connection in 3a3b392793ebf78fd19852a18ff07b08b48a638a remains the caller. No second forward path is added.

## Source and layout

IQ2_XS loads four signed int16 pairs from the existing `packed_iq2` table. IQ3_XXS capacity 1 copies the existing signed `packed_iq3` table into shared memory; capacity 2 uses cached global loads from that table. IQ3 gathers four adjacent grid words using one address, builds packed pair indices, and accumulates integer dot products directly across subgroups.

Both helpers cache the full 2560-value packed input before processing rows. Shared byte intervals are half-open:

| Capacity | Input and scales | IQ3 sign offsets | IQ3 signed table |
|---|---|---|---|
| 1 | [0, 7040) | [7168, 7680) | [8192, 40960) |
| 2 | [0, 14080) | [15360, 15872) | Global |

The nine-input typed fused ABI, fourteen-input generic dispatcher ABI, supported masks/capacities, and 40960-byte dynamic shared requirement are unchanged. Weights retain their GGUF storage layout. No new table initializer or device allocation is added.

## Static instruction counts

These counts use CUDA 12.9 PTX, version 7.4, and CUDA 11.4 ptxas for sm_52. The capacity-1, 512-thread, 16-row-lane probe uses default optimized ptxas compilation. One complete inner iteration processes 32 input values through gate and up, or 64 matrix weights. Counts exclude NOPs and include address updates and the back branch. Input/table staging, outer row setup, reductions, and SiLU are outside these counts.

| Format | Prior cached-address candidate | Signed-stream candidate | Instructions per weight | Registers | Probe stack/spills |
|---|---:|---:|---:|---:|---|
| IQ2_XS | 277 | 189, loop 0x1198..0x1970 | 2.953125 | 89 | 0/0 |
| IQ3_XXS | 315 | 205, loop 0x12b8..0x1b38 | 3.203125 | 54 | 0/0 |

The comparison baseline is an earlier CPU candidate, not llama.cpp or the prior released helper. IQ3 remains above the 2-3 target. Static counts do not establish bandwidth or model speed.

Earlier issue counts 305/208 and 348/268 were wrong because AWK compared associative address keys as strings. `count-backward-loops.awk` uses explicit numeric conversion. It lists every backward loop; divide only the matrix loop count above by 64. It does not assign weight counts to setup loops.

## CPU validation and build

`check-signed-pairs.c` compares all signed table entries with canonical GGML grid/sign values and checks packed IQ3 indices and shared addresses. It reports 262144 IQ2 pair checks, 131072 IQ3 pair checks, and 16777216 IQ3 index checks with zero mismatches.

The shipped library assembles separately with `-O1 --disable-optimizer-constants`. All 12 typed fused functions have zero frame and spill bytes; the generic dispatcher has an 8-byte frame and zero spills. This does not establish resources or instruction counts for the linked persistent worker.

Run these CPU-only commands on archy from the kernel checkout:

```bash
make BUILD=/home/nate/codex/cx-fl-kern-build/iq-signed-canonical /home/nate/codex/cx-fl-kern-build/iq-signed-canonical/selected-probes.cubin
cc -O2 -Wall -Wextra -I/home/nate/llama.cpp-master/ggml/src -I/home/nate/codex/cx-fl-kern-build/iq-signed-canonical evidence/2026-10-10-flash-packed/check-signed-pairs.c -o /home/nate/codex/cx-fl-kern-build/iq-signed-canonical/check-signed-pairs
/home/nate/codex/cx-fl-kern-build/iq-signed-canonical/check-signed-pairs
/opt/cuda-11.4/bin/ptxas -c -arch=sm_52 -O1 --disable-optimizer-constants -v packed.ptx -o /home/nate/codex/cx-fl-kern-build/iq-signed-canonical/packed-device.cubin
```

`splice-fused-functions.awk` replaces the existing typed fused bodies from the selected PTX while preserving every other production byte. SASS and build receipts are committed beside this document.

## Open acceptance gates

No GPU validation or benchmark ran for this candidate. Following CEO comment 6101622060, validation belongs to cx-flash's resident owner through its module reload. Rebuild the current main/worker modules with this source, retain the existing fused caller, and compare actual model speed and canonical logits. Independent GPU jobs remain cancelled; physical dies 2 and 6 remain fenced. PR #1097 remains draft because its IQ performance and correctness gates are open.
