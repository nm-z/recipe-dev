# Resident node capture

This diagnostic reads an operation's output immediately after its collective completion. It copies that output into the existing `NativeRequest.output` allocation, finishes a grid barrier, and lets the complete target forward and every worker job finish. The terminal request-output export is skipped. The diagnostic generates no tokens (`/reply 0`). Traced outputs establish neither model performance nor final-logit agreement.

## Exact source and module

Source: V8 integration `0807217f371959a25b73f3ee9efc0ea9ffbb4b40`.

| Artifact | SHA-256 |
|---|---|
| Original complete PTX | `2e4c7d89533788b43142489d5f9344c49194cb509b524a05f0841292050cb861` |
| Diagnostic complete PTX | `656ea4184a839431595568ddadeb157d2174b5a77f308b19134bcae9971b1da8` |
| Capture manifest | `ef0a7f4e967f7e6f3d554bfbed44999bfb2c37492bd7cb8b492dd2a1d411d7dc` |
| Original native cubin, before owner replacement | `6b05a3c9ff8aed1972b051c1d3e5b47ac4306a1998f88f308671b9ac4ad72c61` |
| Diagnostic native cubin | `e6445ea1cac6699fd0b9085920998a9fe483dcf1466632ad776c4e5f52d2871e` |

Original module: `archy:/home/nate/.cache/recipe/native/recipe-native-73e3eb8b479cb016/artifact.cubin`.
Candidate: `archy:/home/nate/codex/cx-fl-kern-build/trace-v8/trace-embed.cubin`.
Full tools, manifest, PTX, and build receipts: the candidate's directory. The owner must verify the current canonical hash at transaction start. A newer owner module must not be replaced or restored using a stale hash.

## ABI and storage

The complete PTX retains all original arithmetic functions, entry declarations, and global definitions. It adds two scoped PTX blocks to `recipe_model_forward`: zeroing the request output for each chunk and capturing node 0 after its existing call. It adds one unconditional branch over the terminal request-output export. There are no new entries, module globals, shared allocations, kernel arguments, weight reads, or buffer allocations.

`recipe_model_forward` keeps four u64 pointers (samples, weights, values, contexts), followed by five u32 values (rows, threads, begin, end, mode). It keeps the existing 40960-byte dynamic shared requirement. Both linked modules declare 1087640 bytes of global storage and 32 bytes of static shared memory. The diagnostic entry uses 252 registers and 1152 stack bytes; its frame can differ from the canonical frame, but its buffer layout does not.

The existing request-control record begins at context byte 702754952. It contains request begin at +0, position count at +4, output pointer at +8, and hidden pointer at +16. The output pointer names a separate retained request allocation; later arithmetic only overwrites the transient value slots. The canonical request-output export is its only later writer and is disabled in this module.

Node 0 is Gather, with 2560 channels, FP16 storage, value offset 0, and compact channel stride 5. `[precision.flash] embed = "fp16"` and the exact emitted node stores confirm that format. Capture converts each FP16 value to FP32. The first 2560 exported channels contain the embedding; the remaining channels are zero.

For chunk position p, destination index is `channel * request_positions + chunk_begin - request_begin + p`. For the 65-token canonical input, the last chunk contains positions 60 through 64. `/dump_logits` selects position 64 and exports 248320 FP32 channels, with the first 2560 naming the captured embedding. Manifest preflight checks source byte bounds, element alignment, destination channel bounds, duplicate node calls, and overlapping capture ranges. Each capture finishes the same grid count/phase protocol already used by the canonical module, before subsequent operations can reuse the source slot.

## CPU build

The native module assembles and links through the repository Makefile on archy, with `CUDA_VISIBLE_DEVICES` empty, nice 5, and three Make jobs. CUDA 11.4 `ptxas` uses sm_52, `-O1 -c --disable-optimizer-constants`; `nvlink` keeps the original entry set. Build exit is 0. This is compilation and ABI evidence, without GPU execution or capture acceptance.

## Owner transaction

Use the owner's serialized transaction controller after its transaction gate passes. The first line below is followed by the diagnostic controls. Do not include a reload command in the body; the controller owns candidate installation, reload, original restoration, and a consumed restoration report before it runs the next job.

```text
@module main /home/nate/.cache/recipe/native/recipe-native-73e3eb8b479cb016/artifact.cubin 6b05a3c9ff8aed1972b051c1d3e5b47ac4306a1998f88f308671b9ac4ad72c61 /home/nate/codex/cx-fl-kern-build/trace-v8/trace-embed.cubin e6445ea1cac6699fd0b9085920998a9fe483dcf1466632ad776c4e5f52d2871e
/clear
/mtp 0
/lookup 0
/reply 0
Explain how a GPU executes parallel work in five short sentences.
/dump_ids /home/nate/codex/flash-queue/receipts/trace-embed-ids.tsv
/dump_logits /home/nate/codex/flash-queue/receipts/trace-embed-logits.f32
/report /home/nate/codex/flash-queue/receipts/trace-embed-report.tsv
```

The transaction remains pending. cx-flash owns active artifacts, all die contexts, weights, queue consumption, and canonical restoration. fl-logits owns the independent embedding comparison and subsequent node selection. Never submit this transaction to the original command-only controller, and never replay uncertain delivery.
