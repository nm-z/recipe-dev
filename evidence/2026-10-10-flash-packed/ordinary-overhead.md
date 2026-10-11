# Ordinary packed Contraction overhead

Source baseline: `eb0a6848f11156b91f1eca6ac4f4377c8ce3ab46`, current
`flash/1092-integration`. The existing forward branch also retains its previously
published zero-stack worker. This change affects ordinary native Contractions.

## Decode path

- FP32, single-position calls read and write the tape directly. Contiguous tapes
  call the public `packed_prepare` and `packed_matvec`. Strided tapes use private
  single-position derivatives of those same shipped PTX functions.
- The derivatives change activation-read and output-store addresses and apply
  the existing optional ReLU at the output store. They preserve the original
  weight decoding, integer products, scales, and reduction instructions.
- Contractions sharing an uncast FP32 source prepare it once per decode window.
  Prepared pairs and scales use lifetimes from first to last consumer.
- Prepared inputs occupy idle ranges of the existing packed workspace. The
  allocator reserves every general-path write throughout each input's lifetime.
  Smaller graphs can require a larger workspace; the production layouts below
  fit the existing allocation exactly.
- Independent row planes need no grid barrier between matvecs. One completion
  barrier precedes the next consuming operation. Prefill, verification, precision
  conversion, and unsupported-format paths retain their existing general path.

## Emitted PTX barriers

The fixture has three ordinary Q8_0 Contractions, with the second and third
sharing the first one's output. It uses the production emitter and compiler,
256 channels/rows, and separate contiguous and five-position layouts. The
single-position branch has the following counts in both layouts:

| Contraction | Before: executed packed-path grid barriers | After: executed decode-path grid barriers | Prepare calls before / after |
|---|---:|---:|---:|
| Unshared input | 4 | 2 | 1 / 1 |
| First consumer of shared input | 4 | 2 | 1 / 1 |
| Later consumer of shared input | 4 | 1 | 1 / 0 |

The before barriers finish gather, preparation, matvec, and scatter. The after
barriers finish preparation, when this operation performs it, and output writes.
For multiple row planes, the old packed path has `2 + 2 * planes` barriers per
chunk; the new decode path remains 2, or 1 with preparation reused.

These are path counts, not whole-module text counts. LLVM inlines `grid_barrier`
in emitted PTX. Each before operation contains five `grid_barrier.exit` sites:
four on the packed path and one on the alternate direct path. The after
operations contain 7, 7, and 6 sites because the general and direct paths remain
and the added decode branch contains 2, 2, and 1 sites. Each strided matvec also
has one CTA barrier for its CTA-local stride/ReLU parameters. It is a data
dependency, not a grid barrier.

## CPU and compilation evidence

- `cargo build --release --lib --bin recipe`: exit 0 on Archy, three Cargo jobs,
  nice 5, with `CUDA_VISIBLE_DEVICES` empty. Source SHA-256:
  `5948cfc7abeecc683c36a9f5739e02e62d14434bf0d960d31bfd3314e079348a`.
- Before/after contiguous and five-position PTX assemble with CUDA 11.4
  `ptxas -c -arch=sm_52 -O1`. The actual `recipe_model_forward` entry links with
  `nvlink --kernels-used recipe_model_forward`. No GPU context is created.
- Independent CPU address and preparation-ownership check: 128,549,760 checks,
  zero mismatches. It covers strides 1/5/8/128, every valid position, 256/512
  threads, 1/2/16 CTAs, production channel counts, and output row-plane offsets.
- The production target and MTP graphs are lowered through `expert_graphs` with
  the real GGUF bindings, 128 positions, five delta slots, the approved 8 MB
  policy, and five-position windows, using an explicitly constructed CPU device.
  Existing value/context/reset/cast/scratch/route/clock offsets and allocation
  sizes compare byte-for-byte equal before and after:

| Graph | Nodes | Context bytes | Value bytes | Packed workspace offset | Packed route offset |
|---|---:|---:|---:|---:|---:|
| Target | 2616 | 703392320 | 25047124 | 702817760 | 703340000 |
| MTP | 93 | 928884 | 6267212 | 404784 | 927024 |

The target graph has 339 eligible FP32 packed decode Contractions among the 495
inline Contractions. Its 96 shared-input consumers use 48 preparations, removing
48 preparations per decode window. The remaining 156 inline Contractions keep
their current native paths. The MTP graph has 11 eligible calls among its 13
inline calls, with two consumers sharing one preparation.

The kernel entry ABI, weights, and the 40960-byte dynamic shared requirement are
unchanged. This is CPU evidence for using the resident production buffers during
an owner-controlled module reload. It does not establish GPU reload acceptance.

## Evidence and remaining measurement

Complete source, emitter, PTX, relocatable objects, linked cubins, layout dumps,
and logs are at Archy:
`/home/nate/codex/cx-fl-kern-build/overhead/`. Final release receipts are
`overhead-final-release.{log,exit,sha256}` in its parent directory.

The most recent owner report, issue comment 6102976373, measures 75.1458 us per
inline Contraction in the two-layer `eb0a6848` slice. This source change has no
GPU timing or arithmetic receipt. The 20 us average, full-model 60/80 tok/s, and
10% llama.cpp final-logit gates remain for cx-flash's exact-build slice/full runs
and owner-controlled module reloads. Physical dies 2 and 6 remain fenced.
