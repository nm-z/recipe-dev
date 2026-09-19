# What recipe is for

Written 2026-09-19 after a night of measuring rnj-1, Gemma 4, Qwen3.8, LFM2.5 and
Ouro on the desktop card and on archy. This is the target as I understand it from
Nate, stated so that someone who can clear any software obstacle knows what to build
and what not to.

## The one sentence

Run and train the best open models on the cheapest hardware that exists, at the speed
the memory bus allows, with every number format declared once and nothing decided
behind the user's back.

Not the DGX Spark. Not Blackwell, MI355, or anything with fp4 in silicon. The K80,
the M60, the V340L, the MI50, the P40, the RX 6600, the 7700 XT, a Pi 5, a Zen 4
desktop. Cards that cost $30 to $400 and sit in people's houses.

## The numeric contract

**The compute formats are the ones a spec sheet lists, and no others.**

    Vector:  int4  int8  int16  int32  fp16  fp32  fp64  bf16
    Matrix:  int4  int8  fp8   fp16   fp32  fp64  bf16  tf32

Every kernel, helper, table and error message speaks only these. A device supports
some subset of each row; the capability table below says which. Nothing with a
vendor prefix, no int1, no int2, no arbitrary `f(exp, man)`.

**Every file format is a load format, never a compute format.** The GGUF zoo
(Q2_K through Q8_K, every IQ, NF4, F16, BF16, F32) is decoded once, at load, on the
device, into a compute format. After load the kernels see exactly two integer layouts:

    int4:  32 x 4-bit + per-32 scale and min      (the Q4_K inner layout)
    int8:  32 x 8-bit + per-32 scale              (the Q8_0 layout)

and the float tensors as themselves. A Q4_K_M file loads with its Q4_K planes
untouched and its Q6_K planes as int8. Weights are never expanded into device
memory as floats: the codes stay small, the scales ride beside them, and the
result of the sum is scaled once. That is Nate's rule and the point of the
project; llama.cpp's dequantize-to-a-float-buffer is exactly what this is not.
A float suffix on a sum names the multiply type, not a storage type. A file that
is already float stays float. Nothing is decoded inside a dot. Ever.

**The block is 32 and it is one number.** Weights and activations share the step;
it lives in the precision table (`step`) and nowhere else. No block suffix names it.
This is a storage and quantization block size, independent of execution width.
The targeted GCN cards use wave64; RDNA supports wave32 and wave64. Kernels map
32-element blocks onto the device's execution width without changing the layout.
See [AMD's RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/).

**The scale is applied once per block, after the integer sum.** The dot is
`(d * sum(q_i * k_i) - m * sum(k_i)) * s`, integers exact in int32, three float
multiplies per block. The equations in `.docs/nums.ogdl` are the spec.

**No format whose name starts with MX or NV.** MXFP4, MXFP6, MXFP8, NVFP4 and
whatever comes next are rung ladders for silicon that costs more than the machines
this is for. They are not compute formats, not table values, not block suffixes,
not words in the code. If a file arrives in one, the loader decodes it into int4
or int8 like anything else and the name goes no further than the decoder.

## The model definition contract

A precision suffix names the op before it and nothing else: `layer(n).int(8)`,
`.kv(k).fp(16)`, `.gelu().fp(16)`, `res([...]).fp(16)`. An op that names none takes
the table. The suffixes are the spec-sheet formats:

    .int(4|8|16|32)
    .fp(8|16|32|64)
    .bf(16)
    .tf(32)

`.int(1)` and `.f(exp, man)` go: no card lists them. A suffix the device cannot run
in either its vector or its matrix unit is the hard error described below, never a
substitution. That is the whole API for precision.

Gone from the model definition: `.step(n)`, `.qi(n)`, `.iq(n)`, `.quantize(...)`,
run-level `quantize`. The file's format is the file's; the user chooses what to
compute in, not what the file stores. Training writes each block's declared weight
format; integer checkpoint formats use the explicit float training format described
below.

`rnj-1.rs` should read as the Gemma-3 architecture with every op's precision visible,
and `int4-all.rs`, `rnj-1-plain.rs` and the rest of the one-off scripts should not
need to exist.

## The organization and correctness-tracing contract

Recipe's code is already concentrated in a few files. Organization means clear
definitions and data flow within that code. File count is not a measure of
organization, and splitting the code into more modules or adding a wrapper
framework does not establish correctness.

One operation must be traceable through its whole execution:

    declaration -> resolved precision -> storage layout -> kernel arguments -> execution -> result

Use the existing precision and layout structures, including `NativePrecision` and
`NativeLayout`, as the starting point. Each decision has one authoritative
definition. Resolve storage, operand encoding and signedness, accumulator and
gradient types, shape, and the device instruction together. Derive allocations,
offsets, helper declarations, call arguments, and diagnostic descriptions from
that definition. A caller and a callee must not independently choose an argument's
width; an allocator and a kernel must not independently choose a gradient's width.

Pipelines, branches, fusion, and fragments must preserve that trace. Keep the
logical operation's identity and its relationship to the source declaration through
each transformation. A fused kernel must identify the operations it contains;
each tile or fragment must map to its logical tensor region. For every buffer and
execution boundary, make these relationships explicit:

- The producer, consumers, and ownership of each value or region.
- Shape, element type, layout, byte extent, stride, and alignment.
- Lifetime and reuse, including when the last consumer finishes.
- Dependencies, branch joins, and the synchronization and memory visibility needed
  before a consumer can read a producer's writes.

These requirements preserve optimized execution. They do not require serializing
the graph or disabling fusion. They require that someone can follow a value across
the actual pipeline and explain where it came from, how it changed, and who can
read or overwrite it. A diagnostic run that changes fusion, allocation reuse, or
launch geometry must record that change; its success alone does not clear the
original execution path.

Generalize the shared algorithm and contracts before multiplying diagnostics.
For each affected path, consolidate conflicting definitions and remove superseded
paths before adding another independent check. Do not repeat the same precision,
layout, or ABI rule across twelve helpers and then maintain twelve validators for
it. Keep actual hardware differences explicit: instruction selection, register
packing, fragment mapping, and synchronization semantics belong in the applicable
device or instruction-family implementation. They still need device-specific
verification, using shared cases and invariants where possible.

Generate checks and diagnostic records at the shared boundaries. Use compatible
vendor tools and GPU-side checks where they apply, with operation identity carried
into their reports. Hardware diagnostics do not supply the expected mathematical
result, so reference comparisons and captured reproductions remain necessary.
Validate a checker's coverage as well as its output: a parser that skips a type or
truncates a signature cannot establish that the unchecked calls are correct.

The acceptance criterion is a coherent change: changing a resolved format updates
its allocation, kernel interface, call site, and report consistently. A failing
result can be traced to a specific operation or transformation, reduced to a
replayable case, and verified through the original execution path. Add observability
to that organized path rather than using more logs to compensate for conflicting
definitions.

## The instruction contract

**Per device, at compile time, one capability table**, filled from `-mcpu` and target
features, never from a backend enum:

    format   vec dot (name, pairs, sign)                   mat tile (name, MxNxK)      accumulator
    int4     sudot8 / dot8 / none                          wmma iu4 / mma s4 / none    i32
    int8     sudot4 / dp4a / vpdpbusd / sdot / none        wmma iu8 / mma s8 / none    i32
    int16    dot2 i16 / pk_mad_i16 / pmaddwd / smull       none                        i32
    int32    always                                        none                        i32 / i64
    fp16     fdot2 / half2 / none                          wmma f16 / mma f16 / none   f32
    bf16     fdot2.bf16 / dpbf16ps / bfdot / none          wmma bf16 / mma bf16 / none f32
    fp8      none (no vector unit anywhere)                mma e4m3 (sm_89+) / none    f32
    fp32     always                                        native fp32 tile / none    f32
    tf32     none (no vector unit anywhere)                mma tf32 (sm_80+) / none    f32
    fp64     always                                        mma f64 (sm_80+) / none     f64

**Width and shape pick the unit; encoding picks the ladder, with one caveat.** A
precision has a width (4, 8, 16, 32, 64 bits) and an encoding (int, fp, bf, tf).
The vec-or-mat decision reads the width and the op's shape: a 4-bit weight goes
through the 4-bit unit whether its rungs are int4's or fp4's, because the unit
multiplies 4-bit operands into a wide accumulator and the encoding only decides
the unpack and the scale epilogue. The caveat: the encoding's values must map
exactly onto the unit's operands. fp4's ladder doubled is +-12, which does not fit
a signed int4 lane, so fp4 through an int4 tile needs the int8 tile or a lookup;
and fp32 never selects a tf32 tile, because tf32 drops mantissa and that is a
precision change, not a unit choice. Equal width, exact operand match: same unit,
same bytes, same speed; the difference between two such formats is quality only.

**The choice is deterministic and printed.** An op's format never changes. Inside the
format: mat if the device has a compatible tile and the op's shape covers it
(prefill, training, batch), vec otherwise if a compatible vector path exists
(decode). If neither path is available, fail before any kernel builds. One line
per op kind at startup, like the
route line: `ffn_up int8 -> sudot4`, `attn_q int4 -> wmma iu4 (prefill) / sudot8`.

**A format the device lacks is a hard error before any kernel builds**, and the
error prints the device's table. `nv0 (sm_52) has no int8 dot. Vector: fp32 fp64
int32. Matrix: none.` No fallback to another format, no widening the user did not
ask for, no stub that returns zero. Unreachable arms are `trap`.

**A card with no int dot still holds int codes.** On the K80 and M60 (and any
chip without an 8-bit dot) an int sum keeps its 4- or 8-bit weights in memory,
multiplies each code against the activation in fp32 registers, sums in fp32, and
applies the block scale once at the end. The capability table lists this as the
int rows' vector entry for such cards (`int8 -> fp32 multiply, scaled per block`)
and the resolution line prints it. It is not a format substitution: the format is
still int8 in memory and in the checkpoint, and the bandwidth ceiling is still the
int8 byte count. What is forbidden is expanding the weights into a float buffer
in device memory, which quadruples the bytes and is the thing llama.cpp does.

**What LLVM routes on its own, and what it never will.** Generic IR arithmetic on
`half`, `float`, `double`, `i16`, `i32` and their fixed vectors is placed on the
device's unit by every backend, and silently promoted where the unit is missing
(`half` on an M60 runs as fp32 with converts). Sum-of-products chains are matched to
`pmaddwd`, `vpdpbusd`, `sdot` and sometimes `dot4` only when the extension, lane
count and accumulator match the instruction exactly, and fall to scalar without a
word otherwise. Matrix tiles, sub-byte operands, mixed-sign dots, and the scale
epilogue are never formed from generic IR. So the table has two kinds of rows:
floats and int16/int32 ride generic IR; int8, int4 and every matrix tile are named
intrinsics. Recipe validates native format support before emitting that IR so
LLVM's ability to promote a type does not bypass the hard-error contract. Nobody
should expect a loop to become `sudot8`.

**Backends are rows in one table, not files.** Adding a card is adding a row: the
intrinsic name and the register pack for each (format, width). The rows that matter,
in order of hardware people actually own:

    amdgcn gfx11/12   7700 XT, RDNA4      sudot4, sudot8, wmma          (int8 done)
    amdgcn gfx10.3    RX 6600/6700        dot4, dot8
    amdgcn gfx906     MI50 16 GB, 1 TB/s  dot4, dot8, dot2 i16
    amdgcn gfx900     V340L, Vega 56/64   v_pk_mad_i16 (int16 x 2), packed f16
    nvptx sm_61       P40 24 GB, P4       dp4a, dp2a
    nvptx sm_75       T4 16 GB            dp4a, mma s8/s4
    nvptx sm_37/52    K80, M60            no int dot: int codes x fp32 multiply, scaled per block
    x86 VNNI          Zen 4, Alder Lake+  vpdpbusd
    x86 AVX2 / SSE2   everything else     vpmaddubsw / pmaddwd
    aarch64 dotprod   Pi 5, Apple M       sdot / usdot
    aarch64 base      Pi 4                smull

The int8 unsigned-by-signed dot is the common denominator; it is the one to make
perfect first. int4 dots are AMD vector and NVIDIA matrix only.

**Matrix tiles are hand-written once per family, not per card or per type.** The
fragment layout (which lane holds which tile element) is what gets written and
reused where compatible. Each supported type and tile shape must still specify its
operand packing, fragment mapping, signedness, accumulator, and intrinsic. A name
swap is sufficient only when those requirements match.

    AMD WMMA gfx11            done   f16 bf16 i8 i4
    AMD WMMA gfx12            done   f16 bf16 i8 (fp8)
    NVIDIA mma.sync           todo   m16n8k16 for f16/bf16/i8, m8n8k32 for i4 (T4 and up)
    AMD MFMA gfx908/90a       later  MI100+, outside the target
    x86 AMX                   later  Sapphire Rapids, outside the target
    ARM i8mm / bfmmla         later  small tiles, Pi 5 and Apple

## The speed contract

Decode at batch 1 is bound by weight bytes over the memory bus. The target is that
ceiling, stated per card, and the kernel is not done until it is within 1.5x of it:

    7700 XT   432 GB/s   rnj-1 int4 5 GB   ceiling ~85 tok/s   today 33 (64 ctx), 25 (512 ctx)
    MI50      1 TB/s                       ceiling ~200
    V340L     480 GB/s                     ceiling ~95
    P40       346 GB/s                     ceiling ~70
    M60       160 GB/s   int4 in memory    ceiling ~30        today 5.9
    K80       240 GB/s (per half)          ceiling ~45

The ceiling is bandwidth divided by the resident weight bytes per token, scales
included; the 5 GB is the int4 file and it stays 5 GB on every card, because no
card expands it. The figures above are tonight's estimates and measurements, not
a fresh validation; each measurement must print the format and instruction it ran.

Prefill is compute-bound and runs the matrix path where one exists; hundreds of
tokens per second on the 7700 XT, not the 25 measured tonight.

Kernel compile is a build step and must be short: emit the 36 identical layers as a
loop over a layer index, not 36 copies, so the module is 100 KB not 4 MB; split
variants into modules and compile them in parallel. Under 20 s on every backend,
including the NVIDIA driver JIT.

## The correctness contract

- The same script, the same table, the same file give logits within a stated
  tolerance on every backend that has the formats: max |delta logit| <= 1e-3 for fp32
  profiles, <= 1e-2 for fp16/bf16, <= 5e-2 for int sums, measured on the reference
  prompts. The winning token is expected to agree; a flip where the top two logits
  sit inside the tolerance is reported, not failed. Bitwise agreement is required
  only where a profile declares it (`llamacpp` on CPU today) and is a hard failure
  there.
- A profile that names what llama.cpp does must produce llama.cpp's logits, on CPU
  bit for bit, on GPU within the GPU's own rounding, and must never deadlock.
- The context fitter counts device memory only. Host spill is a flag the user sets,
  never a budget the fitter spends.
- A hung kernel on the desktop card is a bug of the highest severity; new dot code
  is proven on CPU and on archy before it touches amd0.

## The training contract

Training is the same graph run backward, with arithmetic and checkpoint formats
resolved explicitly. What is specific to it:

- A block trains in a float format (fp16, bf16, fp32, fp64) with fp32 or fp64
  gradients and optimizer state. For a block declaring int4 or int8, the precision
  table must explicitly supply its float training format. A missing or unsupported
  training format is a hard error before any kernel builds. Training uses that
  float format and writes the block's checkpoint weights as the declared int4 or
  int8 layout. Inference then loads exactly what training wrote when configured
  for those integer formats.
- Every backward kernel obeys the same capability table and the same printed
  resolution as forward. Resolution output distinguishes float training arithmetic,
  gradient and optimizer formats, and checkpoint storage. A backend with no int dot
  can train in the explicitly configured supported float format and write an
  integer checkpoint; that does not authorize integer inference on that backend.
- Multiple cheap cards are one machine. A 27B that does not fit one 8 GB card is
  split across six of them by the placement code (archy: six M60s), for inference
  and for training, with the hop cost measured and printed, not guessed.

## Several cards

Cheap cards have no NVLink, so the split is by layers (pipeline), never by tensor:
an all-reduce per layer over PCIe loses to moving one activation row per cut. The
placement code already does this and measures every link at startup. What it must
add:

- Links are tried in this order and the route line prints which one each link
  got, with its measured latency and bandwidth:
  1. dma-buf: one device exports its buffer, the other imports it; vendor-neutral,
     one mapping, works wherever both drivers speak it (amdgpu does; NVIDIA from
     driver 515 with the open kernel modules).
  2. the driver's own peer access (`cuCtxEnablePeerAccess` / `cuMemcpyPeer`,
     HSA agent-to-agent copies): same vendor only; two GPUs behind one PCIe
     switch get it cheaply, across the root complex it is measured, not assumed.
  3. a DMA bounce through pinned host memory: one DMA down, one DMA up, no
     staging memcpy. This is the floor.
  A pageable host copy, the device -> host buffer -> device path of today, is
  never a link. It is the thing being removed.
- For decode a hop is ~8 KB per token and the bounce costs ~15 us against a
  ~30 ms token; it is noise. For training the exchange is the whole gradient per
  step, and the bounce is where seconds go. P2P is a training feature first.
- Mixed cards are fine: the fastest device leads, each device runs the formats it
  has, and a placement that would need a format some card lacks is the same hard
  error as on one card, printed per range.

**int16, int32, and fp8 on a block.** `.int(16)` and `.int(32)` name the multiply
width only: weights are stored and checkpointed as int8 (the widest integer layout)
and training is the int8 rule (train in the block's float, write int8). `.int(16)`
rounds the activations to int16 with a scale per block and multiplies on a 16-bit
lane. `.int(32)` does not round the activations at all: they stay fp32, each weight
code is multiplied against them in fp32, the block scale is applied once. It is
the no-int-dot path given a name, so a K80 script can say what it does. There is
no int16 or int32 weight layout because no file has one and no bell-shaped weight
needs one, and no i32-code, i64-sum scheme: it would buy nothing over fp32. `.fp(8)` takes
its encoding from the table: `fp8 = "e4m3"` (the default, for weights and
activations) or `"e5m2"` (gradients); the suffix never picks silently.

## The architectures

The builder must know every architecture people actually download as GGUF, so that
`recipe.data("x.gguf")` plus the script is the whole job. Today: llama, gemma3,
qwen2/qwen3/qwen3moe, qwen3.5/qwen3next, qwen4, and lfm2. Missing and needed:
gemma4 (the per-layer output scale and varying attention geometry), phi, mistral,
deepseek2, glm4, and granite. Each is a builder function and a metadata namespace;
none needs a new kernel. Gemma 4's raw-UTF-8, SentencePiece-style BPE tokenizer
must preserve whitespace escaping and newline-run tokens.

## The chat

`rnj-chat` is the demo of all of the above and it should be boring: the model stays
resident between messages, the whole conversation is the prompt, the reply length is
the page's choice, and the stats line under every reply says the device, the formats,
the instructions chosen, and tok/s. One process, one GPU, no reload.

## The measurement tools

Tonight's scripts (`participation.mjs`, `histo-all.mjs`, `overlay.mjs`) should be a
recipe verb: `recipe stats <file.gguf>` prints per-tensor mean |w|, rms, row
participation p, outlier rows, and the histogram, for any format the loader reads.
The fact they found, that trained matmul rows are a bell curve at p = 0.61-0.62
across five labs, is the kind of thing recipe should make a one-liner to check.

## What it is not

- Not a llama.cpp replacement for people with H100s. They have llama.cpp and PyTorch.
- Not a home for every quantization idea. Two integer layouts, five formats, and
  nothing named MX* or NV* anywhere.
- Not Python. Node or Bash for anything that is not Rust or LLVM IR.
- Not a place where an `.int(4)` runs as int8 or fp32. A different compute format
  requires an explicit declaration.

## Progress

Status as of 2026-09-19. Track implementation and verification separately.
"Implemented" means code exists; "verified" names the checks that passed.
Reported failures remain open until reproduced and resolved. Historical timings
are dated observations, not measurements of the current tree. No percentage of
overall completion is claimed.

| Section | Implemented or verified | Remaining acceptance criteria |
| --- | --- | --- |
| Numeric contract | `sum = "int8"` is the local default. Int4 uses 32 codes with scale and minimum (`Q4_1` storage), while int8/int16/int32 weights use Q8_0. Single noncanonical planes convert at load; Q4_K planes can remain in their int4 inner layout. FP8 encoding is explicit in the table. | Mixed Q4_K/Q6_K nodes temporarily retain both legacy planes because the load kernel emits only one target layout per node. Several source codecs still take a host conversion path. Add per-plane target encoding, convert Q6_K to Q8_0, and remove superseded legacy dot helpers. |
| Model definition | The public suffixes are `.fp`, `.int(4|8|16|32)`, `.bf`, and `.tf`. `.int(1)`, `.f()`, `.qi()`, `.iq()`, `.quantize()`, `.profile()`, and `.step()` are removed from `Block` and `Model`; build templates and new bundle precision tokens no longer contain int1 or arbitrary floats. A nested-scope check verifies layer, depthwise convolution, activation, attention, cache, and residual suffix ownership. The table supplies every node's step. Model and product branches no longer carry run-level storage selection, and new bundles write a `model2` header with no quantization field. | Legacy storage codecs and the former model, product, block, and step fields remain read adapters only. |
| Organization and tracing | Precision, layout, node identities, traces, and reference comparisons exist. Their consistency across the full execution path has not been audited. | Consolidate duplicated decisions in the existing code. Derive storage and kernel interfaces from shared definitions, preserve operation identity through branches and fusion, and verify buffer ownership, lifetime, and ordering. Extend diagnostics from those shared boundaries. |
| Instruction contract | One target/format capability table drives pre-build hard errors, matrix eligibility, and startup instruction lines. Invalid storage kinds and impossible legacy packed-dot variants trap instead of returning zero. CPU, generic AMD, gfx11, gfx12, pre-Pascal NVIDIA, and Pascal-or-newer NVIDIA rows describe the routes the code emits. K80/M60 keep Q8_0 weights and int8 activations packed and multiply the codes in fp32 registers; a two-step M60/K80 comparison had zero logit delta. `sm_61+` int8 helpers call NVVM `idp4a`; LLVM 22 emits `dp4a.s32.s32` and `dp4a.u32.s32`. | Split the remaining generic CPU and older AMD rows by features. Implement the NVIDIA int4 matrix route and per-operation matrix selection; int4 still uses int8 activations on the current AMD vector route. |
| Backends as rows | Existing AMD, NVIDIA, and CPU execution paths run bounded fixtures. The pre-Pascal NVIDIA row is bit-identical across an M60 and K80 bounded logit check. The `sm_61+` int8 row has a retained compiler receipt proving the named PTX instructions. | Run and measure the new row on a P40, then implement and measure gfx906, gfx900, sm_75 matrix, VNNI, and NEON rows. Compiler output proves instruction selection, not runtime correctness or speed. |
| Speed | Historical session reports: rnj-1 decode at 25-33 tok/s on the 7700 XT. A current uncached M60 build produced 5.05 MB LLVM IR and took 162.5 s through compile/JIT; the cached run prepared in 4.69 s and decoded at 3.79 tok/s. A later uncached `llamacpp`-profile variant prepared in 238.9 s, then prefetched in 16.54 s and decoded at 2.13 tok/s. Direct Clang-to-PTX also exceeded 240 s. | Emit repeated layers as loops, split variants into parallel modules, and remeasure the current tree with recorded model, format, context, and device. Measure prefill, establish ceilings from actual resident bytes and verified bandwidth, and meet the decode and compile targets. |
| Correctness | Reference recording and comparison are wired into both decode paths. Bounded CPU/M60 fixtures pass tolerance checks; the CPU fixture also passes exact comparison under the `llamacpp` policy. The NVIDIA attention argument-width defect is fixed and verified. A full 8-token `llamacpp`-profile run now completes on the M60 without a kernel deadlock, though its output is corrupt. | Reproduce or close the reported AMD deadlock, retain failing cases, and verify full-model logits against llama.cpp. RNJ output remains wrong: changing YaRN fast from the GGUF's 64 to llama.cpp's Gemma3 default of 32 changes the result materially, while step 32 versus 256 does not repair the first-token mismatch. An exact comparison policy is not proof of llama.cpp parity. |
| Training | Eleven library checks pass on CPU. Targeted M60 checks cover wide gradients, reductions, normalization, attention, built-in recurrence, and mixed accumulators. Fresh fp16 `.int(16)` and bf16 `.int(32)` runs train, save Q8_0 checkpoints, and reload successfully on M60. | Backward contraction and attention use vector kernels; matrix training remains open. Custom `recur([...])` bodies reject differing model and accumulator formats. Verify representative full-model training and convergence, plus capability-table resolution for backward kernels. |
| Several cards | Layer placement exists. Two-M60 data-parallel training verifies gradient aggregation and weight synchronization on a bounded fixture. | That check does not establish model-parallel training or a 27B model split across six cards. Verify those paths and measured hop costs. The reported pageable host-transfer path still needs a current audit and the planned pinned DMA/P2P work. |
| Architectures | Builder entries exist for llama, gemma3, qwen2/3 families, qwen3.5/next, qwen4, and lfm2. The LFM2.5 builder handles per-layer attention or short-convolution selection and produced the same greedy answer as llama.cpp on a complete 1.2B Q8_0 model on an M60; its tokenizer IDs also match. Gemma 4 raw-UTF-8 BPE matches the local Gemma 4 llama.cpp tokenizer exactly on representative whitespace, newline, accent, and Japanese cases. | Add the Gemma 4 graph and its per-layer attention metadata, then the other missing families listed above. Verify LFM2 logits within profile tolerance and verify every other family with a representative file; a builder entry, matching answer, or tokenizer alone does not establish full model correctness. |
| Chat | The server constructs a full conversation prompt and accepts a requested reply budget. Its Rust worker places the script-defined GGUF model once, reads requests over one pipe, reuses the same `Placed` tapes, and reports the device, resolved startup routes, prefill time, decode time, and tok/s under each reply. One M60 process answered two requests after one `READY`, with no reload. | Verify the browser UI end to end, including conversation persistence, cancellation behavior, and the displayed per-reply statistics. |
| Measurement tools | `recipe stats <file.gguf>` streams rows through the loader's decoders and prints per-tensor mean absolute weight, RMS, participation, three-sigma RMS outlier rows, and a normalized histogram. A mixed Q8_0/F32 fixture verifies both block and float decoding without whole-tensor expansion. | Cross-check representative Q2/Q3/Q4/Q5/Q6/IQ/NF4 files against the earlier scripts, then add stable machine-readable output if downstream tools need it. |

Evidence and scope:

- [Initial precision and reference verification](/home/nate/codex/precision-contract-8IP92O/RESULTS.md).
- [Wide-gradient CPU and M60 verification](/home/nate/codex/wide-gradients-8Uc4T7/RESULTS.md),
  including the two-M60 check and fresh checkpoint paths.
- [LFM2.5 tokenizer, builder, and M60 verification](/home/nate/codex/lfm2-verification-20260919/RESULTS.md).
- [RNJ resident chat two-request M60 verification](/home/nate/codex/rnj-chat-resident-20260919/RESULTS.md).
- [NVIDIA sm_61 dp4a compiler verification](/home/nate/codex/nvidia-dp4a-20260919/RESULTS.md).
- [M60 and K80 packed-int8 fp32 widening verification](/home/nate/codex/nvidia-widen-20260919/RESULTS.md).
- Storage retention is implemented in `lower_block` in [recipe.rs](../recipe.rs).
  Chat request handling is in [server.mjs](../rnj-chat/server.mjs); the executable
  selected by [start.sh](../rnj-chat/start.sh) must also be checked.
- The seven fixed attention defects were argument-width mismatches across three
  helper call sites. The helper-call checker reports zero mismatches; bounded
  NVIDIA gradient checks verify the repair. This is correctness work, not an
  item outside the contracts.

The reported `rnj-1` output failure and GPU deadlocks are the next correctness
priorities. Organize and consolidate the affected execution paths as part of that
work, so failures can be traced without maintaining duplicate rules and checks.
Bounded arithmetic checks do not close either issue.

## The first milestone from where the tree is tonight

Completion: **2/8 fully complete under the criteria below**. Items 2, 5, 6, and
8 have partial implementations. Tolerances and resolution printing are not
additional numbered items.

1. **Done:** `.step()` is absent from the public block/model API and new bundle
   records. Lowering obtains every node's step from the selected precision table.
   Legacy 13-field block records still load, with their former step field ignored.
2. **Partial:** Implement the capability table keyed by complete format and
   instruction requirements. The current table drives hard errors, matrix
   eligibility, and printed routes, but several targets still share generic rows
   and matrix selection remains module-wide. Consolidate the remaining precision,
   layout, and kernel interface definitions, following the organization and
   correctness-tracing contract. Separate block size from execution width.
   On K80/M60, declare and print the packed-code route using fp32 register
   arithmetic. Resolve float training arithmetic separately from integer
   checkpoint storage.
3. **Done:** `sum = "int8"` is the committed default. Integer inference keeps
   packed weight codes in device memory. The K80/M60 widening route reads the
   Q8_0 and int8 activation codes into fp32 registers instead of expanding a
   persistent float weight buffer; its bounded cross-device logits are identical.
4. **Open, reported failure:** Determine whether YaRN, softcap, step-256, or another
   cause breaks `rnj-1.rs`. Fix the reproduced cause, verify the intended script,
   and remove superseded one-off scripts.
5. **Partial:** The `llamacpp` profile completed an uncached full-model M60 run,
   including prefill and eight decode steps, without deadlock. Its output was
   corrupt. Reproduce or close the AMD report, then retain correctness evidence
   for both devices.
6. **Partial:** Integer compute converts single noncanonical quantized planes into
   Q4_1 or Q8_0 at load, and a device-load fixture matches host canonical encoding.
   Add per-plane targets so mixed Q4_K/Q6_K nodes keep Q4_K as int4 and convert
   Q6_K to Q8_0. Move codecs that still convert on the host to the device. Keep
   canonical packed layouts for float register arithmetic too; files already
   storing floats remain float. Then delete the Q4_K/Q6_K helper and plane-pair
   paths.
7. **Open:** Implement the int4 activation quantizer and `sudot8` path; measure
   prefill.
8. **Partial:** The `sm_61+` row emits named signed and mixed-sign `dp4a`
   instructions in a compiler check. Run it on a P40. Add and verify gfx906,
   then measure the V340L and P40 against their applicable table entries.
