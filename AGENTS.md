# Repository Guidelines

## Project structure and architecture

Recipe targets inexpensive GPUs and CPUs. Follow `.docs/nums.ogdl`.

- `recipe.rs`: public API, lowering, allocation, execution, and workers.
- `amd-nv-cpu.ll`: shared kernels; `build.rs`: template compilation.
- `Cargo.toml`: precision, encodings, and configuration.
- `cli.rs`: launcher; `rnj-1.rs`: model definition and terminal chat.
- `data/`, `.github/runtime/data/`: fixtures; `target/`: artifacts.

## Build and development commands

- `cargo check --lib --bin recipe`: check compilation.
- `cargo build --release --lib --bin recipe`: build optimized binaries.
- `cargo fmt --check`: check formatting against `rustfmt.toml`.
- `recipe run rnj-1.rs --device archy:nv6.nv7 --context 128`: run the model on Archy's K80.

Edit locally; rsync for remote development. Recipe owns execution and file resolution.

Invoke Recipe directly as `recipe ...`. Do not prefix Recipe commands with `dsh` in commands, scripts, documentation, or verification. DSH is unrelated to Recipe execution.

Never create new files that Recipe depends on. Implement required logic in the existing `recipe.rs`, `amd-nv-cpu.ll`, `build.rs`, and `cli.rs`; keep configuration in `Cargo.toml`. Scratch files belong under `/home/nate/claude/` or `/home/nate/codex/`, never in this repository, and Recipe must not depend on them. If existing work violates this rule, move the required logic into the existing source files and remove the extra dependency.

Recipe must remain dependency-free: no external Rust crates. Terminal handling, CLI code, IR generation, and math do not require additional files or dependencies. Use the existing files, including `cli.rs` for CLI responsibilities; do not treat file count as a bottleneck without evidence.

## Scope, efficiency, and completion

Use Nate's relative token-cost model: `(10 * input + 1 * cache_read + 12 * cache_write + 50 * output) * multiplier`. Count the token categories separately without double-counting. These are planning weights supplied by Nate.

| Mode | Astra | Sol | Terra | Luna |
|---|---:|---:|---:|---:|
| Standard | 1 | 0.4 | 0.2 | 0.02 |

Never use Terra. Its multipliers are recorded for completeness, not authorization.

- Optimize expected total cost to a correct result, including generation, context transfer, retries, supervision, and future rework. Token count alone is insufficient: cached context is inexpensive; generating or transcribing it again is expensive.
- Generate, transform, and compare bulk data with tools. Keep tensors, prediction vectors, binary artifacts, and large logs in files. Inspect an unfamiliar file's format before reading its contents, and bound excerpts by bytes as well as lines. Return only the relevant sample, discrepancy, or summary needed for a decision. Do not hand-transcribe bulk values or copy noisy output into the conversation. Preserve complete evidence on disk.
- Establish the complete requested scope and acceptance criteria. Inspect the relevant data flow before editing, fix shared causes, and batch related changes before expensive validation. Reuse valid results; repeat checks only when a change, failure, or unresolved question requires them.
- When iteration repeatedly encounters the same obstacle, consider improving shared observability, automation, or tooling. Make that bounded investment when its expected savings across the current work and foreseeable reuse justify its cost. Reducing immediate edits is not the objective; reducing total cost while completing the task is.
- When delegation is authorized, choose a capable model using the weighted cost of the whole assignment, including its context, handoff, and review. Give bounded work and request concise evidence. Avoid duplicate investigations and verbose transcripts.
- Complete required work, then hand off. Report unrelated findings without silently expanding the task. Claim completion only when the agreed acceptance criteria are supported by evidence; distinguish implementation, successful execution, and verified correctness. Report remaining gaps explicitly.

## Coding style and design rules

When editing `README.md`, match its current style as closely as possible. Keep it focused on concrete syntax, commands, and examples. Do not add reasoning traces, defensive caveats, implementation diaries, or explanations of the agent's work. Show necessary details in the examples instead of appending prose.

Generalize corrections into their underlying principles. Record and apply the intent across analogous situations, rather than maintaining incident-specific prohibitions or requiring a literal match before honoring the correction.

Apply corrections immediately to the affected work. Recording a preference in a file or session does not complete the correction: repair the current action or artifact in the same turn when it remains incorrect. Do not substitute an acknowledgment or a promise about future behavior for making the correction.

All leading whitespace in every output must use literal tab characters, never spaces. This includes messages, documents, lists, source files, and code examples. Use no indentation where none is needed; otherwise use tabs only.

Use Rust 2024, 200-column formatting, and Rust naming conventions. Use Rust, LLVM IR, Node.js, or Bash; no Python authoring.

Define precision, layout, and kernel interfaces once. Distinguish storage, operands, results, and accumulators. Preserve wide results; widen smaller operands in registers instead of copying large tensors to match types.

Invoke the compiled work exactly once per unit: one prefill, one token-generation step, or one training epoch. This is a hard execution rule, not merely a restriction on observability. Count invocations of the compiled unit of work, not its internal routines. Prefill and generation do not have to share an invocation. Do not add an invocation after the requested work is finished.

Minimal execution computes each needed value once, retains it until its final use, then reuses its storage. Never discard a value that later work must recompute. AOT may evaluate recomputing intermediate operations only after establishing a correct no-recomputation baseline and finding that the model does not fit. That evaluation is separate AOT work, not permission to introduce recomputation into minimal execution.

Remove obsolete APIs completely, including their fields, compatibility branches, no-op stubs, and messages about removed APIs. Saved-model compatibility does not justify retaining obsolete machinery. Validate saved data against the current format without special cases that preserve removed APIs.

## Observability and public reports

Use **machine** in prose, including execution across machines, and **node** in printed machine identifiers. Every participating machine is a node in distributed execution; use **master** and **worker** only for their actual distributed roles. Identify RAM, compute, files, and servers by the machine or node they belong to. Do not use **host** in user-facing, developer-facing, or debug output, or in TOML configuration. Internal code identifiers may use **host**; do not rename internals merely to enforce an output terminology rule.

Do not use **arena** or **arenas** in Recipe. Name the actual memory or buffer and what it contains.

All observability must only ever query, measure, or derive. It must never indicate.

Choose a measurement's presentation before collection and keep it consistent. Report latency only for a single measurement when no more are expected. For a series, report rate from the first measurement; multiple measurements always use rate. During collection, show the current rate at the configured refresh frequency. After collection, show one static average rate: total completed work divided by the collection interval, not an unweighted average of sampled rates. Do not print both latency and rate or switch a latency display into a rate display. Keep parallel activities separate unless their work and measurement intervals support a meaningful combined rate.

Expose facts from the actual execution, not output that merely suggests a requirement was fulfilled. Do not substitute implementation labels, narrated calculations, transport chatter, or inactive-feature placeholders for observations.

Model scripts select printing with `.log([time, epoch, loss])`. Measurements remain accessible through reports:

```rust
let report = recipe.train().run(&model, &data);
println!("loss {}", report.fnl.loss);
println!("r2 {}", report.fnl.r2);
```

Inference uses the same split between selected live output and returned observations:

```rust
let report = recipe.infer().chat([pp, tg, input, out, cached])
	.run(&model, &data);
println!("tok/s {}", report.tg());
```

Printing and measurement access are separate. Common diagnostics must be built-in, conditional public capabilities. Installed users must not need source edits or ad hoc print statements. Errors remain visible.

`report.*` must expose the observations the user wants to report. Store collected report history in machine RAM, not VRAM. Compute GPU metrics on the device, transfer completed results, and reuse their temporary output storage after transfer. Do not retain report history in device buffers or count RAM-held history toward VRAM requirements.

Measurements originate in the executing kernel. For GPU work, use device instructions and device-side reductions, including mixed CPU/GPU runs. Return compact results; do not dump tensors into machine RAM to recompute metrics. Observability must not replay token, epoch, load, or unload work or add measurement-only dispatches. Routine timings must not require rocprof.

## Verification guidelines

Reuse canonical model definitions and existing public entrypoints for verification. Add only the missing check; do not duplicate a complete model or example to attach assertions or prints, including through generated copies. Use the existing observability interfaces to inspect the run.

Inline Rust tests and separate test-only execution paths are banned. Verify through the installed CLI and public API. Scratch programs are allowed under `/home/nate/codex/`, but must use that same user path and observability. Respect selected devices; prove new GPU arithmetic on CPU and Archy before `amd0`. Distinguish execution, correctness, and performance.

## Commits and pull requests

Use imperative subjects and the actual Codex session ID for agent-authored commits. Preserve unrelated changes. PRs describe behavior, link issues, and include user-path receipts naming device, model, precision, and context. Review before pushing or merging.

Every block may name its compute precision. A precision names the operation right
before it: after `layer(n)`, `attn(h)`, or `embed(v, d)` it names that operation;
after `.kv(k)` it names the cache; after `.gelu()`, `.norm(rms)`, `.qk(rms)`,
`.rope(...)`, or `.yarn(...)` it names that operation. File quantization is a
load format and is not selected by the model API. An integer precision uses the
canonical packed layout selected by the precision table. Only the table configures
accumulators, using `acc = "fp32"` or `acc = "fp64"` and per-operation accumulator keys.
The table's `step` is the only activation block size. An
operation that names no precision takes the selected `[precision.<name>]` table.
Without an explicit Rust reply cap, inference stops at the model's end markers
from GGUF metadata and its chat template, or when the available context is full.

## Terminal chat and remote execution

```bash
recipe run rnj-1.rs --device archy:nv6.nv7 --context 128
```

The local source compiles locally. When every selected device is on one remote
machine, Recipe sends the executable and its runtime templates over SSH and runs
the model there. File paths resolve on that machine, including model data, prompt
files, and saved outputs. Relative paths use the same working-directory path as
the launcher; that directory must exist remotely. For development, sync the
checkout there with rsync first. Chains across machines keep the existing device-worker
protocol and open files on the launching machine. This runner requires compatible
executables and the configured compiler toolchain on the machine doing the computation.

`.chat(...)` keeps one placement resident across messages. Each request uses the
whole conversation. `/clear` clears the conversation without reloading weights;
`/exit` or EOF exits. `--message "text"` instead performs one measured request.
`--context` is an explicit capacity: Recipe reports an allocation error if it
cannot fit, rather than silently reducing it. The outstanding full-sequence
intermediate-buffer allocation issue still prevents the RNJ 32K configuration from fitting on the K80.

Recipe streams reply text as tokens arrive. Select live prompt processing, token
generation, input, output, and reused-prefix observations in the chat call:

```rust
use recipe::infer::{cached, input, out, pp, tg, time};
let report = recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
println!("prediction {:?}", report.prediction);
println!("dead buffers {}", report.dead_buffers);
println!("dead bytes {}", report.dead_bytes);
```

Selected values remain visible as `...` until known. `time` counts elapsed seconds live,
with three decimal places. It stops when the model is ready and leaves that load line
in scrollback. Each submitted message starts a new timer, which stops when generation
ends and leaves its final line visible. Waiting for user input is not timed.
`report.load` records preparation time, including compilation; `report.time` records the
last request's elapsed time. `pp` and `tg` divide cumulative completed tokens by elapsed
time in their respective phases at each refresh. One timestamp ends prefill and starts
generation; EOS stops generation. Include the whole phase, not only kernel execution,
and use the same counts and intervals in live output and reports. The final rates use
the frozen phase intervals. The request's `time` timer is separate from these phase timers.
`input` counts cached and uncached input;
`cached` counts prefix tokens actually reused. Inspect planned memory allocations before compiling
kernels with `model.memory(&data, 32768)`, or actual resident allocations with
`placed.memory()`. Both return named fields for input, weights, values, contexts,
and load scratch, using the same layout as execution. `dead_buffers` counts distinct
planned intermediate/workspace buffers with bytes that are never reused and become
dead before a later write. `dead` contains those bytes; `report.dead_bytes` sums
them across the inference placement. These diagnostics do not add to the allocation
total and do not measure peak dead storage. Placement errors print model size,
buffer size, and the available budget used by the pre-check, after its launch reserve:

The command streams one row at a time and prints each tensor's mean absolute
weight, RMS, participation ratio, outlier rows, and normalized histogram. It uses
the same GGUF decoders as model loading and does not expand a whole tensor in
memory.


`.int(16)` stores and checkpoints weights as int8 blocks and quantizes activations
to int16. `.int(32)` keeps the same int8 weight layout and uses unchanged fp32
activations. Both apply each block's scale after its dot product.

Each `[precision.<name>]` table declares these settings:


Integer checkpoint training supports `train = "fp16"`, `"bf16"`, `"fp32"`, and
`"fp64"`. Forward weights and activations use that format. Parameter gradients,
activation derivatives, and backward reduction buffers use the block's fp32 or
fp64 accumulator; optimizer state uses the run's accumulator. Checkpoints retain
the declared integer weight layout.

Backward contractions and attention use vector kernels to preserve wide
derivatives; forward matrix kernels remain eligible. Custom `recur([...])` bodies
whose model and accumulator formats differ still fail explicitly. Built-in RNN,
GRU, LSTM, and delta-rule backward kernels support wide derivatives.

Finite FP8 overflow saturates; E4M3 has a maximum magnitude of 448, and E5M2
preserves infinities.

Set `RECIPE_REFERENCE_WRITE` to a file path to record a decode's full logits.
Set `RECIPE_REFERENCE` to that path on a later run to compare them. Use one variable
at a time. Recording replaces the named file. Both modes feed the reference
winner into the next step, so random sampling cannot change the comparison prompt.
Use the same initial prompt and token budget for both runs.

The comparison reports maximum absolute error and token flips. A flip passes only
if all logits meet the profile's tolerance and the reference scores the two winners
within that tolerance. Other mismatches fail the run. `exact-cpu = true` requires
identical logit bits when every selected device is a CPU; GPU runs use `tolerance`.
Use `0.001` for fp32 profiles, `0.01` for fp16/bf16 profiles, and `0.05` for integer
profiles.

Without an explicit Rust reply cap, inference stops at the model's end markers
from GGUF metadata and its chat template, or when the available context is full.


Attention suffixes are independent. This keeps Q/K normalization narrow while
rotary constants and chained angles stay fp32:
