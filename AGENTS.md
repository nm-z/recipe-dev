# Repository Guidelines

## Project structure and architecture

Recipe targets inexpensive GPUs and CPUs. Follow `.docs/nums.ogdl`.

- `recipe.rs`: public API, lowering, allocation, execution, and workers.
- `amd-nv-cpu.ll`: shared kernels; `build.rs`: template compilation.
- `Cargo.toml`: precision, encodings, and configuration.
- `cli.rs`: launcher; `rnj-1.rs`: model definition; `rnj-chat/`: HTTP frontend.
- `data/`, `.github/runtime/data/`: fixtures; `target/`: artifacts.

## Build and development commands

- `cargo check --lib --bin recipe`: check compilation.
- `cargo build --release --lib --bin recipe`: build optimized binaries.
- `cargo fmt --check`: check formatting against `rustfmt.toml`.
- `recipe run rnj-1.rs --device archy:nv6.nv7 --context 128`: run the model on Archy's K80.

Edit locally; rsync for remote development. Recipe owns execution and file resolution.

Never create new files that Recipe depends on. Implement required logic in the existing `recipe.rs`, `amd-nv-cpu.ll`, `build.rs`, and `cli.rs`; keep configuration in `Cargo.toml`. Scratch files belong under `/home/nate/claude/` or `/home/nate/codex/`, never in this repository, and Recipe must not depend on them. If existing work violates this rule, move the required logic into the existing source files and remove the extra dependency.

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

Generalize corrections into their underlying principles. Record and apply the intent across analogous situations, rather than maintaining incident-specific prohibitions or requiring a literal match before honoring the correction.

Apply corrections immediately to the affected work. Recording a preference in a file or session does not complete the correction: repair the current action or artifact in the same turn when it remains incorrect. Do not substitute an acknowledgment or a promise about future behavior for making the correction.

All leading whitespace in every output must use literal tab characters, never spaces. This includes messages, documents, lists, source files, and code examples. Use no indentation where none is needed; otherwise use tabs only.

Use Rust 2024, 200-column formatting, and Rust naming conventions. Use Rust, LLVM IR, Node.js, or Bash; no Python authoring.

Define precision, layout, and kernel interfaces once. Distinguish storage, operands, results, and accumulators. Preserve wide results; widen smaller operands in registers instead of copying large tensors to match types.

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
let report = recipe.infer().chat([pp, tg, r#in, out, cached])
	.run(&model, &data);
println!("tok/s {}", report.tg());
```

Printing and measurement access are separate. Common diagnostics must be built-in, conditional public capabilities. Installed users must not need source edits or ad hoc print statements. Errors remain visible.

Measurements originate in the executing kernel. For GPU work, use device instructions and device-side reductions, including mixed CPU/GPU runs. Return compact results; do not dump tensors into machine RAM to recompute metrics. Observability must not replay token, epoch, load, or unload work or add measurement-only dispatches. Routine timings must not require rocprof.

## Verification guidelines

Reuse canonical model definitions and existing public entrypoints for verification. Add only the missing check; do not duplicate a complete model or example to attach assertions or prints, including through generated copies. Use the existing observability interfaces to inspect the run.

Inline Rust tests and separate test-only execution paths are banned. Verify through the installed CLI and public API. Scratch programs are allowed under `/home/nate/codex/`, but must use that same user path and observability. Respect selected devices; prove new GPU arithmetic on CPU and Archy before `amd0`. Distinguish execution, correctness, and performance.

## Commits and pull requests

Use imperative subjects and the actual Codex session ID for agent-authored commits. Preserve unrelated changes. PRs describe behavior, link issues, and include user-path receipts naming device, model, precision, and context. Review before pushing or merging.
