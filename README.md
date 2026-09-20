# recipe

GPU/CPU ML training and inference in Rust.

key:<br>
`.`       optional continue<br>
`[...]`   optional children<br>
`|`       chain alternative<br>
`(...)`   multiple children

**Devices**

```bash
recipe run train.rs --device amd0.cpu.archy:cpu.nv7.nv8
```

## **Data**

```rust
let data = recipe.data("measurements/")
	.target(["temperature"])
	.norm(z_score)
	.split(0.8);
```

```rust
data(path|auto)
	.set(add)
	.include([features])|exclude([features])
```

## **Model**

```rust
let model = recipe.model()
	.conv(16, 5).pool(64).gelu()
	.layer(32).gelu()
	.layer(1)
	.loss(mae);
```
```rust
	.loss(mae|mse|...)|.loss(&evaluator)
```r
frozen.blck.atvn.norm.prec = block
	│      │    │    │    └─ precision it computes in
	│      │    │    └────── normalization
	│      │    └─────────── activation
	│      └──────────────── ""
	└─────────────────────── frozen qualifier
```

Every block may name its compute precision. A precision names the operation right
before it: after `layer(n)`, `attn(h)`, or `embed(v, d)` it names that operation;
after `.kv(k)` it names the cache; after `.gelu()`, `.norm(rms)`, `.qk(rms)`,
`.rope(...)`, or `.yarn(...)` it names that operation. File quantization is a
load format and is not selected by the model API. An integer precision uses the
canonical packed layout selected by the precision table. `acc(32)` or `acc(64)`
names the accumulator. The table's `step` is the only activation block size. An
operation that names no precision takes the selected `[precision.<name>]` table.

```rust
let model = recipe.model()
	.embed(tokenizer.ggml.tokens, gemma3.embedding_length).fp(16)
	.layer(gemma3.feed_forward_length).int(4).gelu().fp(16)
	.layer(gemma3.embedding_length).int(8)
	.layer(tokenizer.ggml.tokens).int(8);

recipe.train().run(&model, &data);
recipe.infer().run(&model, &data);
```

Attention suffixes are independent. This keeps Q/K normalization narrow while
rotary constants and chained angles stay fp32:

```rb
attn(heads).int(8)
	.kv(kv_heads).fp(16)
	.qk(rms).fp(16)
	.rope(neox, head_width, rope_base)
	.yarn(factor, original_context, fast, slow).fp(32)
```

	conv(filters, kernel)
	rnn(hidden)
	gru(hidden)
	lstm(hidden)
	perc(width)
	estimators:
		svm()
		bayes()
	trees:
		cbst(trees)
		xgbst(trees)
		lgbm(trees)
	attention:
		attn(heads)
			.width(d)
			.kv(heads)            // key and value heads; a precision right after names the cache: .kv(heads).fp(16)
			.qk(rms|l2)
			.rope(neox, dims, base)
			.yarn(factor, og_ctx, b_fast, b_slow)
			.index(heads, width, block, keep)
			.gate()
atvn:
	relu()
	leak()
	sigmoid()
	tanh()
	selu()
	gelu()
	silu()
	elu()
	prelu()
	cos()
	exp()
	log()
	ln()
	huber()
	tan()
	scale(factor)
	feature reduction:
		pool(size)
		kmeans(clusters)
		knn(neighbors)
norm:
	.norm(batch)
	.norm(layer)
	.norm(rms)
	.norm(l2)
loss:
	.loss(mse|rmse|huber|mae|bce|ce|focal)
exclude:
	.no(bias)
prec:
	.fp(8|16|32|64)
	.int(4|8|16|32)
	.bf(16)
	.tf(32)
```

**compositions**
```rust
	moe(topk, [blocks])
	res([blocks])
	ensemble([blocks])
	recur([layer(width), activation])
	block * block
```

**operations:**
```rust
left * right
.scale(factor)
```

## **Train**

```rust
recipe.train()
	.lr(0.0001)
	.stop(0.1)
	.epochs(100000)
	.save("model.ogdl")
	.run(&model, &data);
```

```rust
.seed(value)
.resume(path)
.rat(history|rolling|online|learned|full, "./evaluate")
.target(value)
observe:
	.log([run, time, epoch, r2, loss, blck, tile, score, choices, window])
	.log(all) // run, time, epoch, r2, loss, blck
	.log(dev) // tile, score, choices, window
```

## **Infer**

```rust
let prediction = recipe.predict("model.ogdl", &input);
recipe.infer().chat(recipe::infer::text).run(&model, &data);
```

```rust
.tokens(count)
```

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
use recipe::infer::{cached, out, pp, r#in, tg};
let report = recipe.infer().chat([pp, tg, r#in, out, cached]).run(&model, &data);
println!("prediction {:?}", report.prediction);
```

The final status uses average rates. `in` counts cached and uncached input;
`cached` counts prefix tokens actually reused. Inspect planned memory allocations before compiling
kernels with `model.memory(&data, 32768)`, or actual resident allocations with
`placed.memory()`. Both return named fields for input, weights, values, contexts,
and load scratch, using the same layout as execution:

```rust
println!("{}", model.memory(&data, 32768).unwrap());
```

The older `rnj-chat/` HTTP frontend remains available, but `rnj-1.rs` now runs
the resident terminal chat directly.

## GGUF statistics

```bash
recipe stats model.gguf
```

The command streams one row at a time and prints each tensor's mean absolute
weight, RMS, participation ratio, outlier rows, and normalized histogram. It uses
the same GGUF decoders as model loading and does not expand a whole tensor in
memory.

## Precision and reference checks

`.int(16)` stores and checkpoints weights as int8 blocks and quantizes activations
to int16. `.int(32)` keeps the same int8 weight layout and uses unchanged fp32
activations. Both apply each block's scale after its dot product.

Each `[precision.<name>]` table declares these settings:

```toml
fp8 = "e4m3"       # Encoding for .fp(8); e5m2 is also supported.
train = "fp32"     # Float arithmetic for integer checkpoint blocks.
tolerance = 0.05   # Maximum absolute logit difference for this integer profile.
exact-cpu = false  # The llamacpp profile sets this to true.
```

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
