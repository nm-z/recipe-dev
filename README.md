# recipe

GPU/CPU ML training and inference in Rust.

```rust
let data = recipe.data("measurements/")
	.target(["temperature"])
	.norm(z_score)
	.split(0.8);

let model = recipe.model()
	.conv(16, 5).pool(64).gelu()
	.layer(32).gelu()
	.layer(1)
	.loss(mae);

recipe.train()
	.fp(32)
	.lr(0.0001)
	.stop(0.1)
	.epochs(100000)
	.save("model.ogdl")
	.run(&model, &data);

let prediction = recipe.infer("model.ogdl", &input);
```

Use `.include([...])` or `.exclude([...])` to select feature columns. A data source cannot use both selectors.

## devices

Use one `--device` flag with a dot-separated chain. Devices before the first host prefix belong to the machine running the command. A host prefix applies to the following devices until another host prefix appears. Commas and repeated `--device` flags are invalid.

```text
recipe run train.rs --device amd0.amd1
recipe run train.rs --device nv0.cpu
recipe run train.rs --device engi:amd0.cpu.archy:cpu.nv7.nv8
recipe --device amd0.archy:nv0 run train.rs
```

`cpu` selects the host's available logical-CPU pool, not an individual socket. Numbered CPU selectors are not supported. The `run` keyword is optional.

Unqualified components shaped like device names select devices: on Engi, `nv0.lan:amd0` means Engi's `nv0` and the SSH host `lan`'s `amd0`. A hostname without `:<device>`, such as the final component in `amd0.archy`, is invalid. Missing devices or SSH hosts are errors, not fallback selections.

## gguf

```rust
let model = recipe.gguf("model-00001-of-00004.gguf");
model.value("general.architecture");
model.tensors();
model.contract("blk.0.ffn_up.weight", &input, 16);
```

Every shard of a split is opened by name and the tensor data stays mapped. A
quantized tensor binds to the tape in its own GGML layout, so the contraction
reads the mapped bytes through the block decoders that read saved `.ogdl` models.

## ngram

```rust
let table = recipe.gguf("ngram.gguf");
let ngram = table.ngram();
let prediction = ngram.infer("model.ogdl", &input, &ids);
```

N-gram embeddings from a table too large for device memory. Each of `ngram.heads`
heads hashes the current token with its previous one, and as many heads with its
previous two, into its own row range of the mapped `[width, rows]` tensor
`ngram.table` names, seeded by `ngram.seeds` and reset at the end-of-sequence id.
Only the addressed rows decode, in any quantization; an `ngram.conv` tensor
convolves them across as many trailing positions; and the gathered vector is
added to the stream before the block `ngram.layer` names. The gather stays on the
host holding the table. The first `--device` runs the blocks before
`ngram.layer` and the last runs the blocks from it on, so one name runs both
ranges on that device and

```text
recipe --device nv0.cpu model.rs
```

gives the host that has the table mapped the range that reads it. A placement
names at most two devices.

## files

```bash
recipe.rs       runtime
amd-nv-cpu.ll   kernels
build.rs        compiler
cli.rs          cli options
```

## blocks

```
frozen.packed.blck.atvn.norm.quant
```

## 18 thingys

```rust
weights:
	layer(neurons)
	conv(filters, kernel)
	attn(heads)
	perc(width)
	rnn(hidden)
	gru(hidden)
	lstm(hidden)

blocks:
	moe(topk, [...])
	res([...])

feature reduction:
	pool(size)
	kmeans(clusters)
	knn(neighbors)

trees:
	forest(trees)
	cbst()
	xgbst()
	lgbm()

estimators:
	svm()
	bayes()
```

Feature generation is banned.

## data

```rust
data(auto)
	.test(source)
	.set(source)
	.include([features])
	.exclude([features])
```

## training

```rust
.seed(value)
.optimizer(adamw)
.log(metrics)
.resume(path)
```

## losses

```rust
.loss(mse|rmse|huber|mae|bce|ce|focal)
```

## 15 activations

```
relu  leak  sigmoid  tanh   selu   gelu   silu   elu
prelu cos   exp      log    ln     huber  tan
```

## 4 normalizations

```rust
.norm(batch)   per-channel statistics over the batch
.norm(layer)   per-row statistics over the channels
.norm(rms)     per-row root mean square, one trainable scale per channel
.norm(l2)      per-row Euclidean norm, floored at the normalization epsilon
```

`.qk(rms|l2)` follows `attn(heads)` and normalizes each head's query and key rows
over its head-width slice, leaving the values untouched:

```rust
.attn(4).qk(rms)
```

## compute precisions

key:<br>
`.`       optional continue<br>
`[...]`   optional children<br>
`|`       chain alternative<br>
`(...)`   multiple children

```rust
.fp(8|16|32|64)
.int(1|4|8)
.bf(16)
.tf(32)
.f(exp, mantissa)
```
## 32 quantizations

```rust
quantized integer:
	.qi(4|5|8).(0|1)
	.qi(2|6|8).k
	.qi(3).k.[s|m|l]
	.qi(4|5).k.[s|m]
	.qi(4).nf
importance quantized:
	.iq(1).(s|m)
	.iq(2|3).(xxs|xs|s|m)
	.iq(4).(xs|nl)
```

## observability

```rust
.log(Run|Loss|R2|Time|Epoch|blck|atvn|norm|tok|quant|tile|all)
let report = recipe.train()
	.run(&model, &data);

report.initial_loss();
report.final_loss();
report.initial_predictions();
report.predictions();
report.r2();
report.tile();
report.epoch_seconds();
```

## clanker docs

### planned

```rust
.embed(vocab, width)
.attn(q, k, v) // n heads
.no(options)     // exclude default model behavior such as bias.
	.no(bias)
.recur([...])
.scale(factor)   // multiply every value from the preceding step by one constant.
.rope(layout, dimensions, base)
	neox
	yarn(factor, og_ctx, b_fast, b_slow)
branching:
	let gate = recipe.model().layer(width).gelu();
	let up = recipe.model().layer(width);
	let output = gate * up;
```
