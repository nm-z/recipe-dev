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

Select one or more local or host-qualified devices by repeating `--device`:

```text
recipe --device amd0 model.rs
recipe --device amd0 --device archy:nv0 model.rs
```

## files

```bash
recipe.rs       runtime
amd-nv-cpu.ll   kernels
build.rs        compiler
cli.rs          cli options
test.rs         combo testing
```

## 18 thingys:
```rust
weights:
	layer(neurons)
	conv(filters, kernel)
	attn(heads)
	attn(q, k, v) // n heads
	perc(width)
	attn(heads)[.width(d)][.kv(heads)][.qk(rms|l2)][.rope(neox, dims, base)][.yarn(factor, og_ctx, b_fast, b_slow)][.index(heads, width, block, keep)][.gate()]
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

`.rope(layout, dimensions, base)` rotates the first `dimensions` channels of
every query and key head by their position. The layout states the pairing: `neox`
pairs channel `i` with channel `i + dimensions / 2`. A model states it so the
same weights cannot silently run under a different pairing.

`.yarn(factor, og_ctx, b_fast, b_slow)` follows a `rope` and scales its
frequencies for an extended context: `factor` is the extension ratio, `og_ctx`
the original training context, and the blend runs between the fast and slow
rotation boundaries. It also applies YaRN's attention-magnitude correction
`0.1 * ln(factor) + 1` for factors above one. It owns no weights and is
invalid without a preceding `rope`.

```rust
.attn(32).rope(neox, 128, 10000.0).yarn(4.0, 8192, 64.0, 1.0)
```

`.qk(rms|l2)` follows `attn(heads)` and normalizes each head's query and key rows
over its head-width slice, leaving the values untouched:

```rust
.attn(4).qk(rms)
```

## sparse attention

`attn(heads)` builds one query, key and value plane per head. `.width(d)` unties
the head width from the block input: without it a head is `channels / heads` wide
and the residual width must divide by the head count, with it a head is `d` wide
whatever the residual width is, and the block projects `heads * d` back to the
residual width on the way out. `.kv(heads)` unties
the key-value head count, so each key-value head serves `heads / kv` query heads.
`.index(heads, width, block, keep)` adds a side projection that scores every group
of `block` keys and keeps the best `keep` blocks per query. `.gate()` multiplies the
attention output by a sigmoid of its own projection of the block input.

```rust
.attn(8).width(64).kv(2).qk(rms).rope(neox, 32, 10000.0).index(2, 16, 32, 4).gate()
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
##### **reporting:**

```rust
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
