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

## samples

A caller selects sources, features and targets. Where the samples begin and end
is read from the source, not declared.

A lone table's rows are its samples. When a folder holds a group of sibling
tables and another table has a column recording their file names, that group is
one sample per file, and the recording column is what both identifies and orders
them — no order is ever taken from the path. A column used that way is identity
and not a feature, so the file name never reaches the model as a value.

```
scans/
	meta.csv        scan,temperature,y      3001 lines: a header and 3000 rows
	scan-0000.csv   magnitude,phase         one sample
	...             (2999 more)
```

That is 3,000 samples. Each one is the whole vector of the scan its row names
plus the row's own selected values. Sources that disagree on the sample count
are refused, saying how each was read; nothing is cycled or repeated to make the
counts match.

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

## files

```bash
recipe.rs       runtime
amd-nv-cpu.ll   kernels
build.rs        compiler
cli.rs          cli options
```

## blocks

```
frozen.packed.blck
```

## 18 thingys

```rust
weights:
	layer(neurons)
	conv(filters, kernel)
	attn(heads) or attn([query_heads, key_heads, value_heads])
	perc(width)
	attn(heads)[.width(d)][.kv(heads)][.qk(rms|l2)][.rope(neox, dims, base)][.yarn(factor, og_ctx, b_fast, b_slow)][.index(heads, width, block, keep)][.gate()]
	rnn(hidden)
	gru(hidden)
	lstm(hidden)

blocks:
	moe(topk, [...])
	res([...])

	A branch step is an ordinary model step, so anything above goes inside one,
	carrying its own activation, normalization, quantization and profile, and a
	branch nests inside a branch:

	res([layer(8), relu(), layer(8)])
	res([norm(rms), layer(8).act(Activation::Relu), layer(8).quantize(0, 8, 0)])
	res([res([layer(8), relu(), layer(8)]), gelu()])
	moe(1, [layer(8), res([layer(8), relu(), layer(8)])])

	norm(rms)          a normalization on its own, computing nothing before it

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

## 16 activations

```
relu  leak  sigmoid  tanh   selu   gelu   silu   elu
prelu cos   exp      log    ln     huber  tan
scale(factor)
```

`scale(factor)` multiplies every value the preceding block produces by one
finite constant. It owns no weights and preserves the shape, so it states an
explicit scalar rescaling directly.

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

`attn(heads)` builds one query, key and value plane per head. The explicit
`attn([query_heads, key_heads, value_heads])` form preserves the three head
counts independently; each key and value count must divide the query count.
`.width(d)` unties
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

## exclusions

```rust
recipe.model().no(bias).layer(64).relu().layer(1)
```

`.no(option)` declares a default the model excludes. `.no(bias)` removes the
bias from every weighted block beneath it — layers, attention projections,
convolutions and recurrent gates — and from every nested branch. Excluded
tensors are not allocated, initialized, trained, saved or loaded, and the
exclusion travels with the saved model.

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
.log(Run|Loss|R2|Time|Epoch|blck|tile|all)
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
