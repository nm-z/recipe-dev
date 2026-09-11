# recipe

GPU/CPU ML training and inference in Rust.

key:<br>
`.`       optional continue<br>
`[...]`   optional children<br>
`|`       chain alternative<br>
`(...)`   multiple children

###### **Data**

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

###### **Model**

```rust
let model = recipe.model()
	.conv(16, 5).pool(64).gelu()
	.layer(32).gelu()
	.layer(1)
	.loss(mae);
```

frozen.packed.blck.atvn.norm.quant = block
  │      │      │    │    │    └─ quantization
  │      │      │    │    └────── normalization
  │      │      │    └─────────── activation
  │      │      └──────────────── ""
  │      └─────────────────────── packed qualifier
  └────────────────────────────── frozen qualifier

```rust
blck:
	layer(neurons)
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
		attn([query_heads, key_heads, value_heads])
		attn(heads).width(d).kv(heads).qk(rms|l2).rope(neox, dims, base).yarn(factor, og_ctx, b_fast, b_slow).index(heads, width, block, keep).gate()
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
quant:
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
composites:
	moe(topk, [blocks])
	res([blocks])
	ensemble([blocks])
	block * block
```
**arithmetic**

`left * right` evaluates two model fragments from the same input and multiplies
their equal-shaped outputs elementwise:

```rust
let gate = recipe.model().layer(width).gelu();
let up = recipe.model().layer(width);
let gated = gate * up;
```

Both branches receive gradients and retain their own weights, bias exclusions,
and block quantization. Nested residuals, ensembles, mixtures, and estimator
blocks use the same lowering and saved-model paths. Configure subsequent blocks
on the product model. Use `.scale(factor)` for multiplication by a scalar.

**losses**

```rust
.loss(mse|rmse|huber|mae|bce|ce|focal)
```

`scale(factor)` multiplies every value the preceding block produces by one
finite constant. It owns no weights and preserves the shape, so it states an
explicit scalar rescaling directly.

**Normalizations**



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

**Sparse attention**

`attn(heads)` builds one query, key, and value plane per head. The explicit
`attn([query_heads, key_heads, value_heads])` form preserves the three head
counts independently; each key and value count must divide the query count.
`.width(d)` unties the head width from the block input: without it a head is
`channels.div_ceil(heads)` wide, so the Q/K/V projections also support residual
widths that are not divisible by the query count; with it a head is `d` wide
whatever the residual width is, and the block projects `heads * d` back to the
residual width on the way out. `.kv(heads)` unties the key-value head count, so
each key-value head serves `heads / kv` query heads.
`.index(heads, width, block, keep)` adds a side projection that scores every
group of `block` keys and keeps the best `keep` blocks per query. `.gate()`
multiplies the attention output by a sigmoid of its own projection of the block
input.

```rust
.attn(8).width(64).kv(2).qk(rms).rope(neox, 32, 10000.0).index(2, 16, 32, 4).gate()
```

**Exclusions**

```rust
.no(bias)
```


**Planned**

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

###### **Train**

```rust
recipe.train()
	.fp(32)
	.lr(0.0001)
	.stop(0.1)
	.epochs(100000)
	.save("model.ogdl")
	.run(&model, &data);
```

```rust
.seed(value)
.optimizer(adamw)
.log(metrics)
.resume(path)
```

**Devices**

```text
recipe run train.rs --device amd0.amd1
recipe run train.rs --device nv0.cpu
recipe run train.rs --device engi:amd0.cpu.archy:cpu.nv7.nv8
recipe --device amd0.archy:nv0 run train.rs
```

Use one `--device` flag with a dot-separated chain. Devices before the first
host prefix belong to the machine running the command. A host prefix applies to
the following devices until another host prefix appears. Commas and repeated
`--device` flags are invalid.

`cpu` selects the host's available logical-CPU pool, not an individual socket.
Numbered CPU selectors are not supported. The `run` keyword is optional.

Unqualified components shaped like device names select devices: on Engi,
`nv0.lan:amd0` means Engi's `nv0` and the SSH host `lan`'s `amd0`. A hostname
without `:<device>`, such as the final component in `amd0.archy`, is invalid.
Missing devices or SSH hosts are errors, not fallback selections.

**Compute precisions**

```rust
.fp(8|16|32|64)
.int(1|4|8)
.bf(16)
.tf(32)
.f(exp, mantissa)
```

**Observability**

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

###### **Infer**

```rust
let prediction = recipe.infer("model.ogdl", &input);
```
