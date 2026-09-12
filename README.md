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
```r
frozen.packed.blck.atvn.norm.quant = block
  │      │      │    │    │    └─ quantization
  │      │      │    │    └────── normalization
  │      │      │    └─────────── activation
  │      │      └──────────────── ""
  │      └─────────────────────── packed qualifier
  └────────────────────────────── frozen qualifier
```

**blocks:**

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
			.width(d)
			.kv(heads)
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
loss:
	.loss(mse|rmse|huber|mae|bce|ce|focal)
exclude:
	.no(bias)
```

**compositions**
```rust
	moe(topk, [blocks])
	route(experts, topk, hidden, activation, scoring, renormalize, shared)
	res([blocks])
	ensemble([blocks])
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
	.fp(32)
	.lr(0.0001)
	.stop(0.1)
	.epochs(100000)
	.save("model.ogdl")
	.run(&model, &data);
```

```rust
.seed(value)
.resume(path)
precisions
	.fp(8|16|32|64)
	.int(1|4|8)
	.bf(16)
	.tf(32)
	.f(exp, mantissa)
observe:
	.log(Run|Loss|R2|Time|Epoch|blck|tile|all)
```

## **Infer**

```rust
let prediction = recipe.infer("model.ogdl", &input);
```
