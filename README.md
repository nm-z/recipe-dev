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
```
```r
frozen.blck.atvn.norm.quant.prec = block
  │      │    │    │    │     └─ precision it computes in
  │      │    │    │    └─────── quantization it is stored in
  │      │    │    └──────────── normalization
  │      │    └───────────────── activation
  │      └────────────────────── ""
  └───────────────────────────── frozen qualifier
```

Every block may name its own `quant` and `prec`: train writes the `quant`, infer reads what the file holds unless the block names another, and both compute in the `prec`. A precision names the op right before it: right after `layer(n)`, `attn(h)` or `embed(v, d)` it is that op's, the sum and how its numbers are stored; after `.kv(k)` it is the cache's; after `.gelu()`, `.norm(rms)`, `.qk(rms)`, `.rope(...)` or `.yarn(...)` it is the block's other ops'. `layer(n).int(8).gelu().fp(32)` is an int8 sum and an fp32 gelu. An int precision on a sum is its storage: int8 weights sit in VRAM as Q8_0 blocks and int4 as Q4_0 (a file plane already at that many bits or fewer, Q4_K or Q6_K under int8, is kept as it is), one step size per 32, multiplied as ints against int8 inputs rounded with their own step per 32, and scaled once per block — the same dot on AMD, NVIDIA and the CPU; every other precision loads its weights into itself once. Nothing decodes a weight in a kernel. `acc(32)` or `acc(64)` after a block names the accumulator its sums and statistics carry (`acc` in the table names the default, `norm-acc` or another `<kind>-acc` one kind's): `layer(n).fp(32).acc(64)` sums fp32 weights in fp64 and stores fp32, and a norm under `acc(64)` sums its fp32 squares in fp64 and finishes in fp32. `step(256)` after an int sum rounds its inputs one step per 256 the way llama.cpp's Q8_K does (`step` in the table names the default, 32), and a Q4_K or Q6_K plane under that step sums in llama.cpp's own order, block by block. A precision on `res([...])` is the add's alone; each part inside names its own. An op that names none takes the run's table: `[precision.<name>]` in Cargo.toml, chosen by `recipe run x.rs --config <name>`, with `default-config` under `[precision]`. Train and infer take no precision, and neither does `recipe.model()` before a block.

```rust
let model = recipe.model()
	.embed(tokenizer.ggml.tokens, gemma3.embedding_length)
	.layer(gemma3.feed_forward_length).gelu().qi(4).k.m.int(4)
	.layer(gemma3.embedding_length).qi(8).0.fp(16)
	.layer(tokenizer.ggml.tokens).qi(6).k.int(8);

recipe.train().run(&model, &data);
recipe.infer().run(&model, &data);
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
prec:
	.fp(8|16|32|64)
	.int(1|4|8)
	.bf(16)
	.tf(32)
	.f(exp, mantissa)
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
	.log(Run|Loss|R2|Time|Epoch|blck|tile|Score|Choices|Window|chat|debug|all|dev)
```

## **Infer**

```rust
let prediction = recipe.predict("model.ogdl", &input);
recipe.infer().log([chat]).run(&model, &data);
```

```rust
.tokens(count)
```
