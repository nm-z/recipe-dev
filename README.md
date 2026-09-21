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

```rust
let model = recipe.model()
	.embed(tokenizer.ggml.tokens, gemma3.embedding_length).fp(16)
	.layer(gemma3.feed_forward_length).int(4).gelu().fp(16)
	.layer(gemma3.embedding_length).int(8)
	.layer(tokenizer.ggml.tokens).int(8);

recipe.train().run(&model, &data);
recipe.infer().run(&model, &data);
```

```rb
attn(heads).int(8)
	.kv(kv_heads).fp(16)
	.qk(rms).fp(16)
	.rope(neox, head_width, rope_base)
	.yarn(factor, original_context, fast, slow).fp(32)
```

```rust
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
			.kv(heads).fp(...)
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
	block + block
```

```rust
let expert = [
	layer(640).silu() * layer(640),
	layer(2560),
];
let routed = moe(10, [expert; 512]);
let shared = expert * layer(1).sigmoid();
let combined = routed + shared;
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

## Terminal chat and remote execution

```bash
recipe run rnj-1.rs --device archy:nv6.nv7 --context 128
```

```rust
use recipe::infer::{cached, input, out, pp, tg, time};
let report = recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
println!("prediction {:?}", report.prediction);
println!("dead buffers {}", report.dead_buffers);
println!("dead bytes {}", report.dead_bytes);
```

```rust
println!("{}", model.memory(&data, 32768).unwrap());
```

## GGUF statistics

```bash
recipe stats model.gguf
```

## Precision and reference checks


```toml
[precision.recipe]
fp8 = "e4m3"       # or e5m2
train = "fp32"
tolerance = 0.05   # Maximum logit difference
exact-cpu = false  # compared to cpu

[storage.recipe]
kv = "fp8"
fp8 = "e4m3"       # or e5m2
```

```bash
RECIPE_REFERENCE_WRITE=/path/reference.bin recipe run model.rs --device archy:nv0
RECIPE_REFERENCE=/path/reference.bin recipe run model.rs --device archy:nv0
```

```rust
println!("reference {:?}", report.reference);
for operation in &report.operations {
	println!("{} {} {:?}", operation.node, operation.operation, operation.fingerprints);
	println!("KV {:?}", operation.cache_fingerprints);
	println!("KV channels {:?}", operation.cache_channel_fingerprints);
}
```

Reference runs collect `(position, fingerprint)` pairs for position-preserving outputs and final length-one outputs in the final execution window. Other outputs and non-reference runs leave `fingerprints` empty.
`cache_fingerprints` contains stored K/V bit fingerprints after attention, covering up to one execution window ending at the operation's `end`, including retained positions before `begin`.
`cache_channel_fingerprints` hashes the same window separately for each K/V channel.
