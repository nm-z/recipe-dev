# recipe

GPU/CPU ML training and inference in Rust.

**Devices**

```bash
recipe run train.rs --config recipe --device amd0.cpu.archy:nv7.nv8 --context 4096 --message "text"
recipe run model.rs --config llamacpp --context 4096 --message "text"
recipe run model.rs --device amd0 export
recipe --worker nv0
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
	.test(sources)
```

```rust
data.value("gemma3.embedding_length")
data.tensor("blk.0.attn_q.weight").unwrap().shape[1]
data.ngram().layer()
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
let attention = recipe.model()
	.delta(48, 4).keys(16, 128).values(128).out(qwen35.embedding_length)
	.delta_activations(Activation::Silu, Activation::Sigmoid)
	.norm(rms);
let model = recipe.model()
	.epsilon(qwen35.attention.layer_norm_rms_epsilon)
	.embed(tokenizer.ggml.tokens, qwen35.embedding_length)
	.hyper(4, 320, &attention);
```

```rust
	conv(filters, kernel)
	dconv(kernel)
		.dilate(steps)
	rnn(hidden)
	gru(hidden)
	lstm(hidden)
	delta(heads, kernel)
		.keys(count, width)
		.values(width)
		.out(width)
		.delta_activations(convolution, output)
	perc(width)
	glu(hidden, activation)
	ple(&ngram)
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
			.head(width)
			.kv(heads).fp(...)
			.qk(rms|l2)
			.rope(neox, dims, base)
			.yarn(factor, og_ctx, b_fast, b_slow)
			.index(heads, width, block, keep)
				.budget(tokens)
				.score(rms|l2, dims)
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
	act(Activation::Silu)
	activate(Activation::Silu)
	feature reduction:
		pool(size)
		kmeans(clusters)
		knn(neighbors)
norm:
	.norm(batch)
	.norm(layer)
	.norm(rms)
	.norm(l2)
	.epsilon(value)
	.scale(factor)
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
	hyper(lanes, rank, &branch)
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

## **Train**

```rust
recipe.train()
	.lr(0.0001)
	.stop(0.1)
	.epochs(100000)
	.save("model.ogdl")
	.run(&model, &data);
```

---

chopping block boundry welcome to sloptown:

---

```rust
train()
	.lr(rate)
	.stop(loss)
	.epochs(count)
	.seed(value)
	.optimizer(adamw)
	.save(path)|.resume(path)
	.rat(history|rolling|online|learned|full, command)
	.target(value)
	.log([run, time, epoch, r2, loss, blck, tile, score, choices, window]|all|dev)
	.run(&model, &data)
all:
	run, time, epoch, r2, loss, blck
dev:
	tile, score, choices, window
```

## **Infer**

```rust
let report = recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
let prediction = recipe.predict("model.ogdl", &input);
```

```rust
infer()
	.tokens(count)
	.mtp(path)
	.chat(text|[time, pp, tg, input, out, cached, mtp])
	.log([chat, debug])
	.run(&model, &data)
predict(path, &input)
gguf(path)
	.tokenizer()
		.encode(text)|.decode(&ids)|.stop_ids()
sampler()
	.temperature(value)|.top_k(count)|.top_p(mass)|.min_p(ratio)|.repeat(penalty, window)|.seed(value)
	.sample(&logits, &previous)
decode(path, &prompt, &mut sampler, &stop, budget)
infer_ids(path, &[&ids])
serve(path, address, requests)
place(path, &[blocks])
	.decode(&prompt, &mut sampler, &stop, budget)
	.serve(address, requests)
	.clear()
	.split()[]|.resident_bytes()[]|.moved_bytes()
	.memory()[]
```

## Terminal chat and remote execution

```bash
recipe run rnj-1.rs --device archy:nv6.nv7 --context 128
```

## GGUF statistics

```bash
recipe stats model.gguf
```

## Precision and reference checks

```toml
[precision]
default-config = "recipe"

[precision.<name>]
sum|embed|attn|rope|atvn|norm|res = "int4"|"int8"|"int16"|"int32"|"fp8"|"fp16"|"bf16"|"tf32"|"fp32"|"fp64"
acc|<kind>-acc = "fp32"|"fp64"
train = "fp16"|"bf16"|"fp32"|"fp64"
fp8 = "e4m3"|"e5m2"
rope-angle = "direct"|"chain"
attn-softmax = "full"|"online"
math = "portable"|"libm"
gelu-table = "none"|"fp16"
exact-cpu = true|false
tolerance = 0.05
step = 32

[storage.<name>]
kv = "fp8"|"fp16"|"bf16"|"fp32"
fp8 = "e4m3"|"e5m2"
```

```bash
RECIPE_REFERENCE_WRITE=/path/reference.bin recipe run model.rs --device archy:nv0
RECIPE_REFERENCE=/path/reference.bin recipe run model.rs --device archy:nv0
```

## Reporting

```rust
let trained = recipe.train().run(&model, &data);
let report = recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
println!("loss {} r2 {}", trained.fnl.loss, trained.fnl.r2);
println!("prediction {:?} tok/s {}", report.prediction, report.tg());
println!("dead {} buffers {} bytes", report.dead_buffers, report.dead_bytes);
println!("reference {:?}", report.reference);
for operation in &report.operations {
	println!("{} {} {:?} {:?}", operation.node, operation.operation, operation.fingerprints, operation.cache_fingerprints);
}
println!("{}", model.memory(&data, 32768));
```

```rust
report.*
	(load|compile).seconds()
	path
	(formats|memory|links|aot|tiles|grids)[]
	llvm.(instructions|intrinsics)[]
train().run().*
	(itl|fnl).*
		(loss|r2)
		predictions[]
		rat.(eval|pred).(r2|reward)
	(loss|r2)[][]
	predictions[][][]
	rat.(eval|pred).(r2|reward)[][]
	(initial_loss|final_loss|r2|epoch_seconds)()
	(initial_predictions|predictions)()[]
	(evaluator_r2|validation_r2|predicted_reward|measured_reward)()
	tile()[]
	rows
infer().run().*
	(pp|tg)()
	time.seconds()
	prediction
	(input_ids|output_ids|logits)[]
	(input|out|cached|reply_limit)
	mtp.(drafted|accepted|verifications)
	(context|requests)
	history[].*
		(time|prediction|input_ids|output_ids|logits|input|out|cached|reply_limit|mtp|reference|operations)
	(dead_buffers|dead_bytes)
	reference.*
		(steps|worst)
		flips[]
			(step, reference, actual, gap)
		failures[]
			(step, message)
	operations[].*
		(device|model|block|node|operation)
		(begin|end|positions|ticks|seconds)
		(fingerprints|cache_fingerprints|cache_channel_fingerprints)[]
			(position, fingerprint)
decode().*
	(ids|logits)[]
	(cached|prefill_seconds|generation_seconds)
	mtp.(drafted|accepted|verifications)
	reference.(steps|worst|flips[]|failures[])
model.memory(&data, positions).*|place().memory()[].*
	(device|input|weights|values|contexts|scratch|dead|dead_buffers|total())
```
