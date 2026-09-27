# recipe

GPU/CPU ML training and inference in Rust.

**Devices**

```bash
recipe run train.rs --cfg recipe --device amd0.cpu.archy:nv7.nv8 --ctx 4096 -p "text"
```

## **Data**

```rust
let data = recipe.data("measurements/")
	.target(["temperature"])
	.norm(z_score)
	.split(0.8);
```

```rust
data("path or file"|auto)
	.set("additional")
	.test("test set")
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
frozen.blck.atvn.prec = block
	│      │    │    └─ precision it computes in
	│      │    └────── activation
	│      └─────────── ""
	└────────────────── frozen qualifier
```

```rust
let model = recipe.model()
	.embed(tokenizer.ggml.tokens, arch.embedding_length).fp(16)
	.layer(arch.feed_forward_length).int(4).gelu().fp(16)
	.layer(arch.embedding_length).int(8)
	.layer(tokenizer.ggml.tokens).int(8);

recipe.train().run(&model, &data);
recipe.infer().run(&model, &data);
```

```rust
let data = recipe.data("model.gguf");
let sampler = recipe.sampler().temperature(general.sampling.temp);
let ratio = arch.attention.compress_ratios[3];
let vocabulary = tokenizer.ggml.tokens.len();
let architecture = general.architecture.get();
let shape = data.tensor(blk[3].attn_q.weight).shape;
let attention = attn(24).q(blk[3].attn_q.weight);
```

```rust
use recipe::tensor::{output, token_embd};
let input = recipe.model()
	.embed(tokenizer.ggml.tokens, arch.embedding_length)
	.scale(arch.embedding_length.sqrt());
let tied = input.layer(tokenizer.ggml.tokens).bind(token_embd.weight)
	.scale(1.0 / arch.final_logit_softcapping).tanh()
	.scale(arch.final_logit_softcapping);
let separate = input.layer(tokenizer.ggml.tokens).bind(output.weight);
model = model.scale(data.scalar(blk[layer].layer_output_scale.weight));
```

```rb
attn(heads).int(8)
	.kv(kv_heads).fp(16)
	.qk(rms).fp(16)
	.rope(neox, head_width, rope_base)
	.yarn(factor, original_context, fast, slow).fp(32)
layer(d).int(8)
```

```rust
let attention = recipe.model()
	.delta(48, 4).keys(16, 128).values(128).out(arch.embedding_length)
	.conv(silu).qk(l2).norm(rms).decay(softplus).output(silu)
	.norm(rms);
let model = recipe.model()
	.e(arch.attention.layer_norm_rms_epsilon)
	.embed(tokenizer.ggml.tokens, arch.embedding_length)
	.hyper(4, &attention)
	.read([norm(rms), layer(320), scale(0.25), silu(), layer(4 * arch.embedding_length), sigmoid()])
	.write([norm(rms), layer(4), scale(0.25), sigmoid(), scale(2.0)]);
```

```rust
let model = model.ple(&ngram)
	.key([layer(lanes * width), group(rms, width), group(rms, width)])
	.factor([fold(lanes).scale(1.0 / (width as f64).sqrt()).signed_sqrt(1e-6).sigmoid()])
	.value([layer(width), group(rms, width)])
	.tail([dconv(ngram.kernel()).dilate(ngram.dilation()).silu()]);
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
		.conv(linear|silu)
		.qk(l2)
		.norm(rms)
		.decay(softplus|sigmoid)
		.output(sigmoid|silu)
	perc(width)
	glu(hidden, activation)
	ple(&ngram).key([blocks]).factor([blocks]).value([blocks]).tail([blocks])
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
			.rope(neox|pairs, dims, base)
			.yarn(factor, og_ctx, b_fast, b_slow)
			.index(heads, width, block, keep)
				.score(rms|l2, dims)
		layer(d)
atvn:
	relu()
	leak()
	sigmoid()
	signed_sqrt(floor)
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
	e(value)
	.norm(batch|layer|rms|l2)
feature reduction:
	pool(size)
	kmeans(clusters)
	knn(neighbors)
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
	block * block  // f(x) * g(x) = z
	block + block  // f(x) + g(x) = z
	.res([blocks]) // f(x) + x = y
	.recur([blocks]])
	.ensemble([blocks])
	.moe(top_k, [experts]).route(softmax|sigmoid).renorm().shared(shared_expert, [gate])
	.hyper(lanes, &branch).read([blocks]).write([blocks])
	.collapse([read_blocks])
```

```rust
let w = blk[layer].shortconv.in_proj.weight;
let short = ((layer(width).bind(w.clone()).rows(0, width)
	* layer(width).bind(w.clone()).rows(2 * width, width))
	.dconv(kernel).bind(blk[layer].shortconv.conv.weight)
	* layer(width).bind(w).rows(width, width))
	.layer(width).bind(blk[layer].shortconv.out_proj.weight);
```

```rust
let expert = (layer(640).silu() * layer(640)).layer(2560);
let mixture = moe(10, vec![expert.clone(); 512])
	.route(softmax).renorm()
	.shared(expert, [layer(1).sigmoid()]);
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

## **RAT**

```rust
	.loss(mae|mse|...)|.loss(&evaluator)
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
data(path)
	.value(key!(other.path))|.number(key!(other.path))|.integer(key!(other.path))
	.tensor(blk[index].attn_q.weight).shape|.has_tensor(blk[index].attn_q.weight)
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
recipe run rnj-1.rs --device archy:nv6.nv7 --ctx 128 "Hello"
```

## GGUF statistics

```bash
recipe stats model.gguf
recipe keys model.gguf
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

## Approved

```rust
.rope(neox|pairs, dims, base)
(attn(heads) * layer(heads * width).sigmoid()).layer(d)
```

## Proposed

```rust
.yarn(factor, og_ctx, b_fast, b_slow).scale(direct|chain)       // .yarn(factor, og_ctx, b_fast, b_slow)


mtp([blocks])                                                   // .mtp(path)
	.file(path)
```
