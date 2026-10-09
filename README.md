# recipe

GPU/CPU ML training and inference in Rust.

**Devices**

```bash
recipe run train.rs --cfg recipe --device amd0.cpu.archy:nv7.nv8 --ctx 4096 -p "text"
```

The AMD backend supports discrete GPUs only and requires gfx8 or newer. AMD APUs and integrated GPUs are unsupported.

## **Data**

```rust
let data = recipe.data("measurements/")
	.target(["temperature"])
	.norm(z_score)
	.split(0.8);
```

```rust
let data = recipe.data(["model.gguf", "rows.csv"]).target("next");
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
	│      │    └────── activation maps in written order
	│      └─────────── ""
	└────────────────── frozen qualifier
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
	.delta_norms(l2, rms)
	.delta_activations(Activation::Silu, Activation::Sigmoid)
	.delta_gates(Activation::Softplus, Activation::Sigmoid)
	.norm(rms);
let gate = HyperGate {
	read: recipe.model().no(bias).norm(rms).layer(320).scale(0.25).silu().layer(4 * qwen35.embedding_length).sigmoid(),
	write: recipe.model().no(bias).norm(rms).layer(4).scale(0.25).sigmoid().scale(2.0),
	mean: 0.25,
};
let model = recipe.model()
	.e(qwen35.attention.layer_norm_rms_epsilon)
	.embed(tokenizer.ggml.tokens, qwen35.embedding_length)
	.hyper_gate(4, &attention, gate);
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
		.delta_norms(l2, rms)
		.out(width)
		.delta_activations(convolution, output)
		.delta_gates(decay, write)
	perc(width)
	glu(hidden, activation)
	ple(&ngram).ple_math(PleMath {
		key_norm: BlockNormalization::Rms, query_norm: BlockNormalization::Rms,
		output_norm: BlockNormalization::Rms,
		gate: PleGate::signed_root_sigmoid(1e-6, true), convolution: Activation::Silu,
	})
	estimators:
		svm()
		bayes()
	trees:
		cbst(trees)
		xgbst(trees)
		lgbm(trees)
	attention:
		attn(heads)
		attn_heads(heads) // head plane for a block product before the output layer
			.width(d)
			.head(width)
			.kv(heads).fp(...)
			.qk(rms|l2)
			.rope(neox|sections([...])|interleaved([...]), dims, base)
			.yarn(factor, og_ctx, b_fast, b_slow)
			.index(heads, width, block, keep)
				.score(rms|l2, dims)
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
	sqrt()
	exp()
	log()
	ln()
	huber()
	tan()
	scale(factor)
	e(value)
	norm(batch|layer|rms|l2)
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
	.moe(topk, [blocks])
	.hyper(lanes, &branch, mean)
	.hyper_gate(lanes, &branch, HyperGate { read, write, mean })
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

```rust
let experts = recipe.model().gguf_moe(256, 8, 512, Activation::Silu, Scoring::Softmax, true,
	SharedExpert::Gated { count: 1, gate: Activation::Sigmoid }, 1.0, false);
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

A GGUF file in the data sources binds the model's weights, and the table sources supply the rows. The projection onto the targets starts from random weights.

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
	.log([chat, debug, hc_values])
	.run(&model, &data)
predict(path, &input)
tokenizer.encode(text)|tokenizer.decode(&ids)|tokenizer.stop_ids()
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

## GGUF tensors

A script names the file tensors a block reads. A delta block names its nine, an attention block its projections, biases, scales, rotary factors and indexer, a mixture-of-experts block its router, expert banks, selection bias and shared expert, a hyper-connection block its gate projections and head mixer, a per-layer embedding block its projections, scales and taps, and a normalization its scale.

```rust
let delta = recipe.model().delta(heads, kernel).keys(key_heads, state).values(state).out(width)
	.delta_norms(l2, rms).delta_activations(Activation::Silu, Activation::Silu).delta_gates(Activation::Softplus, Activation::Sigmoid)
	.delta_from(DeltaTensors::block(layer));
model = model.res([norm(rms).scale_from("blk.0.attn_norm.weight"), Block::from(delta).norm(rms).scale_from("blk.0.post_attention_norm.weight")]);
```

```rust
DeltaTensors { alpha, beta, decay_bias, decay, qkv, conv, norm, gate, out }
DeltaTensors::block(layer)
binding.listing()
```

```rust
let attention = recipe.model().attn_heads(heads).kv(kv).head(width).qk(rms).rope(neox, dims, base)
	.attention_from(AttentionTensors::block(layer));
model = model.res([norm(rms).scale_from("blk.3.attn_norm.weight"), Block::from(attention * gate), layer(width)]);
```

```rust
AttentionTensors { q, k, v, q_bias, k_bias, v_bias, q_norm, k_norm, factors, out, indexer }
IndexerTensors { q_proj, k_proj, q_norm, k_norm }
```

```rust
let experts = recipe.model().gguf_moe(count, used, hidden, Activation::Silu, Scoring::Softmax, true, shared, 1.0, false)
	.moe_from(MoeTensors::block(layer));
```

```rust
MoeTensors { router, selection_bias, gate, up, down, shared }
SharedTensors { gate_input, gate, up, down }
```

```rust
model = model.hyper_gate(lanes, &branch, gate).hyper_from(HyperTensors::mixer(layer, "attn"));
model = model.hyper_head_from(HyperTensors::head()).layer(vocabulary);
model = model.ple(&table).ple_math(math).ple_from(PleTensors::block(layer));
```

```rust
HyperTensors { norm, down, up, inject }
PleTensors { key, norm_key, norm_query, value, norm_conv, conv }
```

## Terminal chat and remote execution

```bash
recipe run rnj-1.rs --device archy:nv6.nv7 --ctx 128 "Hello"
```

## GGUF statistics

```bash
recipe stats model.gguf
recipe keys model.gguf
recipe conventions model.gguf
```

GGUF tensor pairing, feed-forward and expert activations, expert routing, delta math, and per-layer embedding math come from the named `[architecture.<name>]` section in `Cargo.toml`. A new architecture needs an explicit `rope-pairs` value (`halves` or `neighbours`). Models with feed-forward or expert blocks name `feed-forward-activation` or `expert-activation`. An expert model also names `expert-scoring` (`softmax` or `sigmoid`) and `expert-renormalize`; declared GGUF metadata overrides those two manifest choices. Gated-delta models name `delta-convolution`, `delta-output`, `delta-qk-norm`, and `delta-value-norm`; per-layer embedding models name their three `ple-*-norm` fields, convolution activation, gate, floor, and width scaling. Unknown names fail instead of taking another architecture's defaults.

A row also declares the tensor conventions of its blocks: `attention-gate`, `qk-norm`, `rope-factors`, `attention-bias`, `values-from-keys`, `attn-pre-norm`, `attn-post-norm`, `ffn-pre-norm`, `ffn-post-norm`, `output`, `output-norm`, `selection-bias`, `decay-bias` and `indexer-score-norm`. `-` means the architecture has no such block, and `optional` (for `rope-factors` and `output`) means files of the architecture differ, as llama.cpp's optional tensors do. `recipe conventions model.gguf` prints each declaration beside what the file's tensors say, and exits 1 when any differ.

## Dataset schema

```bash
recipe schema probe measurements/ > proposal.json
recipe schema finalize proposal.json answers.json schema.json
```

The answers file names one choice for each question on the active branch:

```json
{"schema_version":1,"status":"ok","answers":[{"id":"parse","choice":"table","probability":0.99}]}
```

A SQLite file uses the `sqlite_table` parse path. The answers name one table and its feature types:

```json
{"id":"sqlite_table","choice":"samples"}
```

A SQLite file in WAL mode is read with the committed frames of its `-wal` file laid over the main file, and a row longer than its page is read through its overflow pages.

The probe records each WAV file's sample rate and frame count, and an `envelope`: 16 hexadecimal digits, one per span of the samples, ranked by mean amplitude from `0` (quietest) to `f` (loudest).

```rust
let data = recipe.data("measurements/").schema("schema.json").target(["temperature"]);
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

Integer checkpoints train in fp32 when the table omits `train`.

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
let report = recipe.infer()
	.log([log::metadata, log::tensors, log::ngram])
	.chat([infer::time, infer::metadata, infer::tensors, infer::ngram])
	.run(&model, &data);
println!("GGUF metadata {}", report.gguf.metadata.len());
println!("GGUF tensors {}", report.gguf.tensors.len());
if let Some(table) = &report.gguf.ngram {
	println!("{} {:?}", table.table.name, table.table.shape);
	println!("head offsets {:?}", table.head_offsets);
}
```

```rust
report.*
	tensors[].*
		(name|device|block|node|row_start|shape|input_window|dtype|bytes)
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
	gguf.*
		metadata[] (key, value)
		tensors[].(name|shape|kind|offset|bytes)
		ngram?.(table|ngram_size|heads_per_ngram|layer|kernel|head_offsets|head_vocab_sizes)
	(pp|tg)()
	time.seconds()
	prediction
	(input_ids|output_ids|logits)[]
	(input|out|cached|reply_limit)
	mtp.(drafted|accepted|verifications)
	(context|requests)
	history[].*
		(time|prediction|input_ids|output_ids|logits|input|out|cached|reply_limit|mtp|reference|operations|tensors)
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
```

## Proposed

```rust
.yarn(factor, og_ctx, b_fast, b_slow).scale(direct|chain)       // .yarn(factor, og_ctx, b_fast, b_slow)

delta(heads, kernel)
	.keys(count, width, tiled)                                  // .keys(count, width)
	.qk(l2|rms)
	.delta_gates(softplus, sigmoid)

let attention = recipe.model().attn_heads(heads).kv(kv).head(head);
let gate = recipe.model().no(bias).layer(heads * head).sigmoid();
let model = recipe.model().block(attention * gate).layer(width);

hyper(lanes, &branch, mean)
hyper_gate(lanes, &branch, HyperGate { read, write, mean })

moe(topk, [experts])                                            // moe(topk, [blocks])
	.route(softmax|sigmoid)
	.renorm()

ple(&ngram).ple_math(PleMath { key_norm, query_norm, output_norm, gate, convolution })
	.silu()

mtp([blocks])                                                   // .mtp(path)
	.file(path)
```
