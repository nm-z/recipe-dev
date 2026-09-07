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

## decode

```rust
let mut sampler = recipe.sampler().temperature(0.8).top_k(40).top_p(0.95).repeat(1.1, 64).seed(7);
let generation = recipe.decode("model.ogdl", &prompt_ids, &mut sampler, &[eos], 64);
generation.ids;
generation.logits;
generation.prefill_seconds;
generation.step_seconds;
```

The model reads a sequence, one value per position, and returns one logit per id. The prefill runs the prompt and the decode then holds that state: a step adds one id, extends the attention keys and values, the recurrent state, and the convolution tail by the one position the id reaches, and samples from the new logits (penalty, top-k, top-p, min-p, temperature, seeded draw; temperature zero is greedy). A step therefore reads what earlier calls left rather than running the sequence again, and the result is the result of one forward of the same ids. The decode stops at a stop id, after the budget, or when the ids fill the model's sequence.

```rust
recipe.serve("model.ogdl", "127.0.0.1:8080", 64);
```

`serve` answers that many decode requests over HTTP and returns. A request names its prompt in the target, as `GET /decode?ids=3,1,4&budget=16&stop=2&temperature=0.8&top_k=40&top_p=0.95&min_p=0.05&penalty=1.1&seed=7`, and each field it leaves out keeps the sampler's default. The answer is chunked and carries one id per chunk as the decode reaches it.

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
	attn(heads)[.width(d)][.kv(heads)][.qk(rms|l2)][.rope(dims, base)][.index(heads, width, block, keep)[.budget(tokens)][.score(rms|l2, dims)]][.gate()]
	attn(q, k, v) // n heads
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

## sparse attention

`attn(heads)` builds one query, key and value plane per head. `.width(d)` unties
the head width from the block input: without it a head is `channels / heads` wide
and the residual width must divide by the head count, with it a head is `d` wide
whatever the residual width is, and the block projects `heads * d` back to the
residual width on the way out. `.kv(heads)` unties
the key-value head count, so each key-value head serves `heads / kv` query heads.
`.index(heads, width, block, keep)` adds a side projection of the block input, its
own weighted node, that scores every group of `block` keys and keeps the best `keep`
blocks per query. A block's representative is the mean of the indexer keys the query
can see, so a query in the middle of a block scores the block's causal prefix, and a
score is the plain dot product of an indexer query head with that mean. Without
`.score` every indexer head normalizes to unit length first; `.score(rms|l2, dims)`
gives the indexer its trained geometry instead: every indexer query and key head
normalizes with its own trained scale, and its leading `dims` channels rotate at the
block's `rope` base, before scoring. A GGUF plan therefore names the indexer
projection as its own entry, in its own layout, after the query-key scales, and the
`.score(rms, ..)` scales after it. `.budget(tokens)` states the admission as a token
count, the way a checkpoint's `attention.indexer.top_k` does, and keeps the blocks
that cover it; a budget that covers the sequence is dense attention. `.gate()` multiplies the
attention output by a sigmoid of its own projection of the block input.

Under `recipe.decode` the indexer is state like the keys and values: the context
arena keeps one running sum of indexer keys per block, a step adds its one key
to the block it lands in and scores only its own query, and the selection equals the
whole-sequence selection of the same prefix.

```rust
.attn(8).width(64).kv(2).qk(rms).rope(32, 10000.0).index(2, 16, 32, 4).gate()
.attn(8).kv(2).qk(rms).rope(32, 10000.0).index(2, 16, 32, 4).budget(128).score(rms, 16).gate()
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
