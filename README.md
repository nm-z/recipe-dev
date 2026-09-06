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

## gguf

```rust
let model = recipe.gguf("model-00001-of-00004.gguf");
model.value("general.architecture");
model.tensors();
model.contract("blk.0.ffn_down.weight", &input, 1024);
model.expert("blk.0.ffn_up_exps.weight", 3, &input, 640);
let q = model.tensor("blk.3.attn_q.weight").unwrap();
let query = (0..24).map(|head| q.rows(head * 512, 256).unwrap()).collect::<Vec<_>>();
let plan = model.plan().node(&query).named(&model, "blk.3.attn_output.weight");
model.infer(&recipe.model().layer(6144).layer(2560), &plan, &input, 2560);
```

Every shard of a split is opened by name and the tensor data stays mapped. A
block-quantized tensor binds to the tape in its own GGML layout, so a node
reads the mapped bytes through the block decoders that read saved `.ogdl`
models and no run holds a decoded copy; an F32 or F16 tensor binds as the
node's values. `infer` compiles the blocks over `input`, whose leading axis is
`channels` wide, and fills every parameterized node from the plan, one entry
per node in lowering order. An entry is one or more views laid end to end in
the order the node's planes are laid out: a whole tensor, one expert of a
`[k, n, experts]` tensor (`expert`), or a run of output rows (`rows`). A
contraction whose views hold exactly its matrix lowers and runs without a bias
row; one whose views hold the matrix plus one output row binds that row as its
bias; any other count, a view that cuts a block, or block-quantized views of
different layouts in one node are rejected before anything runs, naming the
tensor, the node and both counts. A node with any F32 or F16 view, such as a
bias row behind a quantized matrix, decodes every view into its values.
`contract` and `expert` are the one-node plans of `infer`.

```rust
let bound = recipe.gguf("model-00001-of-00004.gguf").model();
bound.blocks(); bound.tensors(); bound.vocabulary();
let logits = bound.infer(&ids);
```

`model` builds the whole model the file describes. `general.architecture`
selects a row of the architecture table (`llama`, `qwen2`, `qwen3`, `qwen2moe`,
`qwen3moe`, `qwen35`, `qwen3next`, `qwen4exp`); the `<architecture>.*` namespace
sizes every block (`block_count`, `embedding_length`, `attention.*`, `rope.*`,
`feed_forward_length`, `expert_*`, `ssm.*`, `full_attention_interval`,
`hyper_connection.*`); and every parameterized node binds by the standard tensor
names: `token_embd`, `blk.<n>.attn_*`, `ssm_*`, `ffn_*`, `hc_*`, `output_norm`
and `output` (tied to the embedding when absent). Each block is an attention or
delta block and its feed-forward, each on the residual stream behind its
pre-normalization, or inside the hyper-connection mixer the file declares. A
row adds only what the names leave implicit, such as which channels the rotary
embedding pairs. A tensor no node reads, a node no tensor fills, a shape that
does not match the block, or a tensor the file lacks is an error before any
device is touched, naming the tensor and the node. `infer` runs one forward
over token ids and returns `vocabulary` rows of one logit per position.

## tokenizer

```rust
let tokenizer = recipe.gguf("model.gguf").tokenizer();
let ids = tokenizer.encode("Hello, world");
let text = tokenizer.decode(&ids);
tokenizer.bos(); tokenizer.eos(); tokenizer.pad();
tokenizer.chat(&[("system", "Be brief."), ("user", "Hi")], true);
```

A byte-level BPE tokenizer built from the GGUF metadata alone: `tokenizer.ggml.tokens`, `token_type`, the special ids, and the pre-tokenizer family named by `tokenizer.ggml.pre` (the GPT-2, Llama 3, and Qwen 2 regex families). Pieces rank by `tokenizer.ggml.merges` when the file lists merges and by `tokenizer.ggml.scores` when it ranks each piece on its own, so both spellings of a vocabulary run the same merge loop. Control and user-defined tokens are matched whole, longest first, and no merge ever spells one out of ordinary pieces. `encode` frames the ids with the sequence tokens `add_bos_token` and `add_eos_token` ask for, and `decode` rejoins bytes split across tokens.

`chat` renders `tokenizer.chat_template` for a conversation of role and content pairs, in the Jinja subset the common templates use: `{% for %}` over `messages`, `{% if %}`/`{% elif %}`/`{% else %}` with `==`, `!=` and `not`, `{{ }}` substitution of `bos_token`, `eos_token`, `add_generation_prompt` and the message fields, and the `-` whitespace controls. Anything outside that subset is named in an error rather than ignored.

## decode

```rust
let mut sampler = recipe.sampler().temperature(0.8).top_k(40).top_p(0.95).repeat(1.1, 64).seed(7);
let generation = recipe.decode("model.ogdl", &prompt_ids, &mut sampler, &[eos], 64);
generation.ids;
generation.logits;
generation.prefill_seconds;
generation.step_seconds;
```

The model reads a sequence of ids and returns one logit per id. The prefill runs the prompt and the decode then holds that state: a step adds one id, extends the attention keys and values, the recurrent state, and the convolution tail by the one position the id reaches, and samples from the new logits (penalty, top-k, top-p, min-p, temperature, seeded draw; temperature zero is greedy). A step therefore reads what earlier calls left rather than running the sequence again, and the result is the result of one forward of the same ids. The decode stops at a stop id, after the budget, or when the ids fill the model's sequence. `recipe.decode` is the one-device case of a placement: the model placed as one range on the primary device, decoded over its one tape.

```rust
recipe.serve("model.ogdl", "127.0.0.1:8080", 64);
```

`serve` answers that many decode requests over HTTP and returns. A request names its prompt in the target, as `GET /decode?ids=3,1,4&budget=16&stop=2&temperature=0.8&top_k=40&top_p=0.95&min_p=0.05&penalty=1.1&seed=7`, and each field it leaves out keeps the sampler's default. The answer is chunked and carries one id per chunk as the decode reaches it.

## ngram

```rust
let table = recipe.gguf("ngram.gguf");
let ngram = table.ngram();
let prediction = ngram.infer("model.ogdl", &input, &ids);
```

N-gram embeddings from a table too large for device memory. Every token hashes
itself with its predecessors into one row per head of the mapped `[width, rows]`
table, the context cut at an end-of-sequence id, and only the addressed rows
decode, in any quantization. One description addresses the rows: the head
ranges as `offsets` and `vocabularies`, the fold, the end id, and the id a
missing predecessor reads as. The `ngram.*` keys fill it with `ngram.heads` heads
per order over equal ranges of `ngram.table`, folded from `ngram.seeds`; the
`<arch>.ple.*` keys of a per-layer embedding fill it with the reference
multiply-xor fold over `layer_multipliers` into `head_offsets` and
`head_vocab_sizes` of `per_layer_token_embd.weight`, cut at `eos_token_id`.
`ngram.rows(&ids, position)` names the rows, `ngram.bytes()` the table bytes one
token reads.

`ngram.infer` is the injection form: an `ngram.conv` tensor convolves the gathered
rows across as many trailing positions, and the gathered vector is added to the
stream before the block `ngram.layer` names, the gather on the host holding the
table and the blocks either side of it on the device.

```rust
let model = recipe.model().embed(vocab, 2560).hyper(4, 320, &branch).ple(&table);
let plan = shards.plan().named(&shards, "per_layer_token_embd.weight").named(&shards, "blk.1.ple_key.weight");
let output = shards.infer(&model, &plan, &ids, 1);
let generation = shards.decode(&model, &plan, 64, &prompt, &mut sampler, &[], 1);
```

`ple(&table)` is the block form, a per-layer embedding at any layer of a model,
on a stream of any width. The host gathers the table's rows for every position
and stages them for the device; everything after the gather is ordinary nodes:
bias-free key and value projections of the gathered vector, root-mean-square
norms grouped over one lane of the stream with a scale over every lane, a
per-lane dot product of the normalized key and stream over the root of the lane
width, the gate `sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))`, the value broadcast
over the lanes under that gate, a third grouped norm, a causal depthwise
convolution of `ple.conv_kernel` taps dilated by the n-gram size, SiLU, and the
sum of both terms added into the stream. A decode step reads the convolution's
earlier positions from the tape, so a chunked prefill equals a single-shot one.
Its weighted nodes bind in order: the table, the key projection, the key norm,
the stream norm, the value projection, the convolution norm, and the taps. The
table's bytes join no device: `place` keeps it on the host and `resident_bytes`
never counts it. `Gguf::decode` runs a bound model as `recipe.decode` runs a saved
one, over a sequence of that many id positions.

## placement

```text
recipe --device nv0 --device nv1 --device cpu model.rs
```

```rust
let placed = recipe.place("model.ogdl", &[]);
let prediction = placed.infer(&input);
placed.split();
placed.resident_bytes();
placed.moved_bytes();
```

Inference blocks across the selected devices. An empty split is measured: each block joins the current device while its parameters and carried state fit that device's free memory, and the CPU is selectable last so a placement can end on the host. A split names the blocks each device takes instead. Every range runs as its own tape on its device and the stream hops between them, so the output equals a single-device run.

## files

```bash
recipe.rs       runtime
amd-nv-cpu.ll   kernels
build.rs        compiler
cli.rs          cli options
test.rs         combo testing
```

## 20 thingys:
```rust
weights:
	layer(neurons)
	conv(filters, kernel)
	dconv(kernel)
	delta(heads, kernel)[.keys(heads, width)][.values(width)][.out(width)]
	attn(heads)[.kv(heads)][.head(width)][.qk(rms|l2)][.rope(dims, base)][.index(heads, width, block, keep)[.budget(tokens)][.score(rms|l2, dims)]][.gate()]
	perc(width)
	embed(vocab, width)
	ple(&table)
	rnn(hidden)
	gru(hidden)
	lstm(hidden)

blocks:
	glu(hidden, activation)
	moe(experts, topk, hidden, activation, scoring, renormalize, shared)
	res([...])
	hyper(lanes, rank, &model)

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

`embed` must be the first block and must carry a quantization. Every input column is one token id below `vocab`, the input reaches the tape as `i32` ids, and the block emits one `width`-channel vector per column. The gather decodes each addressed row out of the packed table, so `width` must be a whole number of the layout's blocks and the run reads one packed row per token instead of the table. The table keeps the values it was quantized from and no optimizer step writes it back.

`hyper` widens the residual stream to `lanes` copies of the width. Each block normalizes the stream with per-lane `rms` statistics under one trainable scale over the whole stream, reads the normalized lanes into a Recipe submodel as their mean under a read gate, writes the submodel output back through one write gate per lane, and the head reads the stream once more before the output projection. The read gate is `sigmoid(up(silu(down(xn) / lanes)))` through a `rank` bottleneck, the write gate is `2 sigmoid(inject(xn) / lanes)`, and no projection carries a bias, so the mixer holds `stream + 2 stream rank + stream lanes` parameters per block and `stream + 2 stream rank` at the head. `rank` zero adds no node and fixes every gate at one: the read is the mean of the raw lanes, which for one lane is the plain residual.

`dconv(kernel)` is a causal depthwise convolution: every channel mixes its own last `kernel` positions with one tap each, left-padded with zeros, so the shape is unchanged and position `t` sees `t - kernel + 1 ..= t`. The taps may sit a dilation apart, as a per-layer embedding's do, so tap `j` reads `t - (kernel - 1 - j) * dilation`; a dilation of one is the plain form.

`ple(&table)` is a per-layer embedding block over the hashed row table a GGUF describes; see the ngram section.

`delta(heads, kernel)` is a gated delta rule. It projects the input to a query, key and value stream, runs `dconv(kernel)` over that stream, normalizes each head's query and key to unit length, and carries one `channels / heads` square state per head with `S <- g S + beta k' (v - k S)`, reading `o = q S`. The decay `g = exp(-softplus(a) exp(A))` and the write gate `beta = sigmoid(b)` come from a second projection, one of each per head; `A` is one trained scale per head. The output takes a per-head `rms` normalization, the gate `sigmoid(z)` from a third projection, and a fourth projection back to the input width. The sequence walks in chunks of `delta-chunk` positions and commits the carried state at each chunk start; a chunk of one is a decode step, and every chunk size gives the same values.

`.keys(heads, width)`, `.values(width)` and `.out(width)` name the extents instead of taking them from the stream. `heads` remains the value head count; each key head serves `heads / keys` of them, the carried state per value head is `key width` by `value width`, and the closing projection ends at `out`. Without them the value width is the stream over `heads`, the keys match the values one for one, and `out` is the stream.

```rust
.delta(48, 4).keys(16, 128).values(128).out(2560)
```

`glu(hidden, activation)` is one gated feed-forward, `down(activation(gate(x)) * up(x))`, through `hidden` and back to the block input width.

`moe` scores every position with one `[width, experts]` router and keeps the `topk` highest scores. `scoring` reads those scores as a softmax over every expert or as a sigmoid of each one, and `renormalize` divides the kept weights by their own total; a plain softmax leaves the dropped experts weighted zero, which is the evaluate-all-then-mask reference. Only the kept experts run: each position gathers its own slices of the `[experts, hidden, width]` gate and up tables and the `[experts, width, hidden]` down table, and takes `down(activation(gate(x)) * up(x))` under its routing weight. A position costs `topk` experts, not `experts`. With `shared` set, one always-on expert of the same shape runs for every position under its own gate: a `[width]` projection of the position with no bias, through a sigmoid, is the routing weight its output joins the sum under, so the block holds, trains and binds a per-position shared-expert gate. A bound `moe` takes its tensors in that order: the router, the gate, up and down tables, then the shared gate vector and the shared gate, up and down tables.

## 2 block qualifiers

```rust
blck.atvn.norm.quant
frozen.blck.atvn.norm.quant
packed.blck.atvn.norm.quant
frozen.packed.blck.atvn.norm.quant
```

`frozen` holds a block's own weights at their current values for the whole run and creates no
optimizer state for them; the block still passes an input adjoint back, so earlier blocks learn.
`packed` keeps a block's weights in their selected quantized representation and decodes each
weight inside the consuming kernel, so inference never holds a decoded copy. `frozen` affects
training only, `packed` affects inference only, and neither changes the selected precision or
quantization. A qualifier on a block that owns no weights is rejected, as is `packed.frozen`.

```rust
let model = recipe.model()
	.frozen().packed().layer(32).qi(4).0.gelu()
	.layer(1);
```

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

A normalization that leads a model normalizes the model input before its first
block, so `hyper(1, 0, &recipe.model().norm(rms).attn(8))` is a residual branch
behind its pre-normalization.

`.qk(rms|l2)` follows `attn(heads)` and normalizes each head's query and key rows
over its head-width slice, leaving the values untouched:

```rust
.attn(4).qk(rms)
```

`.epsilon(value)` sets the model's normalization epsilon, the shift under every
variance and the floor under every norm the model lowers: block norms, query and
key norms, the delta rule's norms, hyper-connection gates, and the attention
node. It sits beside `.loss` on the model, so one value covers every block and
every hyper branch. The default is the `normalization-epsilon` in `Cargo.toml`.
The value is saved in the bundle with the model, so a model reloads with the
epsilon it was trained with on any build, and a bundle written before models
carried one reloads with the default it was lowered with. A checkpoint bound
from GGUF takes its own `attention.layer_norm_rms_epsilon` the same way:

```rust
recipe.model().layer(16).gelu().attn(2).qk(rms).layer(1).epsilon(1e-6)
```

## sparse attention

`attn(heads)` builds one query, key and value plane per head. `.kv(heads)` unties
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
attention output by a sigmoid of its own projection of the block input. `.head(width)`
names the head width, so the heads need not partition the stream: the block attends
over `heads * width`, its gate spans that same width, and the output projection
returns to the stream width. Without it the head width is the stream over `heads`.

Under `recipe.decode` the indexer is state like the keys and values: the context
arena keeps one running sum of indexer keys per block, a step adds its one key
to the block it lands in and scores only its own query, and the selection equals the
whole-sequence selection of the same prefix.

```rust
.attn(8).kv(2).qk(rms).rope(32, 10000.0).index(2, 16, 32, 4).gate()
.attn(8).kv(2).qk(rms).rope(32, 10000.0).index(2, 16, 32, 4).budget(128).score(rms, 16).gate()
.attn(24).kv(2).head(256).gate()
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
