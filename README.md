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

The model reads a sequence of ids and returns one logit per id. The prefill runs the prompt and the decode then holds that state: a step adds one id, extends the attention keys and values, the recurrent state, and the convolution tail by the one position the id reaches, and samples from the new logits (penalty, top-k, top-p, min-p, temperature, seeded draw; temperature zero is greedy). A step therefore reads what earlier calls left rather than running the sequence again, and the result is the result of one forward of the same ids. The decode stops at a stop id, after the budget, or when the ids fill the model's sequence.

```rust
recipe.serve("model.ogdl", "127.0.0.1:8080", 64);
```

`serve` answers that many decode requests over HTTP and returns. A request names its prompt in the target, as `GET /decode?ids=3,1,4&budget=16&stop=2&temperature=0.8&top_k=40&top_p=0.95&min_p=0.05&penalty=1.1&seed=7`, and each field it leaves out keeps the sampler's default. The answer is chunked and carries one id per chunk as the decode reaches it.
## speculate

```rust
let run = recipe.speculate("model.ogdl", "draft.gguf", &prompt, &mut sampler, &[eos], 64);
run.proposed;
run.accepted;
```

`speculate` decodes with a multi-token-prediction draft head read from a second
GGUF: a block set plus the `nextn` tensors, which are a projection that fuses
the embedding of the last id with the model's final hidden state, a norm for
each half, a head-side residual gate, and the shared head. After each step the
head proposes the next id and the step after it accepts the proposal when the
model reaches the same id. The ids are the model's own either way, so they are
the ids `decode` produces from the same prompt and sampler.

## gguf

```rust
let model = recipe.gguf("model-00001-of-00004.gguf");
model.value("general.architecture");
model.tensors();
model.contract("blk.0.ffn_up.weight", &input, 16);
```

Every shard of a split is opened by name and the tensor data stays mapped. A
quantized tensor binds to the tape in its own GGML layout, so the contraction
reads the mapped bytes through the block decoders that read saved `.ogdl` models.

## files

```bash
recipe.rs       runtime
amd-nv-cpu.ll   kernels
build.rs        compiler
cli.rs          cli options
test.rs         combo testing
```

## 19 thingys:
```rust
weights:
	layer(neurons)
	conv(filters, kernel)
	attn(heads)
	perc(width)
	embed(vocab, width)
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

`embed` must be the first block and must carry a quantization. Every input column is one token id below `vocab`, the input reaches the tape as `i32` ids, and the block emits one `width`-channel vector per column. The gather decodes each addressed row out of the packed table, so `width` must be a whole number of the layout's blocks and the run reads one packed row per token instead of the table. The table keeps the values it was quantized from and no optimizer step writes it back.

## 15 activations

```
relu  leak  sigmoid  tanh   selu   gelu   silu   elu
prelu cos   exp      log    ln     huber  tan
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
