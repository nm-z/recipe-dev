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
model.contract("blk.0.ffn_up.weight", &input, 16);
```

Every shard of a split is opened by name and the tensor data stays mapped. A
quantized tensor binds to the tape in its own GGML layout, so the contraction
reads the mapped bytes through the block decoders that read saved `.ogdl` models.

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
	attn(heads)
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
