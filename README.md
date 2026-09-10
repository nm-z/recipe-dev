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

## samples

A caller selects sources, features and targets. Where the samples begin and end
is read from the source, not declared.

A lone table's rows are its samples. When a folder holds a group of sibling
tables and another table has a column recording their file names, that group is
one sample per file, and the recording column is what both identifies and orders
them — no order is ever taken from the path. A column used that way is identity
and not a feature, so the file name never reaches the model as a value.

```
scans/
	meta.csv        scan,temperature,y      3001 lines: a header and 3000 rows
	scan-0000.csv   magnitude,phase         one sample
	...             (2999 more)
```

That is 3,000 samples. Each one is the whole vector of the scan its row names
plus the row's own selected values. Sources that disagree on the sample count
are refused, saying how each was read; nothing is cycled or repeated to make the
counts match.

## devices

Use one `--device` flag with a dot-separated chain. Devices before the first host prefix belong to the machine running the command. A host prefix applies to the following devices until another host prefix appears. Commas and repeated `--device` flags are invalid.

```text
recipe run train.rs --device amd0.amd1
recipe run train.rs --device nv0.cpu
recipe run train.rs --device engi:amd0.cpu.archy:cpu.nv7.nv8
recipe --device amd0.archy:nv0 run train.rs
```

`cpu` selects the host's available logical-CPU pool, not an individual socket. Numbered CPU selectors are not supported. The `run` keyword is optional.

Unqualified components shaped like device names select devices: on Engi, `nv0.lan:amd0` means Engi's `nv0` and the SSH host `lan`'s `amd0`. A hostname without `:<device>`, such as the final component in `amd0.archy`, is invalid. Missing devices or SSH hosts are errors, not fallback selections.

`recipe --worker <device>` serves one local CPU or GPU over stdin and stdout. Recipe starts this transport entrypoint through SSH for a host-qualified selector. It is a protocol endpoint, not a model script invocation.

## RAT

Native AMD tuning is opt-in: set `rat-enabled = true` under `[package.metadata.recipe]` in `Cargo.toml` and rebuild. The default is `false`. The knob model selects a configuration from the queried device's search space. The bench model learns from measured training time and supplies gradients to the knob model. Training retains the fastest measured configuration after the configured observation budget.

Run the normal entrypoint, such as `recipe run model.rs --device amd0`. When enabled, tuning publishes both models together in one atomic `<knob stem>.pair.ogdl` bundle beside the configured knob model. Valid legacy knob/bench pairs can be loaded without overwriting them. Unusable tuner state or unavailable HIP occupancy support leaves heuristic training available. Set `RECIPE_DEBUG=1` for additional diagnostics in `recipe.log`.

Command RAT is separate: `.loss(&evaluator)` requires `.rat(policy, "<script>")`. It does not enable native AMD tuning or use its persisted models.

```rust
recipe.train().rat(history, "./evaluate");
recipe.train().rat(rolling, "./evaluate"); // data must specify .split(fraction)
recipe.train().rat(online, "./evaluate");
recipe.train().rat(learned, "./evaluate");
```

`history` fits every accumulated scored observation. `rolling` retains the newest `floor(source_rows * fraction)` observations, with a minimum capacity of one. `online` retains only the newest observation while preserving model and optimizer state. For command RAT, `.split()` controls only the rolling capacity; it does not create a holdout.

`learned` predicts one continuous selection value per available observation and trains on values at least `0.5`. A second model predicts the resulting primary surrogate R² over the available observations. It uses online fitting, then stays frozen while updating the selector. Both selector models preserve their parameters as the observation count grows. An empty selection leaves the primary surrogate unchanged.

Replay growth changes buffer allocations, not the loaded native program. Each
dispatch passes its active row count; arenas carry their current node offsets.
The learned selector also passes its active sequence length. Growing either
capacity preserves the kernels, weights, and optimizer buffers. Changing the
model structure, arithmetic format, or an explicitly retuned schedule can still
require a different program.

Each command writes one finite raw score to stdout and no stderr. Any stderr output or unsuccessful exit stops training. `.log(score)` prints the raw score. Replay and surrogate fitting use that same raw value without automatic bounding, normalization, or sign changes. Users may transform scores in their own executable. `recipe.train().target(value)` sets the proposer's desired raw score and defaults to zero; it does not replace the evaluator's measured training targets. With MSE and `.target(0.0)`, the evaluator minimizes `(prediction - measurement)^2`, while the proposer minimizes `(prediction - 0)^2` through the frozen evaluator. Native hardware timing models retain their existing log-time units.

### Stateful command RAT

Use `recipe.data(auto)` with command RAT when the executable owns the state.
No table file or data-level `.target()` declaration is needed: the executable supplies
the state fields, action fields, and complete valid choices.

```rust
let data = recipe.data(auto);
let evaluator = recipe.model().layer(16).tanh().loss(mse);
let proposal = recipe.model().layer(4).loss(&evaluator);
recipe.train().rat(history, "./evaluate.lua")
    .target(0.0).epochs(100).log(all).run(&proposal, &data);
```

Recipe starts one process with `RECIPE_RAT_PROTOCOL=1`. Requests and replies
are newline-delimited UTF-8. Flush each complete reply. Example interaction:

```text
Recipe: reset
Evaluator: state position=0,remaining=8
Evaluator: actions destination,amount
Evaluator: choice 1,4
Evaluator: choice 2,8
Evaluator: ready
Recipe: choose destination=1,amount=4
Evaluator: state position=1,remaining=4
Evaluator: actions destination,amount
Evaluator: choice 2,4
Evaluator: ready
Recipe: choose destination=2,amount=4
Evaluator: score 3.25
```

The `Recipe:` and `Evaluator:` prefixes illustrate direction and are not sent.
State and action names, order, and widths remain fixed during training. Values
and the number of valid choices may change. Names must be unique and contain
no whitespace, comma, or equals sign. All values and scores must be finite.
`score` terminates an episode; its meaning belongs to the executable. The
proposer optimizes toward the configured raw target. The executable handles `reset`
for a fresh episode and `close` for a clean exit. Stderr output, malformed
frames, unexpected EOF, or an unsuccessful exit are errors, not low scores.

The proposer emits one real preference per action field. Recipe selects the
listed complete choice with the smallest squared distance after scaling each
field by its current choice range. Constant fields do not contribute; ties
follow the executable's choice order. This enforces supplied choices without
assigning domain meaning to names or inventing constraints.

Each visited state and raw proposal receives the executable's terminal score.
There are no backend progress rewards, timing penalties, or artificial success
scores. The existing surrogate fitter and frozen surrogate backpropagation
are reused, with raw measured targets. One epoch fits the surrogate once and
updates the proposer once using a retained state, then evaluates the updated
policy through another complete episode. Thus N epochs run N+1 episodes.
The models persist across decisions and episodes; changing the choice count
does not recreate their weights.

`history` retains all scored decisions, `online` retains the latest one, and
`full` replaces observations with the latest episode. For stateful `rolling`,
`.split(fraction)` sets capacity from the number of decisions in the latest
completed episode. `learned` uses the existing learned replay selection.
The log's score is the latest raw terminal score; loss and R² describe the
surrogate's fitting observations, not a held-out evaluation or an optimality
claim. `.resume()` and nondefault `.stop()` are unsupported in this mode.
`.log(all)` and `.log(dev)` also print `choices`, the number of decisions in
the rollout that produced that line's score, followed by `window`, the number
of observations fitted in that update. `.log(Choices)` selects the count alone.

## files

```bash
recipe.rs       runtime
amd-nv-cpu.ll   kernels
build.rs        compiler
cli.rs          cli options
```

## blocks

```
frozen.packed.blck
```

## 18 thingys

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

	A branch step is an ordinary model step, so anything above goes inside one,
	carrying its own activation, normalization, quantization and profile, and a
	branch nests inside a branch:

	res([layer(8), relu(), layer(8)])
	res([norm(rms), layer(8).act(Activation::Relu), layer(8).quantize(0, 8, 0)])
	res([res([layer(8), relu(), layer(8)]), gelu()])
	moe(1, [layer(8), res([layer(8), relu(), layer(8)])])

	norm(rms)          a normalization on its own, computing nothing before it

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

## data

```rust
data(auto)
	.test(source)
	.set(source)
	.include([features])
	.exclude([features])
```

## training

```rust
.seed(value)
.optimizer(adamw)
.log(metrics)
.resume(path)
```

## losses

```rust
.loss(mse|rmse|huber|mae|bce|ce|focal)
```

## 16 activations

```
relu  leak  sigmoid  tanh   selu   gelu   silu   elu
prelu cos   exp      log    ln     huber  tan
scale(factor)
```

`scale(factor)` multiplies every value the preceding block produces by one
finite constant. It owns no weights and preserves the shape, so it states an
explicit scalar rescaling directly.

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

## exclusions

```rust
recipe.model().no(bias).layer(64).relu().layer(1)
```

`.no(option)` declares a default the model excludes. `.no(bias)` removes the
bias from every weighted block beneath it — layers, attention projections,
convolutions and recurrent gates — and from every nested branch. Excluded
tensors are not allocated, initialized, trained, saved or loaded, and the
exclusion travels with the saved model.

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

## observability

```rust
.log(Run|Loss|R2|Time|Epoch|blck|tile|all)
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

## clanker docs

### planned

```rust
.embed(vocab, width)
.attn(q, k, v) // n heads
.no(options)     // exclude default model behavior such as bias.
	.no(bias)
.recur([...])
.scale(factor)   // multiply every value from the preceding step by one constant.
.rope(layout, dimensions, base)
	neox
	yarn(factor, og_ctx, b_fast, b_slow)
branching:
	let gate = recipe.model().layer(width).gelu();
	let up = recipe.model().layer(width);
	let output = gate * up;
```
