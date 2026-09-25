# GGUF model definitions: implicit behavior and explicit API shapes

Recipe's automatic GGUF builder (`file.model()`) and its conventional binder make model decisions the model definition never states: which tensor fills which projection, which block each layer uses, which activation, normalization, gate and wrapping apply. Each item below records one such decision, where it is made, and the explicit spelling proposed for it.

The proposed spellings are theoretical API shapes. Each item says which parts exist today.

Examples use two files: Qwen3.8-Flash-Next (`qwen4exp`: 48 blocks, attention on every 4th block, delta on the rest, 512-expert MoE with top-10 routing, 4 hyper-connection lanes with rank 320) and the Gemma3 RNJ model (`gemma3`: attention on every block, plain residual).

## Rules the sketches follow

1. **File keys are values.** Every metadata key and tensor in the opened file is a path in the script, and an unknown path is a compile error. `recipe keys` lists them.

	```rust
	qwen4exp.expert.used.count	// usize, 10
	general.sampling.temp	// f32, 1.0
	blk.3.attn.q.weight	// tensor, [2560, 12288]
	```

2. **Underscores are path separators.** `ffn_gate_inp_shexp` is written `ffn.gate.inp.shexp`. Every path maps back to exactly one key; a file where two keys collapse to one path is rejected when opened. Keyword segments are written raw: `general.file.r#type`.

3. **A node can be a value and a group.** `qwen4exp.attention.head.count` is 24 and `qwen4exp.attention.head.count.kv` is 2; `qwen4exp.embedding.length` is 2560 and `qwen4exp.embedding.length.per.layer.input` is 160. Parameters take Recipe's existing traits (`Width`, `Root`), so these nodes pass wherever a number is accepted.

4. **Build-time names can be path segments.** A loop variable, or a `let` computed from loop variables, may stand in a segment: `blk.a.attn.q.weight`. Recipe unrolls the loop before compiling, so each copy is checked against that block's own keys. The same rule indexes per-layer lists: `qwen4exp.attention.compress.ratios.a`.

5. **Tensors bind in the block that uses them.** `layer(tensor)` is a linear layer whose weights are that tensor; its width comes from the tensor's shape. A 3-D tensor whose last dimension is the expert count is an expert bank.

6. **Declaration order is execution order.** `recipe.model()` starts the model once. Each block is chained onto it (`model = model.res([...])`) and runs in the order declared. A block that is built and not chained is dropped, as Rust drops any unused value.

7. **Inside one block, the chain configures and lowering orders.** `attn(heads).kv(…).q(…).qk(…).rope(…)` sets fields of one attention block; its execution order is fixed:

	```text
	1. Wq, Wk, Wv project x
	2. q/k normalization per head
	3. rope on q and k
	4. index: score key blocks, keep the best (when .index is set)
	5. softmax(q·kᵀ/√d)·v
	6. output projection
	```

8. **Loops and conditions run at build time.** `if a % 4 == 3 { … } else { … }` chooses which blocks are chained; the model itself stays a fixed list. A condition may use loop variables and metadata, never data flowing through the model.

9. **A gate is a product with a sigmoid branch.** Both sides of `*` read the same input, so `x * layer(w).sigmoid()` scales each channel of each token by a value in (0, 1) computed from that token. `.scale(f)` multiplies by one fixed number; a gate multiplies by a learned, per-token factor.

10. **Loop bounds are literal; scalars are keys.** `(3..48).step_by(4)` shows the model's shape at a glance; `attn(qwen4exp.attention.head.count)` says what the number is.

## One cause

Items 1, 3, 4, 5, 6, 7, 8, 11, 14 and 15 share one cause: the builder reads the file and decides what the definition does not state. Once key paths bind (items 2 and 3), each becomes a matter of writing the tensor and the choice in the definition.

## Items

1. **Attention gate: Inferred from query width, and hidden inside `.gate()`.** The builder treats a query tensor with `2 × heads × head` outputs as gated. The gate rows are interleaved per head (`[q₀, g₀, q₁, g₁, …]`, stride `2 × head`), split out, passed through a sigmoid, and multiplied into the attention result before the output projection. `.gate()` names none of this. [Loader](/home/nate/Desktop/recipe-dev/recipe.rs:15022), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18248), [kernel](/home/nate/Desktop/recipe-dev/amd-nv-cpu.ll:5532).

	```rust
	let block = attn(heads).gate();
	```

	A contiguous `split(2)` of `attn_q` would be wrong for this file: it would return heads 0–11's queries and gates mixed together. The gate is either declared on the query, so the binder reads the per-head layout and checks the width:

	```rust
	attn(qwen4exp.attention.head.count)
		.q(blk.a.attn.q.weight).gate(sigmoid)
	```

	or, where a file stores the gate as its own tensor, written as the product it is (rule 9):

	```rust
	(attn(heads)… * layer(blk.a.attn.gate.weight).sigmoid())
		.layer(blk.a.attn.output.weight)
	```

	Exists today: `.gate()` with sigmoid fixed. Proposed: `.gate(sigmoid|silu|tanh)` (README Proposed), `.q(tensor)`, `layer(tensor)`.

2. **Key paths: Hard-coded.** A script reads only the architecture keys Recipe hard-codes, such as `gemma3.embedding_length`. Any other key has to be found with `recipe keys` and copied into the script as a literal. [Namespaces](/home/nate/Desktop/recipe-dev/recipe.rs:15409).

	```rust
	// recipe keys qwen.gguf | rg temp  ->  general.sampling.temp.1.0
	recipe.sampler().temperature(1.0)
	```

	```rust
	recipe.sampler().temperature(general.sampling.temp)
	```

	Every key the file contains resolves, under rules 1 to 4.

3. **Tensor binding: Chosen by conventional names.** `recipe.infer().run(&model, &data)` calls `conventional_plan`, which assigns weights by fixed tensor names and recognized shapes. A public `Binding::named` exists, but the usual `Data` path does not give the script the `Gguf` value it requires. [Inference](/home/nate/Desktop/recipe-dev/recipe.rs:15549), [binding](/home/nate/Desktop/recipe-dev/recipe.rs:14520).

	```rust
	let data = recipe.data(path);
	recipe.infer().run(&model, &data); // conventional_plan chooses tensor names.
	```

	The definition names each tensor where it is used, and the blocks are chained (rule 6):

	```rust
	for a in (3..48).step_by(4) {
		model = model.attn(qwen4exp.attention.head.count)
			.q(blk.a.attn.q.weight)
			.k(blk.a.attn.k.weight)
			.v(blk.a.attn.v.weight)
			.out(blk.a.attn.output.weight);
	}
	```

	A tensor no block binds is an error before execution, as the automatic builder does today.

4. **Which product branch gets `ffn_gate.weight`: Inferred.** The conventional binder assigns `ffn_gate.weight` to the branch that has an activation and `ffn_up.weight` to the other; when that does not distinguish them, it uses branch order. [Binder](/home/nate/Desktop/recipe-dev/recipe.rs:15800).

	```rust
	let product = layer(8).gelu() * layer(8);
	// Hidden: the activated branch binds ffn_gate.weight; the other binds ffn_up.weight.
	```

	```rust
	(layer(blk.a.ffn.gate.weight).gelu() * layer(blk.a.ffn.up.weight))
		.layer(blk.a.ffn.down.weight)
	```

5. **Feed-forward activation: Hard-coded.** The GGUF builder emits `glu(hidden, SiLU)` for ordinary feed-forward blocks. The handwritten Gemma3 RNJ model spells `layer(...).gelu() * layer(...)`. The two definitions disagree in source; this audit did not measure the numerical difference. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15187), [RNJ model](/home/nate/Desktop/recipe-dev/rnj-1.rs:39).

	```rust
	let automatic = file.model(); // Builder inserts GLU with SiLU.
	let handwritten = layer(hidden).gelu() * layer(hidden); // RNJ uses GELU.
	```

	With item 4's spelling, the activation sits on the branch bound to the gate tensor:

	```rust
	let mut model = recipe.model()
		.embed(tokenizer.ggml.tokens, gemma3.embedding.length)
		.scale(gemma3.embedding.length.sqrt());

	for a in 0..26 {
		model = model.res([
			norm(rms, blk.a.ffn.norm.weight),
			(layer(blk.a.ffn.gate.weight).gelu() * layer(blk.a.ffn.up.weight))
				.layer(blk.a.ffn.down.weight),
			norm(rms, blk.a.post.ffw.norm.weight),
		]);
	}
	```

6. **Attention details: Chosen from tensor shape or presence.** The builder decides the gate from query width (item 1), uses `attn_k` as values when `attn_v` is absent, adds Q/K RMS normalization when `attn_q_norm` exists, chooses rotary pairing from an architecture table and reorders Q/K rows to match (item 7), sets a sliding window per layer and disables rope factors on sliding layers, binds `rope_freqs` when it exists, and adds an indexer when metadata names one: its block size from `compress_ratios[layer]`, its kept blocks as ⌈top_k / block⌉, and its score normalization when indexer norm tensors exist. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15034).

	```rust
	let automatic = file.model();
	// Hidden: gate, value fallback, Q/K norm, pairing, window, factors and indexer come from the file.
	```

	```rust
	attn(qwen4exp.attention.head.count)
		.kv(qwen4exp.attention.head.count.kv)
		.head(qwen4exp.attention.key.length)
		.q(blk.a.attn.q.weight).gate(sigmoid)
		.k(blk.a.attn.k.weight)
		.v(blk.a.attn.v.weight)
		.qk(rms, blk.a.attn.q.norm.weight, blk.a.attn.k.norm.weight)
		.rope(neox, qwen4exp.rope.dimension.count, qwen4exp.rope.freq.base)
		.index(qwen4exp.attention.indexer.head.count, qwen4exp.attention.indexer.key.length,
			qwen4exp.attention.compress.ratios.a, qwen4exp.attention.indexer.top.k)
			.q(blk.a.indexer.q.proj.weight).k(blk.a.indexer.k.proj.weight)
			.score(rms, qwen4exp.rope.dimension.count)
		.out(blk.a.attn.output.weight)
	```

	- `.kv(n)` is the key/value head count: 24 query heads share 2 K/V heads, 12 per group. `.head(n)` is each head's width.
	- `.qk(rms, …)` normalizes each query and key head with its trained scale before rope.
	- `.index(heads, width, block, top_k)` is sparse attention: a small indexer (4 heads, 128 wide) scores the keys in blocks of `compress_ratios[a]`, keeps the best ⌈top_k / block⌉ blocks per query, and attention runs only over those. Recipe computes the division; the definition states the file's `top_k`.
	- `.score(rms, dims)` normalizes the indexer's query and key heads with trained scales and rotates their first `dims` channels before scoring.
	- A file without `attn_v` writes `.v(blk.a.attn.k.weight)`.
	- The block's wrapping (residual or hyper) is items 8, 12 and 14.

	Exists today: `attn`, `.kv`, `.head`, `.qk(rms)`, `.rope(neox, …)`, `.index(heads, width, block, keep)`, `.score(norm, dims)`. Proposed: tensor arguments on `.q/.k/.v/.qk/.out`, and `.index` taking `top_k`.

7. **Rotary pairing and YaRN: Not fully declared.** Rope rotates channel pairs by a position-dependent angle. Files pair neighbours `(0,1), (2,3), …` (Llama) or halves `(i, i + dims/2)` (NeoX: Qwen, Gemma, LFM2, `qwen4exp`). The kernel always computes NeoX; for neighbour-paired files the binder permutes Q/K rows at load, chosen by an architecture-name table. The builder reads the `rope.scaling.*` keys but never calls `.yarn(...)`, so a file that needs YaRN runs without it; the RNJ model calls `.yarn(...)` with `YARN_FAST = 32.0` written as a constant. [Pairing table](/home/nate/Desktop/recipe-dev/recipe.rs:14635), [builder](/home/nate/Desktop/recipe-dev/recipe.rs:15054), [scaling keys](/home/nate/Desktop/recipe-dev/recipe.rs:15406), [RNJ model](/home/nate/Desktop/recipe-dev/rnj-1.rs:27).

	```rust
	let automatic = file.model(); // Builder calls rope but not yarn.
	let explicit = attn(heads).rope(neox, dims, base).yarn(factor, context, fast, slow);
	```

	```rust
	// qwen4exp: half-channel pairs; 64 of each head's 256 channels rotate; no YaRN keys.
	attn(qwen4exp.attention.head.count)
		.rope(neox, qwen4exp.rope.dimension.count, qwen4exp.rope.freq.base)

	// gemma3: YaRN from the file's own keys.
	attn(gemma3.attention.head.count)
		.rope(neox, gemma3.attention.key.length, gemma3.rope.freq.base)
		.yarn(gemma3.rope.scaling.factor, gemma3.rope.scaling.original.context.length,
			gemma3.rope.scaling.yarn.beta.fast, gemma3.rope.scaling.yarn.beta.slow)
	```

	A key the file lacks, such as a missing `yarn_beta_fast`, is a compile error instead of a silent default of 0.0; the definition then writes the literal it intends. `qwen4exp.rope.dimension_sections` (multi-section rope) is not read anywhere in Recipe. Exists today: `neox`. Proposed: `pairs` (README Approved).

8. **Per-layer block type: Inferred.** Each layer is a mixer branch followed by a feed-forward branch. The builder picks the mixer per layer: attention when `(layer + 1) % full_attention_interval == 0`, otherwise short convolution when the architecture has one, otherwise delta. It picks the feed-forward from the presence of experts, and one wrapping for every branch of the model: plain residual, or hyper-connections when `hyper_connection.*` keys exist. [Model loop](/home/nate/Desktop/recipe-dev/recipe.rs:14812), [dimensions](/home/nate/Desktop/recipe-dev/recipe.rs:14930).

	```rust
	let automatic = file.model();
	// Hidden: per-layer metadata chooses attention, short convolution, or delta, and residual or hyper wrapping.
	```

	`qwen4exp` repeats one 4-layer group 12 times: three delta layers, then one attention layer, each followed by the MoE. The loop states that pattern directly:

	```rust
	model = model.hyper(qwen4exp.hyper.connection.count, qwen4exp.hyper.connection.low.rank);
	for g in (0..48).step_by(4) {
		for a in g..g + 3 {
			model = model
				.delta(…)	// item 9
				.moe(…);	// item 11
		}
		let a = g + 3;
		model = model
			.attn(…)	// item 6
			.moe(…);	// item 11
	}
	```

	The same model with a condition instead of a nested loop:

	```rust
	for a in 0..48 {
		if a % 4 == 3 {
			model = model.attn(…);
		} else {
			model = model.delta(…);
		}
		model = model.moe(…);
	}
	```

	Gemma3 uses attention on every layer with a plain residual:

	```rust
	for a in 0..26 {
		model = model
			.res([norm(rms, blk.a.attn.norm.weight), attn(gemma3.attention.head.count)…, norm(rms, blk.a.post.attention.norm.weight)])
			.res([norm(rms, blk.a.ffn.norm.weight), (layer(blk.a.ffn.gate.weight).gelu() * layer(blk.a.ffn.up.weight)).layer(blk.a.ffn.down.weight), norm(rms, blk.a.post.ffw.norm.weight)]);
	}
	```

	A sibling file with `block_count = 49` and `nextn_predict_layers = 1` carries a multi-token-prediction block as `blk.48`; its loops stop at 48, and `blk.48` is declared separately (README Proposed `mtp([blocks])`).

9. **Delta block math: Partly private.** Delta is a mixer across positions, like attention. It carries a fixed-size memory per head from token to token instead of comparing each token with every earlier one. Lowering fixes the whole sequence: one projection to queries, keys and values; a causal depthwise convolution over the last `kernel` tokens, then SiLU; per-head L2 normalization of queries and keys; the recurrence; per-head RMS normalization of the output; a product with an output gate; and the output projection. Convolution and output activations come from an architecture-name row. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15139), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18229).

	```rust
	let block = recipe.model().delta(heads, kernel);
	// Hidden: tensor roles, convolution activation, L2, decay transform, RMS, output gate.
	```

	The recurrence, per head, with memory `S`:

	```text
	S ← decay · S                     forget
	S ← S + β · (v − S·k) · kᵀ        correct the memory toward v: the delta rule
	o = S · q                         read
	```

	In `qwen4exp`, `ssm.inner_size` 6144 is 48 value heads × 128, `ssm.group_count` is 16 key heads, and each head's memory is 128 × 128.

	```rust
	(delta(qwen4exp.ssm.time.step.rank, qwen4exp.ssm.conv.kernel)
		.keys(qwen4exp.ssm.group.count, qwen4exp.ssm.state.size)
		.qkv(blk.a.attn.qkv.weight)
		.conv(blk.a.ssm.conv1d.weight).silu()
		.qk(l2)
		.decay(blk.a.ssm.alpha.weight, blk.a.ssm.a, blk.a.ssm.dt.bias)
		.write(blk.a.ssm.beta.weight)
		.norm(rms, blk.a.ssm.norm.weight)
		* layer(blk.a.attn.gate.weight).sigmoid())
		.layer(blk.a.ssm.out.weight)
	```

	The output gate is written as a product (rule 9): sigmoid for `qwen4exp`, a different activation for other delta architectures.

10. **Short convolution: Built as an internal product.** The builder slices one stored projection into B, C and X, then constructs `C × depthwise_conv(B × X)` and an output projection. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15106).

	```rust
	let automatic = file.model();
	// Hidden: shortconv slices B, C, X and builds C * dconv(B * X).
	```

	```rust
	((layer(blk.a.shortconv.in.proj.weight.b)
		* layer(blk.a.shortconv.in.proj.weight.x)).dconv(blk.a.shortconv.conv.weight)
		* layer(blk.a.shortconv.in.proj.weight.c))
		.layer(blk.a.shortconv.out.proj.weight)
	```

	`.b`, `.x` and `.c` name row ranges of one stored tensor. Naming a slice is the same open question as the attention gate rows in item 1.

11. **Mixture of experts: Two unrelated operations.** Recipe has two MoEs.
	- **Public `moe(top_k, [blocks; N])`:** each expert is any block and has its own learned router. Selection happens per output element, every expert is computed, and a softmax over the kept experts weights them.
	- **Private `gguf_moe`, used for GGUF files:** one router (`ffn_gate_inp`, width → experts) scores experts per token. The top `expert_used_count` are kept, with scoring (softmax or sigmoid) and renormalization from metadata. Only those experts run, straight from stacked `[in, out, experts]` tables, each a gated FFN with SiLU fixed. A shared expert runs for every token, weighted by `sigmoid(x · ffn_gate_inp_shexp)`, when `ffn_gate_shexp` exists.

	[Metadata](/home/nate/Desktop/recipe-dev/recipe.rs:14923), [builder](/home/nate/Desktop/recipe-dev/recipe.rs:15216), [public and private forms](/home/nate/Desktop/recipe-dev/recipe.rs:12270), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18582).

	```rust
	let public = recipe.model().moe(top_k, experts);
	let automatic = file.model(); // GGUF uses a different, private MoE operation.
	```

	The file's MoE, written with the public name:

	```rust
	let experts = (layer(blk.a.ffn.gate.exps.weight).silu()
		* layer(blk.a.ffn.up.exps.weight)).layer(blk.a.ffn.down.exps.weight);
	let shared = (layer(blk.a.ffn.gate.shexp.weight).silu()
		* layer(blk.a.ffn.up.shexp.weight)).layer(blk.a.ffn.down.shexp.weight);
	moe(qwen4exp.expert.used.count, experts)
		.route(softmax, layer(blk.a.ffn.gate.inp.weight))
		.renorm()
		.shared(shared, layer(blk.a.ffn.gate.inp.shexp.weight).sigmoid())
	```

	- `experts` is one block built on the stacked `[2560, 640, 512]` tables (rule 5); all three tables must agree on 512. `moe(10, [expert; 512])` keeps its current meaning.
	- `.route(scoring, router)` binds the router. A router of width `experts` means one per-token choice, and only the chosen experts run.
	- `.renorm()` rescales the kept weights to sum to 1. Nothing renormalizes unless it is written.
	- `.shared(expert, gate)` adds an expert every token takes, weighted by a gate that yields one value per token. `*` would not work here: a one-value branch and a 2560-wide branch are different shapes.

	Exists today: `moe(top_k, [blocks])`. Proposed: an expert-bank argument, `.route`, `.renorm` (README Proposed), `.shared`.

12. **Hyper-connections: Fixed internal formula.** Hyper-connections widen the residual stream to `lanes` copies of the model width: 4 × 2560 = 10240 in `qwen4exp`. Every branch then:
	- computes gates from the current stream: RMS over the stream, then `read = sigmoid(W_up · silu(W_down · x / lanes))`, one value per stream channel, and `write = 2 · sigmoid(W_inject · x / lanes)`, one value per lane;
	- reads `mean over lanes of (read · lane)`, a single 2560-wide input;
	- runs once;
	- adds `write[lane] · output` into every lane.

	`rank` is the width of the read gate's bottleneck (`W_down` 10240 → 320 → `W_up`). With `rank = 0` every gate is 1, and each lane is a plain residual. The first hyper widens the stream, and a block that is not lane-aware collapses it through its own read gate (`output_hc_*`). The public `hyper(lanes, rank, &branch)` states the sizes and the branch; lowering supplies the rest. [Lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18792), [gates](/home/nate/Desktop/recipe-dev/recipe.rs:18834).

	```rust
	let model = recipe.model().hyper(lanes, rank, &branch);
	// Hidden: RMS, projections, SiLU, sigmoid, 1/lanes, factor 2, lane read and write, collapse.
	```

	Hyper is a change to the stream, so it is declared once, in order, like `.pool()`. While the stream is widened, each chained block is one branch, and each branch binds its own gates:

	```rust
	model = model.hyper(qwen4exp.hyper.connection.count, qwen4exp.hyper.connection.low.rank);
	model = model
		.attn(…).gates(blk.a.hc.attn)	// blk.a.hc.attn.{norm, down, up, inject}.weight
		.moe(…).gates(blk.a.hc.ffn);
	model = model.norm(rms).gates(output.hc);	// collapse: output.hc.{norm, down, up}.weight
	```

	The gate formula written out, for a definition that changes it:

	```rust
	.read(norm(rms, blk.a.hc.attn.norm.weight).layer(blk.a.hc.attn.down.weight)
		.scale(1.0 / lanes).silu().layer(blk.a.hc.attn.up.weight).sigmoid())
	.write(norm(rms, blk.a.hc.attn.norm.weight).layer(blk.a.hc.attn.inject.weight)
		.scale(1.0 / lanes).sigmoid().scale(2.0))
	```

	A branch must return to the model width, because its output is added back into the lanes. Appendix A traces two branches through two lanes.

13. **Per-layer embedding: Fixed internal formula.** `ple(...)` hides the n-gram lookup, grouped RMS operations, a signed-square-root sigmoid factor, a convolution and a SiLU path. Metadata selects its table and the layers it enters. [Lookup construction](/home/nate/Desktop/recipe-dev/recipe.rs:10007), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18096).

	```rust
	let model = recipe.model().ple(&table);
	// Hidden: n-gram hash, grouped RMS, signed-root sigmoid, convolution, SiLU.
	```

	```rust
	ple(per.layer.token.embd.weight, qwen4exp.ple.ngram.size, qwen4exp.ple.heads.per.ngram)
		.key(layer(blk.a.ple.key.weight).norm(rms, blk.a.ple.norm.key.weight))
		.query(norm(rms, blk.a.ple.norm.query.weight))
		.factor(fold(lanes).signed_sqrt(1e-6).sigmoid())
		.value(layer(blk.a.ple.value.weight))
		.tail(dconv(blk.a.ple.conv1d.weight).norm(rms, blk.a.ple.norm.conv.weight).silu())
	```

14. **Normalization and residual placement: Inferred from tensor names.** On a plain residual, the builder starts each branch with RMS bound to `attn_norm`, or to `ffn_norm` (falling back to `post_attention_norm`). It adds a post-normalization only when `post_attention_norm` or `post_ffw_norm` exists. Under hyper-connections there is no pre-normalization; the RMS belongs to the gates (item 12). The kind is always RMS. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15283).

	```rust
	let automatic = file.model();
	// Hidden: tensor presence chooses pre/post RMS and residual or hyper wrapping.
	```

	```rust
	res([norm(rms, blk.a.attn.norm.weight), attn(…), norm(rms, blk.a.post.attention.norm.weight)])
	res([norm(rms, blk.a.ffn.norm.weight), ffn…, norm(rms, blk.a.post.ffw.norm.weight)])
	```

15. **Embedding and output operations: Some are inserted automatically.** The builder adds a square-root embedding scale for Gemma3 and Gemma4, ties the output to the embedding when `output.weight` is absent, inserts an optional per-layer output scale (`layer_output_scale`), and inserts final logit softcapping when `final_logit_softcapping` exists. Under hyper-connections without `output_norm`, the head collapses the lanes through `output_hc_*`. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:14779), [output](/home/nate/Desktop/recipe-dev/recipe.rs:14842).

	```rust
	let automatic = file.model();
	// Hidden: Gemma scale, optional output tie, layer scale, logit softcap, lane collapse.
	```

	```rust
	let mut model = recipe.model()
		.embed(tokenizer.ggml.tokens, gemma3.embedding.length)
		.scale(gemma3.embedding.length.sqrt());
	// blocks …
	model = model
		.norm(rms, output.norm.weight)
		.layer(output.weight.or(token.embd.weight))
		.scale(1.0 / gemma3.final.logit.softcapping).tanh()
		.scale(gemma3.final.logit.softcapping);
	```

16. **Activation and normalization order: Not an arbitrary visible chain.** A `Block` stores one activation slot and one normalization slot. Another call replaces the slot, and lowering applies them in a fixed order, so a definition cannot express every ordered sequence of maps. [Block](/home/nate/Desktop/recipe-dev/recipe.rs:11891), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:17519), [open issue #888](https://github.com/nm-z/recipe-dev/issues/888).

	```rust
	let block = layer(8).norm(l2).norm(rms);
	// The second norm replaces the first; only RMS reaches lowering.
	```

	```rust
	layer(8).norm(l2).norm(rms)
	// Each map remains in the written order.
	```

17. **Tokenizer and chat behavior: Restricted and partly hard-coded.** The tokenizer accepts a limited set of families. Gemma4 builds its prompt in code instead of rendering the file's chat template. EOS, inferred turn-stop tokens and suppressed tokens affect decoding outside the model definition. [Families](/home/nate/Desktop/recipe-dev/recipe.rs:8729), [Gemma4 prompt](/home/nate/Desktop/recipe-dev/recipe.rs:9044), [stop IDs](/home/nate/Desktop/recipe-dev/recipe.rs:15673).

	```rust
	let coder = file.tokenizer();
	let prompt = coder.chat(&messages, true); // Gemma4 uses a coded prompt form.
	```

	```rust
	tokenizer(tokenizer.ggml).chat(tokenizer.chat.template)
		.stop(tokenizer.ggml.eos.token.id)
		.suppress(tokenizer.ggml.suppress.tokens)
	```

18. **GGUF file coverage: Restricted before model execution.** Recipe accepts GGUF version 3 and a finite list of tensor types. Its automatic builder rejects tensors it did not assign to a node, and its attention dimensions reject differing key and value widths. [Parser](/home/nate/Desktop/recipe-dev/recipe.rs:8453), [types](/home/nate/Desktop/recipe-dev/recipe.rs:8289), [builder checks](/home/nate/Desktop/recipe-dev/recipe.rs:14848).

	```rust
	let data = recipe.data(path);
	// Opening can reject GGUF version, tensor kind, or dimensions before inference.
	```

	```rust
	let data = recipe.data(path);
	let support = data.report(gguf.support);
	```

	Observed with `recipe keys` over a model directory:
	- GGUF version 2 files (`ggml-vocab-aquila.gguf`) are rejected.
	- Shards after the first (`…-00002-of-00004.gguf`) are rejected with `is shard 1, expected shard 0`. Those shards hold no metadata, but they hold tensors a definition binds.

## Appendix A: two branches through two lanes

Made-up values; the gates are illustrative.

```text
a = [1, 2, 3, 4]
	lane 0: [1, 2, 3, 4]
	lane 1: [1, 2, 3, 4]

branch A: layer(2).layer(4)
	read   = blend(lane 0, lane 1) with A's read gate    -> [1, 1, 1.5, 2]
	j      = layer(2) then layer(4)                        -> [2, 0, 1, 3]
	write  (A's write gate: lane 0 ×1.0, lane 1 ×0.6)
		lane 0 = [1, 2, 3, 4] + 1.0·[2, 0, 1, 3] = [3,   2, 4,   7  ]
		lane 1 = [1, 2, 3, 4] + 0.6·[2, 0, 1, 3] = [2.2, 2, 3.6, 5.8]

branch B: layer(3).relu().layer(4)
	gates  computed from the lanes above, so A's output shapes them
	read   = blend(lane 0, lane 1) with B's read gate    -> all gates 1: [2.6, 2, 3.8, 6.4]
	m      = layer(3).relu() then layer(4)
	write  lane 0 += w0·m,  lane 1 += w1·m
```

A branch reaches the next one in two ways: through what it added to the lanes, and through the next branch's gates, which are computed from those lanes.

## Appendix B: the delta rule on one channel

One head, one channel, no decay, β = 0.5, over the sequence `1, 2, 3, 4`. The memory corrects toward each new value, and each output reads it.

```text
token   x   memory s                       output
1       1   0     + 0.5·(1 − 0)     = 0.5       0.5
2       2   0.5   + 0.5·(2 − 0.5)   = 1.25      1.25
3       3   1.25  + 0.5·(3 − 1.25)  = 2.125     2.125
4       4   2.125 + 0.5·(4 − 2.125) = 3.0625    3.0625
```

Token 2 sees token 1 through the memory, and never token 3.
