1. `.gate()` was not typing attn keys, was not typing which blks in the gguf, was not typing where norm happens and was not typing the Wq splitting. It was also not typing the "gate" and the "factor". This led to a separate internal GGUF model builder to have to read and check if the Qout contains extra values if so, then it split and specifically routed to q and the factor mult. [Loader](/home/nate/Desktop/recipe-dev/recipe.rs:15022), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18248), [kernel](/home/nate/Desktop/recipe-dev/amd-nv-cpu.ll:5532).

```rust
let block = attn(heads).gate();
```

```rust
for a in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47] {
	let (c, d) = layer(blk[a].attn_q).split(2);

	let e = attn(24).kv(2)
		.q(c.norm(blk[a].attn_q_norm))
		.k(layer(blk[a].attn_k).norm(blk[a].attn_k_norm))
		.v(layer(blk[a].attn_v));

	(e * d.sigmoid()).layer(blk[a].attn_output);
}
```

2. **all key paths are hardcoded** The GGUF has a key named `qwen4exp.block_count`, but a model script can't even read `qwen4exp.block_count` because Recipe hardcodes all of its architecture keys, except `qwen4exp.block_count`. [Namespaces](/home/nate/Desktop/recipe-dev/recipe.rs:15409).

```rust
// have to go hunt the shit down: recipe keys qwen.gguf | rg temp
// general.sampling.temp.1.0
recipe.sampler().temperature(1.0)
```
Proposed:
```rust
// resolves directly from the gguf
recipe.sampler().temperature(general.sampling.temp)
```

3. [Inference](/home/nate/Desktop/recipe-dev/recipe.rs:15549), [binding](/home/nate/Desktop/recipe-dev/recipe.rs:14520).

```rust
let data = recipe.data(path);
recipe.infer().run(&model, &data); // conventional_plan chooses tensor names.
```

```rust
for a in [0, 3, 5] {
	attn(heads)
		.q(blk.a.attn.q)
		.k(blk.a.attn.k)
		.v(blk.a.attn.v);
}
```

4. both get it [Binder](/home/nate/Desktop/recipe-dev/recipe.rs:15800).

```rust
for a in 0..48 {
	let experts = (layer(blk.a.ffn.gate.exps.weight).silu()
		* layer(blk.a.ffn.up.exps.weight)).layer(blk.a.ffn.down.exps.weight);
	let shared = (layer(blk.a.ffn.gate.shexp.weight).silu()
		* layer(blk.a.ffn.up.shexp.weight)).layer(blk.a.ffn.down.shexp.weight);
	model = model.res([
		norm(rms),
		moe(qwen4exp.expert.used.count, experts)
			.route(softmax, layer(blk.a.ffn.gate.inp.weight))
			.renorm()
			.shared(shared, layer(blk.a.ffn.gate.inp.shexp.weight).sigmoid()),
	]);
}
```


5. GGUF builder emits glu(hidden, SiLU) for ordinary feed-forward blocks and binds ffn_gate, ffn_up and ffn_down by fixed name. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15187), [RNJ model](/home/nate/Desktop/recipe-dev/rnj-1.rs:39).

```rust
let automatic = file.model(); // Builder inserts GLU with SiLU.
let handwritten = layer(hidden).gelu() * layer(hidden); // RNJ uses GELU.
```

```rust
let mut model = recipe.model()
	.embed(tokenizer.ggml.tokens, gemma3.embedding.length)
	.scale(gemma3.embedding.length.sqrt());

for a in 0..gemma3.block.count {
	model = model.res([
		norm(rms),
		(layer(blk.a.ffn.gate.weight).gelu()
			* layer(blk.a.ffn.up.weight))
			.layer(blk.a.ffn.down.weight),
		norm(rms),
	]);
}
```

6. **Attention details: Chosen from tensor shape or presence.** The builder decides whether there is an attention gate from query-tensor width, uses the key tensor as values if `attn_v.weight` is absent, adds Q/K RMS normalization when norm tensors exist, and adds indexer and rotary-factor paths when their metadata or tensors exist. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15016).

```rust
let automatic = file.model();
// Hidden: query shape and optional tensors select a factor projection, Q/K norm, indexer, and factors.
```

```rust
for a in (3..48).step_by(4) {
	model = model.res([
		norm(rms),
		attn(qwen4exp.attention.head.count)
			.kv(qwen4exp.attention.head.count.kv)
			.head(qwen4exp.attention.key.length)
			.q(blk.a.attn.q.weight).gate(sigmoid)
			.k(blk.a.attn.k.weight)
			.v(blk.a.attn.v.weight)
			.qk(rms, blk.a.attn.q.norm.weight, blk.a.attn.k.norm.weight)
			.rope(pairs, qwen4exp.rope.dimension.count, qwen4exp.rope.freq.base)
			.index(qwen4exp.attention.indexer.head.count, qwen4exp.attention.indexer.key.length,
				qwen4exp.attention.compress.ratios.a, qwen4exp.attention.indexer.top.k)
				.q(blk.a.indexer.q.proj.weight).k(blk.a.indexer.k.proj.weight)
				.score(rms, blk.a.indexer.q.norm.weight, blk.a.indexer.k.norm.weight)
			.out(blk.a.attn.output.weight),
	]);
}
```

7. **Rotary pairing and YaRN: Not fully declared by the automatic model.** The architecture table chooses neighboring versus half-channel rotary pairing. The automatic attention builder calls `.rope(...)` but does not call `.yarn(...)`; the handwritten RNJ model does call `.yarn(...)` with a model-specific constant. [Pairing table](/home/nate/Desktop/recipe-dev/recipe.rs:14635), [builder](/home/nate/Desktop/recipe-dev/recipe.rs:15054), [RNJ model](/home/nate/Desktop/recipe-dev/rnj-1.rs:27).

```rust
let automatic = file.model(); // Builder calls rope but not yarn.
let explicit = attn(heads).rope(neox, dims, base).yarn(factor, context, fast, slow);
```


```rust
// qwen4exp: half-channel pairs, 64 of 256 channels rotate, no YaRN in the file.
attn(qwen4exp.attention.head.count)
	.rope(neox, qwen4exp.rope.dimension.count, qwen4exp.rope.freq.base)

// gemma3 (RNJ): the file carries YaRN, so it is written, with every value from the file.
attn(gemma3.attention.head.count)
	.rope(neox, gemma3.attention.key.length, gemma3.rope.freq.base)
	.yarn(gemma3.rope.scaling.factor, gemma3.rope.scaling.original.context.length,
		gemma3.rope.scaling.yarn.beta.fast, gemma3.rope.scaling.yarn.beta.slow)
```


8. **Per-layer block type: Inferred.** Metadata and a fixed interval rule choose attention, short convolution, or delta for each layer. The automatic builder also chooses residual versus hyper-connection wrapping. [Model loop](/home/nate/Desktop/recipe-dev/recipe.rs:14793), [dimensions](/home/nate/Desktop/recipe-dev/recipe.rs:14889).

```rust
let automatic = file.model();
// Hidden: per-layer metadata chooses attention, short convolution, or delta.
```

**Proposed user-facing spelling (API sketch):**

```rust
// qwen4exp: delta everywhere except every 4th layer, MoE feed-forward, hyper-connections.
for a in 0..48 {
	let mixer = if a % 4 == 3 {
		attn(qwen4exp.attention.head.count)…	// item 6
	} else {
		delta(qwen4exp.ssm.group.count, qwen4exp.ssm.conv.kernel)…	// item 9
	};
	model = model
		.hyper(qwen4exp.hyper.connection.count, qwen4exp.hyper.connection.low.rank, mixer)
		.hyper(qwen4exp.hyper.connection.count, qwen4exp.hyper.connection.low.rank,
			moe(qwen4exp.expert.used.count, experts)…);	// item 4
}

// gemma3 (RNJ): attention on every layer, plain residual with pre- and post-norms.
for a in 0..26 {
	model = model
		.res([norm(rms), attn(gemma3.attention.head.count)…, norm(rms)])
		.res([norm(rms), (layer(blk.a.ffn.gate.weight).gelu() * layer(blk.a.ffn.up.weight)).layer(blk.a.ffn.down.weight), norm(rms)]);
}
```




```rust
model = model.hyper(qwen4exp.hyper.connection.count, qwen4exp.hyper.connection.low.rank);
for g in 0..12 {
	for i in 0..3 {
		let a = 4 * g + i;
		model = model
			.delta(qwen4exp.ssm.group.count, qwen4exp.ssm.conv.kernel)…	// item 9
			.moe(qwen4exp.expert.used.count, experts)…;	// item 4
	}
	let a = 4 * g + 3;
	model = model
		.attn(qwen4exp.attention.head.count)…	// item 6
		.moe(qwen4exp.expert.used.count, experts)…;	// item 4
}
model = model.norm(rms)…;	// head: the lanes collapse
```



```rust
a = [1, 2, 3, 4]
	lane 0: [1, 2, 3, 4]
	lane 1: [1, 2, 3, 4]

branch A
	read   = blend(lane 0, lane 1) -> [1, 1, 1.5, 2]
	i      = layer(2)(read)                              -> [.., ..]
	j      = layer(4)(i)                                 -> [2, 0, 1, 3]
	write  (A's write gate: lane 0 ×1.0, lane 1 ×0.6)
		lane 0 = [1, 2, 3, 4] + 1.0·[2, 0, 1, 3] = [3,   2, 4,   7  ]
		lane 1 = [1, 2, 3, 4] + 0.6·[2, 0, 1, 3] = [2.2, 2, 3.6, 5.8]

branch B	(its gates are computed from the lanes above, so A's output affects them)
	read   = blend(lane 0, lane 1) with B's read gate   -> e.g. all gates 1: mean = [2.6, 2, 3.8, 6.4]
	m      = layer(3).relu().layer(4)(read)              -> [..4 numbers..]
	write  lane 0 += w0·m,  lane 1 += w1·m	(B's own write weights)
```
















9. **Delta block math: Partly private.** The lowering fixes its projections, convolution, Q/K L2 normalization, value RMS normalization, decay conversion, and output multiplication. Convolution and output activations come from an architecture-name row or private defaults; the public delta selectors do not expose that full choice. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15139), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18211).

```rust
let block = recipe.model().delta(heads, kernel);
// Hidden: L2, RMS, decay transform, and private activation choices.
```

**Proposed user-facing spelling (API sketch):**

```rust
delta(heads, kernel).conv(silu).qk(l2).values(rms)
	.decay(blk[layer].ssm.a).output(sigmoid)
```


10. **Short convolution: Built as an internal product.** The builder slices one stored projection into B, C, and X, then constructs `C × depthwise_conv(B × X)` and an output projection. That expression is assembled inside the GGUF builder rather than supplied as the user’s model definition. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15106).

```rust
let automatic = file.model();
// Hidden: shortconv slices B, C, X and builds C * dconv(B * X).
```

**Proposed user-facing spelling (API sketch):**

```rust
((layer(blk[layer].shortconv.in_proj.weight.b)
	* layer(blk[layer].shortconv.in_proj.weight.x)).dconv(blk[layer].shortconv.conv.weight)
	* layer(blk[layer].shortconv.in_proj.weight.c))
	.layer(blk[layer].shortconv.out_proj.weight)
```


11. **GGUF MoE: Uses a private model operation.** Expert scoring comes from metadata, renormalization defaults to true, and a shared expert is selected by tensor presence. The GGUF builder calls private `gguf_moe`; public `.moe(...)` lowers through a different route. [Metadata](/home/nate/Desktop/recipe-dev/recipe.rs:14923), [builder](/home/nate/Desktop/recipe-dev/recipe.rs:15195), [public and private forms](/home/nate/Desktop/recipe-dev/recipe.rs:12275).

```rust
let public = recipe.model().moe(top_k, experts);
let automatic = file.model(); // GGUF uses a different, private MoE operation.
```

**Proposed user-facing spelling (API sketch):**

```rust
moe(experts)
	.route(layer(blk[layer].ffn.gate_inp.weight).softmax().topk(used).renorm())
	.shared(layer(blk[layer].ffn.gate_inp_shexp.weight).sigmoid() * shared_expert)
```


12. **Hyper-connections: Fixed internal formula.** The public `hyper(lanes, rank, branch)` names the branch and sizes, while lowering supplies RMS, SiLU, sigmoid read and write factors, a factor of two, and lane reduction. The model definition cannot spell or change that whole sequence. [Lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18817).

```rust
let model = recipe.model().hyper(lanes, rank, &branch);
// Hidden: RMS, projections, SiLU, sigmoid, lane read/write, and factor 2.
```

**Proposed user-facing spelling (API sketch):**

```rust
hyper(lanes, branch)
	.read(norm(rms).layer(blk[layer].hc.attn.down.weight).scale(1.0 / lanes).silu()
		.layer(blk[layer].hc.attn.up.weight).sigmoid())
	.write(norm(rms).layer(blk[layer].hc.attn.inject.weight).scale(1.0 / lanes).sigmoid().scale(2.0))
```


13. **Per-layer embedding: Fixed internal formula.** `ple(...)` hides the n-gram lookup, grouped RMS operations, signed-square-root sigmoid factor, convolution, and SiLU path. Metadata selects its table and placement. [Lookup construction](/home/nate/Desktop/recipe-dev/recipe.rs:10007), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18096).

```rust
let model = recipe.model().ple(&table);
// Hidden: n-gram hash, grouped RMS, signed-root sigmoid, convolution, SiLU.
```

**Proposed user-facing spelling (API sketch):**

```rust
ple(per_layer_token_embd.weight)
	.key(layer(blk[layer].ple.key.weight).norm(rms))
	.factor(fold(lanes).signed_sqrt(1e-6).sigmoid())
	.value(layer(blk[layer].ple.value.weight).norm(rms))
	.tail(dconv(blk[layer].ple.conv1d.weight).silu())
```


14. **Normalization and residual placement: Inferred from tensor names.** The automatic builder chooses RMS pre-normalization and optional post-normalization from which weight tensors exist, then wraps the branch in a residual or hyper operation. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15258).

```rust
let automatic = file.model();
// Hidden: tensor presence chooses pre/post RMS and residual or hyper wrapping.
```

**Proposed user-facing spelling (API sketch):**

```rust
res([norm(rms), attn(heads), norm(rms)])
res([norm(rms), ffn(hidden), norm(rms)])
```


15. **Embedding and output operations: Some are inserted automatically.** The builder adds a square-root embedding scale for Gemma3/4, ties output weights to embedding when `output.weight` is absent, and inserts optional layer-output scaling and final logit softcapping. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:14779), [output](/home/nate/Desktop/recipe-dev/recipe.rs:14817).

```rust
let automatic = file.model();
// Hidden: Gemma scale, optional output tie, layer scale, and logit softcap.
```

**Proposed user-facing spelling (API sketch):**

```rust
let input = embed(tokenizer.ggml.tokens, gguf.arch.embedding_length)
	.scale(gguf.arch.embedding_length.sqrt());
let output = norm(rms).layer(tokenizer.ggml.tokens)
	.weights(output.weight.or(token_embd.weight))
	.scale(1.0 / gguf.arch.final_logit_softcapping).tanh()
	.scale(gguf.arch.final_logit_softcapping);
```


16. **Activation and normalization order: Not an arbitrary visible chain.** A `Block` stores one activation and one normalization slot; another call replaces a slot, and lowering applies them in a fixed order. That prevents the model definition from expressing every ordered sequence of maps. [Block](/home/nate/Desktop/recipe-dev/recipe.rs:11891), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:17519), [open issue #888](https://github.com/nm-z/recipe-dev/issues/888).

```rust
let block = layer(8).norm(l2).norm(rms);
// The second norm replaces the first; only RMS reaches lowering.
```

**Proposed user-facing spelling (API sketch):**

```rust
layer(8).norm(l2).norm(rms)
// Each map remains in the written order.
```


17. **Tokenizer and chat behavior: Restricted and partly hard-coded.** The tokenizer accepts a limited set of families; Gemma4 constructs its prompt in code instead of rendering the file’s chat template. EOS, inferred turn-stop tokens, and suppressed tokens also affect decoding outside the model definition. [Families](/home/nate/Desktop/recipe-dev/recipe.rs:8729), [Gemma4 prompt](/home/nate/Desktop/recipe-dev/recipe.rs:9044), [stop IDs](/home/nate/Desktop/recipe-dev/recipe.rs:15673).

```rust
let coder = file.tokenizer();
let prompt = coder.chat(&messages, true); // Gemma4 uses a coded prompt form.
```

**Proposed user-facing spelling (API sketch):**

```rust
tokenizer(gguf.tokenizer.ggml).chat(gguf.tokenizer.chat_template)
	.stop(gguf.tokenizer.ggml.eos_token_id)
	.suppress(gguf.tokenizer.ggml.suppress_tokens)
```


18. **GGUF file coverage: Restricted before model execution.** Recipe accepts GGUF version 3 and a finite list of tensor types. Its automatic builder rejects tensors it did not assign to a node, and its attention dimensions reject differing key and value widths. [Parser](/home/nate/Desktop/recipe-dev/recipe.rs:8453), [types](/home/nate/Desktop/recipe-dev/recipe.rs:8289), [builder checks](/home/nate/Desktop/recipe-dev/recipe.rs:14848).

```rust
let data = recipe.data(path);
// Opening can reject GGUF version, tensor kind, or dimensions before inference.
```

**Proposed user-facing spelling (API sketch):**

```rust
let data = recipe.data(path);
let support = data.report(gguf.support);
```
