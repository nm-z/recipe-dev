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

2. **All GGUFs with `.gate()`: Also no.** The automatic builder accepts 11 `general.architecture` names and rejects names outside its table. GGUF defines architecture as file metadata and documents other architecture names. [Recipe table](/home/nate/Desktop/recipe-dev/recipe.rs:14635), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

```rust
file.model(); // Hidden: general.architecture
```

```rust
recipe.model(gguf.architecture)
	.blocks(model_definition)
```


3. **Arbitrary metadata paths in a model script: No.** Scripts can use fixed names such as `tokenizer.ggml.tokens` and `gemma3.rope.freq_base`. `Gguf::value(key)` exists, but `recipe.data(...)` keeps its `Gguf` private and does not expose that lookup to the script. The named namespaces cover only eight architecture names. [Data](/home/nate/Desktop/recipe-dev/recipe.rs:11448), [namespaces](/home/nate/Desktop/recipe-dev/recipe.rs:15409).

```rust
let data = recipe.data(path);
let width = gemma3.embedding_length; // No data.value("some.other.key").
```

**Proposed user-facing spelling (API sketch):**

```rust
let width = gguf.arch.embedding_length;
let tokens = tokenizer.ggml.tokens;
```


4. **Explicit GGUF tensor binding from the usual `recipe.infer().run(&model, &data)` path: No.** That path calls `conventional_plan`, which assigns weights using fixed tensor names and recognized model shapes. A public `Binding::named` exists, but the usual `Data` path does not give the script the `Gguf` value it requires. [Inference](/home/nate/Desktop/recipe-dev/recipe.rs:15549), [binding](/home/nate/Desktop/recipe-dev/recipe.rs:14520).

```rust
let data = recipe.data(path);
recipe.infer().run(&model, &data); // conventional_plan chooses tensor names.
```

**Proposed user-facing spelling (API sketch):**

```rust
attn(heads).q(blk[layer].attn.q.weight)
	.k(blk[layer].attn.k.weight)
	.v(blk[layer].attn.v.weight)
```


5. **Which product branch gets `ffn_gate.weight`: Inferred.** The conventional binder checks which branch has an activation; it assigns `ffn_gate.weight` to that branch and `ffn_up.weight` to the other. If that test does not distinguish them, it uses branch order. The model expression does not name either tensor. [Binder](/home/nate/Desktop/recipe-dev/recipe.rs:15800).

```rust
let product = layer(8).gelu() * layer(8);
// Hidden: the activated branch binds ffn_gate.weight; the other binds ffn_up.weight.
```

**Proposed user-facing spelling (API sketch):**

```rust
layer(hidden).weights(blk[layer].ffn.gate.weight).gelu()
	* layer(hidden).weights(blk[layer].ffn.up.weight)
```


6. **Automatic feed-forward activation: Hard-coded.** The GGUF builder emits `glu(hidden, SiLU)` for ordinary feed-forward blocks. The repository’s handwritten Gemma3 RNJ model instead spells `layer(...).gelu() * layer(...)`. Those definitions disagree in source; this audit did not measure the numerical difference. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15187), [RNJ model](/home/nate/Desktop/recipe-dev/rnj-1.rs:39).

```rust
let automatic = file.model(); // Builder inserts GLU with SiLU.
let handwritten = layer(hidden).gelu() * layer(hidden); // RNJ uses GELU.
```

**Proposed user-facing spelling (API sketch):**

```rust
(layer(hidden).weights(blk[layer].ffn.gate.weight).gelu()
	* layer(hidden).weights(blk[layer].ffn.up.weight))
	.layer(blk[layer].ffn.down.weight)
```


7. **Attention details: Chosen from tensor shape or presence.** The builder decides whether there is an attention gate from query-tensor width, uses the key tensor as values if `attn_v.weight` is absent, adds Q/K RMS normalization when norm tensors exist, and adds indexer and rotary-factor paths when their metadata or tensors exist. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15016).

```rust
let automatic = file.model();
// Hidden: query shape and optional tensors select a factor projection, Q/K norm, indexer, and factors.
```

**Proposed user-facing spelling (API sketch):**

```rust
attn(heads).q(blk[layer].attn.q.weight).k(blk[layer].attn.k.weight).v(blk[layer].attn.v.weight)
	.qk(rms).rope(pairs, dims, base)
	.index(gguf.arch.attention.indexer).factors(rope_freqs.weight)
```


8. **Rotary pairing and YaRN: Not fully declared by the automatic model.** The architecture table chooses neighboring versus half-channel rotary pairing. The automatic attention builder calls `.rope(...)` but does not call `.yarn(...)`; the handwritten RNJ model does call `.yarn(...)` with a model-specific constant. [Pairing table](/home/nate/Desktop/recipe-dev/recipe.rs:14635), [builder](/home/nate/Desktop/recipe-dev/recipe.rs:15054), [RNJ model](/home/nate/Desktop/recipe-dev/rnj-1.rs:27).

```rust
let automatic = file.model(); // Builder calls rope but not yarn.
let explicit = attn(heads).rope(neox, dims, base).yarn(factor, context, fast, slow);
```

**Proposed user-facing spelling (API sketch):**

```rust
attn(heads).rope(pairs, gguf.arch.rope.dimension_count, gguf.arch.rope.freq_base)
	.yarn(gguf.arch.rope.scaling)
```


9. **Per-layer block type: Inferred.** Metadata and a fixed interval rule choose attention, short convolution, or delta for each layer. The automatic builder also chooses residual versus hyper-connection wrapping. [Model loop](/home/nate/Desktop/recipe-dev/recipe.rs:14793), [dimensions](/home/nate/Desktop/recipe-dev/recipe.rs:14889).

```rust
let automatic = file.model();
// Hidden: per-layer metadata chooses attention, short convolution, or delta.
```

**Proposed user-facing spelling (API sketch):**

```rust
res([norm(rms), attn(heads)])
res([norm(rms), shortconv(kernel)])
res([norm(rms), delta(delta_heads, kernel)])
```


10. **Delta block math: Partly private.** The lowering fixes its projections, convolution, Q/K L2 normalization, value RMS normalization, decay conversion, and output multiplication. Convolution and output activations come from an architecture-name row or private defaults; the public delta selectors do not expose that full choice. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15139), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18211).

```rust
let block = recipe.model().delta(heads, kernel);
// Hidden: L2, RMS, decay transform, and private activation choices.
```

**Proposed user-facing spelling (API sketch):**

```rust
delta(heads, kernel).conv(silu).qk(l2).values(rms)
	.decay(blk[layer].ssm.a).output(sigmoid)
```


11. **Short convolution: Built as an internal product.** The builder slices one stored projection into B, C, and X, then constructs `C × depthwise_conv(B × X)` and an output projection. That expression is assembled inside the GGUF builder rather than supplied as the user’s model definition. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15106).

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


12. **GGUF MoE: Uses a private model operation.** Expert scoring comes from metadata, renormalization defaults to true, and a shared expert is selected by tensor presence. The GGUF builder calls private `gguf_moe`; public `.moe(...)` lowers through a different route. [Metadata](/home/nate/Desktop/recipe-dev/recipe.rs:14923), [builder](/home/nate/Desktop/recipe-dev/recipe.rs:15195), [public and private forms](/home/nate/Desktop/recipe-dev/recipe.rs:12275).

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


13. **Hyper-connections: Fixed internal formula.** The public `hyper(lanes, rank, branch)` names the branch and sizes, while lowering supplies RMS, SiLU, sigmoid read and write factors, a factor of two, and lane reduction. The model definition cannot spell or change that whole sequence. [Lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18817).

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


14. **Per-layer embedding: Fixed internal formula.** `ple(...)` hides the n-gram lookup, grouped RMS operations, signed-square-root sigmoid factor, convolution, and SiLU path. Metadata selects its table and placement. [Lookup construction](/home/nate/Desktop/recipe-dev/recipe.rs:10007), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18096).

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


15. **Normalization and residual placement: Inferred from tensor names.** The automatic builder chooses RMS pre-normalization and optional post-normalization from which weight tensors exist, then wraps the branch in a residual or hyper operation. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:15258).

```rust
let automatic = file.model();
// Hidden: tensor presence chooses pre/post RMS and residual or hyper wrapping.
```

**Proposed user-facing spelling (API sketch):**

```rust
res([norm(rms), attn(heads), norm(rms)])
res([norm(rms), ffn(hidden), norm(rms)])
```


16. **Embedding and output operations: Some are inserted automatically.** The builder adds a square-root embedding scale for Gemma3/4, ties output weights to embedding when `output.weight` is absent, and inserts optional layer-output scaling and final logit softcapping. [Builder](/home/nate/Desktop/recipe-dev/recipe.rs:14779), [output](/home/nate/Desktop/recipe-dev/recipe.rs:14817).

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


17. **Activation and normalization order: Not an arbitrary visible chain.** A `Block` stores one activation and one normalization slot; another call replaces a slot, and lowering applies them in a fixed order. That prevents the model definition from expressing every ordered sequence of maps. [Block](/home/nate/Desktop/recipe-dev/recipe.rs:11891), [lowering](/home/nate/Desktop/recipe-dev/recipe.rs:17519), [open issue #888](https://github.com/nm-z/recipe-dev/issues/888).

```rust
let block = layer(8).norm(l2).norm(rms);
// The second norm replaces the first; only RMS reaches lowering.
```

**Proposed user-facing spelling (API sketch):**

```rust
layer(8).norm(l2).norm(rms)
// Each map remains in the written order.
```


18. **Tokenizer and chat behavior: Restricted and partly hard-coded.** The tokenizer accepts a limited set of families; Gemma4 constructs its prompt in code instead of rendering the file’s chat template. EOS, inferred turn-stop tokens, and suppressed tokens also affect decoding outside the model definition. [Families](/home/nate/Desktop/recipe-dev/recipe.rs:8729), [Gemma4 prompt](/home/nate/Desktop/recipe-dev/recipe.rs:9044), [stop IDs](/home/nate/Desktop/recipe-dev/recipe.rs:15673).

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


19. **GGUF file coverage: Restricted before model execution.** Recipe accepts GGUF version 3 and a finite list of tensor types. Its automatic builder rejects tensors it did not assign to a node, and its attention dimensions reject differing key and value widths. [Parser](/home/nate/Desktop/recipe-dev/recipe.rs:8453), [types](/home/nate/Desktop/recipe-dev/recipe.rs:8289), [builder checks](/home/nate/Desktop/recipe-dev/recipe.rs:14848).

```rust
let data = recipe.data(path);
// Opening can reject GGUF version, tensor kind, or dimensions before inference.
```

**Proposed user-facing spelling (API sketch):**

```rust
let data = recipe.data(path);
let support = data.report(gguf.support);
```


20. **This is a source audit, not a claim that every listed path has been run.** The direct answer to your design requirement is **no**: today, Recipe cannot represent 100% of every GGUF model’s operations and weight assignments in the user-visible model definition. [Open issue #945](https://github.com/nm-z/recipe-dev/issues/945) tracks the broader hidden-math problem; [#944](https://github.com/nm-z/recipe-dev/issues/944) tracks GGUF contents and named-key access.

```rust
let data = recipe.data(one_path);
recipe.infer().run(&one_model, &data); // One run does not prove all GGUFs.
```

**Proposed user-facing spelling (API sketch):**

```rust
recipe.verify(gguf_corpus).against(reference)
```


## Illustration: what the API says and what runs

The model script can show this attention fragment:

```rust
attn(heads).gate()
```

The `.gate()` call does not show the extra projection or the order of multiplication and output projection:

```text
one input x
├─ Q, K, V projection → attention → context
└─ extra rows in attn_q.weight → factor_logits → sigmoid → factor
context × factor → Wo → output
```

The automatic GGUF builder makes a further choice from the file, rather than from that written fragment:

```text
blk.{layer}.attn_q.weight has heads × head_width output rows → build ordinary attention
blk.{layer}.attn_q.weight has 2 × heads × head_width output rows → build attention with .gate() and split out factor rows
```

This is why two GGUF files can generate different operation graphs from the same builder entrypoint. The file supplies metadata, tensor names, shapes, and values; `Builder::build` supplies architecture rules and operations. In the handwritten `recipe.infer().run(&model, &data)` path, Recipe does not silently add `.gate()` to a model that omitted it. The conventional binder checks the gate declaration against tensor shape and rejects a mismatch.

Another public compound call hides a longer runtime graph:

```rust
model.hyper(lanes, rank, &branch)
```

When `rank` is zero, lowering uses fixed factors of one. With a positive `rank`, lowering inserts RMS normalization, learned projections, SiLU, sigmoid, a factor of two on the write path, lane read and write operations, and the branch. The call exposes the dimensions and branch, but does not spell those operations or their order. [Hyper lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18817).

The general product `layer(8).gelu() * layer(8)` is different: its two branches and elementwise multiplication are explicit in the model definition. The hidden part in GGUF inference is which named file tensor fills each layer. [Product lowering](/home/nate/Desktop/recipe-dev/recipe.rs:18793), [GGUF binding](/home/nate/Desktop/recipe-dev/recipe.rs:15800).
