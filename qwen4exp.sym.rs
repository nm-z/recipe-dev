use recipe::*;
use recipe::infer::{cached, input, mtp, out, pp, tg, time};

const GGUF: &str = "/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf";

// Qwen3.8 Flash-Next: 48 blocks over a 4-lane hyper-connection stream. Every
// block mixes tokens (full attention every fourth block, a gated delta rule
// otherwise), then mixes channels with 512 routed experts and one shared expert.
#[rustfmt::skip]
fn main() {
	let data = recipe.data(std::env::var("GGUF").unwrap_or_else(|_| GGUF.to_owned()));
	let width = qwen4exp.embedding_length;
	let (lanes, rank) = (qwen4exp.hyper_connection.count, qwen4exp.hyper_connection.low_rank);
	let channels = lanes * width;
	let mean = 1.0 / lanes as f64;
	// The mixer gates. The stream normalizes per lane under one scale over all of it;
	// the read gate passes it down to `rank` and back up to a value per stream channel;
	// the write gate is a value per lane, twice a sigmoid, so a zero injection is the
	// plain residual. Every hyper-connection and the head read share the read gate.
	let stream_norm = || norm(rms).per(width);
	let read = || [layer(rank).scale(mean).silu(), layer(channels).sigmoid()];
	let write = || [layer(lanes), linear().scale(mean).sigmoid().scale(2.0)];

	// Hashed n-gram embeddings (Engram-style) add into the stream before the blocks
	// that read them. Each token hashes with its predecessors, cut at the end id,
	// into one row per head of every order from two to ngram_size, from ranges of
	// the host table that the file names; the rows never reach the device whole.
	let rows = ngrams(qwen4exp.ple.ngram_size)
		.heads(qwen4exp.ple.heads_per_ngram)
		.multipliers(&qwen4exp.ple.layer_multipliers)
		.ranges(&qwen4exp.ple.head_offsets, &qwen4exp.ple.head_vocab_sizes)
		.ends(&[qwen4exp.ple.eos_token_id])
		.absent(qwen4exp.ple.eos_token_id)
		.image(qwen4exp.ple.image_token_id)
		.table("per_layer_token_embd.weight");
	// One scalar per lane gates the rows' value: the key the rows project to
	// and the stream, each RMS-normalized per lane, agree by their dot product over
	// the root of the lane width, squashed as sigmoid(sign(s) * sqrt(|s|)).
	let gate = [
		layer(channels).norm(rms).per(width) * stream().norm(rms).per(width),
		fold(width),
		scalar(|s| {
			let s = s * (1.0 / (width as f64).sqrt());
			(s.sign() * s.abs().max(1e-6).sqrt()).sigmoid()
		}),
	];
	// The gated value joins the stream with its own normalized, silu-activated
	// causal convolution, whose taps sit ngram_size positions apart.
	let ple = [
		rows,
		outer(gate, layer(width)),
		identity() + [norm(rms).per(width), dconv(qwen4exp.ple.conv_kernel).dilation(qwen4exp.ple.ngram_size).silu()],
	];

	let mut model = recipe.model()
		.e(qwen4exp.attention.layer_norm_rms_epsilon)
		.embed(tokenizer.ggml.tokens, width);

	for block in 0..qwen4exp.block_count {
		if qwen4exp.ple.layers.contains(&block) { model = model.inject(ple.clone()); }

		// Token mixing.
		model = model.hyper_gated(lanes, stream_norm(), read(), write(), &if (block + 1) % qwen4exp.full_attention_interval == 0 {
			recipe.model()
				.attn(qwen4exp.attention.head_count)
				.kv(qwen4exp.attention.head_count_kv)
				.head(qwen4exp.attention.key_length)
				.gate()
				.qk(rms)
				// Interleaved M-RoPE: the 32 rotary pairs alternate over time, height and
				// width positions by the file's sections. Text gives every axis the token
				// position.
				.rope(interleaved(&qwen4exp.rope.dimension_sections), qwen4exp.rope.dimension_count, qwen4exp.rope.freq_base)
				.index_tokens(
					qwen4exp.attention.indexer.head_count,
					qwen4exp.attention.indexer.key_length,
					qwen4exp.attention.compress_ratios[block].max(1),
					qwen4exp.attention.indexer.top_k
				)
				.score(rms, qwen4exp.rope.dimension_count)
		} else {
			recipe.model()
				.delta(qwen4exp.ssm.time_step_rank, qwen4exp.ssm.conv_kernel)
				.keys(qwen4exp.ssm.group_count, qwen4exp.ssm.state_size)
				.values(qwen4exp.ssm.state_size)
				.out(width)
				// Convolution, then SiLU; unit-length queries and keys per key head; decay
				// exp(-exp(A) * softplus(alpha + dt_bias)) with sigmoid(beta) as the write
				// strength and queries scaled by 1/sqrt(key width); the recurrence output is
				// RMS-normalized per value head, then multiplied by a sigmoid output gate.
				.conv_activation(Activation::Silu)
				.qk(l2)
				.decay(DeltaDecay::Softplus)
				.write(Activation::Sigmoid)
				.query_scale(DeltaScale::InverseRoot)
				.out_norm(rms)
				.output_gate(Activation::Sigmoid)
		});

		// Channel mixing: each position runs its top experts of 512, plus the
		// shared expert behind a per-position sigmoid gate.
		// Router weights are a softmax over all experts, renormalized over the kept ones
		// (llama.cpp qwen4exp); the file holds no gating keys.
		let mut experts = Vec::new();
		for _ in 0..qwen4exp.expert_count {
			experts.push(Block::from([
				layer(qwen4exp.expert_feed_forward_length).silu() * layer(qwen4exp.expert_feed_forward_length),
				layer(width),
			]));
		}
		model = model.hyper_gated(lanes, stream_norm(), read(), write(), &Model::from(
			moe(qwen4exp.expert_used_count, experts).scores(Scoring::Softmax).renormalize(true)
			+ layer(1).sigmoid() * Block::from([
				layer(qwen4exp.expert_shared_feed_forward_length).silu() * layer(qwen4exp.expert_shared_feed_forward_length),
				layer(width),
			])
		).fp(32));
	}

	// Leaving the hyper stream folds its lanes back to one under the same read gate
	// before the vocabulary projection.
	let model = model.collapse(stream_norm(), read()).layer(tokenizer.ggml.tokens);

	let mut infer = recipe.infer();
	if let Ok(path) = std::env::var("MTP") { infer = infer.mtp(path); }
	if std::env::args().any(|arg| arg == "--memory") {
		for (index, memory) in infer.memory(&model, &data, 128).unwrap().iter().enumerate() {
			println!("planned {} {memory}", if index == 0 { "target" } else { "MTP" });
			println!("weights_bytes {} values_bytes {} contexts_bytes {} input_bytes {} scratch_bytes {}", memory.weights, memory.values, memory.contexts, memory.input, memory.scratch);
		}
		return;
	}
	let report = infer.chat([time, pp, tg, input, out, cached, mtp]).run(&model, &data);
	println!("generation tokens {} rate {} tok/s", report.out, report.tg());
}
