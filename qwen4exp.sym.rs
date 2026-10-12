use recipe::*;
use recipe::infer::{cached, input, mtp, out, pp, tg, time};

const GGUF: &str = "/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf";

// Qwen3.8 Flash-Next: 48 blocks over a 4-lane hyper-connection stream. Every
// block mixes tokens (full attention every fourth block, a gated delta rule
// otherwise), then mixes channels with 512 routed experts and one shared expert.
#[rustfmt::skip]
pub fn model() -> Model {
	let width = qwen4exp.embedding_length;
	let (lanes, rank) = (qwen4exp.hyper_connection.count, qwen4exp.hyper_connection.low_rank);

	let mut model = recipe.model()
		.e(qwen4exp.attention.layer_norm_rms_epsilon)
		.embed(tokenizer.ggml.tokens, width);

	for block in 0..qwen4exp.block_count {
		// Hashed n-gram embeddings join the stream before the blocks that read them.
		if qwen4exp.ple.layers.contains(&block) { model = model.ple(&ngram); }

		// Token mixing.
		model = model.hyper(lanes, rank, &if (block + 1) % qwen4exp.full_attention_interval == 0 {
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
		model = model.hyper(lanes, rank, &Model::from(
			moe(qwen4exp.expert_used_count, experts).scores(Scoring::Softmax).renormalize(true)
			+ layer(1).sigmoid() * Block::from([
				layer(qwen4exp.expert_shared_feed_forward_length).silu() * layer(qwen4exp.expert_shared_feed_forward_length),
				layer(width),
			])
		).fp(32));
	}

	// Leaving the hyper stream folds its 4 lanes back to one through the head
	// mixer (output_hc_norm, _down, _up) before the vocabulary projection.
	model.layer(tokenizer.ggml.tokens)
}

fn main() {
	let data = recipe.data(std::env::var("GGUF").unwrap_or_else(|_| GGUF.to_owned()));
	let model = model();
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
