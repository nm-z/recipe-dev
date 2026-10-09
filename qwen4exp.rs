use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/mnt/sentry-nfs/unsloth/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";

fn hyper_gate(lanes: usize, rank: usize, width: usize) -> HyperGate {
	let scale = 1.0 / lanes as f64;
	HyperGate {
		read: recipe.model().no(bias).norm(rms).layer(rank).scale(scale).silu().layer(lanes * width).sigmoid(),
		write: recipe.model().no(bias).norm(rms).layer(lanes).scale(scale).sigmoid().scale(2.0),
	}
}

pub fn model() -> Model {
	let (width, vocabulary) = (qwen4exp.embedding_length, tokenizer.ggml.tokens.len());
	let (lanes, rank) = (qwen4exp.hyper_connection.count, qwen4exp.hyper_connection.low_rank);
	let mut model = recipe.model().embed(vocabulary, width);
	for layer in 0..qwen4exp.block_count {
		if qwen4exp.ple.layers.contains(&layer) { model = model.ple(&ngram); }
		let attention = if (layer + 1) % qwen4exp.full_attention_interval == 0 {
			let group = qwen4exp.attention.compress_ratios[layer].max(1);
			recipe.model().attn(qwen4exp.attention.head_count).kv(qwen4exp.attention.head_count_kv).head(qwen4exp.attention.key_length)
				.rope(neox, qwen4exp.rope.dimension_count, qwen4exp.rope.freq_base)
				.index_tokens(qwen4exp.attention.indexer.head_count, qwen4exp.attention.indexer.key_length, group, qwen4exp.attention.indexer.top_k)
		} else {
			recipe.model().delta(qwen4exp.ssm.time_step_rank, qwen4exp.ssm.conv_kernel)
				.keys(qwen4exp.ssm.group_count, qwen4exp.ssm.state_size).values(qwen4exp.ssm.state_size).out(width)
				.delta_activations(Activation::Silu, Activation::Sigmoid)
		};
		model = if rank == 0 { model.hyper(lanes, &attention) } else { model.hyper_gate(lanes, &attention, hyper_gate(lanes, rank, width)) };
		let scoring = match qwen4exp.expert_gating_func { 1 => Scoring::Softmax, 2 => Scoring::Sigmoid, value => panic!("unknown expert gating function {value}") };
		let experts = recipe.model().gguf_moe(qwen4exp.expert_count, qwen4exp.expert_used_count, qwen4exp.expert_feed_forward_length,
			Activation::Silu, scoring, qwen4exp.expert_weights_norm, qwen4exp.expert_shared_feed_forward_length != 0);
		model = if rank == 0 { model.hyper(lanes, &experts) } else { model.hyper_gate(lanes, &experts, hyper_gate(lanes, rank, width)) };
	}
	model.layer(vocabulary)
}

fn main() {
	let data = recipe.data(std::env::var("GGUF").unwrap_or_else(|_| GGUF.to_owned()));
	let model = model();
	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
