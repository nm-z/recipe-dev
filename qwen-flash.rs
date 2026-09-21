use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/mnt/models/unsloth/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";

fn main() {
	let data = recipe.data(GGUF);
	let ngram = data.ngram();
	let compression = match data.value("qwen4exp.attention.compress_ratios").unwrap() {
		GgufValue::Array(values) => values.iter().map(|value| value.integer().unwrap() as usize).collect::<Vec<_>>(),
		_ => panic!("attention.compress_ratios must be an array"),
	};
	let mut model = recipe.model().epsilon(qwen4exp.attention.layer_norm_rms_epsilon).embed(tokenizer.ggml.tokens, qwen4exp.embedding_length);
	for block in 0..qwen4exp.block_count {
		if block == ngram.layer() { model = model.ple(&ngram); }
		let mut attention = if (block + 1) % 4 == 0 {
			let mut attention = recipe.model().attn(qwen4exp.attention.head_count)
				.kv(qwen4exp.attention.head_count_kv).head(qwen4exp.attention.key_length);
			if data.tensor(&format!("blk.{block}.attn_q.weight")).unwrap().shape[1] == 12288 { attention = attention.gate(); }
			if data.tensor(&format!("blk.{block}.attn_q_norm.weight")).is_some() { attention = attention.qk(rms); }
			attention.rope(neox, qwen4exp.rope.dimension_count, qwen4exp.rope.freq_base)
				.yarn(4.0, 262144, 32.0, 1.0).fp(32)
				.index(4, 128, compression[block].max(1), 1).budget(2048).score(rms, qwen4exp.rope.dimension_count)
		} else {
			recipe.model().delta(48, 4).keys(16, 128).values(128).out(qwen4exp.embedding_length)
				.delta_activations(Activation::Silu, Activation::Sigmoid)
		};
		if data.tensor(&format!("blk.{block}.post_attention_norm.weight")).is_some() { attention = attention.norm(rms); }
		model = model.hyper(4, 320, &attention);
		let expert = [
			layer(640).fp(16).silu() * layer(640).fp(16),
			layer(qwen4exp.embedding_length).fp(16),
		];
		let routed = moe(10, [expert; 512]).fp(16);
		let shared = [
			layer(640).fp(16).silu() * layer(640).fp(16),
			layer(qwen4exp.embedding_length).fp(16),
		] * layer(1).fp(16).sigmoid();
		let mut experts = Model::from(routed + shared);
		if data.tensor(&format!("blk.{block}.post_ffw_norm.weight")).is_some() { experts = experts.norm(rms); }
		model = model.hyper(4, 320, &experts);
	}
	model = model.layer(tokenizer.ggml.tokens);
	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
