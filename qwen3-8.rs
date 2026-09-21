use recipe::*;
use recipe::infer::{cached, input, mtp, out, pp, tg, time};

const GGUF: &str = "/home/nate/models-hdd-backup/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf";
const MTP: &str = "/home/nate/models-hdd-backup/Qwen3.8-27B-GGUF/mtp-Qwen3.8-27B-Q8_0.gguf";

fn main() {
	let data = recipe.data(GGUF);
	let mut model = recipe.model().epsilon(qwen35.attention.layer_norm_rms_epsilon).embed(tokenizer.ggml.tokens, qwen35.embedding_length);
	for _ in 0..qwen35.block_count / 4 {
		for _ in 0..3 {
			model = model
				.res([norm(rms), recipe.model().delta(48, 4).keys(16, 128).values(128).out(qwen35.embedding_length).delta_activations(Activation::Silu, Activation::Silu).into()])
				.res([norm(rms), layer(qwen35.feed_forward_length).silu() * layer(qwen35.feed_forward_length), layer(qwen35.embedding_length)]);
		}
		model = model
			.res([
				norm(rms),
				recipe.model()
					.attn(qwen35.attention.head_count)
					.kv(qwen35.attention.head_count_kv)
					.head(qwen35.attention.key_length)
					.gate()
					.qk(rms)
					.rope(neox, qwen35.rope.dimension_count, qwen35.rope.freq_base)
					.yarn(4.0, 262144, 32.0, 1.0)
					.fp(32)
					.into(),
			])
			.res([norm(rms), layer(qwen35.feed_forward_length).silu() * layer(qwen35.feed_forward_length), layer(qwen35.embedding_length)]);
	}
	model = model.norm(rms).layer(tokenizer.ggml.tokens);
	recipe.infer().mtp(MTP).chat([time, pp, tg, input, out, cached, mtp]).run(&model, &data);
}
