use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/data/models/Phi-3.5-mini-instruct-Q4_K_M.gguf";
#[rustfmt::skip]
fn main() {
	let data = recipe.data(GGUF);

	let mut model = recipe.model().embed(tokenizer.ggml.tokens, phi3.embedding_length);

	for _ in 0..phi3.block_count {
		model = model
			.res([
				norm(rms),
				attn(phi3.attention.head_count)
					.kv(phi3.attention.head_count_kv)
					.rope(
						neox,
						phi3.rope.dimension_count,
						phi3.rope.freq_base
					)
					.rope_factors(phi3.rope.scaling.original_context_length)
					.rope_scale(phi3.rope.scaling.attn_factor).fp(32),
				layer(phi3.embedding_length),
			])
			.res([
				norm(rms),
				layer(phi3.feed_forward_length).silu() * layer(phi3.feed_forward_length),
				layer(phi3.embedding_length),
			]);
	}

	let model = model.norm(rms).layer(tokenizer.ggml.tokens);
	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
