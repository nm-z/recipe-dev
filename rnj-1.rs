use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/home/nate/.lmstudio/models/lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q4_K_M.gguf";
// llama.cpp's Gemma3 loader uses this architecture default and ignores the
// file's stray yarn_beta_fast = 64 metadata.
const YARN_FAST: f64 = 32.0;
#[rustfmt::skip]
pub fn build_model(data: &Data) -> Model {
	let vocab = data.array(key!(tokenizer.ggml.tokens)).len();
	let embedding = data.integer(key!(gemma3.embedding_length));

	let mut model = recipe.model()
		.no(bias)
		.embed(vocab, embedding).fp(16)
		.scale(embedding.sqrt()).fp(16);

	for block in 0..data.integer(key!(gemma3.block_count)) {
		model = model
			.res([
				norm(rms).fp(16),
				attn(data.integer(key!(gemma3.attention.head_count))).int(8)
					.kv(data.integer(key!(gemma3.attention.head_count_kv))).fp(16)
					.q(key!(blk[block].attn_q.weight))
					.k(key!(blk[block].attn_k.weight))
					.v(key!(blk[block].attn_v.weight))
					.qk(rms).fp(16)
					.rope(
						neox,
						data.integer(key!(gemma3.attention.key_length)),
						data.number(key!(gemma3.rope.freq_base))
					)
					.yarn(
						data.number(key!(gemma3.rope.scaling.factor)),
						data.integer(key!(gemma3.rope.scaling.original_context_length)),
						YARN_FAST,
						data.number(key!(gemma3.rope.scaling.yarn_beta_slow))
					).fp(32),
				layer(embedding).int(8),
				norm(rms).fp(16),
			]).fp(16)
			.res([
				norm(rms).fp(16),
				layer(data.integer(key!(gemma3.feed_forward_length))).int(8).gelu().fp(16)
					* layer(data.integer(key!(gemma3.feed_forward_length))).int(8),
				layer(embedding).int(8),
				norm(rms).fp(16),
			]).fp(16);
	}

	model = model
		.norm(rms).fp(16)
		.layer(vocab).int(8)
		.scale(1.0 / data.number(key!(gemma3.final_logit_softcapping))).fp(16)
		.tanh().fp(16)
		.scale(data.number(key!(gemma3.final_logit_softcapping))).fp(16);

	model
}

fn main() {
	let data = recipe.data(GGUF);
	let model = build_model(&data);
	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
