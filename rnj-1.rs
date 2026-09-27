use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};
use recipe::tensor::token_embd;

const GGUF: &str = "/home/nate/.lmstudio/models/lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q4_K_M.gguf";
// llama.cpp's Gemma3 loader uses this architecture default and ignores the
// file's stray yarn_beta_fast = 64 metadata.
const YARN_FAST: f64 = 32.0;
#[rustfmt::skip]
fn main() {
	let data = recipe.data(GGUF);

	let mut model = recipe.model()
		.no(bias)
		.embed(tokenizer.ggml.tokens, arch.embedding_length).fp(16)
		.scale(arch.embedding_length.sqrt()).fp(16);

	for block in 0..arch.block_count.get() {
		model = model
			.res([
				norm(rms).fp(16),
				attn(arch.attention.head_count).int(8)
					.kv(arch.attention.head_count_kv).fp(16)
					.q(blk[block].attn_q.weight)
					.k(blk[block].attn_k.weight)
					.v(blk[block].attn_v.weight)
					.qk(rms).fp(16)
					.rope(
						neox,
						arch.attention.key_length,
						arch.rope.freq_base
					)
					.yarn(
						arch.rope.scaling.factor,
						arch.rope.scaling.original_context_length,
						YARN_FAST,
						arch.rope.scaling.yarn_beta_slow
					).fp(32),
				layer(arch.embedding_length).int(8),
				norm(rms).fp(16),
			]).fp(16)
			.res([
				norm(rms).fp(16),
				layer(arch.feed_forward_length).int(8).gelu().fp(16)
					* layer(arch.feed_forward_length).int(8),
				layer(arch.embedding_length).int(8),
				norm(rms).fp(16),
			]).fp(16);
	}

	model = model
		.norm(rms).fp(16)
		.layer(tokenizer.ggml.tokens).bind(token_embd.weight).int(8)
		.scale(1.0 / arch.final_logit_softcapping).fp(16)
		.tanh().fp(16)
		.scale(arch.final_logit_softcapping).fp(16);

	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
