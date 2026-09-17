use recipe::*;

const GGUF: &str = "/home/nate/.lmstudio/models/lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q4_K_M.gguf";
#[rustfmt::skip]
fn main() {
	let data = recipe.data(GGUF);

	let mut model = recipe.model()
		.no(bias)
		.embed(tokenizer.ggml.tokens, gemma3.embedding_length).bf(16)
		.scale(gemma3.embedding_length.sqrt());

	for _ in 0..gemma3.block_count {
		model = model
			.res([
				norm(rms),
				attn(gemma3.attention.head_count)
					.kv(gemma3.attention.head_count_kv).fp(16)
					.qk(rms)
					.rope(
						neox,
						gemma3.attention.key_length,
						gemma3.rope.freq_base
					)
					.yarn(
						gemma3.rope.scaling.factor,
						gemma3.rope.scaling.original_context_length,
						gemma3.rope.scaling.yarn_beta_fast,
						gemma3.rope.scaling.yarn_beta_slow
					),
				norm(rms),
			]).bf(16)
			.res([
				norm(rms),
				layer(gemma3.feed_forward_length).gelu()
					* layer(gemma3.feed_forward_length),
				layer(gemma3.embedding_length),
				norm(rms),
			]).bf(16);
	}

	model = model
		.norm(rms)
		.layer(tokenizer.ggml.tokens).bf(16)
		.scale(1.0 / gemma3.final_logit_softcapping)
		.tanh()
		.scale(gemma3.final_logit_softcapping);

	recipe.infer().log([chat]).run(&model, &data);
}
