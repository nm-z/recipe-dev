use recipe::*;

const GGUF: &str = "/home/nate/.lmstudio/models/lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q4_K_M.gguf";
// llama.cpp's Gemma3 loader uses this architecture default and ignores the
// file's stray yarn_beta_fast = 64 metadata.
const YARN_FAST: f64 = 32.0;
#[rustfmt::skip]
fn main() {
	let data = recipe.data(GGUF);

	let mut model = recipe.model()
		.no(bias)
		.embed(tokenizer.ggml.tokens, gemma3.embedding_length).fp(16)
		.scale(gemma3.embedding_length.sqrt()).fp(16);

	for _ in 0..gemma3.block_count {
		model = model
			.res([
				norm(rms).fp(16),
				attn(gemma3.attention.head_count).int(8).acc(32)
					.kv(gemma3.attention.head_count_kv).fp(16)
					.qk(rms).fp(16)
					.rope(
						neox,
						gemma3.attention.key_length,
						gemma3.rope.freq_base
					)
					.yarn(
						gemma3.rope.scaling.factor,
						gemma3.rope.scaling.original_context_length,
						YARN_FAST,
						gemma3.rope.scaling.yarn_beta_slow
					).fp(32),
				norm(rms).fp(16),
			]).fp(16)
			.res([
				norm(rms).fp(16),
				layer(gemma3.feed_forward_length).int(8).acc(32).gelu().fp(16)
					* layer(gemma3.feed_forward_length).int(8).acc(32),
				layer(gemma3.embedding_length).int(8).acc(32),
				norm(rms).fp(16),
			]).fp(16);
	}

	model = model
		.norm(rms).fp(16)
		.layer(tokenizer.ggml.tokens).int(8).acc(32)
		.scale(1.0 / gemma3.final_logit_softcapping).fp(16)
		.tanh().fp(16)
		.scale(gemma3.final_logit_softcapping).fp(16);

	recipe.infer().log([chat]).run(&model, &data);
}
