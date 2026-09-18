use recipe::*;

const GGUF: &str = "/home/nate/.lmstudio/models/lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q4_K_M.gguf";
#[rustfmt::skip]
fn main() {
	let data = recipe.data(GGUF);

	let mut model = recipe.model()
		.no(bias)
		.embed(tokenizer.ggml.tokens, gemma3.embedding_length)
		.scale(gemma3.embedding_length.sqrt());

	for _ in 0..1 {
		model = model
			.res([
				norm(rms),
				attn(gemma3.attention.head_count).int(8).acc(64)
					.kv(gemma3.attention.head_count_kv)
					.qk(rms)
					.rope(neox, gemma3.attention.key_length, gemma3.rope.freq_base),
				norm(rms),
			])
			.res([
				norm(rms),
				layer(gemma3.feed_forward_length).int(8).gelu()
					* layer(gemma3.feed_forward_length).int(8),
				layer(gemma3.embedding_length).int(8),
				norm(rms),
			]);
	}

	model = model
		.norm(rms)
		.layer(tokenizer.ggml.tokens);

	recipe.infer().log([chat, debug]).run(&model, &data);
}
