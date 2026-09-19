use recipe::*;
use std::io::{BufRead, Write};

const GGUF: &str = "/home/nate/.lmstudio/models/lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q4_K_M.gguf";
// llama.cpp's Gemma3 loader uses this architecture default and ignores the
// file's stray yarn_beta_fast = 64 metadata.
const YARN_FAST: f64 = 32.0;

fn hex(bytes: &[u8]) -> String {
	const DIGITS: &[u8; 16] = b"0123456789abcdef";
	let mut text = String::with_capacity(bytes.len() * 2);
	for byte in bytes {
		text.push(DIGITS[(byte >> 4) as usize] as char);
		text.push(DIGITS[(byte & 15) as usize] as char);
	}
	text
}

#[rustfmt::skip]
fn main() {
	let _data = recipe.data(GGUF);
	let file = recipe.gguf(GGUF);
	let coder = file.tokenizer();
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
					.rope(neox, gemma3.attention.key_length, gemma3.rope.freq_base)
					.yarn(gemma3.rope.scaling.factor, gemma3.rope.scaling.original_context_length, YARN_FAST, gemma3.rope.scaling.yarn_beta_slow).fp(32),
				norm(rms).fp(16),
			]).fp(16)
			.res([
				norm(rms).fp(16),
				layer(gemma3.feed_forward_length).int(8).acc(32).gelu().fp(16) * layer(gemma3.feed_forward_length).int(8).acc(32),
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

	let positions = std::env::var("RECIPE_CONTEXT").ok().and_then(|value| value.parse().ok()).unwrap_or(1024);
	let placed = file.place(&model, positions, &[]);
	println!("READY\t{positions}");
	std::io::stdout().flush().unwrap();
	let input = std::io::stdin();
	for line in input.lock().lines() {
		let line = line.unwrap();
		let Some((path, budget)) = line.split_once('\t') else { continue };
		let budget = budget.parse::<usize>().unwrap();
		let text = std::fs::read_to_string(path).unwrap();
		let prompt = coder.encode(&text);
		let stop = coder.stop_ids();
		let generation = placed.decode(&prompt, &mut recipe.sampler().temperature(0.0), &stop, budget);
		let mut reply_ids = &generation.ids[prompt.len()..];
		if reply_ids.last().is_some_and(|id| stop.contains(id)) {
			reply_ids = &reply_ids[..reply_ids.len() - 1];
		}
		let reply = coder.decode(reply_ids);
		let decode: f64 = generation.step_seconds.iter().sum();
		let rate = if decode == 0.0 { 0.0 } else { generation.step_seconds.len() as f64 / decode };
		println!("RESULT\t{}\t{:.9}\t{:.9}\t{:.6}\t{}", prompt.len(), generation.prefill_seconds, decode, rate, hex(reply.as_bytes()));
		std::io::stdout().flush().unwrap();
	}
}
