use recipe::*;

// RECIPE_PACKED_DOT=1 enables optional Q8 activation dot products on AMD.
const GGUF: &str = "/home/nate/.lmstudio/models/lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q4_K_M.gguf";

fn main() {
	let precision = std::env::var("RNJ_FP").ok().map(|v| v.parse().unwrap()).unwrap_or(16);
	let data = recipe.gguf(GGUF).fp(precision);
	let integer = |key: &str| data.value(key).unwrap().integer().unwrap() as usize;
	let float = |key: &str| data.value(key).unwrap().float().unwrap();
	let width = integer("gemma3.embedding_length");
	let heads = integer("gemma3.attention.head_count");
	let kv = integer("gemma3.attention.head_count_kv");
	let head = integer("gemma3.attention.key_length");
	let embedding = data.tensor("token_embd.weight").unwrap().clone();
	let mut plan = Binding::default().node(&[embedding.clone()]);
	let mut model = recipe.model().no(bias).epsilon(float("gemma3.attention.layer_norm_rms_epsilon")).embed(embedding.shape[1] as usize, width).scale((width as f64).sqrt());

	for layer_index in 0..integer("gemma3.block_count") {
		model = model
			.res([
				norm(rms),
				attn(heads).kv(kv).width(head).qk(rms).rope(neox, head, float("gemma3.rope.freq_base")).yarn(
					float("gemma3.rope.scaling.factor"),
					integer("gemma3.rope.scaling.original_context_length"),
					float("gemma3.rope.scaling.yarn_beta_fast"),
					float("gemma3.rope.scaling.yarn_beta_slow"),
				),
				norm(rms),
			])
			.res([
				norm(rms),
				recipe.model().no(bias).layer(integer("gemma3.feed_forward_length")).gelu() * recipe.model().no(bias).layer(integer("gemma3.feed_forward_length")),
				layer(width),
				norm(rms),
			]);
		let tensor = |name: &str| data.tensor(&format!("blk.{layer_index}.{name}.weight")).unwrap().clone();
		let qk = std::iter::repeat_n(tensor("attn_q_norm"), heads).chain(std::iter::repeat_n(tensor("attn_k_norm"), kv)).collect::<Vec<_>>();
		plan = plan
			.node(&[tensor("attn_norm")])
			.node(&[tensor("attn_q"), tensor("attn_k"), tensor("attn_v")])
			.node(&qk)
			.node(&[tensor("attn_output")])
			.node(&[tensor("post_attention_norm")])
			.node(&[tensor("ffn_norm")])
			.node(&[tensor("ffn_gate")])
			.node(&[tensor("ffn_up")])
			.node(&[tensor("ffn_down")])
			.node(&[tensor("post_ffw_norm")]);
	}
	let cap = float("gemma3.final_logit_softcapping");
	model = model.norm(rms).last().layer(embedding.shape[1] as usize).scale(1.0 / cap).tanh().scale(cap);
	plan = plan.named(&data, "output_norm.weight").node(&[embedding]);

	let tokenizer = data.tokenizer();
	let message = std::env::args().nth(1).unwrap_or_else(|| "What is the capital of France?".into());
	let message = std::env::var("RNJ_PROMPT_FILE").map(|path| std::fs::read_to_string(path).unwrap()).unwrap_or(message);
	let prompt = tokenizer.encode(&format!("<|start_header_id|>system<|end_header_id|>\nYou are rnj-1, a foundation model trained by Essential AI.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"));
	let count = std::env::var("RNJ_TOKENS").ok().map(|v| v.parse().unwrap()).unwrap_or(32);
	let context = std::env::var("RNJ_CONTEXT").ok().map(|v| v.parse().unwrap()).unwrap_or(16384 + count);
	let bos = integer("tokenizer.ggml.bos_token_id") as u32;
	let mut stop = tokenizer.encode("<|eot_id|>").into_iter().filter(|id| *id != bos).collect::<Vec<_>>();
	stop.push(integer("tokenizer.ggml.eos_token_id") as u32);
	eprintln!("RNJ-1: {} prompt tokens, {context} context positions", prompt.len());
	let generated = data.decode(&model, &plan, context, &prompt, &mut recipe.sampler().temperature(0.0), &stop, count);
	println!("{}", tokenizer.decode(&generated.ids[prompt.len()..]));
	let seconds: f64 = generated.step_seconds.iter().sum();
	eprintln!("prefill {} s; {} decode steps in {} s", generated.prefill_seconds, generated.step_seconds.len(), seconds);
	if seconds > 0.0 {
		eprintln!("{} tok/s", generated.step_seconds.len() as f64 / seconds);
	}
}
