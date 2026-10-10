use recipe::*;
use recipe::infer::{cached, input, mtp, out, pp, tg, time};

const GGUF: &str = "/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf";

pub fn model() -> Model {
	let (width, vocabulary) = (qwen4exp.embedding_length, tokenizer.ggml.tokens.len());
	let mut model = recipe.model().embed(vocabulary, width);
	for layer in 0..qwen4exp.block_count {
		if qwen4exp.ple.layers.contains(&layer) { model = model.ple(&ngram); }
		let attention = if (layer + 1) % qwen4exp.full_attention_interval == 0 {
			let group = qwen4exp.attention.compress_ratios[layer].max(1);
			recipe.model().attn(qwen4exp.attention.head_count).kv(qwen4exp.attention.head_count_kv).head(qwen4exp.attention.key_length)
				.rope(neox, qwen4exp.rope.dimension_count, qwen4exp.rope.freq_base)
				.index_tokens(qwen4exp.attention.indexer.head_count, qwen4exp.attention.indexer.key_length, group, qwen4exp.attention.indexer.top_k)
		} else {
			recipe.model().delta(qwen4exp.ssm.time_step_rank, qwen4exp.ssm.conv_kernel)
				.keys(qwen4exp.ssm.group_count, qwen4exp.ssm.state_size).values(qwen4exp.ssm.state_size).out(width)
				.delta_activations(Activation::Silu, Activation::Sigmoid)
		};
		model = model.hyper(qwen4exp.hyper_connection.count, qwen4exp.hyper_connection.low_rank, &attention);
		let scoring = match qwen4exp.expert_gating_func { 1 => Scoring::Softmax, 2 => Scoring::Sigmoid, value => panic!("unknown expert gating function {value}") };
		let experts = recipe.model().gguf_moe(qwen4exp.expert_count, qwen4exp.expert_used_count, qwen4exp.expert_feed_forward_length,
			Activation::Silu, scoring, qwen4exp.expert_weights_norm, qwen4exp.expert_shared_feed_forward_length != 0).fp(32);
		model = model.hyper(qwen4exp.hyper_connection.count, qwen4exp.hyper_connection.low_rank, &experts);
	}
	model.layer(vocabulary)
}

fn main() {
	let data=recipe.data(GGUF);
	let model=model();
	let mut infer=recipe.infer();
	if let Ok(path)=std::env::var("MTP") {infer=infer.mtp(path);}
	let free=[6113460224usize,8450998272,0,7915765760,8450998272,8450998272,0,8450998272];
	let budgets=std::array::from_fn(|die|ExpertDieBudget {free_bytes:free[die],reserve_bytes:32<<20});
	let plan=infer.expert_plan(&model,&data,128,budgets).unwrap();
	println!("{}",plan.table()); for item in &plan.placements {if item.row_start.is_some() {println!("dense_slice\t{}\t{}\t{}\t{}\t{}\t{}",item.mtp,item.tensor.name,item.die,item.row_start.unwrap(),item.tensor.shape[1],item.tensor.bytes);}}
	assert_eq!(plan.unplaced_bytes,0); assert_eq!(plan.unplaced_experts,0); assert_eq!(plan.main_die,4);
	assert!(plan.placements.iter().all(|item|item.die!=6)); assert!(plan.fits());
}
