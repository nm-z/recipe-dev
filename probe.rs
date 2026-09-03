//! Print the Flash-Next architecture hyperparameters from the GGUF so the model
//! script can be written against real values rather than guesses.

use recipe::*;

fn main() {
	let path = "/mnt/models/unsloth/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";
	let model = recipe.gguf(path);

	let keys = [
		"general.architecture",
		"qwen4exp.block_count",
		"qwen4exp.context_length",
		"qwen4exp.embedding_length",
		"qwen4exp.attention.head_count",
		"qwen4exp.attention.head_count_kv",
		"qwen4exp.attention.key_length",
		"qwen4exp.attention.value_length",
		"qwen4exp.attention.layer_norm_rms_epsilon",
		"qwen4exp.expert_count",
		"qwen4exp.expert_used_count",
		"qwen4exp.expert_feed_forward_length",
		"qwen4exp.expert_shared_feed_forward_length",
		"qwen4exp.rope.freq_base",
		"qwen4exp.rope.dimension_count",
		"qwen4exp.ssm.conv_kernel",
		"qwen4exp.ssm.state_size",
		"qwen4exp.ssm.group_count",
		"qwen4exp.ssm.time_step_rank",
		"qwen4exp.ssm.inner_size",
		"qwen4exp.full_attention_interval",
		"qwen4exp.hyper_connection.count",
		"qwen4exp.hyper_connection.low_rank",
		"qwen4exp.attention.indexer.head_count",
		"qwen4exp.attention.indexer.key_length",
		"qwen4exp.attention.indexer.top_k",
		"qwen4exp.ple.ngram_size",
		"qwen4exp.ple.heads_per_ngram",
		"qwen4exp.ple.conv_kernel",
		"qwen4exp.ple.eos_token_id",
		"qwen4exp.embedding_length_per_layer_input",
	];
	for key in keys {
		match model.value(key) {
			Some(value) => println!("{key} = {value:?}"),
			None => println!("{key} = <absent>"),
		}
	}

	println!("--- tensors: {} ---", model.tensors().len());
	for tensor in model.tensors().iter().take(14) {
		println!("  {} shape={:?} kind={} bytes={}", tensor.name, tensor.shape, tensor.kind, tensor.bytes);
	}
}
