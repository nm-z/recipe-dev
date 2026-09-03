//! Does the DSA indexer actually prune at short context? top_k is in blocks, so
//! blocks_covered = top_k * compress_ratio tokens. If that exceeds the context we
//! intend to run, the selector is a no-op and dense attention is exact.

use recipe::*;

fn main() {
	let path = "/mnt/models/unsloth/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";
	let model = recipe.gguf(path);
	for key in ["qwen4exp.attention.compress_ratios", "qwen4exp.attention.indexer.top_k", "qwen4exp.ple.layer_multipliers", "qwen4exp.ple.head_offsets", "qwen4exp.ple.head_vocab_sizes", "qwen4exp.rope.dimension_sections", "qwen4exp.ple.layers"] {
		match model.value(key) {
			Some(value) => {
				let text = format!("{value:?}");
				println!("{key} = {}", if text.len() > 300 { format!("{}...", &text[..300]) } else { text });
			}
			None => println!("{key} = <absent>"),
		}
	}
	// which layers carry indexer tensors at all
	let mut indexer = Vec::new();
	let mut delta = Vec::new();
	for tensor in model.tensors() {
		if tensor.name.contains("indexer") && tensor.name.ends_with("k_proj.weight") {
			indexer.push(tensor.name.clone());
		}
		if tensor.name.contains("ssm_conv1d") {
			delta.push(tensor.name.clone());
		}
	}
	println!("--- indexer layers: {} ---", indexer.len());
	println!("--- delta-net (ssm_conv1d) layers: {} ---", delta.len());
	// distinct tensor name suffixes on one delta-net layer and one attention layer
	for prefix in ["blk.0.", "blk.3."] {
		let names = model.tensors().iter().filter(|t| t.name.starts_with(prefix)).map(|t| t.name[prefix.len()..].to_owned()).collect::<Vec<_>>();
		println!("{prefix} ({}): {}", names.len(), names.join(" "));
	}
}
