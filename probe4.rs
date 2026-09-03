//! Bind real Flash-Next tensors as parameter buffers. A block-quantized tensor
//! must keep the file's own bytes (no decode, no re-encode); an F32 tensor
//! decodes once. Byte count must equal the mapped region exactly.

use recipe::*;

fn main() {
	let path = "/mnt/models/unsloth/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";
	let model = recipe.gguf(path);
	let names = [
		"blk.0.ffn_gate_exps.weight",  // IQ1_S
		"blk.0.ffn_down_exps.weight",  // IQ4_NL
		"blk.0.attn_qkv.weight",       // Q5_K
		"output.weight",               // Q4_K
		"blk.0.ffn_down_shexp.weight", // Q8_0
		"output_hc_norm.weight",       // F32
		"per_layer_token_embd.weight", // IQ4_NL, 26.8 GiB
	];
	let mut total = 0u64;
	for name in names {
		let Some(tensor) = model.tensor(name) else {
			println!("{name} = <absent>");
			continue;
		};
		match model.bind(tensor) {
			Ok(stored) => {
				let exact = !stored.quantized() || stored.bytes().len() == tensor.bytes;
				println!("{name} kind={} count={} quantized={} bytes={} matches_file={exact}", tensor.kind, stored.count(), stored.quantized(), stored.bytes().len());
				total += tensor.bytes as u64;
			}
			Err(error) => println!("{name} BIND FAILED: {error}"),
		}
	}
	println!("--- bound {:.2} GiB of tensors ---", total as f64 / 1073741824.0);
}
