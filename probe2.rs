//! Decode real quantized tensors from the Flash-Next IQ1_S GGUF to verify the
//! block decoders handle this model's mix before any of it is wired into a graph.
//! ggml kinds here: 0=F32 8=Q8_0 12=Q4_K 13=Q5_K 19=IQ1_S 20=IQ4_NL

use recipe::*;

fn stats(name: &str, values: &[f64]) {
	let finite = values.iter().filter(|v| v.is_finite()).count();
	let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
	let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let mean = values.iter().sum::<f64>() / values.len() as f64;
	// `rms` is a public const in recipe, so the local binding needs another name
	let root = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
	println!("  {name}: n={} finite={finite} min={min:.5} max={max:.5} mean={mean:.5} rms={root:.5}", values.len());
	let head = values.iter().take(8).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(" ");
	println!("      first8: {head}");
}

fn main() {
	let path = "/mnt/models/unsloth/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";
	let model = recipe.gguf(path);

	// one tensor per distinct quantization the model actually uses
	let names = [
		"blk.0.ffn_gate_exps.weight",  // IQ1_S  - the bulk of the model
		"blk.0.ffn_down_exps.weight",  // IQ4_NL
		"blk.0.attn_qkv.weight",       // Q5_K
		"output.weight",               // Q4_K
		"blk.0.ffn_down_shexp.weight", // Q8_0
		"output_hc_norm.weight",       // F32
	];
	for name in names {
		match model.tensor(name) {
			Some(tensor) => {
				println!("{name} kind={} shape={:?} bytes={}", tensor.kind, tensor.shape, tensor.bytes);
				match model.values(tensor) {
					Ok(values) => stats(name, &values[..values.len().min(4096)]),
					Err(error) => println!("  DECODE FAILED: {error}"),
				}
			}
			None => println!("{name} = <absent>"),
		}
	}
}
