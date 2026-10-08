use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/home/nate/models-hdd-backup/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf";
const MTP: &str = "/home/nate/models-hdd-backup/Qwen3.8-27B-GGUF/mtp-Qwen3.8-27B-Q8_0.gguf";

fn main() {
	let data = recipe.data(GGUF);
	let file = recipe.gguf(GGUF);
	let ngram = file.ngram();
	let compression = match file.value("qwen4exp.attention.compress_ratios").unwrap() {
		GgufValue::Array(values) => values.iter().map(|value| value.integer().unwrap() as usize).collect::<Vec<_>>(),
		_ => panic!("qwen4exp.attention.compress_ratios must be an array"),
	};
	let mut model = recipe.model().epsilon(0.000001).embed(248320, 2560);

	for block in 0..48 {
		if block == ngram.layer() {
			model = model.ple(&ngram);
		}

		let mut attention = if (block + 1) % 4 == 0 {
			let mut attention = recipe.model().attn(24).kv(2).head(256);
			if file.tensor(&format!("blk.{block}.attn_q.weight")).unwrap().shape[1] == 12288 {
				attention = attention.gate();
			}
			if file.tensor(&format!("blk.{block}.attn_q_norm.weight")).is_some() {
				attention = attention.qk(rms);
			}
			attention.rope(neox, 64, 10000000.0).index(4, 128, compression[block].max(1), 1).budget(2048)
		} else {
			recipe.model().delta(48, 4).keys(16, 128).values(128).out(2560)
				.delta_block("activations", |delta| {
					delta.conv_activation = Activation::Silu;
					delta.output_activation = Activation::Sigmoid;
				})
		};
		if file.tensor(&format!("blk.{block}.post_attention_norm.weight")).is_some() {
			attention = attention.norm(rms);
		}
		model = model.hyper(4, 320, &attention);

		let shared = file.tensor(&format!("blk.{block}.ffn_gate_shexp.weight")).is_some();
		let mut experts = recipe.model().gguf_moe(512, 10, 640, Activation::Silu, Scoring::Softmax, true, shared);
		if file.tensor(&format!("blk.{block}.post_ffw_norm.weight")).is_some() {
			experts = experts.norm(rms);
		}
		model = model.hyper(4, 320, &experts);
	}

	if file.tensor("output_norm.weight").or_else(|| file.tensor("token_embd_norm.weight")).is_some() {
		model = model.norm(rms);
	}
	model = model.layer(248320);
	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
