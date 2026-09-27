use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/home/nate/models-hdd-backup/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf";
const MTP: &str = "/home/nate/models-hdd-backup/Qwen3.8-27B-GGUF/mtp-Qwen3.8-27B-Q8_0.gguf";

fn hyper(model: Model, branch: &Model) -> Model {
	model.hyper(4, branch)
		.read([norm(rms), layer(320), scale(0.25), silu(), layer(4 * 2560), sigmoid()])
		.write([norm(rms), layer(4), scale(0.25), sigmoid(), scale(2.0)])
}

fn main() {
	let data = recipe.data(GGUF);
	let ngram = data.ngram();
	let mut model = recipe.model().e(0.000001).embed(248320, 2560);

	for block in 0..48 {
		if block == ngram.layer() {
			model = model.ple(&ngram)
				.key([layer(10240), group(rms, 2560), group(rms, 2560)])
				.factor([fold(4).scale(1.0 / (2560.0_f64).sqrt()).signed_sqrt(1e-6).sigmoid()])
				.value([layer(2560), group(rms, 2560)])
				.tail([dconv(ngram.kernel()).dilate(ngram.dilation()).silu()]);
		}

		let mut attention = if (block + 1) % 4 == 0 {
			let mut attention = attn(24).kv(2).width(256)
				.q(key!(blk[block].attn_q.weight))
				.k(key!(blk[block].attn_k.weight))
				.v(key!(blk[block].attn_v.weight));
			if data.has_tensor(key!(blk[block].attn_q_norm.weight)) {
				attention = attention.qk(rms);
			}
			let attention = attention.rope(neox, 64, 10000000.0).index(4, 128, data.integer(key!(qwen4exp.attention.compress_ratios[block])).max(1), 1);
			if data.tensor(key!(blk[block].attn_q.weight)).shape[1] == 12288 {
				(attention * layer(6144).sigmoid()).layer(2560)
			} else {
				attention.layer(2560)
			}
		} else {
			recipe.model().delta(48, 4).keys(16, 128).values(128).out(2560)
				.conv(silu).qk(l2).norm(rms).decay(softplus).output(sigmoid)
		};
		if data.has_tensor(key!(blk[block].post_attention_norm.weight)) {
			attention = attention.norm(rms);
		}
		model = hyper(model, &attention);

		let expert = (layer(640).silu() * layer(640)).layer(2560);
		let mut experts = recipe.model().moe(10, vec![expert.clone(); 512]).route(softmax).renorm();
		if data.has_tensor(key!(blk[block].ffn_gate_shexp.weight)) {
			experts = experts.shared(expert, [layer(1).sigmoid()]);
		}
		if data.has_tensor(key!(blk[block].post_ffw_norm.weight)) {
			experts = experts.norm(rms);
		}
		model = hyper(model, &experts);
	}

	if data.has_tensor(key!(output_hc_norm.weight)) {
		model = model.collapse([norm(rms), layer(320), scale(0.25), silu(), layer(4 * 2560), sigmoid()]);
	} else {
		model = model.collapse([]);
	}
	if data.has_tensor(key!(output_norm.weight)) || data.has_tensor(key!(token_embd_norm.weight)) {
		model = model.norm(rms);
	}
	model = model.layer(248320);
	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
