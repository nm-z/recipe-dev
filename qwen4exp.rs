use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/mnt/models/unsloth/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";

fn hyper(model: Model, branch: &Model) -> Model {
	model.hyper(4, branch)
		.read([norm(rms), layer(320), scale(0.25), silu(), layer(10240), sigmoid()])
		.write([norm(rms), layer(4), scale(0.25), sigmoid(), scale(2.0)])
}

fn main() {
	let data = recipe.data(GGUF);
	let ngram = data.ngram();
	let mut model = recipe.model().e(data.number(key!(arch.attention.layer_norm_rms_epsilon))).embed(248320, 2560);

	for a in 0..data.integer(key!(arch.block_count)) {
		if a == ngram.layer() {
			model = model.ple(&ngram)
				.key([layer(10240), group(rms, 2560), group(rms, 2560)])
				.factor([fold(4).scale(1.0 / (2560.0_f64).sqrt()).signed_sqrt(1e-6).sigmoid()])
				.value([layer(2560), group(rms, 2560)])
				.tail([dconv(ngram.kernel()).dilate(ngram.dilation()).silu()]);
		}

		let mut attention = if a % 4 == 3 {
			let mut block = recipe.model().attn(24).kv(2).width(256)
				.q(key!(blk[a].attn.q.weight))
				.k(key!(blk[a].attn.k.weight))
				.v(key!(blk[a].attn.v.weight));
			if data.has_tensor(key!(blk[a].attn.q.norm.weight)) {
				block = block.qk(rms);
			}
			block = block.rope(neox, 64, 10000000.0);
			let compression = arch.attention.compress_ratios[a].max(1);
			block = block.index(4, 128, compression, 2048_usize.div_ceil(compression));
			if data.has_tensor(key!(blk[a].indexer.q.norm.weight)) {
				block = block.score(rms, 64);
			}
			let query_rows = data.tensor(key!(blk[a].attn.q.weight)).shape[1] as usize;
			if query_rows == 12288 {
				(block * layer(6144).sigmoid()).layer(2560).bind(key!(blk[a].attn.output.weight))
			} else {
				block.layer(2560).bind(key!(blk[a].attn.output.weight))
			}
		} else {
			recipe.model().delta(48, 4).keys(16, 128).values(128).out(2560)
				.conv(silu).qk(l2).norm(rms).decay(softplus).output(sigmoid)
		};
		if data.has_tensor(key!(blk[a].post_attention_norm.weight)) {
			attention = attention.norm(rms);
		}
		model = hyper(model, &attention);

		let expert = (layer(640).silu() * layer(640)).layer(2560);
		let mut experts = recipe.model().moe(10, vec![expert.clone(); 512]).route(softmax).renorm();
		if data.has_tensor(key!(blk[a].ffn_gate_shexp.weight)) {
			experts = experts.shared(expert, [layer(1).sigmoid()]);
		}
		if data.has_tensor(key!(blk[a].post_ffw_norm.weight)) {
			experts = experts.norm(rms);
		}
		model = hyper(model, &experts);
	}

	if data.has_tensor(key!(output_hc_norm.weight)) {
		model = model.collapse([norm(rms), layer(320), scale(0.25), silu(), layer(10240), sigmoid()]);
	} else {
		model = model.collapse([]);
	}
	if data.has_tensor(key!(output_norm.weight)) || data.has_tensor(key!(token_embd_norm.weight)) {
		model = model.norm(rms);
	}
	model = model.layer(248320).bind(key!(output.weight));
	let bound = data.gguf().model();
	if std::env::var_os("RECIPE_TRACE").is_some() {
		bound.memory_for(&model, 1).unwrap();
		return;
	}
	recipe.infer().chat([time, pp, tg, input, out, cached]).run_bound(&model, &data, &bound);
}
