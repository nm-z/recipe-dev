use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "/mnt/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf";

fn main() {
	let data = recipe.data(GGUF);
	let mut model = recipe.model().e(data.number(key!(arch.attention.layer_norm_rms_epsilon))).embed(248320, 5120);

	for a in 0..data.integer(key!(arch.block_count)) {
		if a % 4 == 3 {
			let attention = attn(24).kv(4).width(256)
				.q(key!(blk[a].attn.q.weight))
				.k(key!(blk[a].attn.k.weight))
				.v(key!(blk[a].attn.v.weight))
				.qk(rms).rope(neox, 64, 10000000.0);
			let gated = attention * layer(6144).sigmoid();
			model = model.res([norm(rms), gated, layer(5120).bind(key!(blk[a].attn.output.weight))]);
		} else {
			let delta = recipe.model().delta(48, 4).keys(16, 128).values(128).out(5120)
				.conv(silu).qk(l2).norm(rms).decay(softplus).output(silu);
			model = model.res([norm(rms), delta.into()]);
		}
		let feed_forward = layer(17408).bind(key!(blk[a].ffn_gate.weight)).silu()
			* layer(17408).bind(key!(blk[a].ffn_up.weight));
		model = model.res([norm(rms), feed_forward, layer(5120).bind(key!(blk[a].ffn_down.weight))]);
	}
	model = model.norm(rms).layer(248320).bind(key!(output.weight));
	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
