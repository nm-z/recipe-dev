use recipe::*;
use recipe::infer::{cached, input, mtp, out, pp, tg, time};

const GGUF: &str = "/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf";

#[rustfmt::skip]
fn main() {
	let data = recipe.data(std::env::var("GGUF").unwrap_or_else(|_| GGUF.to_owned()));
	let width = 2560;
	let (lanes, rank) = (4, 320);
	let channels = lanes * width;
	let mean = 1.0 / lanes as f64;
	let stream_norm = || norm(rms).per(width);
	let read = || [layer(rank).scale(mean).silu(), layer(channels).sigmoid()];
	let write = || [layer(lanes), linear().scale(mean).sigmoid().scale(2.0)];

	let rows = ngrams(3)
		.heads(8)
		.multipliers(&[23703573157769_usize, 20109073645365, 8052911324071])
		.ranges(&[0_usize, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275], &[20000003_usize, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171])
		.ends(&[248044])
		.absent(248044)
		.image(248056)
		.table("per_layer_token_embd.weight");
	let gate = [
		layer(channels).norm(rms).per(width) * stream().norm(rms).per(width),
		fold(width),
		scalar(|s| {
			let s = s * (1.0 / (width as f64).sqrt());
			(s.sign() * s.abs().max(1e-6).sqrt()).sigmoid()
		}),
	];
	let ple = [
		rows,
		outer(gate, layer(width)),
		identity() + [norm(rms).per(width), dconv(4).dilation(3).silu()],
	];

	let mut model = recipe.model()
		.e(9.999999974752427e-7)
		.embed(248320, width);

	for block in 0..48 {
		if [1_usize].contains(&block) { model = model.inject(ple.clone()); }

		model = model.hyper_gated(lanes, stream_norm(), read(), write(), &if (block + 1) % 4 == 0 {
			recipe.model()
				.attn(24)
				.kv(2)
				.head(256)
				.gate()
				.qk(rms)
				.rope(interleaved(&[11_usize, 11, 10, 0]), 64, 10000000.0)
				.index_tokens(
					4,
					128,
					[0_usize, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4][block].max(1),
					2048
				)
				.score(rms, 64)
		} else {
			recipe.model()
				.delta(48, 4)
				.keys(16, 128)
				.values(128)
				.out(width)
				.conv_activation(Activation::Silu)
				.qk(l2)
				.decay(DeltaDecay::Softplus)
				.write(Activation::Sigmoid)
				.query_scale(DeltaScale::InverseRoot)
				.out_norm(rms)
				.output_gate(Activation::Sigmoid)
		});

		let mut experts = Vec::new();
		for _ in 0..512 {
			experts.push(Block::from([
				layer(640).silu() * layer(640),
				layer(width),
			]));
		}
		model = model.hyper_gated(lanes, stream_norm(), read(), write(), &Model::from(
			moe(10, experts).scores(Scoring::Softmax).renormalize(true)
			+ layer(1).sigmoid() * Block::from([
				layer(640).silu() * layer(640),
				layer(width),
			])
		).fp(32));
	}

	let model = model.collapse(stream_norm(), read()).layer(248320);

	let mut infer = recipe.infer();
	if let Ok(path) = std::env::var("MTP") { infer = infer.mtp(path); }
	if std::env::args().any(|arg| arg == "--memory") {
		for (index, memory) in infer.memory(&model, &data, 128).unwrap().iter().enumerate() {
			println!("planned {} {memory}", if index == 0 { "target" } else { "MTP" });
			println!("weights_bytes {} values_bytes {} contexts_bytes {} input_bytes {} scratch_bytes {}", memory.weights, memory.values, memory.contexts, memory.input, memory.scratch);
		}
		return;
	}
	let report = infer.chat([time, pp, tg, input, out, cached, mtp]).run(&model, &data);
	println!("generation tokens {} rate {} tok/s", report.out, report.tg());
}
