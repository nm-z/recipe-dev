use recipe::*;
use recipe::infer::{cached, input, mtp, out, pp, tg, time};

const GGUF: &str = "/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf";

#[rustfmt::skip]
fn main() {
	let data = recipe.data(std::env::var("GGUF").unwrap_or_else(|_| GGUF.to_owned()));
	let width = 2560;
	let (lanes, rank) = (4, 320);

	let mut model = recipe.model()
		.e(9.999999974752427e-7)
		.embed(248320, width);

	for block in 0..48 {
		if [1_usize].contains(&block) { model = model.ple(&ngram); }

		model = model.hyper(lanes, rank, &if (block + 1) % 4 == 0 {
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
		model = model.hyper(lanes, rank, &Model::from(
			moe(10, experts).scores(Scoring::Softmax).renormalize(true)
			+ layer(1).sigmoid() * Block::from([
				layer(640).silu() * layer(640),
				layer(width),
			])
		).fp(32));
	}

	let model = model.layer(248320);

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
