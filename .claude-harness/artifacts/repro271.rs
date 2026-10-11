use recipe::*;

fn main() {
	let bundle = std::env::var("REPRO_BUNDLE").expect("REPRO_BUNDLE is unset");
	let data = recipe.data("data/temporal/chronological_splits")
		.target("target")
		.broadcast();
	let model = recipe.model()
		.layer(8)
		.res([conv(8, 1), relu(), layer(8)]).silu().norm(batch).iq(4).nl
		.conv(8, 1).gelu().iq(2).xxs
		.loss(mse);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(24393283136277)
		.epochs(1)
		.log(all)
		.stop(0.8)
		.int(1)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
	println!("TRAIN final_loss={:016x} finite={}", report.final_loss().to_bits(), report.final_loss().is_finite());
}
