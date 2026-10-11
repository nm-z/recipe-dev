use recipe::*;

fn main() {
	let bundle = format!("/tmp/recipe-self-review-271-{}.ogdl", std::process::id());
	let data = recipe
		.data("data/temporal/chronological_splits")
		.target("target")
		.broadcast();
	let model = recipe
		.model()
		.layer(8)
		.res([conv(8, 1), relu(), layer(8)])
		.silu()
		.norm(batch)
		.iq(4)
		.nl
		.conv(8, 1)
		.gelu()
		.iq(2)
		.xxs
		.loss(mse);
	let training = || {
		recipe
			.train()
			.optimizer(adamw)
			.lr(0.001)
			.seed(24393283136277)
			.epochs(1)
			.log(all)
			.stop(0.8)
			.int(1)
	};
	let trained = training().save(&bundle).run(&model, &data);
	assert!(trained.final_loss().is_finite());
	let resumed = training().resume(&bundle).save(&bundle).run(&model, &data);
	assert!(resumed.final_loss().is_finite());
	let output = recipe.infer(&bundle, &[0.0; 20]);
	assert!(!output.is_empty());
	assert!(output.iter().all(|value| value.is_finite()));
	println!(
		"trained={:016x} resumed={:016x} outputs={} infer={:016x} bundle={}",
		trained.final_loss().to_bits(),
		resumed.final_loss().to_bits(),
		output.len(),
		output[0].to_bits(),
		std::fs::metadata(&bundle).unwrap().len()
	);
	std::fs::remove_file(bundle).unwrap();
}
