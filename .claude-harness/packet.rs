use recipe::*;

fn main() {
	let bundle = ".claude-harness/packet-bundle.ogdl";
	let data = recipe.data("data/image/arrays_npz.npz")
		.target("target")
		.norm(z_score);
	let model = recipe.model()
		.bayes().prelu().norm(batch).iq(1).m
		.kmeans(2).gelu().iq(3).xxs
		.loss(bce);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(87409099861703)
		.epochs(1)
		.log(all)
		.stop(0.0)
		.tf(32)
		.save(bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
	let resumed = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(87409099861703)
		.epochs(1)
		.log(all)
		.stop(0.0)
		.tf(32)
		.resume(bundle)
		.save(bundle)
		.run(&model, &data);
	assert!(resumed.final_loss().is_finite());
}
