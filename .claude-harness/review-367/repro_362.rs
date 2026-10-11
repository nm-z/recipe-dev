use recipe::*;

fn main() {
	let bundle = format!("/home/nate/.local/share/claude-harness/worktrees/claude-harness-2/.claude-harness/review-367/repro-{}.ogdl", std::process::id());
	let data = recipe.data("data/ordinal/string_levels.csv")
		.target("target")
		.norm(z_score);
	let model = recipe.model()
		.knn(3).prelu().iq(3).m
		.kmeans(2).exp().qi(2).k
		.loss(mae);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.0001)
		.seed(57143384586813)
		.epochs(1)
		.log(all)
		.stop(0.8)
		.int(8)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
}
