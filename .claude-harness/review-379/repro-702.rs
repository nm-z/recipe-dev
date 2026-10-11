use recipe::*;

fn main() {
	let bundle = "/home/nate/.local/share/claude-harness/worktrees/claude-harness-6/.claude-harness/review-379/repro-702.ogdl";
	let data = recipe.data("data/temporal/series_sqlite.sqlite")
		.target("target")
		.split(0.8);
	let model = recipe.model()
		.pool(2).sigmoid().qi(4).1
		.gru(8).tanh().norm(batch).qi(8).1
		.loss(bce);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(20662076354778)
		.epochs(1)
		.log(all)
		.stop(0.0)
		.tf(32)
		.save(bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
}
