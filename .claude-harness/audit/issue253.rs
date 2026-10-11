use recipe::*;

fn main() {
	let bundle = format!("/home/nate/.local/share/claude-harness/worktrees/claude-harness-12/.claude-harness/audit/issue253-{}.ogdl", std::process::id());
	let data = recipe.data("data/ordinal/string_levels.csv")
		.target("target");
	let model = recipe.model()
		.pool(2).exp().iq(2).xs
		.layer(8)
		.moe(1, [layer(8), layer(8)]).ln().qi(8).0
		.loss(mae);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(26771436245506)
		.epochs(1)
		.log(all)
		.stop(0.8)
		.int(8)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
	std::fs::remove_file(&bundle).ok();
}
