use recipe::*;

fn main() {
	let bundle = format!("/home/nate/.local/share/claude-harness/worktrees/claude-harness-13/.claude-harness/repro254-{}.ogdl", std::process::id());
	let data = recipe.data("data/numeric/samples_sqlite.sqlite")
		.target("target");
	let model = recipe.model()
		.bayes().prelu().norm(batch).qi(6).k
		.perc(8).cos().norm(batch).qi(3).k.m
		.loss(huber);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.0001)
		.seed(104592427676686)
		.epochs(1)
		.log(all)
		.stop(0.0)
		.fp(64)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
	println!("final_loss={}", report.final_loss());
}
