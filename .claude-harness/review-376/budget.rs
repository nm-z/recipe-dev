use recipe::*;

fn main() {
	let data = recipe.data("/home/nate/.local/share/claude-harness/worktrees/claude-harness-16/data/numeric/single_csv.csv").target("target");
	let model = recipe.model().layer(64).relu().layer(1).loss(mse);
	let report = recipe.train().optimizer(adamw).lr(0.0092).seed(17).epochs(2).fp(32).log(all).run(&model, &data);
	println!("initial_loss={} final_loss={} r2={} tile={:?}", report.initial_loss(), report.final_loss(), report.r2(), report.tile());
}
