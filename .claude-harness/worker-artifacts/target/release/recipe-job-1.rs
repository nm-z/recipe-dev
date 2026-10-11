use recipe::*;

fn main() {
	let data = recipe.data("/home/nate/.local/share/claude-harness/worktrees/claude-harness-14/data/numeric/single_csv.csv").target("target");
	let model = recipe.model().layer(256).relu().layer(256).relu().layer(1).loss(mse);
	let report = recipe.train().seed(17).epochs(8).lr(0.001).stop(0.0).run(&model, &data);
	println!("VERDICT loss {:.12} r2 {:.12} epoch_seconds {:.6}", report.final_loss(), report.r2(), report.epoch_seconds());
}
