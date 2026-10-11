use recipe::*;
fn main() {
	let data = recipe.data(auto).set("/home/nate/.local/share/claude-harness/worktrees/claude-harness-4/.claude-harness/recipe-73-text-413266.txt");
	let model = recipe.model().lstm(8).layer(1).loss(mse);
	let report = recipe.train().seed(17).epochs(6).fp(8).run(&model, &data);
	println!("report loss={:.12} r2={:.12} epoch-seconds={:.6}", report.final_loss(), report.r2(), report.epoch_seconds());
}
