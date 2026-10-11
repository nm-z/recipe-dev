use recipe::*;
fn main() {
	let data = recipe.data("data/numeric/sample_subfolders").target("target");
	let model = recipe.model().layer(32).relu().layer(16).norm(rms).res([layer(16), relu()]).layer(8).tanh().layer(1).loss(mse);
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(2).fp(32).run(&model, &data);
	eprintln!("RESULT loss={}", report.final_loss());
}
