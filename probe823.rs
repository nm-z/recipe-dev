use recipe::*;
fn main() {
	let data = recipe.data("data/numeric/sample_subfolders").target("target");
	let model = recipe.model().layer(16).res([layer(16), relu()]).moe(1, [layer(8)]).ensemble([layer(4), layer(4)]).layer(1).loss(mse);
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(1).fp(32).run(&model, &data);
	eprintln!("RESULT loss={} prediction={}", report.final_loss(), report.predictions()[0]);
}
