use recipe::*;
fn main() {
	let path = std::env::var("DATA").unwrap();
	let targets: Vec<String> = std::env::var("TARGETS").unwrap_or("target".into()).split(',').map(String::from).collect();
	let refs: Vec<&str> = targets.iter().map(String::as_str).collect();
	let data = recipe.data(path.as_str()).target(refs.as_slice());
	let model = recipe.model().layer(16).layer(1).loss(mse);
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(2).fp(32).run(&model, &data);
	eprintln!("RESULT data={path} initial={} final={} predictions={}", report.initial_loss(), report.final_loss(), report.predictions().len());
}
