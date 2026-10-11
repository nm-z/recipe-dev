use recipe::*;

fn env(name: &str) -> String { std::env::var(name).unwrap_or_else(|_| panic!("{name} is unset")) }

fn main() {
	let path = env("AUDIT_DATA");
	let targets: Vec<String> = env("AUDIT_TARGETS").split(',').map(|v| v.to_owned()).collect();
	let bundle = env("AUDIT_BUNDLE");
	let width = targets.len();
	let data = recipe.data(path.as_str()).target(targets);
	let model = recipe.model().layer(4).relu().layer(width).loss(mse);
	let report = recipe.train().optimizer(adamw).lr(0.01).seed(17).epochs(1).save(&bundle).run(&model, &data);
	println!("OK rows_predictions={}", report.predictions().len());
}
