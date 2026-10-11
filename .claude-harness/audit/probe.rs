use recipe::*;

fn env(name: &str) -> String { std::env::var(name).unwrap_or_else(|_| panic!("{name} is unset")) }

fn main() {
	let path = env("AUDIT_DATA");
	let targets: Vec<String> = env("AUDIT_TARGETS").split(',').map(|v| v.to_owned()).collect();
	let out = env("AUDIT_OUT");
	let bundle_a = format!("{out}/stage.ogdl");
	let bundle_b = format!("{out}/final.ogdl");
	let width = targets.len();
	let mut data = recipe.data(path.as_str()).target(targets.clone());
	if let Ok(test) = std::env::var("AUDIT_TEST") {
		data = data.test(test.as_str());
	} else {
		data = data.split(0.8);
	}
	if std::env::var("AUDIT_EXCLUDE").is_ok() {
		data = data.exclude(env("AUDIT_EXCLUDE").split(',').map(|v| v.to_owned()).collect::<Vec<_>>());
	}
	let model = recipe.model().layer(8).relu().layer(width).loss(mse);
	let first = recipe.train().optimizer(adamw).lr(0.01).seed(17).epochs(5).log(all).save(&bundle_a).run(&model, &data);
	println!("STAGE initial={:.6} final={:.6} predictions={}", first.initial_loss(), first.final_loss(), first.predictions().len());
	let second = recipe.train().optimizer(adamw).lr(0.01).seed(17).epochs(5).log(all).resume(&bundle_a).save(&bundle_b).run(&model, &data);
	println!("RESUME initial={:.6} final={:.6} predictions={}", second.initial_loss(), second.final_loss(), second.predictions().len());
	let features: usize = env("AUDIT_FEATURES").parse().expect("AUDIT_FEATURES");
	let input: Vec<f64> = (0..features).map(|i| 0.5 + i as f64 * 0.1).collect();
	let output = recipe.infer(&bundle_b, &input);
	println!("INFER declared_targets={width} output_len={} values={:?}", output.len(), output);
	println!("DONE");
}
