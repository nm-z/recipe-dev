use recipe::*;
use std::path::PathBuf;
fn main() {
	let root = std::env::var_os("RECIPE_TRIAL_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
	let bundle = root.join(format!("place-probe-{}.ogdl", std::process::id()));
	let data = recipe.data("data/numeric/sample_subfolders").target("target");
	let model = recipe.model().layer(32).relu().layer(16).norm(rms).res([layer(16), relu()]).layer(8).tanh().layer(1).loss(mse);
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(2).fp(32).save(&bundle).run(&model, &data);
	let input: Vec<f64> = (0..477).map(|i| (i as f64 * 0.01).sin()).collect();
	let single = recipe.infer(&bundle, &input);
	let split: Vec<usize> = std::env::var("SPLIT").map(|s| s.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or_default();
	let placed = recipe.place(&bundle, &split);
	let output = placed.infer(&input);
	let bits = |v: &Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
	eprintln!("RESULT loss={} split={:?} resident={:?} moved={} single={single:?} placed={output:?} bit_identical={}", report.final_loss(), placed.split(), placed.resident_bytes(), placed.moved_bytes(), bits(&single) == bits(&output));
	std::fs::remove_file(&bundle).unwrap();
}
