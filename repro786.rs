use recipe::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn input_width(path: &Path) -> usize {
	let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
	let mut shape = text.lines().find_map(|line| line.trim().strip_prefix("shape ")).expect("saved model has no input shape").split_whitespace();
	let channels = shape.next().expect("saved model has no input channels").parse::<usize>().expect("saved input channels are invalid");
	let length = shape.next().expect("saved model has no input length").parse::<usize>().expect("saved input length is invalid");
	channels.checked_mul(length).expect("saved input width overflows")
}

fn main() {
	let root = std::env::var_os("RECIPE_TRIAL_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
	let bundle = root.join(format!("composition-candidate-{}.ogdl", std::process::id()));
	let data = recipe.data("data/numeric/sample_subfolders").target("target");
	let case = std::env::var("CASE").unwrap_or("786".into());
	let model = recipe.model().layer(16);
	let model = match case.as_str() {
		"786" => model.res([pool(1).norm(rms), layer(24), lgbm(8).norm(layer), bayes(), svm().act(Activation::Log).quantize(0, 8, 3), cbst(8).norm(rms)]).layer(1).loss(mse),
		"781" => model.res([pool(1), perc(32), norm(l2), knn(3).quantize(0, 5, 3), pool(1), pool(1), xgbst(2), svm()]).ln().norm(layer).layer(1).loss(mse),
		"775" => model.res([gru(8), perc(32).norm(batch).quantize(1, 4, 5), pool(1).act(Activation::Tanh), knn(3).norm(rms)]).layer(1).loss(mse),
		"one-est" => model.res([svm()]).layer(1).loss(mse),
		"two-est" => model.res([svm(), bayes()]).layer(1).loss(mse),
		_ => panic!("unknown case"),
	};
	let start = Instant::now();
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(1).log(all).fp(32).save(&bundle).run(&model, &data);
	let train_seconds = start.elapsed().as_secs_f64();
	let input = vec![0.0; input_width(&bundle)];
	let start = Instant::now();
	let output = recipe.infer(&bundle, &input);
	let infer_seconds = start.elapsed().as_secs_f64();
	eprintln!("RESULT case={case} train_seconds={train_seconds:.1} infer_seconds={infer_seconds:.1} loss={} output={:?}", report.final_loss(), output);
	std::fs::remove_file(&bundle).unwrap_or_else(|error| panic!("cannot remove {}: {error}", bundle.display()));
}
