use recipe::*;
use std::path::{Path, PathBuf};

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
	let model = recipe.model()
		.layer(16)
		.moe(1, [
			lgbm(8).packed().norm(l2),
			rnn(24).packed(),
		])
		.layer(1)
		.loss(mse);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(17)
		.epochs(1)
		.log(all)
		.fp(32)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
	println!("final_loss={}", report.final_loss());
	let input = vec![0.0; input_width(&bundle)];
	let output = recipe.infer(&bundle, &input);
	assert!(!output.is_empty());
	assert!(output.iter().all(|value| value.is_finite()));
	println!("bundle_bytes={} output={:?}", std::fs::metadata(&bundle).map(|m| m.len()).unwrap_or(0), output);
	std::fs::remove_file(&bundle).unwrap_or_else(|error| panic!("cannot remove {}: {error}", bundle.display()));
}
