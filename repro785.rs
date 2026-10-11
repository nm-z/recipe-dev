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
	let case = std::env::var("CASE").unwrap_or("785".into());
	let model = recipe.model().layer(16);
	let model = match case.as_str() {
		"785" => model.moe(1, [svm()]).exp().layer(1).loss(mse),
		"783" => model.res([svm().act(Activation::Exp).quantize(1, 3, 3)]).tan().layer(1).loss(mse),
		"787" => model.res([svm().act(Activation::Exp)]).qi(5).k.s.layer(1).loss(mse),
		"svm-exp" => model.res([svm().act(Activation::Exp)]).layer(1).loss(mse),
		"svm" => model.moe(1, [svm()]).layer(1).loss(mse),
		"layer-exp" => model.moe(1, [layer(16)]).exp().layer(1).loss(mse),
		_ => panic!("unknown case"),
	};
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(1).log(all).fp(32).save(&bundle).run(&model, &data);
	eprintln!("RESULT case={case} loss={} initial={} finite={}", report.final_loss(), report.initial_loss(), report.final_loss().is_finite());
	let input = vec![0.0; input_width(&bundle)];
	let output = recipe.infer(&bundle, &input);
	eprintln!("RESULT case={case} output={:?}", output);
	std::fs::remove_file(&bundle).unwrap_or_else(|error| panic!("cannot remove {}: {error}", bundle.display()));
}
