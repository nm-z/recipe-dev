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
	let widths: Vec<usize> = std::env::var("WIDTHS").unwrap_or("16,16,16,16,16,16,16,16".into()).split(',').map(|v| v.parse().unwrap()).collect();
	let top_k: usize = std::env::var("TOPK").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
	let root = std::env::var_os("RECIPE_TRIAL_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
	let bundle = root.join(format!("composition-candidate-{}.ogdl", std::process::id()));
	let data = recipe.data("data/numeric/sample_subfolders").target("target");
	let l = |i: usize| layer(widths[i]);
	let model = recipe.model().layer(16);
	let model = match widths.len() {
		1 => model.moe(top_k, [l(0)]),
		2 => model.moe(top_k, [l(0), l(1)]),
		3 => model.moe(top_k, [l(0), l(1), l(2)]),
		4 => model.moe(top_k, [l(0), l(1), l(2), l(3)]),
		5 => model.moe(top_k, [l(0), l(1), l(2), l(3), l(4)]),
		6 => model.moe(top_k, [l(0), l(1), l(2), l(3), l(4), l(5)]),
		7 => model.moe(top_k, [l(0), l(1), l(2), l(3), l(4), l(5), l(6)]),
		8 => model.moe(top_k, [l(0), l(1), l(2), l(3), l(4), l(5), l(6), l(7)]),
		n => panic!("unsupported expert count {n}"),
	};
	let model = model.layer(1).loss(mse);
	let start = Instant::now();
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(17)
		.epochs(1)
		.log(all)
		.fp(32)
		.save(&bundle)
		.run(&model, &data);
	let train_seconds = start.elapsed().as_secs_f64();
	assert!(report.final_loss().is_finite());
	let bytes = std::fs::metadata(&bundle).map(|m| m.len()).unwrap_or(0);
	let input = vec![0.0; input_width(&bundle)];
	let start = Instant::now();
	let output = recipe.infer(&bundle, &input);
	let infer_seconds = start.elapsed().as_secs_f64();
	assert!(!output.is_empty());
	assert!(output.iter().all(|value| value.is_finite()));
	eprintln!("RESULT widths={} top_k={top_k} train_seconds={train_seconds:.1} infer_seconds={infer_seconds:.1} loss={} bundle_bytes={bytes} output={:?}", widths.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(","), report.final_loss(), output);
	std::fs::remove_file(&bundle).unwrap_or_else(|error| panic!("cannot remove {}: {error}", bundle.display()));
}
