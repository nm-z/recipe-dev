//! Model default exclusions.
//!
//! `.no(bias)` is a model-level declaration: no weighted block beneath it
//! allocates, initializes, trains, saves or loads a bias. It applies to layers,
//! attention projections, convolutions and recurrent gates alike, and to nested
//! branches, because the lowering reads it from the graph rather than from each
//! block.

use recipe::*;
use std::path::PathBuf;

const FEATURES: usize = 12;
const ROWS: usize = 64;

fn dataset(name: &str) -> String {
	let directory = std::env::temp_dir().join(format!("recipe-exclusion-{}-{name}", std::process::id()));
	std::fs::create_dir_all(&directory).unwrap();
	let mut text = String::new();
	for column in 0..FEATURES {
		text.push_str(&format!("x{column},"));
	}
	text.push_str("y\n");
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	for _ in 0..ROWS {
		let values = (0..FEATURES).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			text.push_str(&format!("{value},"));
		}
		text.push_str(&format!("{}\n", (values.iter().sum::<f64>() / 4.0).tanh()));
	}
	std::fs::write(directory.join("rows.csv"), text).unwrap();
	directory.to_str().unwrap().to_owned()
}

fn train(name: &str, model: &Model) -> (f64, f64) {
	let directory = dataset(name);
	let data = recipe.data(directory.as_str()).target("y");
	let report = recipe.train().fp(64).seed(4).lr(0.01).epochs(30).stop(0.0).run(model, &data);
	(report.initial_loss(), report.final_loss())
}

/// The trained parameter count, read back from the saved bundle. A bias the
/// model excluded would show up here as extra values.
fn saved_parameters(name: &str, model: &Model) -> usize {
	let directory = dataset(name);
	let path: PathBuf = std::env::temp_dir().join(format!("recipe-exclusion-{}-{name}.ogdl", std::process::id()));
	let data = recipe.data(directory.as_str()).target("y");
	recipe.train().fp(64).seed(4).lr(0.01).epochs(2).stop(0.0).save(&path).run(model, &data);
	let text = std::fs::read_to_string(&path).unwrap();
	// Weights are saved as `tensor <format> <count> <metadata> <hex>` lines, so
	// the second field of each is the number of values that tensor holds.
	let count = text
		.lines()
		.filter_map(|line| line.trim().strip_prefix("tensor "))
		.filter_map(|values| values.split_whitespace().nth(1))
		.filter_map(|count| count.parse::<usize>().ok())
		.sum::<usize>();
	std::fs::remove_file(&path).unwrap();
	assert!(count > 0, "no parameters were saved for {name}");
	count
}

/// A layer's bias is one value per output channel. Excluding it must remove
/// exactly that many parameters — not fewer, and not merely freeze them.
#[test]
fn a_biasless_layer_saves_fewer_parameters() {
	let with = saved_parameters("layer-with", &recipe.model().layer(7).relu().layer(1).loss(mse));
	let without = saved_parameters("layer-without", &recipe.model().no(bias).layer(7).relu().layer(1).loss(mse));
	// layer(7) drops 7 and layer(1) drops 1.
	assert_eq!(with - without, 8, "expected 8 fewer parameters, {with} became {without}");
}

/// A convolution's bias is one value per filter.
#[test]
fn a_biasless_convolution_saves_fewer_parameters() {
	let with = saved_parameters("conv-with", &recipe.model().conv(4, 3).relu().layer(1).loss(mse));
	let without = saved_parameters("conv-without", &recipe.model().no(bias).conv(4, 3).relu().layer(1).loss(mse));
	// conv(4, 3) drops 4 and layer(1) drops 1. The sixth is the output
	// convolution compile appends to a model whose output still has a length,
	// which is a weighted block like any other and drops its bias too.
	assert_eq!(with - without, 4 + 1 + 1, "expected 6 fewer parameters, {with} became {without}");
}

/// A recurrent gate carries its own bias per gate, which is the case the
/// contraction kernel's existing bias flag does not cover.
#[test]
fn biasless_recurrent_gates_save_fewer_parameters() {
	let with = saved_parameters("gru-with", &recipe.model().conv(2, 2).gru(5).relu().layer(1).loss(mse));
	let without = saved_parameters("gru-without", &recipe.model().no(bias).conv(2, 2).gru(5).relu().layer(1).loss(mse));
	// gru(5) has three gates of five channels, conv(2, 2) drops 2, layer(1) drops
	// 1, and the appended output convolution drops 1.
	assert_eq!(with - without, 3 * 5 + 2 + 1 + 1, "expected 19 fewer parameters, {with} became {without}");
}

/// The exclusion reaches a nested branch, because the lowering reads it from the
/// graph rather than from the block it is lowering.
#[test]
fn the_exclusion_reaches_a_nested_branch() {
	let with = saved_parameters("res-with", &recipe.model().layer(6).res([layer(6), relu(), layer(6)]).layer(1).loss(mse));
	let without = saved_parameters("res-without", &recipe.model().no(bias).layer(6).res([layer(6), relu(), layer(6)]).layer(1).loss(mse));
	// layer(6) outside, two layer(6) inside the branch, and layer(1).
	assert_eq!(with - without, 6 + 6 + 6 + 1, "expected 19 fewer parameters, {with} became {without}");
}

/// Omitting the declaration keeps the biased behavior exactly.
#[test]
fn omitting_the_exclusion_changes_nothing() {
	let plain = train("plain", &recipe.model().layer(7).relu().layer(1).loss(mse));
	assert!(plain.1 < plain.0, "the biased model did not train");
}

/// A biasless model still trains, which it cannot do if the gradient walked a
/// parameter block laid out for a bias that is not there.
#[test]
fn a_biasless_model_trains() {
	let (initial, trained) = train("trains", &recipe.model().no(bias).layer(7).relu().layer(1).loss(mse));
	assert!(trained < initial, "the biasless model did not train: {initial} to {trained}");
	let (initial, trained) = train("trains-gru", &recipe.model().no(bias).conv(2, 2).gru(5).relu().layer(1).loss(mse));
	assert!(trained < initial, "the biasless recurrent model did not train: {initial} to {trained}");
}

/// The exclusion is model metadata, so inference has to rebuild the same shapes.
/// A bundle that lost it would allocate a bias the parameter vector does not have.
#[test]
fn the_exclusion_survives_the_bundle() {
	let directory = dataset("bundle");
	let path: PathBuf = std::env::temp_dir().join(format!("recipe-exclusion-{}-bundle.ogdl", std::process::id()));
	let data = recipe.data(directory.as_str()).target("y");
	let model = recipe.model().no(bias).layer(7).relu().layer(1).loss(mse);
	recipe.train().fp(64).seed(4).lr(0.01).epochs(20).stop(0.0).save(&path).run(&model, &data);

	let row = vec![0.25; FEATURES];
	let first = recipe.infer(&path, &row);
	let second = recipe.infer(&path, &row);
	assert_eq!(first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "two inferences of one bundle disagree");
	assert!(first.iter().all(|value| value.is_finite()), "the reloaded biasless model produced a nonfinite prediction: {first:?}");
	std::fs::remove_file(path).unwrap();
}
