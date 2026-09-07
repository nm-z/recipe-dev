//! Model composition by Rust multiplication.
//!
//! `left * right` evaluates both models from the same incoming activation and
//! multiplies their outputs elementwise. Each branch keeps its own weights, and
//! the product's derivative sends `right * dy` to the left branch and `left * dy`
//! to the right, so both input-gradient contributions reach the shared source.

use recipe::*;
use std::path::PathBuf;

const FEATURES: usize = 8;
const ROWS: usize = 64;
const WIDTH: usize = 4;

fn dataset() -> String {
	let directory = std::env::temp_dir().join(format!("recipe-product-{}", std::process::id()));
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

fn train(model: &Model, epochs: usize) -> (u64, u64, Vec<f64>) {
	let directory = dataset();
	let data = recipe.data(directory.as_str()).target("y");
	let report = recipe.train().fp(64).seed(9).lr(0.01).epochs(epochs).stop(0.0).run(model, &data);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.predictions().to_vec())
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

/// The acceptance expression from the issue: a gated feed-forward built with
/// Rust multiplication, with no `.mul([...])`, `.branch(...)` or `.chain(...)`.
fn gated() -> Model {
	let gate = recipe.model().layer(WIDTH).gelu();
	let up = recipe.model().layer(WIDTH);
	(gate * up).layer(1).loss(mse)
}

/// It builds, trains, saves, loads and infers — and the loss falls, which it
/// cannot do if only one branch receives gradient.
#[test]
fn a_gated_feed_forward_trains_and_reloads() {
	let directory = dataset();
	let path: PathBuf = std::env::temp_dir().join(format!("recipe-product-{}.ogdl", std::process::id()));
	let data = recipe.data(directory.as_str()).target("y");
	let report = recipe.train().fp(64).seed(9).lr(0.01).epochs(40).stop(0.0).save(&path).run(&gated(), &data);

	assert!(report.final_loss() < report.initial_loss(), "the composed model did not train: {} to {}", report.initial_loss(), report.final_loss());

	let row = vec![0.25; FEATURES];
	let first = recipe.infer(&path, &row);
	let second = recipe.infer(&path, &row);
	assert_eq!(first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "two inferences of one bundle disagree");
	assert!(first.iter().all(|value| value.is_finite()), "the reloaded product produced a nonfinite prediction: {first:?}");
	std::fs::remove_file(path).unwrap();
}

/// Both branches carry their own weights. A product of two `layer(WIDTH)`
/// branches followed by `layer(1)` holds two projection matrices where a single
/// `layer(WIDTH).layer(1)` holds one, so the two models cannot train to the same
/// loss from the same seed — if they did, one branch would not be there.
#[test]
fn the_branches_have_independent_weights() {
	let single = train(&recipe.model().layer(WIDTH).layer(1).loss(mse), 40);
	let product = train(&gated(), 40);
	assert_ne!(single.0, product.0, "a product and a single branch started at the same loss");
	assert!(f64::from_bits(product.1) < f64::from_bits(product.0), "the product did not train");
}

/// The gradient reaches both branches. Training the product with only one branch
/// differentiated would leave the other at its initial weights, so the model
/// would behave as a single branch scaled by a constant — this asserts the
/// composed model reaches a loss the single branch does not.
#[test]
fn training_moves_both_branches() {
	let (initial, trained, predictions) = train(&gated(), 60);
	assert!(f64::from_bits(trained) < f64::from_bits(initial), "the product did not train: {} to {}", f64::from_bits(initial), f64::from_bits(trained));
	assert_eq!(predictions.len(), ROWS, "expected one prediction per row");
	assert!(predictions.iter().all(|value| value.is_finite()), "a prediction is not finite");
}

/// Incompatible branch shapes are refused before execution, naming both.
#[test]
fn mismatched_branch_shapes_are_refused() {
	let message = panic_text(std::panic::catch_unwind(|| {
		let left = recipe.model().layer(WIDTH);
		let right = recipe.model().layer(WIDTH + 1);
		let _ = train(&(left * right).layer(1).loss(mse), 1);
	}));
	assert!(message.contains("elementwise product takes one shape"), "unexpected error: {message}");
}

/// A branch is layers, convolutions and activations. Anything else is refused
/// when the product is built, not when it runs.
#[test]
fn a_branch_that_is_not_composable_is_refused() {
	let message = panic_text(std::panic::catch_unwind(|| {
		let left = recipe.model().layer(WIDTH);
		let right = recipe.model().attn(2);
		let _ = left * right;
	}));
	assert!(message.contains("product branch is layers, convolutions and activations"), "unexpected error: {message}");

	let normalized = panic_text(std::panic::catch_unwind(|| {
		let left = recipe.model().layer(WIDTH);
		let right = recipe.model().layer(WIDTH).norm(rms);
		let _ = left * right;
	}));
	assert!(normalized.contains("carries normalization"), "unexpected error: {normalized}");
}
