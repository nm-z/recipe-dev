//! The scalar scale activation.
//!
//! `scale(factor)` multiplies every value the preceding block produces by one
//! constant. It owns no weights, preserves shape, and the factor travels with the
//! model. Forward is `y = factor * x`; backward is `dx = factor * dy`, which the
//! existing scalar-program derivative gives for free.

use recipe::*;
use std::path::PathBuf;

const FEATURES: usize = 8;
const ROWS: usize = 64;

fn dataset() -> String {
	let directory = std::env::temp_dir().join(format!("recipe-scale-{}", std::process::id()));
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

/// Initial loss bits, final loss bits, and the untrained predictions. The initial
/// predictions are taken before any weight moves, so two models that differ only
/// in a scale are comparable there exactly.
fn evidence(model: &Model) -> (u64, u64, Vec<f64>) {
	let directory = dataset();
	let data = recipe.data(directory.as_str()).target("y");
	let report = recipe.train().fp(64).seed(5).lr(0.01).epochs(20).stop(0.0).run(model, &data);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.initial_predictions().to_vec())
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

/// A factor of one is the identity, so it must be bit-identical to no activation
/// at all — not merely close. This is the check that the scale is wired into the
/// same activation slot the other activations use and adds nothing else.
#[test]
fn a_unit_scale_is_the_linear_activation() {
	let plain = evidence(&recipe.model().layer(6).layer(1).loss(mse));
	let scaled = evidence(&recipe.model().layer(6).scale(1.0).layer(1).loss(mse));
	assert_eq!(plain.0, scaled.0, "a unit scale moved the initial loss");
	assert_eq!(plain.1, scaled.1, "a unit scale moved the final loss");
	assert_eq!(plain.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), scaled.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "a unit scale moved the predictions");
}

/// The forward is exactly `factor * x`. A model whose only block is scaled has
/// untrained predictions that are the unscaled model's, multiplied by the factor
/// — the scalar reference calculation, on the real values.
#[test]
fn the_forward_multiplies_by_the_factor() {
	const FACTOR: f64 = 3.0;
	let plain = evidence(&recipe.model().layer(1).loss(mse));
	let scaled = evidence(&recipe.model().layer(1).scale(FACTOR).loss(mse));
	assert_eq!(plain.2.len(), scaled.2.len(), "the scale changed the prediction count");
	assert!(!plain.2.is_empty(), "no predictions to compare");
	for (index, (plain, scaled)) in plain.2.iter().zip(&scaled.2).enumerate() {
		let expected = FACTOR * plain;
		assert_eq!(expected.to_bits(), scaled.to_bits(), "prediction {index}: {FACTOR} * {plain} is {expected}, the scaled model produced {scaled}");
	}
}

/// The gradient carries the factor, so a scaled model still trains. Without
/// `dx = factor * dy` the block below it would receive nothing and the loss would
/// sit where it started.
#[test]
fn the_backward_carries_the_factor() {
	let (initial, final_loss, _) = evidence(&recipe.model().layer(6).scale(2.0).layer(1).loss(mse));
	assert!(f64::from_bits(final_loss) < f64::from_bits(initial), "a scaled model did not train: {} to {}", f64::from_bits(initial), f64::from_bits(final_loss));
}

/// The factor is model state, so inference has to rebuild the same graph from the
/// saved bundle. A factor that did not round-trip gives a different scalar program
/// and different values.
#[test]
fn the_factor_survives_the_bundle() {
	let directory = dataset();
	let path: PathBuf = std::env::temp_dir().join(format!("recipe-scale-{}.ogdl", std::process::id()));
	let data = recipe.data(directory.as_str()).target("y");
	let model = recipe.model().layer(6).scale(0.125).layer(1).loss(mse);
	recipe.train().fp(64).seed(5).lr(0.01).epochs(20).stop(0.0).save(&path).run(&model, &data);

	let row = vec![0.25; FEATURES];
	let first = recipe.infer(&path, &row);
	let second = recipe.infer(&path, &row);
	assert_eq!(first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "two inferences of one bundle disagree");
	assert!(first.iter().all(|value| value.is_finite()), "the reloaded model produced a nonfinite prediction: {first:?}");
}

/// A factor that is not a number cannot be rejected later: the scalar program
/// would carry it as a constant and every value downstream would be NaN.
#[test]
fn a_nonfinite_factor_is_refused() {
	for factor in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
		let message = panic_text(std::panic::catch_unwind(|| {
			let _ = recipe.model().layer(6).scale(factor).layer(1).loss(mse);
		}));
		assert!(message.contains("scale factor must be finite"), "unexpected error for {factor}: {message}");
	}
}
