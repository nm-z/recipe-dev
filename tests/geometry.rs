//! Attention head geometry: the head width is declared, not derived.
//!
//! `attn(heads)` alone splits the residual width evenly over the heads, so the
//! residual width has to divide by the head count. `.width(d)` states the width
//! instead, and the block projects `heads * d` back to the residual width on the
//! way out. The two must agree exactly where they overlap, and the declared form
//! must accept geometries the derived one cannot express.

use recipe::*;
use std::path::PathBuf;

const FEATURES: usize = 16;
const ROWS: usize = 48;

fn dataset() -> String {
	let directory = std::env::temp_dir().join(format!("recipe-geometry-{}", std::process::id()));
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
		let sum = values.iter().enumerate().map(|(index, value)| value * (index as f64 / 4.0).cos()).sum::<f64>();
		text.push_str(&format!("{}\n", (sum / 4.0).tanh()));
	}
	std::fs::write(directory.join("rows.csv"), text).unwrap();
	directory.to_str().unwrap().to_owned()
}

fn evidence(model: &Model) -> (u64, u64, Vec<u64>) {
	let directory = dataset();
	let data = recipe.data(directory.as_str()).target("y");
	let report = recipe.train().fp(32).seed(17).lr(0.01).epochs(20).stop(0.0).run(model, &data);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.predictions().iter().map(|value| value.to_bits()).collect())
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

/// The default is exactly the derivation it replaced. `FEATURES / 4` is the width
/// `attn(4)` has always used, so declaring it must not move a single bit — if it
/// does, the declared path is not a generalization of the derived one.
#[test]
fn a_declared_width_equal_to_the_derived_one_changes_nothing() {
	let derived = evidence(&recipe.model().attn(4).relu().layer(1).loss(mse));
	let declared = evidence(&recipe.model().attn(4).width(FEATURES / 4).relu().layer(1).loss(mse));
	assert_eq!(derived, declared, "declaring the derived head width changed the result");
}

/// The same, with the gate on. Every kernel defines the gate plane as the query
/// plane, so this is the case that catches a gate sized from the residual width
/// rather than from `heads * width`: it would shift the row stride and corrupt
/// the indexer reads with no error anywhere.
#[test]
fn a_declared_width_changes_nothing_with_the_gate_on() {
	let derived = evidence(&recipe.model().attn(4).gate().relu().layer(1).loss(mse));
	let declared = evidence(&recipe.model().attn(4).width(FEATURES / 4).gate().relu().layer(1).loss(mse));
	assert_eq!(derived, declared, "declaring the derived head width changed the gated result");
}

/// The point of the change. 16 does not divide by 3, so `attn(3)` alone cannot
/// express this geometry at all; with a declared width the head count and the
/// residual width are independent.
#[test]
fn a_declared_width_frees_the_residual_width_from_the_head_count() {
	let message = panic_text(std::panic::catch_unwind(|| {
		let _ = evidence(&recipe.model().attn(3).relu().layer(1).loss(mse));
	}));
	assert!(message.contains("attention head partition is invalid"), "unexpected error: {message}");

	let (initial, final_loss, predictions) = evidence(&recipe.model().attn(3).width(5).relu().layer(1).loss(mse));
	assert_eq!(predictions.len(), ROWS, "expected one prediction per row");
	assert!(f64::from_bits(final_loss) < f64::from_bits(initial), "loss did not fall: {} to {}", f64::from_bits(initial), f64::from_bits(final_loss));
}

/// The qwen4exp partition in miniature: a head count that does not divide the
/// residual width, a head width wider than `channels / heads`, untied key-value
/// heads, query and key normalization, partial rotary and the output gate, all at
/// once. This is the geometry the derived width cannot reach.
#[test]
fn an_untied_geometry_trains() {
	let model = recipe.model().attn(6).width(8).kv(2).qk(rms).rope(4, 10000.0).gate().relu().layer(1).loss(mse);
	let (initial, final_loss, predictions) = evidence(&model);
	assert_eq!(predictions.len(), ROWS, "expected one prediction per row");
	assert!(f64::from_bits(final_loss) < f64::from_bits(initial), "loss did not fall: {} to {}", f64::from_bits(initial), f64::from_bits(final_loss));
}

/// A width that is not written into the bundle is a width inference rebuilds
/// differently, which reads the saved parameters at the wrong offsets.
#[test]
fn the_declared_width_survives_the_bundle() {
	let directory = dataset();
	let path = std::env::temp_dir().join(format!("recipe-geometry-{}.ogdl", std::process::id()));
	let data = recipe.data(directory.as_str()).target("y");
	let model = recipe.model().attn(3).width(5).kv(1).gate().relu().layer(1).loss(mse);
	let report = recipe.train().fp(32).seed(17).lr(0.01).epochs(20).stop(0.0).save(&path).run(&model, &data);

	let row = vec![0.25; FEATURES];
	let reloaded = recipe.infer(&path, &row);
	assert_eq!(reloaded.len(), 1, "expected one prediction");
	// The training predictions are on the training rows, so this only asserts the
	// reload produced a finite prediction of the right shape from a graph that
	// rebuilt at the declared width; a wrong width panics on the parameter count
	// long before this line.
	assert!(reloaded[0].is_finite(), "reloaded prediction is not finite: {}", reloaded[0]);
	assert!(report.final_loss().is_finite());
	std::fs::remove_file(path).unwrap();
}
