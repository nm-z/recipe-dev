//! YaRN frequency scaling on a rotary embedding.
//!
//! YaRN blends two frequencies per rotated dimension: the unscaled one, kept for
//! the fast-rotating dimensions, and the same one divided by the context
//! extension factor, used for the slow ones. The blend runs over a ramp between
//! the two rotation boundaries.
//!
//! It scales the frequency, not the value, so the rotation and its reverse are
//! the ones `rope` already had and the backward carries no new term.

use recipe::*;
use std::fmt::Write as _;

const COLUMNS: usize = 16;

fn dataset() -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-yarn-{}.csv", std::process::id()));
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	let mut text = (0..COLUMNS).fold(String::new(), |mut text, column| {
		let _ = write!(text, "x{column},");
		text
	});
	text.push_str("y\n");
	for _ in 0..131 {
		let values = (0..COLUMNS).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			let _ = write!(text, "{value:.6},");
		}
		let _ = writeln!(text, "{:.6}", values.iter().enumerate().map(|(index, value)| value * (index + 1) as f64 / 64.0).sum::<f64>());
	}
	std::fs::write(&path, text).unwrap();
	path
}

fn evidence(model: &Model, tag: &str, path: &std::path::Path) -> (u64, u64, Vec<u64>) {
	let bundle = std::env::temp_dir().join(format!("recipe-yarn-{}-{tag}.ogdl", std::process::id()));
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let report = recipe.train().fp(64).seed(21).lr(0.01).epochs(20).stop(0.0).save(&bundle).run(model, &data);
	let _ = std::fs::remove_file(&bundle);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.predictions().iter().map(|value| value.to_bits()).collect())
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

/// A factor of one extends the context by nothing, so every frequency is the one
/// rope already used and the model must be bit-identical to the unscaled one.
#[test]
fn a_factor_of_one_matches_unscaled_rope() {
	let path = dataset();
	let plain = evidence(&recipe.model().conv(4, 3).attn(2).rope(neox, 4, 10000.0).relu().layer(1).loss(mse), "plain", &path);
	let unity = evidence(&recipe.model().conv(4, 3).attn(2).rope(neox, 4, 10000.0).yarn(1.0, 8192, 32.0, 1.0).relu().layer(1).loss(mse), "unity", &path);
	assert_eq!(plain, unity, "a yarn factor of one changed the model");
	let _ = std::fs::remove_file(path);
}

/// A real extension changes the frequencies, so it must change the result. If it
/// did not, the scaling would not be reaching the kernel at all.
#[test]
fn an_extension_factor_changes_the_result() {
	let path = dataset();
	let plain = evidence(&recipe.model().conv(4, 3).attn(2).rope(neox, 4, 10000.0).relu().layer(1).loss(mse), "plain2", &path);
	let scaled = evidence(&recipe.model().conv(4, 3).attn(2).rope(neox, 4, 10000.0).yarn(4.0, 8192, 32.0, 1.0).relu().layer(1).loss(mse), "scaled", &path);
	assert_ne!(plain.2, scaled.2, "a yarn factor of four produced the unscaled predictions");
	assert!(f64::from_bits(scaled.1) < f64::from_bits(scaled.0), "the scaled model did not train");
	let _ = std::fs::remove_file(path);
}

/// The four values are model state, so inference has to rebuild the same
/// frequencies. A bundle that lost them would rotate at the unscaled rate.
#[test]
fn the_scaling_survives_the_bundle() {
	let path = dataset();
	let bundle = std::env::temp_dir().join(format!("recipe-yarn-{}-bundle.ogdl", std::process::id()));
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let model = recipe.model().conv(4, 3).attn(2).rope(neox, 4, 10000.0).yarn(4.0, 8192, 32.0, 1.0).relu().layer(1).loss(mse);
	recipe.train().fp(64).seed(21).lr(0.01).epochs(20).stop(0.0).save(&bundle).run(&model, &data);

	let saved = std::fs::read_to_string(&bundle).unwrap();
	let attention = saved.lines().find(|line| line.contains("attn,")).unwrap_or_else(|| panic!("no attention block in the bundle"));
	for value in ["4", "8192", "32", "1"] {
		assert!(attention.contains(value), "the saved attention block does not carry {value}: {attention}");
	}

	let first = recipe.infer(&bundle, &[0.25; COLUMNS]);
	let second = recipe.infer(&bundle, &[0.25; COLUMNS]);
	assert_eq!(first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "two inferences of one bundle disagree");
	let _ = std::fs::remove_file(bundle);
	let _ = std::fs::remove_file(path);
}

/// YaRN configures a rope. Without one there is no frequency to scale.
#[test]
fn yarn_needs_a_rope() {
	let message = panic_text(std::panic::catch_unwind(|| {
		let _ = recipe.model().attn(2).yarn(4.0, 8192, 32.0, 1.0);
	}));
	assert!(message.contains("yarn configures a preceding rope"), "unexpected error: {message}");
}

/// Every value is checked when the model is built, not when it runs.
#[test]
fn invalid_scaling_is_refused() {
	let rope = || recipe.model().attn(2).rope(neox, 4, 10000.0);
	for (factor, context, fast, slow, expected) in [
		(0.5, 8192, 32.0, 1.0, "yarn factor must be finite and at least one"),
		(f64::NAN, 8192, 32.0, 1.0, "yarn factor must be finite and at least one"),
		(4.0, 0, 32.0, 1.0, "yarn original context must be positive"),
		(4.0, 8192, f64::INFINITY, 1.0, "yarn boundaries must be finite"),
		(4.0, 8192, 1.0, 32.0, "must exceed the slow boundary"),
		(4.0, 8192, 8.0, 8.0, "must exceed the slow boundary"),
	] {
		let message = panic_text(std::panic::catch_unwind(move || {
			let _ = rope().yarn(factor, context, fast, slow);
		}));
		assert!(message.contains(expected), "for ({factor}, {context}, {fast}, {slow}) expected {expected:?}, got: {message}");
	}
}
