//! The hyper-connection mixer, checked against the two properties #673 states.
//!
//! `hyper(1, 0, ...)` must stay the plain residual, bit for bit, so the mixer
//! cannot change what a model without lanes computes. And a nonzero rank must
//! hold exactly the mixer parameters the lowering declares, so a checkpoint
//! that carries `hc_*` tensors binds without a spare bias row.

use recipe::*;
use std::fmt::Write as _;

fn dataset(rows: usize, columns: usize) -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-hyper-{rows}x{columns}.csv"));
	if path.exists() {
		return path;
	}
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	let mut text = String::with_capacity(rows * (columns + 1) * 8);
	for column in 0..columns {
		let _ = write!(text, "x{column},");
	}
	text.push_str("y\n");
	for _ in 0..rows {
		for _ in 0..columns {
			let _ = write!(text, "{:.6},", random());
		}
		let _ = writeln!(text, "{:.6}", random());
	}
	std::fs::write(&path, text).unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
	path
}

/// The bundle records the run, so the comparison covers every line that is not
/// a property of when it ran.
const VOLATILE: &[&str] = &["artifact", "run", "seconds", "time"];

fn stable_bundle(path: &std::path::Path) -> String {
	let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
	let mut kept = String::with_capacity(text.len());
	for line in text.lines() {
		let key = line.trim_start().split_whitespace().next().unwrap_or_default();
		if !VOLATILE.contains(&key) {
			kept.push_str(line);
			kept.push('\n');
		}
	}
	kept
}

fn train(name: &str, model: &Model, rows: usize, columns: usize) -> (u64, u64, Vec<u64>, String) {
	let bundle = std::env::temp_dir().join(format!("recipe-hyper-{name}-{}.ogdl", std::process::id()));
	let data = recipe.data(dataset(rows, columns).to_string_lossy().as_ref()).target("y");
	let report = recipe.train().optimizer(adamw).lr(0.01).seed(17).epochs(3).fp(32).save(&bundle).run(model, &data);
	let predictions = report.predictions().iter().map(|value| value.to_bits()).collect();
	let text = stable_bundle(&bundle);
	let _ = std::fs::remove_file(&bundle);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), predictions, text)
}

/// Every parameter the bundle records: each `tensor` line states how many
/// values it carries before the values themselves.
fn weight_count(bundle: &str) -> usize {
	let mut total = 0;
	for line in bundle.lines() {
		let mut fields = line.trim_start().split_whitespace();
		if fields.next() != Some("tensor") {
			continue;
		}
		let mut fields = fields.skip(1);
		let count = fields.next().unwrap_or_else(|| panic!("a tensor line states no length: {line}"));
		total += count.parse::<usize>().unwrap_or_else(|error| panic!("a tensor length is malformed: {error}"));
	}
	assert!(total != 0, "the bundle records no tensors");
	total
}

/// `hyper(1, 0, &m)` adds no normalization, fixes every gate at one, and reads
/// the mean of a single lane, so it must be the plain residual bit for bit.
#[test]
fn static_single_lane_is_the_plain_residual() {
	let hyper = recipe.model().layer(9).relu().hyper(1, 0, &recipe.model().layer(9).relu().layer(9)).relu().layer(1).loss(mse);
	let residual = recipe.model().layer(9).relu().res([layer(9), relu(), layer(9)]).relu().layer(1).loss(mse);
	let (hyper_initial, hyper_final, hyper_predictions, hyper_bundle) = train("static", &hyper, 131, 17);
	let (residual_initial, residual_final, residual_predictions, residual_bundle) = train("residual", &residual, 131, 17);
	assert_eq!(hyper_initial, residual_initial, "initial loss differs: hyper {hyper_initial:016x}, residual {residual_initial:016x}");
	assert_eq!(hyper_final, residual_final, "final loss differs: hyper {hyper_final:016x}, residual {residual_final:016x}");
	assert_eq!(hyper_predictions.len(), residual_predictions.len(), "prediction count differs");
	for (index, (left, right)) in hyper_predictions.iter().zip(&residual_predictions).enumerate() {
		assert_eq!(left, right, "prediction {index} differs: hyper {left:016x}, residual {right:016x}");
	}
	assert_eq!(weight_count(&hyper_bundle), weight_count(&residual_bundle), "a static hyper block holds mixer parameters the residual does not");
}

/// The mixer holds `stream + 2 stream rank + stream lanes` parameters per block
/// and `stream + 2 stream rank` at the head, and no projection carries a bias.
#[test]
fn a_ranked_mixer_holds_exactly_its_declared_parameters() {
	let (lanes, rank, width) = (4_usize, 3_usize, 8_usize);
	let stream = lanes * width;
	let branch = || recipe.model().layer(width).relu();
	let mixed = recipe.model().layer(width).hyper(lanes, rank, &branch()).layer(1).loss(mse);
	let plain = recipe.model().layer(width).hyper(lanes, 0, &branch()).layer(1).loss(mse);
	let (.., mixed_bundle) = train("mixed", &mixed, 131, 17);
	let (.., plain_bundle) = train("plain", &plain, 131, 17);
	// One block and the head: the block carries a read gate, a write gate and a
	// scale; the head carries a read gate and a scale. No bias rows anywhere.
	let block = stream + 2 * stream * rank + stream * lanes;
	let head = stream + 2 * stream * rank;
	assert_eq!(
		weight_count(&mixed_bundle) - weight_count(&plain_bundle),
		block + head,
		"a rank {rank} mixer over {lanes} lanes of {width} does not hold {block} mixer parameters per block and {head} at the head"
	);
}
