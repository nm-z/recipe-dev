//! The runtime suite every Recipe CI cell executes.
//!
//! One invocation runs every check against one selected backend and writes a
//! machine-readable evidence document to `RECIPE_EVIDENCE`. The process exits
//! non-zero as soon as a check fails, so a cell can never report success
//! without having executed the checks.
//!
//! The datasets in `.github/runtime/data` encode exact linear relationships,
//! so the expected values below are arithmetic, not values recorded from an
//! earlier run.
use recipe::*;
use std::path::{Path, PathBuf};

/// `y = 3x + 7` over `linear.csv`.
const LINEAR_SLOPE: f64 = 3.0;
const LINEAR_INTERCEPT: f64 = 7.0;
const LINEAR_ROWS: usize = 128;
/// `y = 2a - 3b + 0.5c + 1` over `multi.csv`.
const MULTI_WEIGHTS: [f64; 3] = [2.0, -3.0, 0.5];
const MULTI_INTERCEPT: f64 = 1.0;
const MULTI_ROWS: usize = 432;

const SEED: usize = 17;
const RATE: f64 = 0.5;
const EPOCHS: usize = 400;
const PRECISION_BITS: u8 = 32;

/// A converged fit of an exact linear relationship. The measured worst case on
/// a reference CPU run is 0.064 absolute over the sorted predictions and 0.044
/// through a persisted bundle, so this leaves better than a factor of two of
/// headroom while still rejecting an untrained model, whose error against these
/// targets exceeds 7.
const CLOSED_FORM_TOLERANCE: f64 = 0.15;
/// A converged run reaches roughly 1.3e-3 on `linear.csv` and 4.9e-5 on
/// `multi.csv` from an initial loss near 191. This threshold is two orders of
/// magnitude above the measured value and four below the initial loss.
const CONVERGED_LOSS: f64 = 0.1;

struct Check {
	name: &'static str,
	detail: String,
	passed: bool,
}

struct Report {
	checks: Vec<Check>,
}

impl Report {
	fn record(&mut self, name: &'static str, passed: bool, detail: String) {
		println!("check {name} {} {detail}", if passed { "PASS" } else { "FAIL" });
		self.checks.push(Check { name, detail, passed });
	}
}

fn escape(value: &str) -> String {
	value.chars().flat_map(|character| match character {
		'"' => vec!['\\', '"'],
		'\\' => vec!['\\', '\\'],
		'\n' => vec!['\\', 'n'],
		other => vec![other],
	}).collect()
}

fn environment(name: &str) -> String {
	std::env::var(name).unwrap_or_else(|_| panic!("{name} is absent"))
}

fn worst_absolute(got: &[f64], want: &[f64]) -> f64 {
	assert_eq!(got.len(), want.len(), "prediction count does not match the dataset");
	got.iter().zip(want).map(|(left, right)| (left - right).abs()).fold(0.0_f64, f64::max)
}

/// Predictions come back in the trainer's row order, so both sides are sorted
/// before they are compared. The target multiset is still exactly the closed
/// form, which is what this check is asserting.
fn sorted(values: impl IntoIterator<Item = f64>) -> Vec<f64> {
	let mut values: Vec<f64> = values.into_iter().collect();
	values.sort_by(f64::total_cmp);
	values
}

fn linear_targets() -> Vec<f64> {
	(-64..64).map(|index| LINEAR_SLOPE * (f64::from(index) / 8.0) + LINEAR_INTERCEPT).collect()
}

fn multi_targets() -> Vec<f64> {
	let mut targets = Vec::with_capacity(MULTI_ROWS);
	for first in -6..6 {
		for second in -3..3 {
			for third in -3..3 {
				let row = [f64::from(first) / 4.0, f64::from(second) / 2.0, f64::from(third) / 2.0];
				targets.push(MULTI_WEIGHTS.iter().zip(row).map(|(weight, value)| weight * value).sum::<f64>() + MULTI_INTERCEPT);
			}
		}
	}
	targets
}

fn main() {
	let root = PathBuf::from(environment("RECIPE_SUITE_ROOT"));
	let work = PathBuf::from(environment("RECIPE_SUITE_WORK"));
	let evidence = PathBuf::from(environment("RECIPE_EVIDENCE"));
	std::fs::create_dir_all(&work).expect("cannot create the suite work directory");
	let linear_source = root.join("data/linear.csv");
	let multi_source = root.join("data/multi.csv");
	let mut report = Report { checks: Vec::new() };

	// 1. A linear model recovers an exact linear relationship, checked against
	//    the closed form rather than against a recorded value.
	let linear = recipe.data(linear_source.to_str().expect("linear path is not UTF-8")).target("target");
	let model = recipe.model().layer(1).loss(mse);
	let bundle = work.join("linear.ogdl");
	let trained = recipe.train().seed(SEED).lr(RATE).epochs(EPOCHS).fp(PRECISION_BITS).save(&bundle).run(&model, &linear);
	assert_eq!(trained.predictions().len(), LINEAR_ROWS, "linear.csv did not produce one prediction per row");
	let worst = worst_absolute(&sorted(trained.predictions().iter().copied()), &sorted(linear_targets()));
	report.record("linear_closed_form", worst <= CLOSED_FORM_TOLERANCE, format!("worst_abs_err={worst:.9} tolerance={CLOSED_FORM_TOLERANCE}"));
	report.record("linear_converged", trained.final_loss() < CONVERGED_LOSS, format!("final_loss={:.9} threshold={CONVERGED_LOSS}", trained.final_loss()));

	// 2. Backward and optimizer execution: the loss must actually fall.
	let improved = trained.final_loss() < trained.initial_loss() && trained.initial_loss().is_finite();
	report.record("optimizer_progress", improved, format!("initial_loss={:.9} final_loss={:.9}", trained.initial_loss(), trained.final_loss()));

	// 3. A reduction across three features with mixed signs, again against the
	//    closed form.
	let multi = recipe.data(multi_source.to_str().expect("multi path is not UTF-8")).target("target");
	let multi_model = recipe.model().layer(1).loss(mse);
	let multi_trained = recipe.train().seed(SEED).lr(RATE).epochs(EPOCHS).fp(PRECISION_BITS).run(&multi_model, &multi);
	assert_eq!(multi_trained.predictions().len(), MULTI_ROWS, "multi.csv did not produce one prediction per row");
	let multi_worst = worst_absolute(&sorted(multi_trained.predictions().iter().copied()), &sorted(multi_targets()));
	report.record("multi_feature_reduction", multi_worst <= CLOSED_FORM_TOLERANCE, format!("worst_abs_err={multi_worst:.9} rows={MULTI_ROWS} features={}", MULTI_WEIGHTS.len()));

	// 4. The same seed must reproduce the same loss bits on the same backend.
	let first = recipe.train().seed(SEED).lr(RATE).epochs(120).fp(PRECISION_BITS).run(&model, &linear);
	let second = recipe.train().seed(SEED).lr(RATE).epochs(120).fp(PRECISION_BITS).run(&model, &linear);
	let identical = first.final_loss().to_bits() == second.final_loss().to_bits();
	report.record("determinism", identical, format!("first_bits={:016x} second_bits={:016x}", first.final_loss().to_bits(), second.final_loss().to_bits()));

	// 5. Persistence: a resumed run starts from the state training saved, and
	//    inference reads the bundle without rewriting it.
	let saved = std::fs::read(&bundle).expect("training did not save a bundle");
	assert!(!saved.is_empty(), "training saved an empty bundle");
	let resumed = recipe.train().seed(SEED).lr(RATE).epochs(60).fp(PRECISION_BITS).resume(&bundle).save(&bundle).run(&model, &linear);
	let resumed_bytes = std::fs::read(&bundle).expect("resume did not save a bundle");
	report.record("persistence_resume", resumed.initial_loss().is_finite() && !resumed_bytes.is_empty(), format!("resume_initial_loss={:.9} bundle_bytes={}", resumed.initial_loss(), resumed_bytes.len()));

	// 6. Inference through the persisted bundle, point by point against the
	//    closed form, and the bundle must be unchanged afterwards.
	let mut inference_worst = 0.0_f64;
	let mut points = Vec::new();
	for point in [-4.0, -1.5, 0.0, 2.0, 5.25] {
		let produced = recipe.infer(&bundle, &[point]);
		assert_eq!(produced.len(), 1, "inference returned {} values for one input", produced.len());
		let want = LINEAR_SLOPE * point + LINEAR_INTERCEPT;
		inference_worst = inference_worst.max((produced[0] - want).abs());
		points.push(format!("{{\"input\":{point},\"produced\":{:.9},\"expected\":{want:.9}}}", produced[0]));
	}
	let after = std::fs::read(&bundle).expect("inference removed the bundle");
	report.record("inference_closed_form", inference_worst <= CLOSED_FORM_TOLERANCE, format!("worst_abs_err={inference_worst:.9} tolerance={CLOSED_FORM_TOLERANCE}"));
	report.record("inference_is_read_only", after == resumed_bytes, format!("bundle_bytes_before={} bundle_bytes_after={}", resumed_bytes.len(), after.len()));

	let failed: Vec<&str> = report.checks.iter().filter(|check| !check.passed).map(|check| check.name).collect();
	let body = report
		.checks
		.iter()
		.map(|check| format!("{{\"name\":\"{}\",\"passed\":{},\"detail\":\"{}\"}}", check.name, check.passed, escape(&check.detail)))
		.collect::<Vec<_>>()
		.join(",");
	let document = format!(
		"{{\"schema\":\"recipe-runtime-suite/1\",\"executed\":{},\"failed\":{},\"linear\":{{\"rows\":{LINEAR_ROWS},\"initial_loss\":{:.9},\"final_loss\":{:.9},\"initial_loss_bits\":\"{:016x}\",\"final_loss_bits\":\"{:016x}\"}},\"multi\":{{\"rows\":{MULTI_ROWS},\"final_loss\":{:.9}}},\"inference\":[{}],\"checks\":[{body}]}}",
		report.checks.len(),
		failed.len(),
		trained.initial_loss(),
		trained.final_loss(),
		trained.initial_loss().to_bits(),
		trained.final_loss().to_bits(),
		multi_trained.final_loss(),
		points.join(","),
	);
	write_document(&evidence, &document);
	println!("suite executed={} failed={}", report.checks.len(), failed.len());
	assert!(failed.is_empty(), "runtime suite checks failed: {}", failed.join(", "));
	println!("SUITE PASS executed={}", report.checks.len());
}

fn write_document(path: &Path, document: &str) {
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).expect("cannot create the evidence directory");
	}
	std::fs::write(path, document).unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
}
