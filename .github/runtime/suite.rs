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

/// `small.csv` is `linear.csv` with both columns scaled by this factor, so the
/// relation is `y = 3x + 7 * NARROW_SCALE`. Its gradients sit near the bottom of
/// the fp16 range, where a narrow accumulator would round them to zero.
const NARROW_SCALE: f64 = 1.0 / 1024.0;
/// Worst error, in units of `y`, a narrow format may leave on `small.csv`.
const NARROW_TOLERANCE: f64 = 1.0;

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

fn prediction_bits(values: &[f64]) -> Vec<u64> { values.iter().map(|value| value.to_bits()).collect() }

fn nonzero_tile(extent: [u32; 3]) -> bool { extent.iter().all(|value| *value != 0) }

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


fn small_targets() -> Vec<f64> {
	linear_targets().into_iter().map(|value| value * NARROW_SCALE).collect()
}

fn gguf_text(out: &mut Vec<u8>, value: &str) {
	out.extend_from_slice(&(value.len() as u64).to_le_bytes());
	out.extend_from_slice(value.as_bytes());
}

/// Writes a synthetic checkpoint: a native Q8_0 n-gram row table, a three-tap
/// convolution dilated by the order-3 n-gram, and dense tensors for the stream
/// around one per-layer embedding block.
fn write_stepper_checkpoint(path: &Path) {
	let tensors: Vec<(&str, Vec<f32>)> = vec![
		("token_embd.weight", vec![1.0, 2.0]),
		("blk.0.attn_q.weight", vec![0.0]),
		("blk.0.attn_k.weight", vec![0.0]),
		("blk.0.attn_v.weight", vec![0.0]),
		("blk.0.attn_output.weight", vec![0.0]),
		("blk.0.ffn_gate.weight", vec![2.0]),
		("blk.0.ffn_up.weight", vec![3.0]),
		("blk.0.ffn_down.weight", vec![1.0]),
		("output.weight", vec![1.0, -1.0]),
		("ngram.table", Vec::new()),
		("blk.0.ple_key.weight", vec![0.01; 64]),
		("blk.0.ple_norm_key.weight", vec![1.0]),
		("blk.0.ple_norm_query.weight", vec![1.0]),
		("blk.0.ple_value.weight", vec![0.02; 64]),
		("blk.0.ple_norm_conv.weight", vec![1.0]),
		("blk.0.ple_conv1d.weight", vec![0.6, -0.2, 0.1]),
	];
	let mut metadata = Vec::new();
	for (name, value) in [("general.architecture", "llama"), ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "gpt-2")] {
		gguf_text(&mut metadata, name);
		metadata.extend_from_slice(&8u32.to_le_bytes());
		gguf_text(&mut metadata, value);
	}
	for (name, value) in [("llama.context_length", 160u32), ("tokenizer.ggml.bos_token_id", 0), ("tokenizer.ggml.eos_token_id", 1), ("ngram.heads", 1), ("ngram.kernel", 3), ("ngram.layer", 0)] {
		gguf_text(&mut metadata, name);
		metadata.extend_from_slice(&4u32.to_le_bytes());
		metadata.extend_from_slice(&value.to_le_bytes());
	}
	gguf_text(&mut metadata, "tokenizer.ggml.add_bos_token");
	metadata.extend_from_slice(&7u32.to_le_bytes());
	metadata.push(0);
	for (name, values) in [("tokenizer.ggml.tokens", &["a", "b"][..]), ("tokenizer.ggml.merges", &[][..])] {
		gguf_text(&mut metadata, name);
		metadata.extend_from_slice(&9u32.to_le_bytes());
		metadata.extend_from_slice(&8u32.to_le_bytes());
		metadata.extend_from_slice(&(values.len() as u64).to_le_bytes());
		for value in values {
			gguf_text(&mut metadata, value);
		}
	}
	let mut out = b"GGUF".to_vec();
	out.extend_from_slice(&3u32.to_le_bytes());
	out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
	out.extend_from_slice(&12u64.to_le_bytes());
	out.extend(metadata);
	let mut data = Vec::new();
	for (name, values) in &tensors {
		gguf_text(&mut out, name);
		out.extend_from_slice(&2u32.to_le_bytes());
		let table = *name == "ngram.table";
		let columns: u64 = match *name {
			"ngram.table" => 32,
			"blk.0.ple_key.weight" | "blk.0.ple_value.weight" => 64,
			"blk.0.ple_conv1d.weight" => 3,
			_ => 1,
		};
		let rows: u64 = if table { 4 } else { values.len() as u64 / columns };
		out.extend_from_slice(&columns.to_le_bytes());
		out.extend_from_slice(&rows.to_le_bytes());
		out.extend_from_slice(&(if table { 8u32 } else { 0 }).to_le_bytes());
		out.extend_from_slice(&(data.len() as u64).to_le_bytes());
		if table {
			for code in [32i8, 96, -80, 48] {
				data.extend_from_slice(&0x2400u16.to_le_bytes());
				data.extend_from_slice(&[code as u8; 32]);
			}
		} else {
			for value in values {
				data.extend_from_slice(&value.to_le_bytes());
			}
		}
		while data.len() % 32 != 0 {
			data.push(0);
		}
	}
	while out.len() % 32 != 0 {
		out.push(0);
	}
	out.extend(data);
	std::fs::write(path, out).expect("cannot write the stepper checkpoint");
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
	let model = recipe.model().layer(1).fp(PRECISION_BITS).loss(mse);
	let bundle = work.join("linear.ogdl");
	let training = recipe.train().seed(SEED).lr(RATE).epochs(EPOCHS).save(&bundle);
	let training = if std::env::var_os("RECIPE_SUITE_PROGRESS").is_some() { training.log([Epoch, Loss, debug]) } else { training };
	let trained = training.run(&model, &linear);
	let device = trained.memory.first().and_then(|line| line.split_whitespace().next()).expect("training reported no device memory");
	println!("suite device {device}");
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
	let multi_model = recipe.model().layer(1).fp(PRECISION_BITS).loss(mse);
	let multi_trained = recipe.train().seed(SEED).lr(RATE).epochs(EPOCHS).run(&multi_model, &multi);
	assert_eq!(multi_trained.predictions().len(), MULTI_ROWS, "multi.csv did not produce one prediction per row");
	let multi_worst = worst_absolute(&sorted(multi_trained.predictions().iter().copied()), &sorted(multi_targets()));
	report.record("multi_feature_reduction", multi_worst <= CLOSED_FORM_TOLERANCE, format!("worst_abs_err={multi_worst:.9} rows={MULTI_ROWS} features={}", MULTI_WEIGHTS.len()));

	// 4. The same seed must reproduce the report bits and tile on the same backend.
	let first = recipe.train().seed(SEED).lr(RATE).epochs(120).run(&model, &linear);
	let second = recipe.train().seed(SEED).lr(RATE).epochs(120).run(&model, &linear);
	let first_tile = first.tile();
	let second_tile = second.tile();
	let predictions_identical = prediction_bits(first.predictions()) == prediction_bits(second.predictions());
	let identical = first.initial_loss().to_bits() == second.initial_loss().to_bits()
		&& first.final_loss().to_bits() == second.final_loss().to_bits()
		&& predictions_identical
		&& first_tile == second_tile
		&& nonzero_tile(first_tile);
	report.record(
		"determinism",
		identical,
		format!(
			"first_bits={:016x} second_bits={:016x} initial_bits={:016x}/{:016x} prediction_bits_equal={} tile={:?}/{:?}",
			first.final_loss().to_bits(),
			second.final_loss().to_bits(),
			first.initial_loss().to_bits(),
			second.initial_loss().to_bits(),
			predictions_identical,
			first_tile,
			second_tile,
		),
	);

	// 5. Persistence: a resumed run starts from the state training saved, and
	//    inference reads the bundle without rewriting it.
	let saved = std::fs::read(&bundle).expect("training did not save a bundle");
	assert!(!saved.is_empty(), "training saved an empty bundle");
	let resumed = recipe.train().seed(SEED).lr(RATE).epochs(60).resume(&bundle).save(&bundle).run(&model, &linear);
	let resumed_bytes = std::fs::read(&bundle).expect("resume did not save a bundle");
	let trained_tile = trained.tile();
	let resumed_tile = resumed.tile();
	let resume_predictions_match = prediction_bits(resumed.initial_predictions()) == prediction_bits(trained.predictions());
	let resume_matches = resumed.initial_loss().is_finite()
		&& resumed.initial_loss().to_bits() == trained.final_loss().to_bits()
		&& resume_predictions_match
		&& !resumed_bytes.is_empty()
		&& trained_tile == resumed_tile
		&& nonzero_tile(trained_tile);
	report.record(
		"persistence_resume",
		resume_matches,
		format!(
			"resume_initial_loss={:.9} bundle_bytes={} resume_initial_bits={:016x} trained_final_bits={:016x} prediction_bits_equal={} tile={:?}/{:?}",
			resumed.initial_loss(),
			resumed_bytes.len(),
			resumed.initial_loss().to_bits(),
			trained.final_loss().to_bits(),
			resume_predictions_match,
			trained_tile,
			resumed_tile,
		),
	);

	// 6. Inference through the persisted bundle, point by point against the
	//    closed form, and the bundle must be unchanged afterwards.
	let mut inference_worst = 0.0_f64;
	let mut points = Vec::new();
	for point in [-4.0, -1.5, 0.0, 2.0, 5.25] {
		let produced = recipe.predict(&bundle, &[point]);
		assert_eq!(produced.len(), 1, "inference returned {} values for one input", produced.len());
		let want = LINEAR_SLOPE * point + LINEAR_INTERCEPT;
		inference_worst = inference_worst.max((produced[0] - want).abs());
		points.push(format!("{{\"input\":{point},\"produced\":{:.9},\"expected\":{want:.9}}}", produced[0]));
	}
	let after = std::fs::read(&bundle).expect("inference removed the bundle");
	report.record("inference_closed_form", inference_worst <= CLOSED_FORM_TOLERANCE, format!("worst_abs_err={inference_worst:.9} tolerance={CLOSED_FORM_TOLERANCE}"));
	report.record("inference_is_read_only", after == resumed_bytes, format!("bundle_bytes_before={} bundle_bytes_after={}", resumed_bytes.len(), after.len()));


	// 7. Narrow precisions: gradients near the bottom of the fp16 range must
	//    survive, and a saved bundle must resume from exactly the state it left,
	//    at every precision family the bundle stores.
	let small_source = root.join("data/small.csv");
	let small = recipe.data(small_source.to_str().expect("small path is not UTF-8")).target("target");
	let narrow = [
		("narrow_fp16", recipe.model().layer(1).fp(16).loss(mse)),
		("narrow_bf16", recipe.model().layer(1).bf(16).loss(mse)),
	];
	for (name, narrow_model) in &narrow {
		let saved = work.join(format!("{name}.ogdl"));
		let run = recipe.train().seed(SEED).lr(RATE).epochs(EPOCHS).save(&saved).run(narrow_model, &small);
		let worst = worst_absolute(&sorted(run.predictions().iter().copied()), &sorted(small_targets())) / NARROW_SCALE;
		let resumed = recipe.train().seed(SEED).lr(RATE).epochs(10).resume(&saved).save(&saved).run(narrow_model, &small);
		let resumes = resumed.initial_loss().is_finite() && resumed.initial_loss() <= 2.0 * run.final_loss();
		report.record(*name, worst <= NARROW_TOLERANCE && resumes, format!("worst_abs_err_in_y={worst:.9} tolerance={NARROW_TOLERANCE} resumes={resumes} final_loss={:.9e} resume_initial_loss={:.9e}", run.final_loss(), resumed.initial_loss()));
	}

	// 8. A per-layer embedding stepper: a whole sequence, a prefill, and a
	//    prefill followed by single steps must produce the same logits bits.
	let stepper = work.join("stepper.gguf");
	write_stepper_checkpoint(&stepper);
	let file = recipe.gguf(stepper.to_str().expect("stepper path is not UTF-8"));
	let table = file.ngram();
	let gate = layer(1).fp(64).silu().fp(64);
	let up = layer(1).fp(64);
	let embedding_math = PleMath {
		key_norm: BlockNormalization::Rms,
		query_norm: BlockNormalization::Rms,
		output_norm: BlockNormalization::Rms,
		gate: PleGate::signed_root_sigmoid(1e-6, true),
		convolution: Activation::Silu,
	};
	let stream = recipe.model().no(bias).embed(2, 1).fp(64).ple(&table).ple_math(embedding_math).fp(64);
	let stepped_model = stream.res([attn(1).fp(64)]).fp(64).res([(gate * up).fp(64), layer(1).fp(64)]).fp(64).layer(2).fp(64);
	let placed = file.place(&stepped_model, 160, &[]);
	let ids: Vec<u32> = (0..160).map(|index| (index % 2) as u32).collect();
	let whole = placed.infer(&ids.iter().map(|id| f64::from(*id)).collect::<Vec<_>>());
	let direct = placed.prefill(&ids[..13]);
	let prefix = placed.prefill(&ids[..12]);
	let prefix_then_step = placed.step(ids[12]);
	let mut splits_match = prediction_bits(&direct) == prediction_bits(&prefix_then_step) && prediction_bits(&direct) != prediction_bits(&prefix);
	for split in [1usize, 2, 4, 8, 16, 32, 64, 128] {
		let mut logits = placed.prefill(&ids[..split]);
		for id in &ids[split..] {
			logits = placed.step(*id);
		}
		splits_match &= prediction_bits(&logits) == prediction_bits(&whole);
	}
	report.record("ple_prefill_steps_match_whole", splits_match && whole.iter().all(|value| value.is_finite()), format!("positions=160 logits={}", whole.len()));

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
