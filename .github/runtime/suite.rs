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
/// The gated delta model trains for a few epochs at a small rate, which is enough for its loss to fall.
const DELTA_RATE: f64 = 0.01;
const DELTA_EPOCHS: usize = 20;

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
	gguf_text(&mut metadata, "tokenizer.ggml.suppress_tokens");
	metadata.extend_from_slice(&9u32.to_le_bytes());
	metadata.extend_from_slice(&4u32.to_le_bytes());
	metadata.extend_from_slice(&1u64.to_le_bytes());
	metadata.extend_from_slice(&0u32.to_le_bytes());
	let mut out = b"GGUF".to_vec();
	out.extend_from_slice(&3u32.to_le_bytes());
	out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
	out.extend_from_slice(&13u64.to_le_bytes());
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
	if let Ok(mode) = std::env::var("RECIPE_SUITE_CONTRACT") {
		contract_child(&mode, &root, &work);
		return;
	}
	let trace_path = std::env::var_os("RECIPE_TRACE_PATH").map(PathBuf::from).unwrap_or_else(|| work.join("phases.log"));
	// Set the log destination before Recipe starts any workers.
	unsafe { std::env::set_var("RECIPE_TRACE_PATH", &trace_path); }
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
	let mut sampler = recipe.sampler().temperature(0.0);
	let generated = placed.decode(&[0], &mut sampler, &[], 8);
	let excludes = whole[0] > whole[1] && sampler.suppressed() == [0] && generated.ids.len() == 9 && generated.ids[1..].iter().all(|id| !sampler.suppressed().contains(id));
	report.record("sampler_suppression", excludes, format!("suppressed={:?} generated={:?}", sampler.suppressed(), &generated.ids[1..]));


	// 9. A gated delta model trains at each precision: it is not refused, its loss is finite and falls,
	//    and the same seed repeats the report bits.
	for (name, bits) in [("delta_fp32", 32_u8), ("delta_fp16", 16_u8)] {
		let delta = recipe.model().layer(32).fp(bits).delta(1, 2).fp(bits).keys(1, 32).values(32).out(32).delta_norms(l2, rms).delta_activations(Activation::Silu, Activation::Sigmoid).delta_gates(Activation::Softplus, Activation::Sigmoid).layer(1).fp(bits).loss(mse);
		let first = recipe.train().seed(SEED).lr(DELTA_RATE).epochs(DELTA_EPOCHS).run(&delta, &multi);
		let second = recipe.train().seed(SEED).lr(DELTA_RATE).epochs(DELTA_EPOCHS).run(&delta, &multi);
		let falls = first.initial_loss().is_finite() && first.final_loss().is_finite() && first.final_loss() < first.initial_loss();
		let repeats = first.initial_loss().to_bits() == second.initial_loss().to_bits() && first.final_loss().to_bits() == second.final_loss().to_bits() && prediction_bits(first.predictions()) == prediction_bits(second.predictions());
		report.record(name, falls && repeats, format!("initial_loss={:.9} final_loss={:.9} final_bits={:016x} repeats={repeats}", first.initial_loss(), first.final_loss(), first.final_loss().to_bits()));
	}

	additional_contracts(&mut report, &root, &work);
	// The shared device-clock writer must preserve full-window and step outputs.
	let _ = recipe.infer().log([debug]);
	let traced = file.place(&stepped_model, 160, &[]);
	let traced_whole = traced.infer(&ids.iter().map(|id| f64::from(*id)).collect::<Vec<_>>());
	traced.prefill(&ids[..159]);
	let traced_step = traced.step(ids[159]);
	let unchanged = prediction_bits(&traced_whole) == prediction_bits(&whole) && prediction_bits(&traced_step) == prediction_bits(&whole);
	let trace = std::fs::read_to_string(&trace_path).expect("traced placement did not write its log");
	let window = trace.lines().rev().find(|line| line.starts_with("range window ") && line.contains("positions=159..160"));
	let seconds = window.and_then(|line| line.split_whitespace().find_map(|field| field.strip_prefix("window_seconds=")?.parse::<f64>().ok()));
	let clocks = trace.lines().rev().find(|line| line.starts_with("clocks window 159..160 "));
	let stamps = clocks.map(|line| line.split_whitespace().filter(|field| field.starts_with('n') && field.contains(':')).collect::<Vec<_>>()).unwrap_or_default();
	let measured = !stamps.is_empty() && stamps.iter().all(|field| field.split_once(':').is_some_and(|(_, ticks)| ticks.parse::<u64>().is_ok()))
		&& seconds.is_some_and(|seconds| seconds.is_finite() && seconds > 0.0) && window.is_some_and(|line| line.contains("device=")) && clocks.is_some_and(|line| line.contains("device="));
	report.record("traced_window_and_step", unchanged && measured, format!("output_bits_equal={unchanged} node_clocks={} window_seconds={seconds:?}", stamps.len()));
	if traced.split().len() == 1 {
		report.record("single_device_has_no_exchange", !trace.lines().any(|line| line.starts_with("exchange phase ")), "ordinary placement has no exchange phase records".to_owned());
	} else {
		println!("check single_device_has_no_exchange SKIP {} selected devices", traced.split().len());
	}

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
fn write_features(path: &Path) {
	let mut csv = (0..32).map(|column| format!("x{column}")).collect::<Vec<_>>().join(",");
	csv.push_str(",target\n");
	for row in 0..64 {
		let values = (0..32).map(|column| ((row * (column * 2 + 1) + column) % 31) as f64 / 32.0 - 0.5).collect::<Vec<_>>();
		let target = values.iter().sum::<f64>() / 32.0;
		csv.push_str(&values.iter().map(ToString::to_string).collect::<Vec<_>>().join(","));
		csv.push_str(&format!(",{target}\n"));
	}
	std::fs::write(path, csv).expect("cannot write 32-feature data");
}

fn write_large(path: &Path) {
	let mut csv = String::from("x,target\n");
	for index in -64..64 {
		let x = f64::from(index) / 8.0;
		csv.push_str(&format!("{},{}\n", x * 256.0, (LINEAR_SLOPE * x + LINEAR_INTERCEPT) * 256.0));
	}
	std::fs::write(path, csv).expect("cannot write large-gradient data");
}

fn contract_model(kind: &str, bits: u8, brain: bool) -> Model {
	let format = |model: Model| if brain { model.bf(16) } else { model.fp(bits) };
	let block_format = |block: Block| if brain { block.bf(16) } else { block.fp(bits) };
	let model = match kind {
		"attention" => format(format(recipe.model().layer(4)).res([block_format(attn(1)) * block_format(layer(4).sigmoid())])).last().layer(1),
		"rnn" => format(recipe.model().rnn(4)).last().layer(1),
		"gru" => format(recipe.model().gru(4)).last().layer(1),
		"lstm" => format(recipe.model().lstm(4)).last().layer(1),
		"normalization" => format(recipe.model().layer(32)).norm(rms).layer(1),
		_ => panic!("unknown narrow contract {kind}"),
	};
	format(model).scale(NARROW_SCALE).fp(32).loss(mse)
}

fn compare_narrow(report: &mut Report, data: &Data, kind: &'static str, names: [&'static str; 2]) {
	let wide = recipe.train().seed(SEED).lr(0.01).epochs(40).run(&contract_model(kind, 32, false), data);
	for (name, brain) in names.into_iter().zip([false, true]) {
		let narrow = recipe.train().seed(SEED).lr(0.01).epochs(40).run(&contract_model(kind, 16, brain), data);
		let bound = wide.final_loss() * 4.0 + 1e-12;
		let passes = narrow.initial_loss().is_finite() && narrow.final_loss().is_finite()
			&& narrow.final_loss() < narrow.initial_loss() && narrow.final_loss() <= bound;
		report.record(name, passes, format!("initial_loss={:.9e} final_loss={:.9e} fp32_loss={:.9e} bound={bound:.9e}", narrow.initial_loss(), narrow.final_loss(), wide.final_loss()));
	}
}

fn run_child(mode: &str, profile: Option<&str>, device: Option<&str>) -> std::process::Output {
	let mut command = std::process::Command::new(std::env::current_exe().expect("cannot locate suite executable"));
	command.env("RECIPE_SUITE_CONTRACT", mode);
	if let Some(profile) = profile { command.env("RECIPE_CONFIG", profile); }
	if let Some(device) = device { command.env("RECIPE_DEVICE", device); }
	command.output().expect("cannot execute contract in a separate process")
}

fn child_pass(report: &mut Report, name: &'static str, mode: &str, profile: Option<&str>) {
	let output = run_child(mode, profile, None);
	let stdout = String::from_utf8_lossy(&output.stdout);
	let stderr = String::from_utf8_lossy(&output.stderr);
	report.record(name, output.status.success(), format!("status={} {} {}", output.status, stdout.trim(), stderr.trim()));
}

fn suffix_model(brain_cache: bool) -> Model {
	let model = recipe.model().layer(32).fp(32).attn(1).int(8).kv(1);
	let model = if brain_cache { model.bf(16) } else { model.fp(16) };
	model.qk(rms).fp(16).rope(neox, 32, 10000.0).fp(32).last().layer(1).fp(32).loss(mse)
}

fn contract_child(mode: &str, root: &Path, work: &Path) {
	match mode {
		"quantized" => quantized_contract(work),
		"rope" => rope_contract(work),
		"missing-train" => {
			let path = work.join("features32.csv");
			let data = recipe.data(path.to_str().unwrap()).target("target");
			let model = recipe.model().no(bias).layer(1).int(16).loss(mse);
			recipe.train().seed(SEED).epochs(1).run(&model, &data);
		},
		"fp8-change" => {
			let path = root.join("data/linear.csv");
			let data = recipe.data(path.to_str().unwrap()).target("target");
			let model = recipe.model().layer(1).fp(8).loss(mse);
			recipe.train().seed(SEED).epochs(1).resume(work.join("fp8.ogdl")).run(&model, &data);
		},
		"suffix" => {
			let path = work.join("features32.csv");
			let data = recipe.data(path.to_str().unwrap()).target("target");
			let model = suffix_model(true);
			let bundle = work.join("attention-suffix.ogdl");
			let trained = recipe.train().seed(SEED).lr(0.01).epochs(4).save(&bundle).run(&model, &data);
			let resumed = recipe.train().seed(SEED).lr(0.01).epochs(1).resume(&bundle).run(&model, &data);
			let tolerance = 0.02 * trained.final_loss().abs().max(1e-6);
			assert!((resumed.initial_loss() - trained.final_loss()).abs() <= tolerance, "attention suffix resume loss differs");
			println!("saved_loss={:.9e} resumed_loss={:.9e} tolerance={tolerance:.9e}", trained.final_loss(), resumed.initial_loss());
		}
		"suffix-change" => {
			let path = work.join("features32.csv");
			let data = recipe.data(path.to_str().unwrap()).target("target");
			let bundle = work.join("attention-suffix.ogdl");
			let model = suffix_model(false);
			recipe.train().seed(SEED).lr(0.01).epochs(1).resume(&bundle).run(&model, &data);
		}
		"double" => {
			let path = root.join("data/small.csv");
			let data = recipe.data(path.to_str().unwrap()).target("target");
			for (name, model) in [("fp16", recipe.model().layer(1).fp(16).loss(mse)), ("bf16", recipe.model().layer(1).bf(16).loss(mse))] {
				let run = recipe.train().seed(SEED).lr(RATE).epochs(EPOCHS).run(&model, &data);
				assert!(run.final_loss().is_finite() && run.final_loss() < run.initial_loss(), "double accumulator loss does not fall");
				std::fs::write(work.join(format!("double-{name}.loss")), run.final_loss().to_string()).unwrap();
				println!("{name}_loss={:.9e}", run.final_loss());
			}
		}
		"fp8" => {
			let path = root.join("data/linear.csv");
			let data = recipe.data(path.to_str().unwrap()).target("target");
			let model = recipe.model().layer(1).fp(8).loss(mse);
			let bundle = work.join("fp8.ogdl");
			let run = recipe.train().seed(SEED).lr(0.01).epochs(10).save(&bundle).run(&model, &data);
			let resumed = recipe.train().seed(SEED).lr(0.01).epochs(1).resume(&bundle).run(&model, &data);
			assert!(resumed.initial_loss().is_finite() && (resumed.initial_loss() - run.final_loss()).abs() <= 0.1 * run.final_loss().abs().max(1e-6));
		}
		"shard" => {
			let path = work.join("features32.csv");
			let data = recipe.data(path.to_str().unwrap()).target("target");
			for (name, model) in [("fp16", recipe.model().no(bias).layer(1).fp(16).loss(mse)), ("bf16", recipe.model().no(bias).layer(1).bf(16).loss(mse))] {
				let bundle = work.join(format!("shard-{name}.ogdl"));
				let trained = recipe.train().seed(SEED).lr(0.01).epochs(10).save(&bundle).run(&model, &data);
				let selected = recipe::device_names(&environment("RECIPE_DEVICE")).unwrap().len();
				assert_eq!(trained.memory.len(), selected, "training does not execute on every selected device");
				let weights = (0..32).map(|column| { let mut basis = vec![0.0; 32]; basis[column] = 1.0; recipe.predict(&bundle, &basis)[0].to_string() }).collect::<Vec<_>>().join(" ");
				println!("{name} {weights}");
			}
		}
		_ => panic!("unknown suite contract {mode}"),
	}
}

fn additional_contracts(report: &mut Report, root: &Path, work: &Path) {
	let features = work.join("features32.csv");
	write_features(&features);
	let data32 = recipe.data(features.to_str().unwrap()).target("target");
	for (name, bits) in [("canonical_int4", 4), ("canonical_int8", 8), ("canonical_int16", 16), ("canonical_int32", 32)] {
		let model = recipe.model().no(bias).layer(1).int(bits).loss(mse);
		let run = recipe.train().seed(SEED).lr(0.01).epochs(20).run(&model, &data32);
		report.record(name, run.initial_loss().is_finite() && run.final_loss().is_finite() && run.final_loss() < run.initial_loss(), format!("features=32 initial_loss={:.9e} final_loss={:.9e}", run.initial_loss(), run.final_loss()));
	}
	let narrow_path = work.join("narrow-sequence.csv");
	let source = std::fs::read_to_string(root.join("data/multi.csv")).unwrap();
	let mut rows = source.lines();
	let mut narrow_csv = format!("{}\n", rows.next().unwrap());
	for row in rows {
		let (inputs, target) = row.rsplit_once(',').unwrap();
		narrow_csv.push_str(&format!("{inputs},{}\n", target.parse::<f64>().unwrap() * NARROW_SCALE));
	}
	std::fs::write(&narrow_path, narrow_csv).unwrap();
	let narrow_data = recipe.data(narrow_path.to_str().unwrap()).target("target");
	compare_narrow(report, &narrow_data, "attention", ["attention_fp16", "attention_bf16"]);
	compare_narrow(report, &narrow_data, "rnn", ["rnn_fp16", "rnn_bf16"]);
	compare_narrow(report, &narrow_data, "gru", ["gru_fp16", "gru_bf16"]);
	compare_narrow(report, &narrow_data, "lstm", ["lstm_fp16", "lstm_bf16"]);
	compare_narrow(report, &narrow_data, "normalization", ["normalization_fp16", "normalization_bf16"]);
	let large_path = work.join("large.csv");
	write_large(&large_path);
	let large = recipe.data(large_path.to_str().unwrap()).target("target");
	for (name, model) in [("large_fp16", recipe.model().layer(1).fp(16).scale(256.0).fp(32).loss(mse)), ("large_bf16", recipe.model().layer(1).bf(16).scale(256.0).fp(32).loss(mse))] {
		let run = recipe.train().seed(SEED).lr(0.02).epochs(800).run(&model, &large);
		let wanted = linear_targets().into_iter().map(|value| value * 256.0).collect::<Vec<_>>();
		let worst = worst_absolute(&sorted(run.predictions().iter().copied()), &sorted(wanted)) / 256.0;
		report.record(name, run.final_loss().is_finite() && run.final_loss() < run.initial_loss() && worst <= 1.0, format!("scale=256 worst_abs_err_in_y={worst:.9} tolerance=1 initial_loss={:.9e} final_loss={:.9e}", run.initial_loss(), run.final_loss()));
	}
	child_pass(report, "attention_suffix_resume", "suffix", None);
	child_pass(report, "quantized_source_conversion", "quantized", None);
	child_pass(report, "proportional_rope_factors", "rope", None);
	let changed = run_child("suffix-change", None, None);
	let reason = String::from_utf8_lossy(&changed.stderr);
	report.record("attention_suffix_mismatch", !changed.status.success() && reason.contains("model mismatch not overwritten"), format!("status={} {}", changed.status, reason.trim()));
	child_pass(report, "double_accumulator", "double", Some("check64"));
	let small_path = root.join("data/small.csv");
	let small = recipe.data(small_path.to_str().unwrap()).target("target");
	for (name, label, model) in [("double_fp16_matches_fp32_accumulator", "fp16", recipe.model().layer(1).fp(16).loss(mse)), ("double_bf16_matches_fp32_accumulator", "bf16", recipe.model().layer(1).bf(16).loss(mse))] {
		let baseline = recipe.train().seed(SEED).lr(RATE).epochs(EPOCHS).run(&model, &small);
		let wide = std::fs::read_to_string(work.join(format!("double-{label}.loss"))).unwrap().parse::<f64>().unwrap();
		report.record(name, wide.is_finite() && wide <= baseline.final_loss() * 1.01 + 1e-15, format!("fp64_accumulator_loss={wide:.9e} fp32_accumulator_loss={:.9e}", baseline.final_loss()));
	}
	let selection = std::env::var("RECIPE_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
	let devices = recipe::device_names(&selection).expect("invalid selected devices");
	let fp8 = run_child("fp8", None, None);
	let reason = String::from_utf8_lossy(&fp8.stderr);
	if !fp8.status.success() && reason.contains("no fp8 instruction") {
		report.record("fp8_e4m3_capability_error", !fp8.status.success() && reason.contains("no fp8 instruction") && reason.contains("Matrix: none"), format!("device={selection} status={} {}", fp8.status, reason.trim()));
		let other = run_child("fp8", Some("e5m2"), None);
		let reason = String::from_utf8_lossy(&other.stderr);
		report.record("fp8_e5m2_capability_error", !other.status.success() && reason.contains("no fp8 instruction") && reason.contains("Matrix: none"), format!("device={selection} status={} {}", other.status, reason.trim()));
	} else {
		report.record("fp8_saved_encoding", fp8.status.success(), format!("status={} {}", fp8.status, reason.trim()));
		if fp8.status.success() {
			let changed = run_child("fp8-change", Some("e5m2"), None);
			let reason = String::from_utf8_lossy(&changed.stderr);
			report.record("fp8_encoding_mismatch", !changed.status.success() && reason.contains("saved precision profile"), format!("status={} {}", changed.status, reason.trim()));
		}
	}
	let absent = run_child("missing-train", Some("inference"), None);
	let reason = String::from_utf8_lossy(&absent.stderr);
	report.record("integer_training_requires_float_precision", !absent.status.success() && reason.contains("integer checkpoint blocks require an explicit train ="), format!("status={} {}", absent.status, reason.trim()));
	if devices.len() < 2 {
		println!("check narrow_sharded_training SKIP one selected device: {selection}");
	} else {
		let single = run_child("shard", None, Some(&devices[0]));
		let sharded = run_child("shard", None, None);
		let parse = |output: &std::process::Output| String::from_utf8_lossy(&output.stdout).lines().filter(|line| line.starts_with("fp16 ") || line.starts_with("bf16 ")).flat_map(|line| line.split_whitespace().skip(1).map(|value| value.parse::<f64>().unwrap()).collect::<Vec<_>>()).collect::<Vec<_>>();
		let (want, got) = (parse(&single), parse(&sharded));
		let relative = got.iter().zip(&want).map(|(got, want)| (got - want).abs() / want.abs().max(1e-12)).fold(0.0_f64, f64::max);
		report.record("narrow_sharded_training", single.status.success() && sharded.status.success() && want.len() == 64 && got.len() == want.len() && relative <= 2e-5, format!("selected={selection} weights={} worst_relative={relative:.9e} tolerance=2e-5", got.len()));
	}
}
fn write_contract_gguf(path: &Path, tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)>) {
	let mut out = b"GGUF".to_vec();
	out.extend_from_slice(&3u32.to_le_bytes());
	out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
	out.extend_from_slice(&1u64.to_le_bytes());
	gguf_text(&mut out, "general.architecture");
	out.extend_from_slice(&8u32.to_le_bytes());
	gguf_text(&mut out, "llama");
	let mut data = Vec::new();
	for (name, shape, kind, bytes) in tensors {
		gguf_text(&mut out, name);
		out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
		for dimension in shape { out.extend_from_slice(&dimension.to_le_bytes()); }
		out.extend_from_slice(&kind.to_le_bytes());
		out.extend_from_slice(&(data.len() as u64).to_le_bytes());
		data.extend(bytes);
		while data.len() % 32 != 0 { data.push(0); }
	}
	while out.len() % 32 != 0 { out.push(0); }
	out.extend(data);
	std::fs::write(path, out).expect("cannot write contract GGUF");
}

fn quantized_contract(work: &Path) {
	let path = work.join("q4-source.gguf");
	let mut bytes = 0x3800u16.to_le_bytes().to_vec();
	bytes.extend((0u8..16).map(|code| code | (code << 4)));
	write_contract_gguf(&path, vec![("source.weight", vec![32, 1], 2, bytes)]);
	let file = recipe.gguf(path.to_str().unwrap());
	let model = recipe.model().no(bias).layer(1).int(4);
	let plan = file.plan().named(&file, "source.weight");
	let got = file.infer(&model, &plan, &[1.0; 32], 32);
	let want = 2.0 * (0..16).map(|code| f64::from(code - 8) * 0.5).sum::<f64>();
	assert!(got.len() == 1 && (got[0] - want).abs() <= 1e-5, "q4_0 conversion produces {got:?}, expected {want}");
	println!("q4_0_to_int4={:.9} analytic={want:.9} tolerance=1e-5", got[0]);
}

fn rope_contract(work: &Path) {
	let path = work.join("rope-factors.gguf");
	let identity = (0..16).flat_map(|index| (if index / 4 == index % 4 { 1f32 } else { 0f32 }).to_le_bytes()).collect::<Vec<_>>();
	let factors = [1f32, 2f32].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
	write_contract_gguf(&path, vec![
		("blk.0.attn_q.weight", vec![4, 4], 0, identity.clone()),
		("blk.0.attn_k.weight", vec![4, 4], 0, identity.clone()),
		("blk.0.attn_v.weight", vec![4, 4], 0, identity.clone()),
		("blk.0.attn_output.weight", vec![4, 4], 0, identity),
		("rope_freqs.weight", vec![2], 0, factors),
	]);
	let file = recipe.gguf(path.to_str().unwrap());
	let model = recipe.model().no(bias).attn(1).fp(32).rope(neox, 4, 10000.0).fp(32).last();
	let plan = file.plan().node(&[file.tensor("blk.0.attn_q.weight").unwrap().clone(), file.tensor("blk.0.attn_k.weight").unwrap().clone(), file.tensor("blk.0.attn_v.weight").unwrap().clone()])
		.named(&file, "rope_freqs.weight").named(&file, "blk.0.attn_output.weight");
	let got = file.infer(&model, &plan, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 4);
	let first = [0.1, 0.3, 0.5, 0.7];
	let last = [0.2, 0.4, 0.6, 0.8];
	let mut query = last;
	for (pair, angle) in [1.0_f64, 0.005].into_iter().enumerate() {
		query[pair] = last[pair] * angle.cos() - last[pair + 2] * angle.sin();
		query[pair + 2] = last[pair + 2] * angle.cos() + last[pair] * angle.sin();
	}
	let cross = query.iter().zip(first).map(|(q, k)| q * k).sum::<f64>() / 2.0;
	let own = last.iter().map(|value| value * value).sum::<f64>() / 2.0;
	let previous = 1.0 / (1.0 + (own - cross).exp());
	let want = first.into_iter().zip(last).map(|(a, b)| previous * a + (1.0 - previous) * b).collect::<Vec<_>>();
	let worst = worst_absolute(&got, &want);
	assert!(worst <= 1e-5, "proportional rotary attention produces {got:?}, expected {want:?}, error={worst}");
	println!("analytic_attention_worst_abs={worst:.9e} factors=[1,2] tolerance=1e-5");
}
