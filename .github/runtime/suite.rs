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

/// A three-channel identity projection makes the affine normalization's output directly observable.
fn write_layer_norm_checkpoint(path: &Path, scale: bool, shift: bool) {
	let mut tensors = vec![
		("token_embd.weight", vec![3u64, 3], vec![1.0f32, 2.0, 4.0, -2.0, 0.0, 3.0, 7.0, 7.0, 7.0]),
		("output.weight", vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
	];
	if scale { tensors.push(("output_norm.weight", vec![3], vec![2.0, -3.0, 0.5])); }
	if shift { tensors.push(("output_norm.bias", vec![3], vec![0.25, -0.5, 1.0])); }
	let mut out = b"GGUF".to_vec();
	out.extend_from_slice(&3u32.to_le_bytes());
	out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
	out.extend_from_slice(&2u64.to_le_bytes());
	gguf_text(&mut out, "general.architecture");
	out.extend_from_slice(&8u32.to_le_bytes());
	gguf_text(&mut out, "llama");
	gguf_text(&mut out, "llama.attention.layer_norm_epsilon");
	out.extend_from_slice(&12u32.to_le_bytes());
	out.extend_from_slice(&0.25f64.to_le_bytes());
	let mut data = Vec::new();
	for (name, shape, values) in tensors {
		gguf_text(&mut out, name);
		out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
		for dimension in shape { out.extend_from_slice(&dimension.to_le_bytes()); }
		out.extend_from_slice(&0u32.to_le_bytes());
		out.extend_from_slice(&(data.len() as u64).to_le_bytes());
		for value in values { data.extend_from_slice(&value.to_le_bytes()); }
		while data.len() % 32 != 0 { data.push(0); }
	}
	while out.len() % 32 != 0 { out.push(0); }
	out.extend(data);
	std::fs::write(path, out).expect("cannot write the layer normalization checkpoint");
}

fn check_layer_norm(report: &mut Report, work: &Path) {
	for (name, scale, shift) in [
		("layer_norm_affine", true, true),
		("layer_norm_default_scale", false, true),
		("layer_norm_default_shift", true, false),
		("layer_norm_defaults", false, false),
	] {
		let path = work.join(format!("{name}.gguf"));
		write_layer_norm_checkpoint(&path, scale, shift);
		let file = Gguf::open(&path).expect("cannot open the layer normalization checkpoint");
		let model = recipe.model().no(bias).embed(3, 3).fp(32).norm(layer).fp(32).scale_from("output_norm.weight").layer(3).fp(32);
		let placed = file.place(&model, 1, &[]);
		let got = (0..3).flat_map(|position| placed.infer(&[position as f64])).collect::<Vec<_>>();
		let mut want = vec![0.0; 9];
		for (position, input) in [[1.0f64, 2.0, 4.0], [-2.0, 0.0, 3.0], [7.0, 7.0, 7.0]].into_iter().enumerate() {
			let mean = input.iter().sum::<f64>() / 3.0;
			let variance = input.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / 3.0;
			for channel in 0..3 {
				let gamma = if scale { [2.0, -3.0, 0.5][channel] } else { 1.0 };
				let beta = if shift { [0.25, -0.5, 1.0][channel] } else { 0.0 };
				want[position * 3 + channel] = (input[channel] - mean) / (variance + 0.25).sqrt() * gamma + beta;
			}
		}
		let worst = worst_absolute(&got, &want);
		report.record(name, worst < 2e-6, format!("worst_abs_err={worst:.9} scale={scale} shift={shift} epsilon=0.25 positions=3"));
		let sequence = file.place(&model.last(), 3, &[]).infer(&[0.0, 1.0, 2.0]);
		let sequence_worst = worst_absolute(&sequence, &want[6..9]);
		report.record("layer_norm_sequence", sequence_worst < 2e-6, format!("worst_abs_err={sequence_worst:.9} scale={scale} shift={shift} positions=3"));
		let explicit = model.e(0.5);
		let got = file.place(&explicit, 1, &[]).infer(&[0.0]);
		let want = (1.0 - 7.0 / 3.0) / (14.0f64 / 9.0 + 0.5).sqrt() * if scale { 2.0 } else { 1.0 } + if shift { 0.25 } else { 0.0 };
		report.record("layer_norm_explicit_epsilon", (got[0] - want).abs() < 2e-6, format!("produced={:.9} expected={want:.9}", got[0]));
	}
	// With no other trainable block, only gamma and beta can fit these targets.
	for (name, gamma, beta) in [("layer_norm_train_scale", 2.0, 0.0), ("layer_norm_train_shift", 1.0, 0.75)] {
		let path = work.join(format!("{name}.csv"));
		let mut csv = "a,b,c,y0,y1,y2\n".to_owned();
		for row in 0..24 {
			let input = [row as f64 / 8.0, (row % 5) as f64 - 2.0, (row % 7) as f64 / 3.0];
			let mean = input.iter().sum::<f64>() / 3.0;
			let denominator = (input.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / 3.0 + 0.25).sqrt();
			let target = input.map(|value| (value - mean) / denominator * gamma + beta);
			csv.push_str(&format!("{},{},{},{},{},{}\n", input[0], input[1], input[2], target[0], target[1], target[2]));
		}
		std::fs::write(&path, csv).expect("cannot write the affine training dataset");
		let data = recipe.data(path.to_str().unwrap()).target(["y0", "y1", "y2"]);
		let model = recipe.model().e(0.25).norm(layer).fp(32).loss(mse);
		let bundle = work.join(format!("{name}.ogdl"));
		let trained = recipe.train().seed(SEED).lr(0.03).stop(0.0).epochs(EPOCHS).save(&bundle).run(&model, &data);
		let persisted = recipe.predict(&bundle, &[0.0, -2.0, 0.0]);
		let denominator = (8.0f64 / 9.0 + 0.25).sqrt();
		let want = [2.0 / 3.0, -4.0 / 3.0, 2.0 / 3.0].map(|value| value / denominator * gamma + beta);
		let worst = worst_absolute(&persisted, &want);
		report.record(name, trained.final_loss() < 1e-5 && worst < 0.01, format!("initial_loss={:.9} final_loss={:.9} persisted_worst={worst:.9}", trained.initial_loss(), trained.final_loss()));
	}
}

/// Compare a public GPT-2 GGUF with teacher-forced CPU logits from llama.cpp b10705.
/// The reference stores GPTNORM1, four u32 counts (prompt, sequence, vocabulary,
/// steps), sequence u32 token IDs, then step-major fp32 logits, all little-endian.
fn check_gpt2_layer_norm(report: &mut Report) {
	let Some(path) = std::env::var_os("RECIPE_LAYER_NORM_GGUF") else { return };
	let reference = std::fs::read(environment("RECIPE_LAYER_NORM_REFERENCE")).expect("cannot read the GPT-2 reference");
	assert_eq!(&reference[..8], b"GPTNORM1");
	let count = |at| u32::from_le_bytes(reference[at..at + 4].try_into().unwrap()) as usize;
	let (prompt, length, vocabulary, steps) = (count(8), count(12), count(16), count(20));
	assert_eq!(length, prompt + steps - 1);
	assert_eq!(reference.len(), 24 + 4 * length + 4 * steps * vocabulary);
	let ids = (0..length).map(|position| count(24 + 4 * position)).collect::<Vec<_>>();
	let file = Gguf::open(Path::new(&path)).expect("cannot open the GPT-2 checkpoint");
	assert_eq!(file.value("general.architecture").unwrap().text(), Some("gpt2"));
	let integer = |key| file.value(key).unwrap().integer().unwrap() as usize;
	let (width, heads, hidden, blocks) = (integer("gpt2.embedding_length"), integer("gpt2.attention.head_count"), integer("gpt2.feed_forward_length"), integer("gpt2.block_count"));
	let mut used = std::collections::BTreeSet::new();
	let mut tensor = |name: &str| {
		used.insert(name.to_owned());
		file.tensor(name).unwrap_or_else(|| panic!("missing {name}")).clone()
	};
	let embedding = tensor("token_embd.weight");
	let positional = tensor("position_embd.weight");
	assert_eq!(embedding.shape, [width as u64, vocabulary as u64]);
	let mut plan = file.plan();
	let mut model = recipe.model();
	for block in 0..blocks {
		model = model.res([norm(layer).fp(32), attn(heads).kv(heads).fp(32).width(width / heads).fp(32)])
			.res([norm(layer).fp(32), layer(hidden).fp(32).gelu().fp(32), layer(width).fp(32)]);
		for stem in ["attn_norm", "attn_qkv", "attn_output", "ffn_norm", "ffn_up", "ffn_down"] {
			plan = plan.node(&[tensor(&format!("blk.{block}.{stem}.weight")), tensor(&format!("blk.{block}.{stem}.bias"))]);
		}
	}
	model = model.norm(layer).fp(32).layer(vocabulary).fp(32);
	plan = plan.node(&[tensor("output_norm.weight"), tensor("output_norm.bias")]);
	plan = if file.tensor("output.weight").is_some() { plan.node(&[tensor("output.weight")]) } else { plan.node(&[embedding.clone()]) };
	let unread = file.tensors().iter().filter(|tensor| !used.contains(&tensor.name)).map(|tensor| &tensor.name).collect::<Vec<_>>();
	assert!(unread.is_empty(), "unread {unread:?}");
	assert!(plan.defaulted().is_empty());
	// The public row lookup reads both embedding tables once. The compiled model
	// evaluates all teacher-forced positions in one forward, with no replay.
	let mut input = vec![0.0; width * length];
	for (position, id) in ids.iter().enumerate() {
		let token = file.row(&embedding, *id).unwrap();
		let position_values = file.row(&positional, position).unwrap();
		for channel in 0..width { input[channel * length + position] = f64::from(token[channel] as f32 + position_values[channel] as f32); }
	}
	let got = file.infer(&model, &plan, &input, width);
	assert_eq!(got.len(), length * vocabulary);
	let mut worst = 0.0f64;
	for step in 0..steps {
		let position = prompt - 1 + step;
		let at = 24 + 4 * length + 4 * step * vocabulary;
		let expected = reference[at..at + 4 * vocabulary].chunks_exact(4).map(|bytes| f64::from(f32::from_le_bytes(bytes.try_into().unwrap()))).collect::<Vec<_>>();
		assert!(expected.iter().all(|value| value.is_finite()));
		let range = expected.iter().copied().fold(f64::NEG_INFINITY, f64::max) - expected.iter().copied().fold(f64::INFINITY, f64::min);
		let error = expected.iter().enumerate().map(|(channel, value)| {
			let value = (got[channel * length + position] - value).abs();
			if value.is_finite() { value } else { f64::INFINITY }
		}).fold(0.0f64, f64::max) / range;
		worst = worst.max(error);
		println!("gpt2 step={step} position={position} max_error_of_range={error:.9}");
	}
	report.record("gpt2_layer_norm_logits", worst <= 0.1,
		format!("steps={steps} logits={} worst_error_of_range={worst:.9} tensors={} defaulted={:?} unread={unread:?}", steps * vocabulary, used.len(), plan.defaulted()));
}

fn main() {
	let root = PathBuf::from(environment("RECIPE_SUITE_ROOT"));
	let work = PathBuf::from(environment("RECIPE_SUITE_WORK"));
	let evidence = PathBuf::from(environment("RECIPE_EVIDENCE"));
	std::fs::create_dir_all(&work).expect("cannot create the suite work directory");
	let trace_path = std::env::var_os("RECIPE_TRACE_PATH").map(PathBuf::from).unwrap_or_else(|| work.join("phases.log"));
	// Set the log destination before Recipe starts any workers.
	unsafe { std::env::set_var("RECIPE_TRACE_PATH", &trace_path); }
	let linear_source = root.join("data/linear.csv");
	let multi_source = root.join("data/multi.csv");
	let mut report = Report { checks: Vec::new() };
	// Select the LayerNorm group while keeping the checks on the public API.
	if let Ok(group) = std::env::var("RECIPE_SUITE_GROUP") && matches!(group.as_str(), "layer-norm" | "gpt2") {
		if group == "layer-norm" { check_layer_norm(&mut report, &work); }
		else { assert!(std::env::var_os("RECIPE_LAYER_NORM_GGUF").is_some(), "the GPT-2 group requires RECIPE_LAYER_NORM_GGUF"); }
		check_gpt2_layer_norm(&mut report);
		let failed = report.checks.iter().filter(|check| !check.passed).map(|check| check.name).collect::<Vec<_>>();
		write_document(&evidence, &format!("{{\"schema\":\"recipe-layer-norm/1\",\"executed\":{},\"failed\":{}}}", report.checks.len(), failed.len()));
		assert!(failed.is_empty(), "layer normalization checks failed: {failed:?}");
		println!("LAYER NORM PASS executed={}", report.checks.len());
		return;
	}

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

	check_layer_norm(&mut report, &work);
	check_gpt2_layer_norm(&mut report);

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
	report.record("single_device_has_no_exchange", !trace.lines().any(|line| line.starts_with("exchange phase ")), "ordinary placement has no exchange phase records".to_owned());

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
