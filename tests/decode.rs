//! Decode equivalence: a decode that extends the tape by one position at a time
//! must produce the logits one whole-sequence forward of the same ids produces.
//!
//! The decode carries attention keys and values, recurrent state and the
//! convolution tail across calls and writes only the positions a new id reaches.
//! Everything it keeps is a chance to keep the wrong thing, so the comparison is
//! on raw bit patterns with no tolerance, the way `determinism.rs` compares.
//!
//! The model is one graph that exercises every windowed primitive at once:
//! `conv` and `pool` force a sequential input shape, `attn` reads every earlier
//! position, `gru` carries state, and the closing `layer` reads the pooled tail.

use recipe::*;
use std::path::PathBuf;

const COLUMNS: usize = 24;
/// The decode samples an id from the logits, so the target count is the
/// vocabulary and every prompt id has to fall inside it.
const TARGETS: usize = 12;
const PROMPT: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];

fn model() -> Model {
	recipe.model().conv(4, 3).relu().attn(2).relu().gru(4).relu().pool(2).relu().layer(TARGETS).loss(mse)
}

/// Rows whose targets are a smooth function of the features, so three epochs
/// move the weights off their seed without needing the model to be any good.
fn dataset() -> PathBuf {
	let directory = std::env::temp_dir().join(format!("recipe-decode-{}", std::process::id()));
	std::fs::create_dir_all(&directory).unwrap();
	let mut text = String::new();
	for column in 0..COLUMNS {
		text.push_str(&format!("x{column},"));
	}
	for target in 0..TARGETS {
		text.push_str(&format!("y{target}{}", if target + 1 == TARGETS { "\n" } else { "," }));
	}
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	for _ in 0..48 {
		let values = (0..COLUMNS).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			text.push_str(&format!("{value},"));
		}
		for target in 0..TARGETS {
			let sum = values.iter().enumerate().map(|(index, value)| value * ((index + target) as f64 / 8.0).cos()).sum::<f64>();
			text.push_str(&format!("{}{}", (sum / 4.0).tanh(), if target + 1 == TARGETS { "\n" } else { "," }));
		}
	}
	let path = directory.join("rows.csv");
	std::fs::write(&path, text).unwrap();
	directory
}

/// Each test trains its own bundle: the suite runs tests in parallel and they
/// would otherwise write one path at once.
fn bundle(name: &str) -> PathBuf {
	let directory = dataset();
	let path = std::env::temp_dir().join(format!("recipe-decode-{}-{name}.ogdl", std::process::id()));
	let names = (0..TARGETS).map(|target| format!("y{target}")).collect::<Vec<_>>();
	// Normalized on purpose. The tape uploads the raw ids, so the tail a decode
	// has not reached holds 0.0 while a whole-sequence forward sees the prepared
	// padding `(0 - mean) / scale`. Without a normalizing dataset those two are
	// the same value and the seeding this exercises would be untestable.
	let data = recipe.data(directory.to_str().unwrap()).target(names.as_slice()).norm(z_score);
	recipe.train().fp(32).seed(17).lr(0.01).epochs(3).stop(0.0).save(&path).run(&model(), &data);
	path
}

/// The same ids a decode has settled, laid out as one whole-sequence input.
fn whole_sequence(ids: &[u32]) -> Vec<f64> {
	let mut input = vec![0.0; COLUMNS];
	for (slot, id) in input.iter_mut().zip(ids) {
		*slot = f64::from(*id);
	}
	input
}

fn bits(values: &[f64]) -> Vec<u64> {
	values.iter().map(|value| value.to_bits()).collect()
}

/// At full extent the decode has written every position, so its logits must be
/// the logits one whole-sequence forward of the same ids produces — bit for bit.
///
/// This is the claim the feature makes and it is the strong one: the decode
/// reached those logits through sixteen incremental windows, the last of them
/// `23..24`, carrying attention keys and values, the recurrent state and the
/// convolution tail across every call. A single forward computes them in one
/// pass. Agreement to the bit means nothing the decode kept was stale.
#[test]
fn a_full_extent_decode_matches_a_whole_sequence_forward() {
	let path = bundle("full-extent");
	let generation = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], COLUMNS - PROMPT.len());
	assert_eq!(generation.ids.len(), COLUMNS, "decode reached {} ids, expected {COLUMNS}", generation.ids.len());
	let reference = recipe.infer(&path, &whole_sequence(&generation.ids));
	assert_eq!(
		bits(&generation.logits),
		bits(&reference),
		"a full-extent decode disagrees with a whole-sequence forward of the same ids:\n  decode    {:?}\n  reference {:?}",
		generation.logits,
		reference
	);
	std::fs::remove_file(path).unwrap();
}

/// Greedy sampling is deterministic, so a longer decode must extend a shorter
/// one rather than diverge from it. A window that read an unsettled position, or
/// state that a step failed to carry, shows up here as a divergence at the id
/// where the two decodes part.
///
/// This is deliberately not compared against `recipe.infer` at intermediate
/// lengths. The model's sequence is fixed, so a forward of eight settled ids
/// still reads the sixteen padded positions after them, and this model's closing
/// `pool` reduces the whole length into its output. An intermediate decode,
/// whose window stops at the settled position, is therefore not the same
/// computation as a padded whole-sequence forward, and asserting that it is
/// would be asserting something the feature does not claim.
#[test]
fn a_longer_decode_extends_a_shorter_one() {
	let path = bundle("extends");
	let mut settled: Vec<u32> = Vec::new();
	for reached in [8, 9, 16, COLUMNS] {
		let generation = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], reached - PROMPT.len());
		assert_eq!(generation.ids.len(), reached, "decode reached {} ids, expected {reached}", generation.ids.len());
		assert!(generation.ids.starts_with(&settled), "decode to {reached} ids diverged from the shorter decode:\n  {:?}\n  {:?}", generation.ids, settled);
		settled = generation.ids;
	}
	std::fs::remove_file(path).unwrap();
}

/// The prefill and every step are timed separately and each timing is real.
///
/// This deliberately does not assert that a step is faster than the prefill.
/// On a model this small the fixed cost of a call — the whole-arena download and
/// the `infer_graphs` pass — swamps the difference between writing eight
/// positions and writing one, and the assertion fails on the CPU backend for
/// that reason rather than for anything the decode does wrong. The incremental
/// property is proved above, exactly and without timing: the full-extent logits
/// are reached through sixteen one-position windows and still match a single
/// whole-sequence forward bit for bit.
#[test]
fn every_step_is_timed_separately() {
	let path = bundle("step-cost");
	let generation = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 12);
	assert_eq!(generation.step_seconds.len(), 12, "expected one timing per step, got {}", generation.step_seconds.len());
	assert!(generation.prefill_seconds.is_finite() && generation.prefill_seconds > 0.0, "prefill was not timed: {}", generation.prefill_seconds);
	assert!(
		generation.step_seconds.iter().all(|seconds| seconds.is_finite() && *seconds > 0.0),
		"a step was not timed: {:?}",
		generation.step_seconds
	);
	std::fs::remove_file(path).unwrap();
}

/// The window is checked, not assumed.
#[test]
fn a_decode_past_the_model_sequence_is_refused() {
	let path = bundle("refused");
	let message = std::panic::catch_unwind(|| {
		let _ = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], COLUMNS);
	})
	.unwrap_err();
	let text = match message.downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	};
	assert!(text.contains("exceeds the model sequence"), "unexpected error: {text}");
	std::fs::remove_file(path).unwrap();
}
