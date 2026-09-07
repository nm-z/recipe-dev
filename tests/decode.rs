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
	let data = recipe.data(directory.to_str().unwrap()).target(names.as_slice());
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

/// A decode of `steps` ids must leave the logits one forward of the settled ids
/// leaves, at every split point between the prompt and the full sequence.
#[test]
fn decode_matches_a_whole_sequence_forward() {
	let path = bundle("whole-sequence");
	let mut settled = Vec::new();
	for reached in [8, 9, 16, 23, 24] {
		let steps = reached - PROMPT.len();
		let generation = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], steps);
		assert_eq!(generation.ids.len(), reached, "decode reached {} ids, expected {reached}", generation.ids.len());

		let reference = recipe.infer(&path, &whole_sequence(&generation.ids));
		assert_eq!(
			bits(&generation.logits),
			bits(&reference),
			"decode to {reached} ids disagrees with a whole-sequence forward of the same ids:\n  decode    {:?}\n  reference {:?}",
			generation.logits,
			reference
		);

		// Greedy sampling is deterministic, so every longer decode must extend the
		// shorter one rather than diverge from it.
		assert!(generation.ids.starts_with(&settled), "decode to {reached} ids diverged from the shorter decode:\n  {:?}\n  {:?}", generation.ids, settled);
		settled = generation.ids;
	}
	std::fs::remove_file(path).unwrap();
}

/// A step extends the tape by one position; the prefill runs the whole prompt.
/// If a step were re-running the sequence the two would not separate.
#[test]
fn a_step_costs_less_than_the_prefill() {
	let path = bundle("step-cost");
	let generation = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 12);
	assert_eq!(generation.step_seconds.len(), 12, "expected one timing per step");
	let mean = generation.step_seconds.iter().sum::<f64>() / generation.step_seconds.len() as f64;
	assert!(mean < generation.prefill_seconds, "mean step {mean:.6}s is not below the prefill {:.6}s", generation.prefill_seconds);
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
