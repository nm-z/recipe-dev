//! The sparse-attention indexer under decode.
//!
//! The indexer now keeps state: one running sum of indexer keys per block in
//! the context arena, extended by the positions each window reaches. A step
//! that extends the wrong block, or re-sums a block it already holds, changes
//! which keys a query attends to — and a wrong selection is invisible in an
//! aggregate loss. So the comparison is on raw bit patterns, as `decode.rs`
//! compares: a decode driven one position at a time must reach the logits one
//! whole-sequence forward of the same ids reaches.

use recipe::*;
use std::path::PathBuf;

const COLUMNS: usize = 24;
const TARGETS: usize = 12;
const PROMPT: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
/// `conv(4, 3)` leaves 22 positions, which `block` 4 covers in six blocks.
const BLOCKS: usize = 6;

/// A block with the indexer's own scoring geometry: its planes normalize under
/// their trained scale and rotate at the block's rope base before scoring.
fn indexed(keep: usize, budget: usize) -> Model {
	let attention = recipe.model().conv(4, 3).relu().attn(2).width(8).qk(rms).rope(4, 10000.0).index(2, 8, 4, keep).score(rms, 4);
	let attention = if budget == 0 { attention } else { attention.budget(budget) };
	attention.relu().gru(4).relu().pool(2).relu().layer(TARGETS).loss(mse)
}

fn dataset(name: &str) -> PathBuf {
	let directory = std::env::temp_dir().join(format!("recipe-indexer-{}-{name}", std::process::id()));
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

fn bundle(name: &str, model: &Model) -> PathBuf {
	let directory = dataset(name);
	let path = std::env::temp_dir().join(format!("recipe-indexer-{}-{name}.ogdl", std::process::id()));
	let names = (0..TARGETS).map(|target| format!("y{target}")).collect::<Vec<_>>();
	let data = recipe.data(directory.to_str().unwrap()).target(names.as_slice()).norm(z_score);
	recipe.train().fp(32).seed(17).lr(0.01).epochs(3).stop(0.0).save(&path).run(model, &data);
	path
}

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

/// The claim #679 makes. A full-extent decode reaches its logits through
/// sixteen one-position windows; each one extends exactly the block its new key
/// lands in and scores exactly its own query. A single forward computes the
/// same logits in one pass, re-summing every block. Bit-for-bit agreement means
/// the running sums the steps built are the sums the whole forward computes,
/// and the selection a step made is the selection the whole forward makes.
#[test]
fn an_indexed_decode_matches_a_whole_sequence_forward() {
	let path = bundle("decode", &indexed(2, 0));
	let generation = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], COLUMNS - PROMPT.len());
	assert_eq!(generation.ids.len(), COLUMNS, "decode reached {} ids, expected {COLUMNS}", generation.ids.len());
	let reference = recipe.infer(&path, &whole_sequence(&generation.ids));
	assert_eq!(
		bits(&generation.logits),
		bits(&reference),
		"an indexed full-extent decode disagrees with a whole-sequence forward of the same ids:\n  decode    {:?}\n  reference {:?}",
		generation.logits,
		reference
	);
	std::fs::remove_file(path).unwrap();
}

/// A longer decode extends a shorter one. A block whose running sum a step
/// failed to extend, or extended twice, parts the two decodes at the id where
/// the selection first differs.
#[test]
fn a_longer_indexed_decode_extends_a_shorter_one() {
	let path = bundle("extends", &indexed(2, 0));
	let mut settled: Vec<u32> = Vec::new();
	for reached in [8, 9, 16, COLUMNS] {
		let generation = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], reached - PROMPT.len());
		assert_eq!(generation.ids.len(), reached, "decode reached {} ids, expected {reached}", generation.ids.len());
		assert!(generation.ids.starts_with(&settled), "decode to {reached} ids diverged from the shorter decode:\n  {:?}\n  {:?}", generation.ids, settled);
		settled = generation.ids;
	}
	std::fs::remove_file(path).unwrap();
}

/// A token budget is a block admission. A budget covering the sequence admits
/// every block, so it must be the same computation as naming every block by
/// count — same graph, same parameters, only the admission stated differently.
#[test]
fn a_covering_budget_admits_every_block() {
	let counted = bundle("counted", &indexed(BLOCKS, 0));
	let budgeted = bundle("budgeted", &indexed(1, COLUMNS * 4));
	let input = whole_sequence(&PROMPT);
	let (left, right) = (recipe.infer(&counted, &input), recipe.infer(&budgeted, &input));
	assert_eq!(
		bits(&left),
		bits(&right),
		"a budget covering the sequence differs from keeping every block:\n  keep({BLOCKS}) {left:?}\n  budget        {right:?}"
	);
	std::fs::remove_file(counted).unwrap();
	std::fs::remove_file(budgeted).unwrap();
}
