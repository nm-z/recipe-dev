//! A placed decode is a decode: the same logits, whatever the model is split
//! across.
//!
//! `Placed` now owns one tape per range for its whole life, so the state a
//! decode carries — attention keys and values, the recurrent state, the
//! convolution tail — lives in those tapes across every step and every hop.
//! That is a new place to keep the wrong thing, so the comparisons here are on
//! raw bit patterns with no tolerance, as `decode.rs` compares.
//!
//! With one device selected the model is one range and this checks that the
//! placed path is the path `recipe.decode` runs. Select two or more devices
//! (`RECIPE_DEVICE=nv0,nv1`) and the same assertions cover a real hop: the
//! stream crossing devices once per token.

use recipe::*;
use std::path::PathBuf;

const COLUMNS: usize = 24;
const TARGETS: usize = 12;
const PROMPT: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
/// Six blocks, so a two-device placement has a boundary to fall on.
fn model() -> Model {
	recipe.model().conv(4, 3).relu().attn(2).relu().gru(4).relu().pool(2).relu().layer(TARGETS).loss(mse)
}

fn dataset() -> PathBuf {
	let directory = std::env::temp_dir().join(format!("recipe-placement-{}", std::process::id()));
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

fn bundle(name: &str) -> PathBuf {
	let directory = dataset();
	let path = std::env::temp_dir().join(format!("recipe-placement-{}-{name}.ogdl", std::process::id()));
	let names = (0..TARGETS).map(|target| format!("y{target}")).collect::<Vec<_>>();
	let data = recipe.data(directory.to_str().unwrap()).target(names.as_slice()).norm(z_score);
	recipe.train().fp(32).seed(17).lr(0.01).epochs(3).stop(0.0).save(&path).run(&model(), &data);
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

/// A full-extent decode over the placed model must equal one whole-sequence
/// forward of the same ids, bit for bit. Every range wrote its window and
/// handed on only that window's rows; agreement means no range read a position
/// an earlier hop had not delivered.
#[test]
fn a_placed_decode_matches_a_whole_sequence_forward() {
	let path = bundle("placed-full-extent");
	let placed = recipe.place(&path, &[]);
	let generation = placed.decode(&PROMPT, &mut recipe.sampler().temperature(0.0), &[], COLUMNS - PROMPT.len());
	assert_eq!(generation.ids.len(), COLUMNS, "decode reached {} ids, expected {COLUMNS}", generation.ids.len());
	let reference = placed.infer(&whole_sequence(&generation.ids));
	assert_eq!(
		bits(&generation.logits),
		bits(&reference),
		"a placed full-extent decode disagrees with a whole-sequence forward of the same ids over {} ranges:\n  decode    {:?}\n  reference {:?}",
		placed.split().len(),
		generation.logits,
		reference
	);
	std::fs::remove_file(path).unwrap();
}

/// The placed decode is the decode `recipe.decode` runs, so the two agree to
/// the bit on one device and must keep agreeing when the model is split.
#[test]
fn a_placed_decode_equals_the_unplaced_one() {
	let path = bundle("placed-equals");
	let placed = recipe.place(&path, &[]);
	let steps = COLUMNS - PROMPT.len();
	let left = placed.decode(&PROMPT, &mut recipe.sampler().temperature(0.0), &[], steps);
	let right = recipe.decode(&path, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], steps);
	assert_eq!(left.ids, right.ids, "the placed decode sampled different ids");
	assert_eq!(bits(&left.logits), bits(&right.logits), "the placed decode reached different logits:\n  placed   {:?}\n  unplaced {:?}", left.logits, right.logits);
	std::fs::remove_file(path).unwrap();
}

/// The tapes live across the whole decode, so a second decode on the same
/// placement has to start from nothing the first one left. Two decodes of the
/// same prompt must give the same ids and the same logits.
#[test]
fn a_placement_starts_each_sequence_clean() {
	let path = bundle("placed-reset");
	let placed = recipe.place(&path, &[]);
	let steps = COLUMNS - PROMPT.len();
	let first = placed.decode(&PROMPT, &mut recipe.sampler().temperature(0.0), &[], steps);
	let second = placed.decode(&PROMPT, &mut recipe.sampler().temperature(0.0), &[], steps);
	assert_eq!(first.ids, second.ids, "a second decode on the same placement sampled different ids, so a range kept the first decode's state");
	assert_eq!(bits(&first.logits), bits(&second.logits), "a second decode reached different logits:\n  first  {:?}\n  second {:?}", first.logits, second.logits);
	std::fs::remove_file(path).unwrap();
}

/// The placement reports what it holds: every named device takes blocks, every
/// range's tape holds bytes, and a hop moves one token's stream, not the
/// sequence.
#[test]
fn a_placement_reports_what_each_device_holds() {
	let path = bundle("placed-report");
	let placed = recipe.place(&path, &[]);
	let ranges = placed.split().len();
	assert!(ranges != 0, "the placement names no range");
	assert!(placed.split().iter().all(|blocks| *blocks != 0), "a device took no block: {:?}", placed.split());
	assert_eq!(placed.resident_bytes().len(), ranges, "resident bytes name {} devices for {ranges} ranges", placed.resident_bytes().len());
	assert!(placed.resident_bytes().iter().take(ranges).all(|bytes| *bytes != 0), "a range holds no bytes: {:?}", placed.resident_bytes());
	// One range never hops; more than one moves a token's stream once per boundary.
	if ranges == 1 {
		assert_eq!(placed.moved_bytes(), 0, "a single range moved {} bytes", placed.moved_bytes());
	} else {
		assert!(placed.moved_bytes() != 0, "a {ranges}-range placement moves nothing");
		assert!(
			placed.moved_bytes() < COLUMNS * placed.resident_bytes()[0],
			"a hop moved {} bytes, which is not one token's stream",
			placed.moved_bytes()
		);
	}
	std::fs::remove_file(path).unwrap();
}
