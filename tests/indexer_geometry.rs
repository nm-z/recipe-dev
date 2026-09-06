//! A public GGUF attention run that distinguishes the reference indexer geometry
//! from normalizing each key before pooling and from rectifying a sum of heads.
use recipe::*;
use std::path::{Path, PathBuf};

const BASE: f64 = 10_000.0;
const EPSILON: f64 = 0.00001;

struct TemporaryFile(PathBuf);
impl Drop for TemporaryFile {
	fn drop(&mut self) {
		let _ = std::fs::remove_file(&self.0);
	}
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend_from_slice(&value.to_le_bytes());
}
fn push_u64(bytes: &mut Vec<u8>, value: u64) {
	bytes.extend_from_slice(&value.to_le_bytes());
}

/// GGML contractions consume output-major values. GGUF descriptors still use
/// `[inputs, outputs]`, so this helper writes one row for each output channel.
fn output_major(inputs: usize, rows: &[&[f64]]) -> Vec<f64> {
	assert!(rows.iter().all(|row| row.len() == inputs));
	rows.iter().flat_map(|row| row.iter().copied()).collect()
}

fn write_f64_gguf(path: &Path, tensors: &[(&str, &[u64], &[f64])]) {
	let mut header = Vec::new();
	push_u32(&mut header, 0x4655_4747); // GGUF
	push_u32(&mut header, 3);
	push_u64(&mut header, tensors.len() as u64);
	push_u64(&mut header, 0); // metadata pairs
	let mut offsets = Vec::with_capacity(tensors.len());
	let mut offset = 0_u64;
	for (name, shape, values) in tensors {
		assert_eq!(shape.iter().product::<u64>() as usize, values.len());
		push_u64(&mut header, name.len() as u64);
		header.extend_from_slice(name.as_bytes());
		push_u32(&mut header, shape.len() as u32);
		for dimension in *shape {
			push_u64(&mut header, *dimension);
		}
		push_u32(&mut header, 28); // F64
		push_u64(&mut header, offset);
		offsets.push(offset);
		offset += values.len() as u64 * 8;
	}
	let data_start = header.len().div_ceil(32) * 32;
	header.resize(data_start, 0);
	for ((_, _, values), tensor_offset) in tensors.iter().zip(offsets) {
		assert_eq!(header.len() as u64 - data_start as u64, tensor_offset);
		for value in *values {
			header.extend_from_slice(&value.to_le_bytes());
		}
	}
	std::fs::write(path, header).unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
}

fn rms2(value: [f64; 2]) -> [f64; 2] {
	let magnitude = ((value[0] * value[0] + value[1] * value[1]) / 2.0 + EPSILON).sqrt();
	[value[0] / magnitude, value[1] / magnitude]
}
fn rope(value: [f64; 2], position: usize) -> [f64; 2] {
	let angle = position as f64 * BASE.powf(-0.0); // dims = 2, so the one frequency is one.
	let (cosine, sine) = (angle.cos(), angle.sin());
	[value[0] * cosine - value[1] * sine, value[1] * cosine + value[0] * sine]
}
fn dot(left: [f64; 2], right: [f64; 2]) -> f64 {
	left[0] * right[0] + left[1] * right[1]
}
fn mean(keys: &[[f64; 2]]) -> [f64; 2] {
	let count = keys.len() as f64;
	[keys.iter().map(|key| key[0]).sum::<f64>() / count, keys.iter().map(|key| key[1]).sum::<f64>() / count]
}
fn corrected_representative(keys: &[[f64; 2]], block: usize, gamma: [f64; 2]) -> [f64; 2] {
	let unit = rms2(mean(keys));
	rope([unit[0] * gamma[0], unit[1] * gamma[1]], block * 2)
}
fn per_key_representative(keys: &[[f64; 2]], start: usize, gamma: [f64; 2]) -> [f64; 2] {
	let transformed = keys
		.iter()
		.enumerate()
		.map(|(index, key)| {
			let unit = rms2(*key);
			rope([unit[0] * gamma[0], unit[1] * gamma[1]], start + index)
		})
		.collect::<Vec<_>>();
	mean(&transformed)
}
fn score(query: [f64; 2], representative: [f64; 2]) -> f64 {
	// Two query heads are opposites. ReLU before the head reduction therefore
	// produces the absolute dot product, while ReLU after the reduction gives 0.
	let value = dot(query, representative);
	value.max(0.0) + (-value).max(0.0)
}
fn score_after_head_sum(query: [f64; 2], representative: [f64; 2]) -> f64 {
	(dot(query, representative) + dot([-query[0], -query[1]], representative)).max(0.0)
}

#[test]
fn scored_attention_matches_pooled_reference_and_per_head_relu() {
	// Keep this test runnable on hosts without an accelerator while preserving an
	// explicit RECIPE_DEVICE or RECIPE_FORCE_CPU requested by the caller.
	if std::env::var_os("RECIPE_DEVICE").is_none() && std::env::var_os("RECIPE_FORCE_CPU").is_none() {
		unsafe { std::env::set_var("RECIPE_FORCE_CPU", "1") };
	}

	// Positions 0 and 1 form the first block. Positions 2 and 3 form the
	// second. The second block's raw keys have unequal magnitudes, so the
	// reference (pool, then RMS/RoPE) and the old (RMS/RoPE, then pool) paths
	// rank the two blocks differently.
	let input = vec![
		8.0, 8.0, 10.0, 0.0, // raw key dimension 0
		6.0, 6.0, 0.0, 1.0, // raw key dimension 1
		1.0, 1.0, 1.0, 1.0, // constant query feature
	];
	let raw_blocks = [[8.0, 6.0], [8.0, 6.0], [10.0, 0.0], [0.0, 1.0]];
	// This query is the inverse block-3 rotation of the second block's pooled
	// representative. The second head is its negation, which makes head-wise
	// ReLU observable independently of the pooling correction.
	let query = [3.2045682331608485, 0.21809619538928263];
	let main_rows: [&[f64]; 6] = [
		&[0.0, 0.0, 0.0],
		&[0.0, 0.0, 0.0],
		&[0.0, 0.0, 0.0],
		&[0.0, 0.0, 0.0],
		&[1.0, 2.0, 0.0], // value head 0 = key_dim0 + 2 * key_dim1
		&[0.0, 0.0, 0.0],
	];
	let index_rows: [&[f64]; 6] = [&[0.0, 0.0, query[0]], &[0.0, 0.0, query[1]], &[0.0, 0.0, -query[0]], &[0.0, 0.0, -query[1]], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]];
	let output_rows: [&[f64]; 3] = [&[1.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]];
	let main_values = output_major(3, &main_rows);
	let index_values = output_major(3, &index_rows);
	let output_values = output_major(2, &output_rows);
	let scales = [1.0, 1.0, 1.0, 1.0, 1.1, 20.0];
	let path = std::env::temp_dir().join(format!("recipe-indexer-{}.gguf", std::process::id()));
	let _temporary = TemporaryFile(path.clone());
	write_f64_gguf(&path, &[("main", &[3, 6], &main_values), ("index", &[3, 6], &index_values), ("scale", &[6], &scales), ("output", &[2, 3], &output_values)]);

	let file = recipe.gguf(&path);
	let plan = file.plan().named(&file, "main").named(&file, "index").named(&file, "scale").named(&file, "output");
	let blocks = recipe.model().attn(1).head(2).rope(2, BASE).index(2, 2, 2, 1).score(recipe::rms, 2);
	let actual = file.infer(&blocks, &plan, &input, 3);

	let transformed_query = rope(rms2(query), 3);
	let key_gamma = [1.1, 20.0];
	let first = corrected_representative(&raw_blocks[0..2], 0, key_gamma);
	let second = corrected_representative(&raw_blocks[2..4], 1, key_gamma);
	let corrected_scores = [score(transformed_query, first), score(transformed_query, second)];
	let old_first = per_key_representative(&raw_blocks[0..2], 0, key_gamma);
	let old_second = per_key_representative(&raw_blocks[2..4], 2, key_gamma);
	let old_scores = [score(transformed_query, old_first), score(transformed_query, old_second)];
	let summed_scores = [score_after_head_sum(transformed_query, first), score_after_head_sum(transformed_query, second)];
	let corrected_selected = corrected_scores.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0))).map(|(index, _)| index).unwrap();
	let old_selected = old_scores.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0))).map(|(index, _)| index).unwrap();
	assert_eq!(corrected_selected, 1, "host reference selects scores {corrected_scores:?}");
	assert_eq!(old_selected, 0, "the pre-pooling geometry should select the other block: {old_scores:?}");
	assert_eq!(summed_scores, [0.0, 0.0], "head-wise ReLU must precede the head reduction: {summed_scores:?}");
	assert!(corrected_scores[1] > corrected_scores[0] + 0.5, "pooled reference lacks a discriminating margin: {corrected_scores:?}");
	assert!(old_scores[0] > old_scores[1], "old geometry unexpectedly agrees: {old_scores:?}");

	// The main Q and K planes are zero, so attention over the selected causal
	// block is uniform. Its value rows are [10, 2], giving 6 when block 1 is
	// admitted. The old paths admit [20, 20] and would return 20 instead.
	let channel0 = &actual[..4];
	assert!((channel0[3] - 6.0).abs() < 1e-9, "public output {:?} does not match selected block 1", channel0);
	assert!((channel0[3] - 20.0).abs() > 5.0, "old block selection was not excluded: {:?}", channel0);
}
