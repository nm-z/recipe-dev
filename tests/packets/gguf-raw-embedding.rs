//! Standalone packet for the raw F32/F16 embedding-table path.
//!
//! Compile this source against the Recipe library, then run it from a writable
//! directory. It writes two tiny GGUF files, binds each through `Gguf::model`,
//! and checks the mapped gather output bit for bit.

use recipe::*;
use std::path::{Path, PathBuf};

const WIDTH: usize = 4;
const VOCABULARY: usize = 3;

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend(value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
	bytes.extend(value.to_le_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
	push_u64(bytes, value.len() as u64);
	bytes.extend(value.as_bytes());
}

fn align(bytes: &mut Vec<u8>, boundary: usize) {
	bytes.resize(bytes.len().div_ceil(boundary) * boundary, 0);
}

fn push_metadata_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
	push_string(bytes, key);
	push_u32(bytes, 8); // GGUF_TYPE_STRING
	push_string(bytes, value);
}

fn push_metadata_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
	push_string(bytes, key);
	push_u32(bytes, 4); // GGUF_TYPE_UINT32
	push_u32(bytes, value);
}

fn push_tensor(bytes: &mut Vec<u8>, name: &str, kind: u32, offset: usize) {
	push_string(bytes, name);
	push_u32(bytes, 2);
	push_u64(bytes, WIDTH as u64);
	push_u64(bytes, VOCABULARY as u64);
	push_u32(bytes, kind);
	push_u64(bytes, offset as u64);
}

fn write_gguf(path: &Path, embedding_kind: u32, embedding: &[u8], output: &[u8]) {
	let mut bytes = Vec::new();
	push_u32(&mut bytes, 0x4655_4747);
	push_u32(&mut bytes, 3);
	push_u64(&mut bytes, 2); // tensor count
	push_u64(&mut bytes, 5); // metadata count
	push_metadata_string(&mut bytes, "general.architecture", "llama");
	push_metadata_u32(&mut bytes, "llama.embedding_length", WIDTH as u32);
	push_metadata_u32(&mut bytes, "llama.attention.head_count", 1);
	push_metadata_u32(&mut bytes, "llama.feed_forward_length", WIDTH as u32);
	push_metadata_u32(&mut bytes, "llama.block_count", 0);
	let output_offset = embedding.len().div_ceil(32) * 32;
	push_tensor(&mut bytes, "token_embd.weight", embedding_kind, 0);
	push_tensor(&mut bytes, "output.weight", 0, output_offset);
	align(&mut bytes, 32);
	bytes.extend(embedding);
	align(&mut bytes, 32);
	bytes.extend(output);
	align(&mut bytes, 32);
	std::fs::write(path, bytes).unwrap();
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
	values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn f16_bytes(bits: &[u16]) -> Vec<u8> {
	bits.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
	assert_eq!(actual.len(), expected.len(), "{label} length");
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert_eq!(actual.to_bits(), expected.to_bits(), "{label}[{index}] = {actual:?}, expected {expected:?}");
	}
}

fn check_f32(path: &Path) {
	let rows = [1.25_f32, -2.5, 3.75, -4.5, 5.25, -6.5, 7.75, -8.25, 9.5, -10.75, 11.125, -12.25];
	let output_weights = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
	write_gguf(path, 0, &f32_bytes(&rows), &f32_bytes(&output_weights));
	let file = recipe.gguf(path);
	let tensor = file.tensor("token_embd.weight").unwrap();
	assert_eq!((tensor.kind, tensor.shape.clone(), tensor.bytes), (0, vec![4, 3], rows.len() * 4));
	assert_close(&file.values(tensor).unwrap(), &rows.iter().map(|value| f64::from(*value)).collect::<Vec<_>>(), "f32 values");
	assert_close(&file.row(tensor, 2).unwrap(), &rows[8..12].iter().map(|value| f64::from(*value)).collect::<Vec<_>>(), "f32 row");
	let bound = file.model();
	let output = bound.infer(&[0, 2]);
	println!("f32 raw output={output:?}");
	let expected = [rows[0], rows[8], rows[1], rows[9], rows[2], rows[10]].iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
	assert_close(&output, &expected, "f32 mapped gather");
	println!("f32 kind={} tensor_bytes={} rows={rows:?} output={output:?}", tensor.kind, tensor.bytes);
}

fn check_f16(path: &Path) {
	// These are exact, finite IEEE-754 binary16 values: 1..12 with alternating signs.
	let bits = [0x3c00_u16, 0xc000, 0x4200, 0xc400, 0x4500, 0xc600, 0x4700, 0xc800, 0x4880, 0xc900, 0x4980, 0xca00];
	let rows = [1.0_f64, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0];
	let output_weights = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
	write_gguf(path, 1, &f16_bytes(&bits), &f32_bytes(&output_weights));
	let file = recipe.gguf(path);
	let tensor = file.tensor("token_embd.weight").unwrap();
	assert_eq!((tensor.kind, tensor.shape.clone(), tensor.bytes), (1, vec![4, 3], bits.len() * 2));
	assert_close(&file.values(tensor).unwrap(), &rows, "f16 values");
	assert_close(&file.row(tensor, 1).unwrap(), &rows[4..8], "f16 row");
	let bound = file.model();
	let output = bound.infer(&[1, 2]);
	println!("f16 raw output={output:?}");
	let expected = [rows[4], rows[8], rows[5], rows[9], rows[6], rows[10]];
	assert_close(&output, &expected, "f16 mapped gather");
	println!("f16 kind={} tensor_bytes={} rows={rows:?} output={output:?}", tensor.kind, tensor.bytes);
}

fn main() {
	let directory = PathBuf::from("fixtures");
	std::fs::create_dir_all(&directory).unwrap();
	let f32_path = directory.join("raw-embedding-f32.gguf");
	let f16_path = directory.join("raw-embedding-f16.gguf");
	check_f32(&f32_path);
	check_f16(&f16_path);
	println!("raw F32 and F16 mapped embedding gathers passed");
}
