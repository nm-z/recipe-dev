//! The mapped GGUF reader's bounds.
//!
//! A tensor descriptor states a shape and a quantization, and the reader turns
//! those into a byte extent. That arithmetic has to be bounded: a product that
//! wraps lands back inside the mapping, so the range check accepts it and the
//! tensor is read as a short, possibly empty, one instead of being refused.

use recipe::*;
use std::path::PathBuf;

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend(value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
	bytes.extend(value.to_le_bytes());
}

/// One tensor named `x`, of the given shape and F32 kind, with `data` after a
/// 64-byte header.
fn fixture(name: &str, shape: u64, data: &[u8]) -> PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-gguf-{name}-{}-{}.gguf", std::process::id(), std::thread::current().name().unwrap_or("test")));
	let mut bytes = Vec::new();
	push_u32(&mut bytes, 0x4655_4747);
	push_u32(&mut bytes, 3);
	push_u64(&mut bytes, 1);
	push_u64(&mut bytes, 0);
	push_u64(&mut bytes, 1);
	bytes.push(b'x');
	push_u32(&mut bytes, 1);
	push_u64(&mut bytes, shape);
	push_u32(&mut bytes, 0);
	push_u64(&mut bytes, 0);
	bytes.resize(64, 0);
	bytes.extend(data);
	std::fs::write(&path, bytes).unwrap();
	path
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

#[test]
fn reader_exposes_a_mapped_f32_descriptor() {
	let path = fixture("mapped", 1, &1.5_f32.to_le_bytes());
	let gguf = recipe.gguf(&path);
	let tensor = gguf.tensor("x").unwrap();
	assert_eq!((tensor.shape.as_slice(), tensor.kind, tensor.offset, tensor.bytes), (&[1_u64][..], 0, 0, 4));
	assert_eq!(gguf.data(tensor), 1.5_f32.to_le_bytes());
	std::fs::remove_file(path).unwrap();
}

/// 2^62 F32 elements is 2^64 bytes, which wraps to zero. Unbounded, the reader
/// accepted the tensor and reported an empty extent that sits inside the file.
#[test]
fn reader_rejects_a_wrapped_tensor_extent() {
	let path = fixture("wrapped", 1_u64 << 62, &[]);
	let message = panic_text(std::panic::catch_unwind(|| drop(recipe.gguf(&path))));
	assert!(message.contains("tensor x byte extent exceeds the address space"), "unexpected error: {message}");
	std::fs::remove_file(path).unwrap();
}
