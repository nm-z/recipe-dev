use recipe::*;
use std::path::PathBuf;

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend(value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
	bytes.extend(value.to_le_bytes());
}

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

#[test]
fn reader_rejects_an_overflowed_tensor_extent() {
	let path = fixture("overflow", 1_u64 << 62, &[]);
	let message = panic_text(std::panic::catch_unwind(|| drop(recipe.gguf(&path))));
	assert!(message.contains("tensor x byte extent overflows"), "unexpected error: {message}");
	std::fs::remove_file(path).unwrap();
}
