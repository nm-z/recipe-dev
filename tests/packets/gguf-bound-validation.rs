//! Standalone packet for GGUF builder errors and bound-model serving.
//!
//! Compile this source against the Recipe library, then run it from a writable
//! directory. It writes tiny llama GGUF fixtures, checks that malformed plans
//! fail before a tape is allocated, and compares one HTTP serve decode with a
//! direct bound decode.

use recipe::*;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

const WIDTH: usize = 4;
const VOCABULARY: usize = 3;

#[derive(Clone)]
struct Tensor {
	name: &'static str,
	kind: u32,
	shape: Vec<u64>,
	bytes: Vec<u8>,
	offset: usize,
}

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

fn metadata_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
	push_string(bytes, key);
	push_u32(bytes, 8); // GGUF_TYPE_STRING
	push_string(bytes, value);
}

fn metadata_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
	push_string(bytes, key);
	push_u32(bytes, 4); // GGUF_TYPE_UINT32
	push_u32(bytes, value);
}

fn tensor_header(bytes: &mut Vec<u8>, tensor: &Tensor) {
	push_string(bytes, tensor.name);
	push_u32(bytes, tensor.shape.len() as u32);
	for dimension in &tensor.shape {
		push_u64(bytes, *dimension);
	}
	push_u32(bytes, tensor.kind);
	push_u64(bytes, tensor.offset as u64);
}

fn write_gguf(path: &Path, mut tensors: Vec<Tensor>) {
	let mut bytes = Vec::new();
	push_u32(&mut bytes, 0x4655_4747);
	push_u32(&mut bytes, 3);
	push_u64(&mut bytes, tensors.len() as u64);
	push_u64(&mut bytes, 5); // metadata count
	metadata_string(&mut bytes, "general.architecture", "llama");
	metadata_u32(&mut bytes, "llama.embedding_length", WIDTH as u32);
	metadata_u32(&mut bytes, "llama.attention.head_count", 1);
	metadata_u32(&mut bytes, "llama.feed_forward_length", WIDTH as u32);
	metadata_u32(&mut bytes, "llama.block_count", 0);
	let mut at = 0;
	for tensor in &mut tensors {
		tensor.offset = at;
		at += tensor.bytes.len().div_ceil(32) * 32;
	}
	for tensor in &tensors {
		tensor_header(&mut bytes, tensor);
	}
	align(&mut bytes, 32);
	let data_offset = bytes.len();
	for tensor in &tensors {
		align(&mut bytes, 32);
		assert_eq!(bytes.len(), data_offset + tensor.offset);
		bytes.extend(&tensor.bytes);
	}
	align(&mut bytes, 32);
	std::fs::write(path, bytes).unwrap();
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
	values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn tensor(name: &'static str, shape: Vec<u64>, bytes: Vec<u8>) -> Tensor {
	Tensor { name, kind: 0, shape, bytes, offset: 0 }
}

fn valid_tensors() -> Vec<Tensor> {
	let embedding = [1.25_f32, -2.5, 3.75, -4.5, 5.25, -6.5, 7.75, -8.25, 9.5, -10.75, 11.125, -12.25];
	let output = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
	vec![tensor("token_embd.weight", vec![WIDTH as u64, VOCABULARY as u64], f32_bytes(&embedding)), tensor("output.weight", vec![WIDTH as u64, VOCABULARY as u64], f32_bytes(&output))]
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
	if let Some(text) = payload.downcast_ref::<String>() {
		return text.clone();
	}
	if let Some(text) = payload.downcast_ref::<&str>() {
		return (*text).to_owned();
	}
	"non-string panic".to_owned()
}

fn assert_model_error(path: &Path, needle: &str) {
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recipe.gguf(path).model()));
	let error = match result {
		Ok(_) => panic!("malformed GGUF unexpectedly built"),
		Err(error) => error,
	};
	let text = panic_text(error);
	assert!(text.contains(needle), "error {text:?} does not contain {needle:?}");
	println!("expected model error: {text}");
}

fn check_model_errors(directory: &Path) {
	let missing = directory.join("missing.gguf");
	let mut tensors = valid_tensors();
	tensors.remove(0);
	write_gguf(&missing, tensors);
	assert_model_error(&missing, "token_embd.weight is absent");

	let unread = directory.join("unread.gguf");
	let mut tensors = valid_tensors();
	tensors.push(tensor("unexpected.weight", vec![1, 1], f32_bytes(&[0.0])));
	write_gguf(&unread, tensors);
	assert_model_error(&unread, "1 tensors are read by no node");

	let shape = directory.join("shape.gguf");
	let mut tensors = valid_tensors();
	tensors[0].shape[0] = 3;
	write_gguf(&shape, tensors);
	assert_model_error(&shape, "token_embd.weight has shape");

	let plan = directory.join("plan.gguf");
	write_gguf(&plan, valid_tensors());
	let file = recipe.gguf(&plan);
	let binding = file.plan().named(&file, "output.weight").named(&file, "output.weight");
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| file.infer(&recipe.model().layer(VOCABULARY), &binding, &[1.0, 2.0, 3.0, 4.0], WIDTH)));
	let error = match result {
		Ok(_) => panic!("extra plan entry unexpectedly ran"),
		Err(error) => error,
	};
	let text = panic_text(error);
	assert!(text.contains("plan names 1 more weights"), "error {text:?} does not identify the unbound plan entry");
	println!("expected plan error: {text}");
}

fn chunk_ids(response: &str) -> Vec<u32> {
	let body = response.split_once("\r\n\r\n").map_or(response, |(_, body)| body);
	let lines = body.split("\r\n").collect::<Vec<_>>();
	let mut ids = Vec::new();
	let mut at = 0;
	while at + 1 < lines.len() {
		let size = usize::from_str_radix(lines[at], 16).unwrap();
		if size == 0 {
			break;
		}
		assert_eq!(lines[at + 1].len(), size, "chunk payload length");
		ids.push(lines[at + 1].trim_end_matches('\n').parse().unwrap());
		at += 2;
	}
	ids
}

fn check_serve(path: &Path) {
	let file = recipe.gguf(path);
	let bound = file.model();
	let mut sampler = recipe.sampler().temperature(0.0);
	let direct = bound.decode(4, &[], &[0], &mut sampler, &[], 1);
	assert_eq!(direct.logits.len(), bound.vocabulary() * 4);
	let expected = direct.ids[1..].to_vec();
	let server = recipe.gguf(path).model();
	let address = "127.0.0.1:38127";
	let handle = std::thread::spawn(move || server.serve(4, &[], address, 1));
	let mut stream = None;
	for _ in 0..200 {
		match std::net::TcpStream::connect(address) {
			Ok(connection) => {
				stream = Some(connection);
				break;
			}
			Err(_) => std::thread::sleep(Duration::from_millis(20)),
		}
	}
	let mut stream = stream.expect("bound serve did not listen");
	stream.write_all(b"GET /decode?ids=0&budget=1&temperature=0 HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
	let mut response = String::new();
	stream.read_to_string(&mut response).unwrap();
	handle.join().expect("bound serve panicked");
	let served = chunk_ids(&response);
	assert_eq!(served, expected, "served ids differ from direct decode");
	println!("bound serve ids {served:?} match direct ids {expected:?}");
}

fn main() {
	unsafe { std::env::set_var("RECIPE_FORCE_CPU", "1") };
	let directory = PathBuf::from("fixtures");
	std::fs::create_dir_all(&directory).unwrap();
	check_model_errors(&directory);
	let valid = directory.join("valid.gguf");
	write_gguf(&valid, valid_tensors());
	check_serve(&valid);
	println!("GGUF bound error and serve checks passed");
}
