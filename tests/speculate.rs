//! Speculative decoding with a multi-token-prediction draft head.
//!
//! The head proposes the next id and the step after it accepts the proposal when
//! the model reaches the same id. The ids are the model's own either way, so the
//! whole claim is that `speculate` produces exactly the ids `decode` produces
//! from the same prompt and sampler. That is what this asserts, and it asserts it
//! on the id sequence, not on a tolerance.

use recipe::*;
use std::path::PathBuf;

const COLUMNS: usize = 24;
const VOCABULARY: usize = 8;
const INNER: usize = 16;
const PROMPT: [u32; 4] = [3, 1, 4, 1];

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend(value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
	bytes.extend(value.to_le_bytes());
}

fn push_str(bytes: &mut Vec<u8>, text: &str) {
	push_u64(bytes, text.len() as u64);
	bytes.extend(text.as_bytes());
}

/// A GGUF of F32 tensors, written in the order given. Shapes are GGUF-order, so
/// `[rows, columns]` here is what the reader reports as the tensor's shape.
fn write_gguf(path: &PathBuf, tensors: &[(&str, Vec<u64>, Vec<f32>)]) {
	let mut bytes = Vec::new();
	push_u32(&mut bytes, 0x4655_4747);
	push_u32(&mut bytes, 3);
	push_u64(&mut bytes, tensors.len() as u64);
	push_u64(&mut bytes, 0);
	let mut offset = 0_u64;
	for (name, shape, values) in tensors {
		push_str(&mut bytes, name);
		push_u32(&mut bytes, shape.len() as u32);
		for dimension in shape {
			push_u64(&mut bytes, *dimension);
		}
		push_u32(&mut bytes, 0);
		push_u64(&mut bytes, offset);
		offset += 4 * values.len() as u64;
	}
	while bytes.len() % 32 != 0 {
		bytes.push(0);
	}
	for (_, _, values) in tensors {
		for value in values {
			bytes.extend(value.to_le_bytes());
		}
	}
	std::fs::write(path, bytes).unwrap();
}

/// Every tensor the head reads, one element each. `Draft::open` looks up each
/// tensor's dimension before it checks any shape, so a fixture that names fewer
/// than all ten reports an absent tensor rather than the shape it wanted.
fn placeholders() -> Vec<(&'static str, Vec<u64>, Vec<f32>)> {
	[
		"token_embd.weight",
		"nextn.enorm.weight",
		"nextn.hnorm.weight",
		"nextn.eh_proj.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"nextn.shared_head.gate.weight",
		"nextn.shared_head.norm.weight",
		"nextn.shared_head.head.weight",
		"nextn.shared_head.head.bias",
	]
	.into_iter()
	.map(|name| (name, vec![1, 1], vec![0.0]))
	.collect()
}

/// Small deterministic values, spread so the head's logits are not all equal.
fn weights(count: usize, seed: u64) -> Vec<f32> {
	let mut state = seed | 1;
	(0..count)
		.map(|_| {
			state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			((state >> 40) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.25
		})
		.collect()
}

fn dataset() -> String {
	let directory = std::env::temp_dir().join(format!("recipe-speculate-{}", std::process::id()));
	std::fs::create_dir_all(&directory).unwrap();
	let mut text = String::new();
	for column in 0..COLUMNS {
		text.push_str(&format!("x{column},"));
	}
	for target in 0..VOCABULARY {
		text.push_str(&format!("y{target}{}", if target + 1 == VOCABULARY { "\n" } else { "," }));
	}
	for row in 0..48 {
		let t = row as f64 / 48.0;
		for column in 0..COLUMNS {
			text.push_str(&format!("{},", (t + column as f64 / COLUMNS as f64).sin()));
		}
		for target in 0..VOCABULARY {
			text.push_str(&format!("{}{}", (t * (target + 1) as f64).cos(), if target + 1 == VOCABULARY { "\n" } else { "," }));
		}
	}
	std::fs::write(directory.join("rows.csv"), text).unwrap();
	directory.to_str().unwrap().to_owned()
}

fn bundle(name: &str) -> PathBuf {
	let directory = dataset();
	let path = std::env::temp_dir().join(format!("recipe-speculate-{}-{name}.ogdl", std::process::id()));
	let names = (0..VOCABULARY).map(|target| format!("y{target}")).collect::<Vec<_>>();
	let data = recipe.data(directory.as_str()).target(names.as_slice());
	let model = recipe.model().conv(4, 3).relu().layer(VOCABULARY).loss(mse);
	recipe.train().fp(32).seed(17).lr(0.01).epochs(3).stop(0.0).save(&path).run(&model, &data);
	path
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

/// The draft head fuses the model's final hidden state, whose width is a property
/// of the compiled graph rather than of anything the DSL states. Rather than
/// hard-coding it, ask: a deliberately wrong `token_embd` makes `Draft::open`
/// report the shape it wanted, and the test builds the real head from that.
fn hidden_width(model: &PathBuf) -> usize {
	let probe = std::env::temp_dir().join(format!("recipe-speculate-{}-probe.gguf", std::process::id()));
	write_gguf(&probe, &placeholders());
	let message = panic_text(std::panic::catch_unwind(|| {
		let _ = recipe.speculate(model, &probe, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 1);
	}));
	std::fs::remove_file(&probe).unwrap();
	let takes = message.split("the head takes [").nth(1).unwrap_or_else(|| panic!("unexpected probe error: {message}"));
	let width = takes.split(',').next().unwrap().trim();
	width.parse().unwrap_or_else(|error| panic!("cannot read the head width from {message:?}: {error}"))
}

fn draft(name: &str, width: usize) -> PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-speculate-{}-{name}.gguf", std::process::id()));
	write_gguf(
		&path,
		&[
			("token_embd.weight", vec![width as u64, VOCABULARY as u64], weights(width * VOCABULARY, 11)),
			("nextn.enorm.weight", vec![width as u64], vec![1.0; width]),
			("nextn.hnorm.weight", vec![width as u64], vec![1.0; width]),
			("nextn.eh_proj.weight", vec![2 * width as u64, width as u64], weights(2 * width * width, 23)),
			("blk.0.ffn_up.weight", vec![width as u64, INNER as u64], weights(width * INNER, 37)),
			("blk.0.ffn_down.weight", vec![INNER as u64, width as u64], weights(INNER * width, 41)),
			("nextn.shared_head.gate.weight", vec![width as u64], vec![1.0; width]),
			("nextn.shared_head.norm.weight", vec![width as u64], vec![1.0; width]),
			("nextn.shared_head.head.weight", vec![width as u64, VOCABULARY as u64], weights(width * VOCABULARY, 53)),
			("nextn.shared_head.head.bias", vec![VOCABULARY as u64], weights(VOCABULARY, 59)),
		],
	);
	path
}

/// The claim. A draft head changes which ids are *proposed*, never which ids are
/// *produced*: every id a speculating run emits is the model's own, so the run
/// must end on exactly the ids `decode` reaches from the same prompt and sampler.
#[test]
fn speculating_produces_the_ids_decode_produces() {
	let model = bundle("ids");
	let head = draft("ids", hidden_width(&model));

	let plain = recipe.decode(&model, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 12);
	let speculated = recipe.speculate(&model, &head, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 12);

	assert_eq!(speculated.ids, plain.ids, "a draft head changed the ids the model produced");
	assert_eq!(
		speculated.logits.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
		plain.logits.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
		"a draft head changed the model's final logits"
	);
	std::fs::remove_file(model).unwrap();
	std::fs::remove_file(head).unwrap();
}

/// The head is actually consulted, and the accounting is coherent. Without this
/// the test above would pass just as well against a draft head that never ran.
#[test]
fn the_draft_head_proposes_and_the_model_verifies() {
	let model = bundle("counts");
	let head = draft("counts", hidden_width(&model));

	let speculated = recipe.speculate(&model, &head, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 12);
	let plain = recipe.decode(&model, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 12);

	assert!(speculated.proposed > 0, "the draft head proposed nothing");
	assert!(speculated.accepted <= speculated.proposed, "accepted {} of {} proposals", speculated.accepted, speculated.proposed);
	assert_eq!((plain.proposed, plain.accepted), (0, 0), "a decode without a draft head counted proposals");
	std::fs::remove_file(model).unwrap();
	std::fs::remove_file(head).unwrap();
}

/// A head whose tensors do not fit the model it drafts for is refused by name,
/// rather than read at the wrong offsets.
#[test]
fn a_draft_head_of_the_wrong_shape_is_refused() {
	let model = bundle("shape");
	let wrong = std::env::temp_dir().join(format!("recipe-speculate-{}-wrong.gguf", std::process::id()));
	let mut tensors = placeholders();
	tensors[0] = ("token_embd.weight", vec![3, 5], weights(15, 7));
	write_gguf(&wrong, &tensors);

	let message = panic_text(std::panic::catch_unwind(|| {
		let _ = recipe.speculate(&model, &wrong, &PROMPT, &mut recipe.sampler().temperature(0.0), &[], 4);
	}));
	assert!(message.contains("draft head tensor token_embd.weight is [3, 5]"), "unexpected error: {message}");
	std::fs::remove_file(model).unwrap();
	std::fs::remove_file(wrong).unwrap();
}
