//! GGUF reading through the public `recipe.gguf` path: the checked-in sample
//! that llama.cpp's `gguf` example wrote, its independent header dump, the
//! four-shard split `gguf-split` cut from it with a metadata-only first shard,
//! and a Q8_0 tensor decoded against a scalar reading of the format.

use recipe::*;
use std::fmt::Write as _;

/// The dump `llama-gguf data/gguf/sample.gguf r` printed, reduced to the
/// header facts it lists: metadata keys in order and every tensor's name,
/// byte size, and data offset.
fn independent_dump() -> (u64, usize, Vec<String>, Vec<(String, usize, usize)>) {
	let text = std::fs::read_to_string("data/gguf/sample.txt").unwrap_or_else(|error| panic!("cannot read the independent dump: {error}"));
	let (mut alignment, mut data, mut keys, mut tensors) = (0_u64, 0_usize, Vec::new(), Vec::new());
	for line in text.lines().filter_map(|line| line.strip_prefix("gguf_ex_read_0: ")) {
		if let Some(value) = line.strip_prefix("alignment:") {
			alignment = value.trim().parse().unwrap();
		} else if let Some(value) = line.strip_prefix("data offset:") {
			data = value.trim().parse().unwrap();
		} else if line.starts_with("kv[") {
			keys.push(line.split("key = ").nth(1).unwrap().to_owned());
		} else if line.starts_with("tensor[") {
			let mut fields = line.split(", ");
			let name = fields.next().unwrap().split("name = ").nth(1).unwrap().to_owned();
			let size = fields.next().unwrap().split("size = ").nth(1).unwrap().parse().unwrap();
			let offset = fields.next().unwrap().split("offset = ").nth(1).unwrap().parse().unwrap();
			tensors.push((name, size, offset));
		}
	}
	(alignment, data, keys, tensors)
}

#[test]
fn sample_matches_the_independent_dump() {
	let (alignment, data, keys, tensors) = independent_dump();
	let model = recipe.gguf("data/gguf/sample.gguf");
	// The example writes no alignment key, so the dump reports the default.
	assert_eq!(model.value("general.alignment").and_then(GgufValue::integer).unwrap_or(32), alignment);
	assert_eq!(model.metadata().iter().map(|(key, _)| key.clone()).collect::<Vec<_>>(), keys);
	assert_eq!(model.tensors().iter().map(|tensor| (tensor.name.clone(), tensor.bytes, tensor.offset)).collect::<Vec<_>>(), tensors);
	assert_eq!(model.value("some.parameter.string"), Some(&GgufValue::String("hello world".to_owned())));
	assert_eq!(model.value("some.parameter.arr.i16"), Some(&GgufValue::Array(vec![GgufValue::I16(1), GgufValue::I16(2), GgufValue::I16(3), GgufValue::I16(4)])));
	assert_eq!(model.value("some.parameter.arr.str"), Some(&GgufValue::Array(vec![GgufValue::String("hello".to_owned()), GgufValue::String("world".to_owned()), GgufValue::String("!".to_owned())])));
	assert_eq!(model.value("some.parameter.bool"), Some(&GgufValue::Bool(true)));
	let file = std::fs::read("data/gguf/sample.gguf").unwrap();
	for tensor in model.tensors() {
		assert_eq!(model.data(tensor), &file[data + tensor.offset..data + tensor.offset + tensor.bytes], "{} maps the wrong bytes", tensor.name);
		assert_eq!(tensor.kind, 0, "{} is not the F32 the example writes", tensor.name);
		let values = model.values(tensor).unwrap();
		assert_eq!(values.len() * 4, tensor.bytes);
		// The example fills tensor_i with the constant 100 + i.
		let index = tensor.name.strip_prefix("tensor_").unwrap().parse::<usize>().unwrap();
		assert!(values.iter().all(|value| *value == (100 + index) as f64), "{} values differ from the example fill", tensor.name);
	}
}

#[test]
fn split_shards_hold_the_same_tensors() {
	let whole = recipe.gguf("data/gguf/sample.gguf");
	let split = recipe.gguf("data/gguf/split/sample-00001-of-00004.gguf");
	assert_eq!(split.value("split.count").and_then(GgufValue::integer), Some(4));
	assert_eq!(split.value("split.tensors.count").and_then(GgufValue::integer), Some(whole.tensors().len() as u64));
	assert_eq!(std::fs::metadata("data/gguf/split/sample-00001-of-00004.gguf").unwrap().len(), 768, "the first shard is expected to hold metadata only");
	assert_eq!(split.tensors().len(), whole.tensors().len());
	for tensor in whole.tensors() {
		let shard = split.tensor(&tensor.name).unwrap_or_else(|| panic!("{} is absent from the split", tensor.name));
		assert_eq!((&shard.shape, shard.kind, shard.bytes), (&tensor.shape, tensor.kind, tensor.bytes), "{} descriptor differs", tensor.name);
		assert_eq!(split.data(shard), whole.data(tensor), "{} bytes differ", tensor.name);
	}
}

/// Writes one GGUF file: version 3, the given metadata strings, and tensors of
/// the given type and shape whose data lands at 32-byte alignment.
fn write_gguf(path: &std::path::Path, metadata: &[(&str, &str)], tensors: &[(&str, u32, Vec<u64>, Vec<u8>)]) {
	let mut header = Vec::new();
	header.extend(0x4655_4747_u32.to_le_bytes());
	header.extend(3_u32.to_le_bytes());
	header.extend((tensors.len() as u64).to_le_bytes());
	header.extend((metadata.len() as u64).to_le_bytes());
	let string = |output: &mut Vec<u8>, text: &str| {
		output.extend((text.len() as u64).to_le_bytes());
		output.extend(text.as_bytes());
	};
	for (key, value) in metadata {
		string(&mut header, key);
		header.extend(8_u32.to_le_bytes());
		string(&mut header, value);
	}
	let mut offset = 0_u64;
	for (name, kind, shape, data) in tensors {
		string(&mut header, name);
		header.extend((shape.len() as u32).to_le_bytes());
		for dimension in shape {
			header.extend(dimension.to_le_bytes());
		}
		header.extend(kind.to_le_bytes());
		header.extend(offset.to_le_bytes());
		offset += (data.len() as u64).div_ceil(32) * 32;
	}
	header.resize(header.len().div_ceil(32) * 32, 0);
	for (_, _, _, data) in tensors {
		header.extend(data);
		header.resize(header.len().div_ceil(32) * 32, 0);
	}
	std::fs::write(path, header).unwrap();
}

#[test]
fn q8_0_decodes_like_the_scalar_format() {
	// Two Q8_0 blocks with half scales that are exact in f16: 0.5 and 2.0.
	let mut q8 = Vec::new();
	let mut expected = Vec::new();
	for (scale_bits, scale) in [(0x3800_u16, 0.5_f64), (0x4000, 2.0)] {
		q8.extend(scale_bits.to_le_bytes());
		for lane in 0..32_i32 {
			let code = (lane * 7 - 100) as i8;
			q8.push(code as u8);
			expected.push(scale * f64::from(code));
		}
	}
	let f32s = (0..8).map(|index| (index as f32 * 0.25 - 1.0).to_le_bytes()).flatten().collect::<Vec<_>>();
	let path = std::env::temp_dir().join(format!("recipe-gguf-{}.gguf", std::process::id()));
	write_gguf(&path, &[("general.architecture", "recipe")], &[("blk.0.ffn_up.weight", 8, vec![32, 2], q8.clone()), ("output_norm.weight", 0, vec![8], f32s.clone())]);
	let model = recipe.gguf(&path);
	let _ = std::fs::remove_file(&path);
	assert_eq!(model.value("general.architecture"), Some(&GgufValue::String("recipe".to_owned())));
	let weight = model.tensor("blk.0.ffn_up.weight").unwrap();
	assert_eq!((weight.kind, weight.bytes, &weight.shape), (8, 68, &vec![32, 2]));
	assert_eq!(model.data(weight), &q8[..]);
	assert_eq!(model.values(weight).unwrap(), expected);
	let output_norm = model.tensor("output_norm.weight").unwrap();
	assert_eq!(model.values(output_norm).unwrap(), (0..8).map(|index| f64::from(index as f32 * 0.25 - 1.0)).collect::<Vec<_>>());
	let mut summary = String::new();
	let _ = writeln!(summary, "gguf: {} tensors, {} metadata pairs", model.tensors().len(), model.metadata().len());
	assert!(!summary.is_empty());
}
