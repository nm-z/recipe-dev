//! Issue #559's third acceptance bullet: a quantized tensor bound as a parameter
//! buffer must carry exactly the bytes the file holds and, once decompressed,
//! the same values `values()` reports. Binding decodes nothing, so the two paths
//! agreeing is what says the bound buffer is the same weights.

use recipe::*;

/// Write a GGUF holding one Q8_0 tensor of `blocks` blocks of 32 weights.
fn q8_0_file(path: &std::path::Path, blocks: usize) -> Vec<f64> {
	let (mut bytes, mut expected) = (Vec::new(), Vec::new());
	for block in 0..blocks {
		let scale = 0.0125_f32 * (block as f32 + 1.0);
		bytes.extend_from_slice(&half_bits(scale).to_le_bytes());
		for lane in 0..32 {
			let level = ((lane as i32 * 7 + block as i32 * 13) % 255 - 127) as i8;
			bytes.push(level as u8);
			expected.push(f64::from(level) * f64::from(half_round(scale)));
		}
	}
	write_gguf(path, "recipe", &[("weight", 8, vec![32, blocks as u64], bytes)]);
	expected
}

/// f32 -> f16 bits, and the value that f16 actually represents.
fn half_bits(value: f32) -> u16 {
	let bits = value.to_bits();
	let (sign, exponent, mantissa) = (bits >> 31, ((bits >> 23) & 0xFF) as i32, bits & 0x007F_FFFF);
	let exponent = exponent - 127 + 15;
	((sign as u16) << 15) | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

fn half_round(value: f32) -> f32 {
	let bits = half_bits(value);
	let (sign, exponent, mantissa) = (u32::from(bits >> 15), i32::from((bits >> 10) & 0x1F), u32::from(bits & 0x3FF));
	f32::from_bits((sign << 31) | (((exponent - 15 + 127) as u32) << 23) | (mantissa << 13))
}

fn write_gguf(path: &std::path::Path, architecture: &str, tensors: &[(&str, u32, Vec<u64>, Vec<u8>)]) {
	let mut head = Vec::new();
	head.extend_from_slice(&0x4655_4747_u32.to_le_bytes());
	head.extend_from_slice(&3_u32.to_le_bytes());
	head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
	head.extend_from_slice(&1_u64.to_le_bytes());
	let key = b"general.architecture";
	head.extend_from_slice(&(key.len() as u64).to_le_bytes());
	head.extend_from_slice(key);
	head.extend_from_slice(&8_u32.to_le_bytes());
	head.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
	head.extend_from_slice(architecture.as_bytes());
	let mut offset = 0_u64;
	for (name, kind, shape, data) in tensors {
		head.extend_from_slice(&(name.len() as u64).to_le_bytes());
		head.extend_from_slice(name.as_bytes());
		head.extend_from_slice(&(shape.len() as u32).to_le_bytes());
		for extent in shape {
			head.extend_from_slice(&extent.to_le_bytes());
		}
		head.extend_from_slice(&kind.to_le_bytes());
		head.extend_from_slice(&offset.to_le_bytes());
		offset += data.len() as u64;
	}
	while head.len() % 32 != 0 {
		head.push(0);
	}
	for (_, _, _, data) in tensors {
		head.extend_from_slice(data);
	}
	std::fs::write(path, head).unwrap();
}

#[test]
fn a_bound_q8_0_tensor_keeps_the_file_bytes_and_decodes_the_same_values() {
	let path = std::env::temp_dir().join(format!("recipe-bind-{}.gguf", std::process::id()));
	let expected = q8_0_file(&path, 6);
	let model = recipe.gguf(&path);
	let tensor = model.tensor("weight").expect("the file holds one tensor");

	let bound = model.bind(tensor).expect("a Q8_0 tensor binds");
	assert!(bound.quantized(), "a block-quantized tensor keeps its own encoding");
	assert_eq!(bound.count(), expected.len(), "the buffer spans every weight");
	assert_eq!(bound.bytes(), model.data(tensor), "binding keeps the file's bytes, decoding nothing");

	// the decoded path and the bound path must describe the same weights
	let values = model.values(tensor).expect("the tensor decodes");
	assert_eq!(values.len(), expected.len());
	for (index, (value, want)) in values.iter().zip(&expected).enumerate() {
		assert_eq!(value.to_bits(), want.to_bits(), "weight {index} differs: {value} vs {want}");
	}
	let _ = std::fs::remove_file(&path);
}

#[test]
fn an_unquantized_tensor_binds_as_arithmetic() {
	let path = std::env::temp_dir().join(format!("recipe-bind-f32-{}.gguf", std::process::id()));
	let values = [1.5_f32, -2.25, 0.0, 7.75];
	let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
	write_gguf(&path, "recipe", &[("norm", 0, vec![4], bytes)]);
	let model = recipe.gguf(&path);
	let tensor = model.tensor("norm").expect("the file holds one tensor");
	let bound = model.bind(tensor).expect("an F32 tensor binds");
	assert!(!bound.quantized(), "an unquantized tensor has no block encoding to keep");
	assert_eq!(bound.count(), 4);
	assert!(bound.bytes().is_empty(), "an unquantized buffer holds arithmetic, not blocks");
	let _ = std::fs::remove_file(&path);
}
