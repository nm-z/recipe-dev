//! N-gram embeddings through the public `recipe.gguf(..).ngram()` path: rows
//! hashed from the bigram and trigram context into per-head ranges, gathered
//! from an F32 table and from a Q8_0 table row by row.

use recipe::*;

/// Writes one GGUF file with typed metadata and 32-byte aligned tensors.
fn write_gguf(path: &std::path::Path, metadata: &[(&str, GgufValue)], tensors: &[(&str, u32, Vec<u64>, Vec<u8>)]) {
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
		match value {
			GgufValue::U32(value) => {
				header.extend(4_u32.to_le_bytes());
				header.extend(value.to_le_bytes());
			}
			GgufValue::String(value) => {
				header.extend(8_u32.to_le_bytes());
				string(&mut header, value);
			}
			GgufValue::Array(items) => {
				header.extend(9_u32.to_le_bytes());
				header.extend(10_u32.to_le_bytes());
				header.extend((items.len() as u64).to_le_bytes());
				for item in items {
					header.extend(item.integer().unwrap().to_le_bytes());
				}
			}
			other => panic!("the test writer has no encoding for {other:?}"),
		}
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
fn rows_hash_per_head_and_reset_at_the_end_id() {
	// Row r of the F32 table [4, 64] holds r, r + 0.25, r + 0.5, r + 0.75.
	let table = (0..64_u32).flat_map(|row| (0..4).map(move |lane| (row as f32 + lane as f32 * 0.25).to_le_bytes())).flatten().collect::<Vec<_>>();
	let path = std::env::temp_dir().join(format!("recipe-ngram-f32-{}.gguf", std::process::id()));
	write_gguf(&path, &[("ngram.heads", GgufValue::U32(2)), ("tokenizer.ggml.eos_token_id", GgufValue::U32(9))], &[("ngram.table", 0, vec![4, 64], table)]);
	let model = recipe.gguf(&path);
	let _ = std::fs::remove_file(&path);
	let ngram = model.ngram();
	assert_eq!((ngram.heads(), ngram.width()), (2, 16));
	let ids = [5, 7, 9, 7, 3];
	let values = ngram.lookup(&ids).unwrap();
	assert_eq!(values.len(), ids.len() * 16);
	for position in 0..ids.len() {
		let rows = ngram.rows(&ids, position);
		assert_eq!(rows.len(), 4);
		for (head, row) in rows.iter().enumerate() {
			assert!((head * 16..(head + 1) * 16).contains(row), "token {position} head {head} left its range with row {row}");
			let gathered = &values[position * 16 + head * 4..position * 16 + head * 4 + 4];
			assert_eq!(gathered, &[*row as f64, *row as f64 + 0.25, *row as f64 + 0.5, *row as f64 + 0.75], "token {position} head {head} gathered another row");
		}
	}
	// The same context addresses the same rows; a different context does not.
	assert_eq!(ngram.rows(&ids, 1), ngram.rows(&[5, 7], 1));
	assert_ne!(ngram.rows(&ids, 1), ngram.rows(&[2, 6, 7], 2), "the bigram ignored the previous token");
	assert_eq!(ngram.rows(&ids, 1)[..2], ngram.rows(&[6, 5, 7], 2)[..2], "the bigram heads looked past the previous token");
	// Token 7 after the end id 9 sees no previous token, like a first token.
	assert_eq!(ngram.rows(&ids, 3), ngram.rows(&[7], 0));
	assert_eq!(ngram.rows(&ids, 4), ngram.rows(&[7, 3], 1), "the trigram reached across the end id");
}

#[test]
fn q8_0_rows_decode_from_their_own_bytes() {
	// A Q8_0 table [32, 8]: one block per row, scale 2^(row - 4), codes row + lane.
	let mut table = Vec::new();
	for row in 0..8_i32 {
		table.extend(((row - 4 + 15) as u16 * 0x400).to_le_bytes());
		table.extend((0..32).map(|lane| (row * 3 + lane - 40) as i8 as u8));
	}
	let path = std::env::temp_dir().join(format!("recipe-ngram-q8-{}.gguf", std::process::id()));
	write_gguf(
		&path,
		&[("ngram.heads", GgufValue::U32(1)), ("ngram.table", GgufValue::String("blk.memory".to_owned())), ("ngram.seeds", GgufValue::Array(vec![GgufValue::U64(11), GgufValue::U64(17)]))],
		&[("blk.memory", 8, vec![32, 8], table)],
	);
	let model = recipe.gguf(&path);
	let _ = std::fs::remove_file(&path);
	let tensor = model.tensor("blk.memory").unwrap();
	let whole = model.values(tensor).unwrap();
	for row in 0..8 {
		assert_eq!(model.row(tensor, row).unwrap(), whole[row * 32..(row + 1) * 32], "row {row} differs from the whole decode");
	}
	assert_eq!(model.row(tensor, 8).unwrap_err().to_string(), "tensor blk.memory has 8 rows, row 8 requested");
	let ngram = model.ngram();
	let rows = ngram.rows(&[1, 2], 1);
	assert!(rows[0] < 4 && (4..8).contains(&rows[1]));
	assert_eq!(ngram.lookup(&[1, 2]).unwrap()[64..], [whole[rows[0] * 32..(rows[0] + 1) * 32].to_vec(), whole[rows[1] * 32..(rows[1] + 1) * 32].to_vec()].concat());
}
