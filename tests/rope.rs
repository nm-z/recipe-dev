//! Rotary position embedding through the public `attn(..).rope(dims, base)`
//! path. At position zero the rotation is the identity, so a model whose
//! sequences have one position must train, predict, and infer bit for bit
//! like the same model without rope; on longer sequences the rotation acts.

use recipe::*;
use std::fmt::Write as _;

fn dataset() -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-rope-{}.csv", std::process::id()));
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	let mut text = (0..16).fold(String::new(), |mut text, column| {
		let _ = write!(text, "x{column},");
		text
	});
	text.push_str("y\n");
	for _ in 0..131 {
		let values = (0..16).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			let _ = write!(text, "{value:.6},");
		}
		let _ = writeln!(text, "{:.6}", values.iter().enumerate().map(|(index, value)| value * (index + 1) as f64 / 64.0).sum::<f64>());
	}
	std::fs::write(&path, text).unwrap();
	path
}

/// Loss, prediction, and inference bits of one 20-epoch training.
fn evidence(model: &Model, tag: &str, path: &std::path::Path) -> (u64, u64, Vec<u64>, u64) {
	let bundle = std::env::temp_dir().join(format!("recipe-rope-{}-{tag}.ogdl", std::process::id()));
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let report = recipe.train().fp(32).lr(0.01).epochs(20).save(&bundle).run(model, &data);
	let output = recipe.infer(&bundle, &[0.25; 16]);
	let _ = std::fs::remove_file(&bundle);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.predictions().iter().map(|value| value.to_bits()).collect(), output[0].to_bits())
}

#[test]
fn position_zero_is_the_identity_and_later_positions_rotate() {
	let path = dataset();
	// One position per sequence: attention over 16 channels of length 1.
	let flat = evidence(&recipe.model().attn(2).relu().layer(1).loss(mse), "flat", &path);
	let flat_rope = evidence(&recipe.model().attn(2).rope(2, 10000.0).relu().layer(1).loss(mse), "flat-rope", &path);
	assert_eq!(flat, flat_rope, "rope at position zero changed the model");
	// A convolution first makes the 16 columns a sequence of 14 positions.
	let sequence = evidence(&recipe.model().conv(4, 3).attn(2).relu().layer(1).loss(mse), "sequence", &path);
	let sequence_rope = evidence(&recipe.model().conv(4, 3).attn(2).rope(2, 10000.0).relu().layer(1).loss(mse), "sequence-rope", &path);
	let _ = std::fs::remove_file(&path);
	assert_ne!(sequence.1, sequence_rope.1, "rope over 14 positions left the loss unchanged");
	assert!(f64::from_bits(sequence_rope.1) < f64::from_bits(sequence_rope.0), "the rotated model did not train");
}

#[test]
fn rope_needs_an_attention_block() {
	let result = std::panic::catch_unwind(|| recipe.model().layer(4).rope(2, 10000.0));
	let message = result.err().and_then(|payload| payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|text| (*text).to_owned()))).unwrap_or_default();
	assert_eq!(message, "rope requires a preceding attn block");
}
