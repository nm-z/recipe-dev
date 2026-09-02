//! The causal depthwise convolution through the public `dconv(kernel)` path:
//! it keeps the shape, trains on a sequence whose target lives in the order
//! of its positions, and refuses an empty kernel.

use recipe::*;
use std::fmt::Write as _;

fn dataset(tag: &str) -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-dconv-{}-{tag}.csv", std::process::id()));
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
		// Each position minus the one before it, summed: a two-tap causal filter.
		let _ = writeln!(text, "{:.6}", values.windows(2).map(|pair| (pair[1] - pair[0]) * pair[1]).sum::<f64>() / 8.0);
	}
	std::fs::write(&path, text).unwrap();
	path
}

#[test]
fn depthwise_taps_reach_the_previous_position() {
	let path = dataset("train");
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let bundle = std::env::temp_dir().join(format!("recipe-dconv-{}.ogdl", std::process::id()));
	// Per-position layers under a linear head cannot form the neighbour product;
	// two causal taps per channel can.
	let baseline = recipe.train().fp(32).lr(0.02).epochs(150).run(&recipe.model().conv(4, 1).relu().layer(1).loss(mse), &data);
	let taps = recipe.train().fp(32).lr(0.02).epochs(150).save(&bundle).run(&recipe.model().conv(4, 1).dconv(2).relu().layer(1).loss(mse), &data);
	assert!(baseline.final_loss() > baseline.initial_loss() * 0.9, "the baseline learned the neighbour product without taps: {} -> {}", baseline.initial_loss(), baseline.final_loss());
	assert!(taps.final_loss() < taps.initial_loss() * 0.2, "the taps did not learn the neighbour product: {} -> {}", taps.initial_loss(), taps.final_loss());
	let output = recipe.infer(&bundle, &[0.25; 16]);
	assert!(output.len() == 1 && output[0].is_finite());
	let _ = std::fs::remove_file(&bundle);
	let _ = std::fs::remove_file(&path);
}

#[test]
fn the_kernel_must_be_positive() {
	let path = dataset("kernel");
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let model = recipe.model().dconv(0).layer(1).loss(mse);
	let result = std::panic::catch_unwind(|| recipe.train().fp(32).epochs(1).run(&model, &data));
	let _ = std::fs::remove_file(&path);
	let message = result.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
	assert_eq!(message, "depthwise convolution kernel must be positive");
}
