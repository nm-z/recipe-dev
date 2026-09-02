//! Mixture routing through the public `moe(topk, [...])` path: one router
//! per position, the top-k experts under a softmax, and a sum of expert
//! outputs. Many experts must stay cheap to compile and train.

use recipe::*;
use std::fmt::Write as _;

fn dataset(tag: &str) -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-moe-{}-{tag}.csv", std::process::id()));
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	let mut text = (0..17).fold(String::new(), |mut text, column| {
		let _ = write!(text, "x{column},");
		text
	});
	text.push_str("y\n");
	for _ in 0..131 {
		let values = (0..17).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			let _ = write!(text, "{value:.6},");
		}
		// Two regimes by the sign of the first column, so routing has something to learn.
		let _ = writeln!(text, "{:.6}", if values[0] > 0.0 { values[1] - values[2] } else { values[3] * values[4] });
	}
	std::fs::write(&path, text).unwrap();
	path
}

#[test]
fn routed_experts_train_and_infer() {
	let path = dataset("train");
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let bundle = std::env::temp_dir().join(format!("recipe-moe-{}.ogdl", std::process::id()));
	let model = recipe.model().layer(9).relu().moe(2, [layer(9), layer(9), layer(9), layer(9)]).relu().layer(1).loss(mse);
	let report = recipe.train().fp(32).lr(0.01).epochs(60).save(&bundle).run(&model, &data);
	assert!(report.final_loss() < report.initial_loss() * 0.5, "routing did not train: {} -> {}", report.initial_loss(), report.final_loss());
	let output = recipe.infer(&bundle, &[0.25; 17]);
	assert_eq!(output.len(), 1);
	assert!(output[0].is_finite());
	let _ = std::fs::remove_file(&bundle);
	let _ = std::fs::remove_file(&path);
}

#[test]
fn thirty_two_experts_compile_and_train() {
	let path = dataset("many");
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let experts: [Residual; 32] = std::array::from_fn(|_| layer(9));
	let model = recipe.model().layer(9).relu().moe(2, experts).relu().layer(1).loss(mse);
	let started = std::time::Instant::now();
	let report = recipe.train().fp(32).lr(0.01).epochs(3).run(&model, &data);
	let _ = std::fs::remove_file(&path);
	assert!(report.final_loss().is_finite());
	assert!(started.elapsed().as_secs() < 120, "32 experts took {:?} for three epochs", started.elapsed());
}

#[test]
fn top_k_is_bounded_by_the_experts() {
	let path = dataset("bounds");
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let model = recipe.model().layer(9).moe(3, [layer(9), layer(9)]).layer(1).loss(mse);
	let result = std::panic::catch_unwind(|| recipe.train().fp(32).epochs(1).run(&model, &data));
	let _ = std::fs::remove_file(&path);
	let message = result.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
	assert_eq!(message, "moe top-k is invalid");
}
