//! Hyper-connections through the public `hyper(lanes, rank, [...])` path:
//! one lane with fixed gates is the plain residual bit for bit, and a
//! widened stream with data-dependent gates trains and infers.

use recipe::*;
use std::fmt::Write as _;

fn dataset(tag: &str) -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-hyper-{}-{tag}.csv", std::process::id()));
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
		let _ = writeln!(text, "{:.6}", values.iter().map(|value| value * value).sum::<f64>() / 17.0 - 0.3);
	}
	std::fs::write(&path, text).unwrap();
	path
}

/// Loss, prediction, and inference bits of one 30-epoch training.
fn evidence(model: &Model, tag: &str, path: &std::path::Path) -> (u64, u64, Vec<u64>, u64) {
	let bundle = std::env::temp_dir().join(format!("recipe-hyper-{}-{tag}.ogdl", std::process::id()));
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let report = recipe.train().fp(32).lr(0.01).epochs(30).save(&bundle).run(model, &data);
	let output = recipe.infer(&bundle, &[0.25; 17]);
	let _ = std::fs::remove_file(&bundle);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.predictions().iter().map(|value| value.to_bits()).collect(), output[0].to_bits())
}

#[test]
fn one_lane_with_fixed_gates_is_the_residual() {
	let path = dataset("residual");
	let residual = evidence(&recipe.model().layer(8).res([layer(8), relu()]).layer(1).loss(mse), "residual", &path);
	let hyper = evidence(&recipe.model().layer(8).hyper(1, 0, [layer(8), relu()]).layer(1).loss(mse), "hyper-one", &path);
	assert_eq!(residual, hyper, "one lane with gates of one differs from the residual");
	let wide = evidence(&recipe.model().layer(8).hyper(4, 6, [layer(8), relu()]).hyper(4, 6, [layer(8), gelu()]).layer(1).loss(mse), "hyper-wide", &path);
	let _ = std::fs::remove_file(&path);
	assert!(f64::from_bits(wide.1) < f64::from_bits(wide.0), "the widened stream did not train");
	assert_ne!(wide.1, hyper.1, "four gated lanes trained like one fixed lane");
}

#[test]
fn lanes_must_agree_along_the_stream() {
	let path = dataset("lanes");
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let model = recipe.model().layer(8).hyper(2, 0, [layer(8)]).hyper(3, 0, [layer(8)]).layer(1).loss(mse);
	let result = std::panic::catch_unwind(|| recipe.train().fp(32).epochs(1).run(&model, &data));
	let _ = std::fs::remove_file(&path);
	let message = result.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
	assert_eq!(message, "hyper-connections with 3 lanes follow a stream of 2");
}
