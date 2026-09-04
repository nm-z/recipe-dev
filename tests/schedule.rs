//! The measured contraction schedule through the public training path: a
//! rerun of the same model on the same target reuses the cached choice, so
//! it trains, saves, and infers to the same bits, and the reported tile is
//! the dominant weight-gradient tile of that choice.

use recipe::*;
use std::fmt::Write as _;

fn dataset() -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-schedule-test-{}.csv", std::process::id()));
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
	for _ in 0..1031 {
		let values = (0..17).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			let _ = write!(text, "{value:.6},");
		}
		let _ = writeln!(text, "{:.6}", values.iter().map(|value| value * value).sum::<f64>() / 17.0);
	}
	std::fs::write(&path, text).unwrap();
	path
}

#[test]
fn a_rerun_reuses_the_measured_schedule_and_trains_to_the_same_bits() {
	let path = dataset();
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let model = recipe.model().layer(48).relu().layer(24).relu().layer(1).loss(mse);
	let run = |tag: &str| {
		let bundle = std::env::temp_dir().join(format!("recipe-schedule-test-{}-{tag}.ogdl", std::process::id()));
		let report = recipe.train().fp(32).lr(0.01).epochs(6).save(&bundle).run(&model, &data);
		let output = recipe.infer(&bundle, &[0.25; 17]);
		let _ = std::fs::remove_file(&bundle);
		(report.tile(), report.final_loss().to_bits(), report.predictions().iter().map(|value| value.to_bits()).collect::<Vec<_>>(), output[0].to_bits())
	};
	let first = run("first");
	let second = run("second");
	let _ = std::fs::remove_file(&path);
	assert_eq!(first, second, "the rerun did not train to the same bits with the cached schedule");
	assert!(first.0.iter().all(|extent| *extent != 0), "the reported tile is empty: {:?}", first.0);
}
