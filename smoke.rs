//! Does the kernel template still compile after the delta additions? Trains a
//! trivial model, which exercises the whole template on the selected device.

use recipe::*;
use std::fmt::Write as _;

fn main() {
	let path = std::env::temp_dir().join("recipe-smoke.csv");
	let mut text = "x0,x1,y\n".to_owned();
	for i in 0..64 {
		let (a, b) = (i as f64 * 0.01, (64 - i) as f64 * 0.02);
		let _ = writeln!(text, "{a:.4},{b:.4},{:.4}", a * 2.0 + b);
	}
	std::fs::write(&path, text).unwrap();
	let data = recipe.data(path.to_string_lossy().as_ref()).target(["y"]);
	let model = recipe.model().layer(4).relu().layer(1).loss(mse);
	let report = recipe.train().fp(32).lr(0.01).epochs(20).run(&model, &data);
	println!("template OK: initial={:.6} final={:.6}", report.initial_loss(), report.final_loss());
}
