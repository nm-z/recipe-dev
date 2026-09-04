//! GPU training throughput benchmark: measures achieved T(FL)OPS per arithmetic format on a dense stack of WIDTH x WIDTH layers and prints a summary table.
//! Runs on a plain `cargo test`; the default configuration takes a few minutes of GPU time.
//! Environment overrides: BENCH_PRECISIONS (comma list of fp64,fp32,fp16,bf16,int8,int4), BENCH_WIDTH, BENCH_LAYERS, BENCH_ROWS, BENCH_EPOCHS_A, BENCH_EPOCHS_B,
//! and BENCH_PEAKS (comma list of precision=tflops used for the utilization column; the default holds the Radeon RX 7700 XT WMMA peaks and its fp32/fp64 vector peaks).
//! FLOPs use the dense-layer convention of 2*n*n forward and 4*n*n backward per row per layer; the final projection and the optimizer are negligible and uncounted.

use recipe::*;
use std::fmt::Write as _;
use std::io::Write as _;

// The libtest harness captures the print macros on passing tests, so write reports straight to the inherited stderr handle.
fn report(text: String) {
	let _ = std::io::stderr().lock().write_all(text.as_bytes());
}

fn natural(name: &str, fallback: usize) -> usize {
	std::env::var(name).ok().map(|value| value.parse().unwrap_or_else(|_| panic!("{name} must be a positive integer"))).unwrap_or(fallback)
}

fn dataset(rows: usize, columns: usize) -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-throughput-{rows}x{columns}.csv"));
	if path.exists() {
		return path;
	}
	let mut state = 0x1234_5678_9abc_def0_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	let mut text = String::with_capacity(rows * (columns + 1) * 8);
	for column in 0..columns {
		let _ = write!(text, "x{column},");
	}
	text.push_str("y\n");
	for _ in 0..rows {
		for _ in 0..columns {
			let _ = write!(text, "{:.4},", random());
		}
		let _ = writeln!(text, "{:.4}", random());
	}
	std::fs::write(&path, text).unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
	path
}

fn stack(width: usize, layers: usize) -> Model {
	let mut model = recipe.model();
	for _ in 0..layers {
		model = model.layer(width).relu();
	}
	model.loss(mse)
}

fn train(precision: &str, epochs: usize) -> Train {
	let train = recipe.train().optimizer(adamw).lr(0.0001).seed(17).epochs(epochs).log(all);
	match precision {
		"fp64" => train.fp(64),
		"fp32" => train.fp(32),
		"fp16" => train.fp(16),
		"bf16" => train.bf(16),
		"int8" => train.int(8),
		"int4" => train.int(4),
		other => panic!("unknown precision {other:?}: expected fp64, fp32, fp16, bf16, int8, or int4"),
	}
}

#[test]
fn training_throughput() {
	let width = natural("BENCH_WIDTH", 1024);
	let layers = natural("BENCH_LAYERS", 4);
	let rows = natural("BENCH_ROWS", 2048);
	let epochs_a = natural("BENCH_EPOCHS_A", 1);
	let epochs_b = natural("BENCH_EPOCHS_B", 3);
	assert!(epochs_b > epochs_a, "BENCH_EPOCHS_B must exceed BENCH_EPOCHS_A");
	let precisions = std::env::var("BENCH_PRECISIONS").unwrap_or_else(|_| "fp16,bf16,int8,int4,fp32,fp64".to_owned());
	let peaks = std::env::var("BENCH_PEAKS").unwrap_or_else(|_| "fp16=70.34,bf16=70.34,int8=70.34,int4=140.7,fp32=35.17,fp64=1.1".to_owned());
	let peak = |precision: &str| peaks.split(',').filter_map(|entry| entry.trim().split_once('=')).find(|(name, _)| *name == precision).and_then(|(_, value)| value.parse::<f64>().ok());
	let data = recipe.data(dataset(rows, width).to_string_lossy().as_ref()).target("y");
	let model = stack(width, layers);
	let flops_per_epoch = (rows * layers * 6 * width * width) as f64;
	let mut results = Vec::new();
	for precision in precisions.split(',').map(str::trim).filter(|value| !value.is_empty()) {
		let first = train(precision, epochs_a).run(&model, &data);
		let second = train(precision, epochs_b).run(&model, &data);
		let seconds_a = first.epoch_seconds();
		let seconds_b = second.epoch_seconds();
		let per_epoch = seconds_b / epochs_b as f64;
		if per_epoch <= 0.0 {
			report(format!("throughput {precision}: measured epoch time {per_epoch:.3}s is invalid\n"));
			continue;
		}
		let tflops = flops_per_epoch / per_epoch / 1e12;
		report(format!(
			"throughput {precision}: {tflops:.3} tflops ({per_epoch:.3}s/epoch, epoch totals {seconds_a:.2}s/{seconds_b:.2}s, tiles {:?}/{:?}, loss {:.5})\n",
			first.tile(),
			second.tile(),
			second.final_loss()
		));
		results.push((precision.to_owned(), tflops, per_epoch, second.tile()));
	}
	let mut table = format!("\n{:<9} {:>10} {:>10} {:>12} {:>10}  {}\n", "precision", "tflops", "s/epoch", "peak", "of peak", "tile");
	for (precision, tflops, per_epoch, extent) in &results {
		let (peak_text, share) = match peak(precision) {
			Some(peak) => (format!("{peak:.2}"), format!("{:.4}%", tflops / peak * 100.0)),
			None => ("-".to_owned(), "-".to_owned()),
		};
		let _ = writeln!(table, "{precision:<9} {tflops:>10.3} {per_epoch:>10.3} {peak_text:>12} {share:>10}  {extent:?}");
	}
	report(table);
}
