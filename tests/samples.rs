//! Sample construction without a caller-facing broadcast.
//!
//! `Data::broadcast()` used to let a caller ask for shorter tables to be cycled
//! until their row counts matched. It is gone, and sample boundaries are derived
//! from the source structure instead: when one table's column records the file
//! names of a group of sibling tables, that group is one sample per file and the
//! recording column both identifies and orders them.
//!
//! The VNA case is the acceptance shape — a folder of scan files beside one CSV
//! with a header and one data row per scan.

use recipe::*;
use std::fmt::Write as _;
use std::path::PathBuf;

/// The acceptance count from the issue: 3,000 scans, 3,001 physical CSV lines.
const SCANS: usize = 3000;
/// Frequency points per scan. Each scan becomes one sample vector of
/// `POINTS * 2` values, since every point carries a magnitude and a phase.
const POINTS: usize = 4;

fn scan_values(index: usize) -> Vec<f64> {
	let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ (index as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	(0..POINTS * 2).map(|_| random()).collect()
}

/// The target is a function of the scan's own values and its temperature, so it
/// is only learnable when each row is paired with the scan it names.
fn target(values: &[f64], temperature: f64) -> f64 {
	(values.iter().enumerate().map(|(index, value)| value * ((index + 1) as f64 / 8.0).cos()).sum::<f64>() / 4.0 + temperature / 8.0).tanh()
}

/// Build a scan folder. `name` maps a scan's index to its file stem, so the same
/// dataset can be written with the on-disk order matching the CSV row order or
/// running against it. The CSV row order is the same either way.
fn folder(tag: &str, name: impl Fn(usize) -> String, scans: usize) -> String {
	let directory = std::env::temp_dir().join(format!("recipe-samples-{}-{tag}", std::process::id()));
	let _ = std::fs::remove_dir_all(&directory);
	std::fs::create_dir_all(&directory).unwrap();
	let mut meta = String::from("scan,temperature,y\n");
	for index in 0..scans {
		let values = scan_values(index);
		let temperature = (index % 32) as f64 / 32.0;
		let mut scan = String::from("magnitude,phase\n");
		for point in values.chunks(2) {
			let _ = writeln!(scan, "{:.6},{:.6}", point[0], point[1]);
		}
		std::fs::write(directory.join(format!("{}.csv", name(index))), scan).unwrap();
		let _ = writeln!(meta, "{},{temperature:.6},{:.6}", name(index), target(&values, temperature));
	}
	std::fs::write(directory.join("meta.csv"), meta).unwrap();
	directory.to_str().unwrap().to_owned()
}

fn evidence(directory: &str) -> (u64, u64, Vec<u64>) {
	let data = recipe.data(directory).target("y");
	let model = recipe.model().layer(8).tanh().layer(1).loss(mse);
	let report = recipe.train().fp(32).seed(17).lr(0.01).epochs(8).stop(0.0).run(&model, &data);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.predictions().iter().map(|value| value.to_bits()).collect())
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

/// The acceptance case. 3,000 scan files beside a CSV of 3,001 physical lines
/// build exactly 3,000 samples, with no `.broadcast()` to call and none needed.
/// The header is not a sample.
#[test]
fn three_thousand_scans_and_three_thousand_and_one_lines_are_three_thousand_samples() {
	let directory = folder("vna", |index| format!("scan-{index:04}"), SCANS);
	let (initial, final_loss, predictions) = evidence(&directory);
	let _ = std::fs::remove_dir_all(&directory);
	assert_eq!(predictions.len(), SCANS, "expected one sample per scan file");
	assert!(f64::from_bits(final_loss) < f64::from_bits(initial), "the paired dataset did not train: {} to {}", f64::from_bits(initial), f64::from_bits(final_loss));
}

/// The pairing comes from the recorded name, not from the order the files sort
/// in. Numbering the scans backwards reverses the on-disk order while leaving
/// every CSV row exactly where it was, so a loader that took its order from the
/// path would pair every row with the wrong scan. The two datasets describe the
/// same 3,000 pairs and must train to the same bits.
#[test]
fn the_pairing_follows_the_recorded_name_and_not_the_path_order() {
	let forward = folder("forward", |index| format!("scan-{index:04}"), SCANS);
	let backward = folder("backward", |index| format!("scan-{:04}", SCANS - 1 - index), SCANS);
	let first = evidence(&forward);
	let second = evidence(&backward);
	let _ = std::fs::remove_dir_all(&forward);
	let _ = std::fs::remove_dir_all(&backward);
	assert_eq!(first, second, "reversing the on-disk order of the scans changed the training");
	// Not two collapsed constants agreeing: the pairing has to be learnable.
	assert!(f64::from_bits(first.1) < f64::from_bits(first.0), "the pairing did not train, so the comparison proves nothing: {} to {}", f64::from_bits(first.0), f64::from_bits(first.1));
}

/// The column that resolves the files is identity, not a value. Renaming every
/// scan changes those bytes and nothing else, so a run that encoded the file
/// name as a feature would train differently; one that resolved it as a
/// reference cannot tell the two datasets apart.
#[test]
fn the_recording_column_is_not_a_feature() {
	let long = folder("long-names", |index| format!("scan-{index:04}"), 256);
	let short = folder("short-names", |index| format!("s{index}"), 256);
	let first = evidence(&long);
	let second = evidence(&short);
	let _ = std::fs::remove_dir_all(&long);
	let _ = std::fs::remove_dir_all(&short);
	assert_eq!(first, second, "renaming the scan files changed the training, so the file name reached the model as a value");
	assert!(f64::from_bits(first.1) < f64::from_bits(first.0), "the dataset did not train, so the comparison proves nothing: {} to {}", f64::from_bits(first.0), f64::from_bits(first.1));
}

/// A row count that does not match the file count is refused, and the message
/// says how each source was read. Cycling the shorter one to fit is exactly the
/// behaviour the public broadcast used to produce.
#[test]
fn an_unequal_sample_count_is_refused_rather_than_cycled() {
	let directory = folder("short", |index| format!("scan-{index:04}"), 64);
	// Drop one data row from the metadata table, leaving 64 scans and 63 rows.
	let meta = PathBuf::from(&directory).join("meta.csv");
	let text = std::fs::read_to_string(&meta).unwrap();
	let mut lines = text.lines().collect::<Vec<_>>();
	lines.pop();
	std::fs::write(&meta, format!("{}\n", lines.join("\n"))).unwrap();

	let message = panic_text(std::panic::catch_unwind(|| {
		let _ = evidence(&directory);
	}));
	let _ = std::fs::remove_dir_all(&directory);
	assert!(message.contains("samples"), "unexpected error: {message}");
	assert!(message.contains("63") || message.contains("64"), "the message did not name the counts it disagreed on: {message}");
}
