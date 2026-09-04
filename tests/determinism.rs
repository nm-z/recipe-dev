//! Cross-backend byte identity: the same program, data, and seed must produce
//! the same bytes on the CPU backend and on the GPU backend.
//!
//! Every case below trains twice, once per backend, and compares the raw bit
//! patterns of the reported losses, of every prediction, and of the saved
//! bundle. Nothing is compared as a decimal string and nothing is compared with
//! a tolerance: a single differing bit fails the case and the report names the
//! first differing element and both bit patterns.
//!
//! Device selection is read once per process, so the CPU side runs in a child
//! process of this binary. The child is this same test executable with
//! `RECIPE_DETERMINISM_CASE` set, which makes it run one case and print the
//! digest instead of running the suite.

use recipe::*;
use std::fmt::Write as _;
use std::io::Write as _;

// The libtest harness captures the print macros on passing tests, so write reports straight to the inherited stderr descriptor.
fn report(text: String) {
	use std::os::fd::FromRawFd;
	let mut stderr = unsafe { std::fs::File::from_raw_fd(2) };
	let _ = stderr.write_all(text.as_bytes());
	std::mem::forget(stderr);
}

/// Rows and columns are prime-adjacent on purpose: they force partial M, N, and
/// K tiles, a partial register block, and a partition count that does not divide
/// the element count evenly.
fn dataset(rows: usize, columns: usize) -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-determinism-{rows}x{columns}.csv"));
	if path.exists() {
		return path;
	}
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
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
			let _ = write!(text, "{:.6},", random());
		}
		let _ = writeln!(text, "{:.6}", random());
	}
	std::fs::write(&path, text).unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
	path
}

/// One case is a topology, an arithmetic format, and a data shape.
struct Case {
	name: &'static str,
	shape: &'static str,
	precision: &'static str,
	rows: usize,
	columns: usize,
}

const CASES: &[Case] = &[
	Case { name: "linear-narrow", shape: "linear", precision: "fp32", rows: 1, columns: 1 },
	Case { name: "linear-thin", shape: "linear", precision: "fp32", rows: 4, columns: 3 },
	Case { name: "linear-wide", shape: "linear", precision: "fp32", rows: 257, columns: 65 },
	Case { name: "linear-tail", shape: "linear", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "deep-fp64", shape: "deep", precision: "fp64", rows: 131, columns: 17 },
	Case { name: "deep-fp32", shape: "deep", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "deep-tf32", shape: "deep", precision: "tf32", rows: 131, columns: 17 },
	Case { name: "deep-fp16", shape: "deep", precision: "fp16", rows: 131, columns: 17 },
	Case { name: "deep-bf16", shape: "deep", precision: "bf16", rows: 131, columns: 17 },
	Case { name: "deep-int8", shape: "deep", precision: "int8", rows: 131, columns: 17 },
	Case { name: "deep-int4", shape: "deep", precision: "int4", rows: 131, columns: 17 },
	Case { name: "hyper", shape: "hyper", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "hyper-static", shape: "hyper-static", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "scalar-parameter", shape: "prelu", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "scalar-parameter-bf16", shape: "prelu", precision: "bf16", rows: 131, columns: 17 },
	Case { name: "transcendental-tanh", shape: "tanh", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "transcendental-gelu", shape: "gelu", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "transcendental-silu", shape: "silu", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "transcendental-sigmoid", shape: "sigmoid", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "transcendental-fp64", shape: "tanh", precision: "fp64", rows: 131, columns: 17 },
	Case { name: "normalization", shape: "norm", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "convolution", shape: "conv", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "residual", shape: "residual", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "residual-bf16", shape: "residual", precision: "bf16", rows: 131, columns: 17 },
	Case { name: "deep-fp8", shape: "deep", precision: "fp8", rows: 131, columns: 17 },
	Case { name: "deep-int1", shape: "deep", precision: "int1", rows: 131, columns: 17 },
	Case { name: "deep-custom-float", shape: "deep", precision: "f6.9", rows: 131, columns: 17 },
	Case { name: "attention", shape: "attention", precision: "fp32", rows: 131, columns: 18 },
	Case { name: "scan-gru", shape: "gru", precision: "fp32", rows: 131, columns: 18 },
	Case { name: "scan-lstm", shape: "lstm", precision: "fp32", rows: 131, columns: 18 },
	Case { name: "pool", shape: "pool", precision: "fp32", rows: 131, columns: 18 },
	Case { name: "moe", shape: "moe", precision: "fp32", rows: 131, columns: 17 },
	Case { name: "persistence", shape: "deep", precision: "fp32", rows: 131, columns: 17 },
];

fn build(case: &Case) -> Model {
	match case.shape {
		"linear" => recipe.model().layer(1).loss(mse),
		"deep" => recipe.model().layer(9).relu().layer(5).relu().layer(1).loss(mse),
		"prelu" => recipe.model().layer(9).prelu().layer(5).prelu().layer(1).loss(mse),
		"tanh" => recipe.model().layer(9).tanh().layer(5).tanh().layer(1).loss(mse),
		"gelu" => recipe.model().layer(9).gelu().layer(5).gelu().layer(1).loss(mse),
		"silu" => recipe.model().layer(9).silu().layer(5).silu().layer(1).loss(mse),
		"sigmoid" => recipe.model().layer(9).sigmoid().layer(5).sigmoid().layer(1).loss(mse),
		"norm" => recipe.model().layer(9).relu().norm(batch).layer(5).relu().layer(1).loss(mse),
		"conv" => recipe.model().conv(3, 3).relu().conv(2, 1).relu().layer(1).loss(mse),
		"residual" => recipe.model().layer(9).relu().res([layer(9), layer(9)]).relu().layer(1).loss(mse),
		"hyper" => recipe.model().layer(8).hyper(4, 6, [layer(8), relu()]).hyper(4, 6, [layer(8), gelu()]).layer(1).loss(mse),
		"hyper-static" => recipe.model().layer(8).hyper(2, 0, [layer(8), relu()]).layer(1).loss(mse),
		"attention" => recipe.model().attn(2).relu().layer(1).loss(mse),
		"gru" => recipe.model().gru(6).relu().layer(1).loss(mse),
		"lstm" => recipe.model().lstm(6).relu().layer(1).loss(mse),
		"pool" => recipe.model().conv(4, 3).relu().pool(2).relu().layer(1).loss(mse),
		"moe" => recipe.model().layer(9).relu().moe(1, [layer(9), layer(9)]).relu().layer(1).loss(mse),
		other => panic!("unknown topology {other:?}"),
	}
}

fn configure(precision: &str) -> Train {
	let train = recipe.train().optimizer(adamw).lr(0.01).seed(17).epochs(3).log(all);
	match precision {
		"f6.9" => train.f(6, 9),
		"fp8" => train.fp(8),
		"int1" => train.int(1),
		"fp64" => train.fp(64),
		"fp32" => train.fp(32),
		"tf32" => train.tf(32),
		"fp16" => train.fp(16),
		"bf16" => train.bf(16),
		"int8" => train.int(8),
		"int4" => train.int(4),
		other => panic!("unknown precision {other:?}"),
	}
}

/// Everything the case produced, as bytes. Predictions are recorded and
/// compared in the order the runtime reports them, so the assignment of values
/// to rows is part of what the backends must agree on.
struct Evidence {
	initial: u64,
	final_loss: u64,
	predictions: Vec<u64>,
	bundle: Vec<u8>,
}

impl Evidence {
	fn encode(&self) -> String {
		let mut text = format!("{:016x} {:016x} {:016x}", self.initial, self.final_loss, self.predictions.len());
		for value in &self.predictions {
			let _ = write!(text, " {value:016x}");
		}
		let _ = write!(text, " |");
		for byte in &self.bundle {
			let _ = write!(text, "{byte:02x}");
		}
		text
	}
	fn decode(text: &str) -> Self {
		let mut fields = text.split_whitespace();
		let mut number = |role: &str| u64::from_str_radix(fields.next().unwrap_or_else(|| panic!("{role} is absent")), 16).unwrap_or_else(|error| panic!("{role} is malformed: {error}"));
		let initial = number("initial loss");
		let final_loss = number("final loss");
		let count = number("prediction count") as usize;
		let predictions = (0..count).map(|index| number(&format!("prediction {index}"))).collect();
		let bundle = fields.next().unwrap_or_else(|| panic!("bundle is absent"));
		let bundle = bundle.strip_prefix('|').unwrap_or_else(|| panic!("bundle marker is absent"));
		let bundle =
			(0..bundle.len() / 2).map(|index| u8::from_str_radix(&bundle[index * 2..index * 2 + 2], 16).unwrap_or_else(|error| panic!("bundle byte {index} is malformed: {error}"))).collect();
		Self { initial, final_loss, predictions, bundle }
	}
}

/// The bundle records when it was written, how long it took, and which cached
/// artifact produced it. Those fields are properties of the run, not of the
/// computation, so the comparison covers every line that is not one of them.
/// Everything the optimiser computed, including the frozen mask and the best
/// checkpoint, is compared byte for byte.
const VOLATILE: &[&str] = &["artifact", "run", "seconds", "time"];

fn stable_bundle(path: &std::path::Path) -> Vec<u8> {
	let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
	let mut kept = String::with_capacity(text.len());
	for line in text.lines() {
		let key = line.trim_start().split(|value: char| value.is_whitespace()).next().unwrap_or_default();
		if !VOLATILE.contains(&key) {
			kept.push_str(line);
			kept.push('\n');
		}
	}
	kept.into_bytes()
}

fn run(case: &Case) -> Evidence {
	let bundle = std::env::temp_dir().join(format!("recipe-determinism-{}-{}.ogdl", case.name, std::process::id()));
	let data = recipe.data(dataset(case.rows, case.columns).to_string_lossy().as_ref()).target("y");
	// The persistence case trains, saves, reloads the bundle, trains again, and
	// saves again, so the evidence covers the save, reload, resume, and rerun
	// path rather than one uninterrupted run.
	let report = if case.name == "persistence" {
		let stage = std::env::temp_dir().join(format!("recipe-determinism-{}-stage-{}.ogdl", case.name, std::process::id()));
		configure(case.precision).save(&stage).run(&build(case), &data);
		let resumed = configure(case.precision).resume(&stage).save(&bundle).run(&build(case), &data);
		let _ = std::fs::remove_file(&stage);
		resumed
	} else {
		configure(case.precision).save(&bundle).run(&build(case), &data)
	};
	// Predictions are compared in reported order: a backend that assigned a
	// prediction to the wrong row would fail even if the multiset matched.
	let predictions = report.predictions().iter().map(|value| value.to_bits()).collect::<Vec<_>>();
	let evidence = Evidence { initial: report.initial_loss().to_bits(), final_loss: report.final_loss().to_bits(), predictions, bundle: stable_bundle(&bundle) };
	let _ = std::fs::remove_file(&bundle);
	evidence
}

fn compare(case: &Case, gpu: &Evidence, cpu: &Evidence) -> Option<String> {
	if gpu.initial != cpu.initial {
		return Some(format!("initial loss differs: gpu {:016x}, cpu {:016x}", gpu.initial, cpu.initial));
	}
	if gpu.final_loss != cpu.final_loss {
		return Some(format!("final loss differs: gpu {:016x}, cpu {:016x}", gpu.final_loss, cpu.final_loss));
	}
	if gpu.predictions.len() != cpu.predictions.len() {
		return Some(format!("prediction count differs: gpu {}, cpu {}", gpu.predictions.len(), cpu.predictions.len()));
	}
	for (index, (left, right)) in gpu.predictions.iter().zip(&cpu.predictions).enumerate() {
		if left != right {
			return Some(format!("prediction {index} of {} differs: gpu {left:016x}, cpu {right:016x}", gpu.predictions.len()));
		}
	}
	if gpu.bundle.len() != cpu.bundle.len() {
		return Some(format!("bundle length differs: gpu {}, cpu {}", gpu.bundle.len(), cpu.bundle.len()));
	}
	for (index, (left, right)) in gpu.bundle.iter().zip(&cpu.bundle).enumerate() {
		if left != right {
			let context = String::from_utf8_lossy(&gpu.bundle[index.saturating_sub(48)..(index + 16).min(gpu.bundle.len())]).replace('\n', "\\n");
			return Some(format!("bundle byte {index} differs: gpu {left:#04x}, cpu {right:#04x}, near {context:?}"));
		}
	}
	let _ = case;
	None
}

fn child(case: &Case) -> Evidence {
	let executable = std::env::current_exe().unwrap_or_else(|error| panic!("cannot locate the test binary: {error}"));
	let output = std::process::Command::new(&executable)
		.env("RECIPE_DETERMINISM_CASE", case.name)
		.env("RECIPE_FORCE_CPU", "1")
		.arg("--exact")
		.arg("--nocapture")
		.arg("cross_backend_byte_identity")
		.output()
		.unwrap_or_else(|error| panic!("cannot run the CPU side of {}: {error}", case.name));
	let text = String::from_utf8_lossy(&output.stdout);
	let line = text
		.lines()
		.find_map(|line| line.strip_prefix("EVIDENCE "))
		.unwrap_or_else(|| panic!("the CPU side of {} produced no evidence:\n{}\n{}", case.name, text, String::from_utf8_lossy(&output.stderr)));
	Evidence::decode(line)
}

#[test]
fn cross_backend_byte_identity() {
	if let Ok(name) = std::env::var("RECIPE_DETERMINISM_CASE") {
		let case = CASES.iter().find(|case| case.name == name).unwrap_or_else(|| panic!("unknown case {name:?}"));
		println!("EVIDENCE {}", run(case).encode());
		return;
	}
	let mut failures = Vec::new();
	let mut summary = String::from("\ncross-backend byte identity\n");
	for case in CASES {
		let gpu = run(case);
		let cpu = child(case);
		match compare(case, &gpu, &cpu) {
			None => {
				let _ = writeln!(summary, "  {:<24} {:>4} rows {:>3} columns  {:<5}  identical, {} predictions", case.name, case.rows, case.columns, case.precision, gpu.predictions.len());
			}
			Some(difference) => {
				let _ = writeln!(summary, "  {:<24} {:>4} rows {:>3} columns  {:<5}  DIFFERS: {difference}", case.name, case.rows, case.columns, case.precision);
				failures.push(format!("{}: {difference}", case.name));
			}
		}
	}
	report(summary);
	assert!(failures.is_empty(), "backends disagree on {} of {} cases:\n{}", failures.len(), CASES.len(), failures.join("\n"));
}
