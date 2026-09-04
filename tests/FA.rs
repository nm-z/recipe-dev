use recipe::*;
use std::fmt::Write as _;

struct Evidence {
	outputs: Vec<u64>,
	moments: Vec<u64>,
}

impl Evidence {
	fn encode(&self) -> String {
		let mut text = String::new();
		for value in &self.outputs {
			let _ = write!(text, "{value:016x} ");
		}
		text.push(';');
		for value in &self.moments {
			let _ = write!(text, "{value:016x} ");
		}
		text
	}
	fn decode(text: &str) -> Self {
		let (outputs, moments) = text.split_once(';').unwrap_or_else(|| panic!("FA evidence has no gradient marker"));
		let outputs = outputs.split_whitespace().map(|value| u64::from_str_radix(value, 16).unwrap_or_else(|error| panic!("FA output value is malformed: {error}"))).collect();
		let moments = moments.split_whitespace().map(|value| u64::from_str_radix(value, 16).unwrap_or_else(|error| panic!("FA gradient value is malformed: {error}"))).collect();
		Self { outputs, moments }
	}
}

fn saved_moments(path: &std::path::Path) -> Vec<u64> {
	let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
	let mut moments = Vec::new();
	for line in text.lines() {
		let mut fields = line.trim_start().split_whitespace();
		let key = fields.next().unwrap_or_default();
		if key == "moments" {
			moments.extend(fields.map(|value| value.parse::<f64>().unwrap_or_else(|error| panic!("FA saved gradient is malformed: {error}")).to_bits()));
		}
	}
	moments
}

fn run(width: usize, bits: u8) -> (Evidence, f64) {
	let bundle = std::env::temp_dir().join(format!("recipe-fa-{width}-{}.ogdl", std::process::id()));

	let data = recipe.data(auto).set("data/temporal/window_subfolders/window-0000");
	let model = recipe.model().conv(width, 1);
	let model = model.attn(1);
	let model = model.layer(1).loss(mse);
	let report = recipe
		.train()
		.seed(17)
		.epochs(1)
		.fp(std::env::var("RECIPE_FA_REFERENCE_BITS").map_or(bits, |bits| bits.parse().unwrap_or_else(|error| panic!("FA reference precision is malformed: {error}"))))
		.save(&bundle)
		.run(&model, &data);

	assert!(report.final_loss().is_finite(), "attention produced a nonfinite loss");
	assert!(!report.predictions().is_empty(), "attention produced no predictions");
	assert!(report.predictions().iter().all(|value| value.is_finite()), "attention produced a nonfinite output");
	let outputs = std::iter::once(report.initial_loss().to_bits()).chain(std::iter::once(report.final_loss().to_bits())).chain(report.predictions().iter().map(|value| value.to_bits())).collect();
	let moments = saved_moments(&bundle);
	let evidence = Evidence { outputs, moments };
	let _ = std::fs::remove_file(bundle);
	(evidence, report.epoch_seconds())
}

fn reference(test: &str, bits: u8) -> Evidence {
	let executable = std::env::current_exe().unwrap_or_else(|error| panic!("cannot locate the FA test binary: {error}"));
	let output = std::process::Command::new(executable)
		.env("RECIPE_FA_CPU_REFERENCE", "1")
		.env("RECIPE_FA_REFERENCE_BITS", bits.to_string())
		.env("RECIPE_FORCE_CPU", "1")
		.arg("--exact")
		.arg(test)
		.arg("--nocapture")
		.output()
		.unwrap_or_else(|error| panic!("cannot run the CPU FA reference: {error}"));
	assert!(output.status.success(), "CPU FA reference failed:\n{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
	let text = String::from_utf8_lossy(&output.stdout);
	let line = text.lines().find_map(|line| line.strip_prefix("FA_EVIDENCE ")).unwrap_or_else(|| panic!("CPU FA reference produced no evidence:\n{text}"));
	Evidence::decode(line)
}

fn maximum_error(values: &[u64], reference: &[u64]) -> f64 {
	assert_eq!(values.len(), reference.len(), "FA evidence widths differ");
	values.iter().zip(reference).map(|(&value, &reference)| (f64::from_bits(value) - f64::from_bits(reference)).abs()).fold(0.0, f64::max)
}

fn compare(head_width: usize, bits: u8, test: &str) {
	let width = head_width;
	let (evidence, seconds) = run(width, bits);
	if let Ok(runs) = std::env::var("RECIPE_FA_BENCH") {
		let runs = runs.parse::<usize>().unwrap_or_else(|error| panic!("FA benchmark run count is malformed: {error}"));
		assert!(runs != 0, "FA benchmark run count is zero");
		let mut timings = vec![seconds];
		timings.extend((1..runs).map(|_| run(width, bits).1));
		timings.sort_by(f64::total_cmp);
		let seconds = timings[timings.len() / 2];
		println!("FA_SECONDS head={head_width} items={} full={seconds}", evidence.outputs.len() - 2);
		if let Ok(reference) = std::env::var("RECIPE_FA_UPSTREAM_SECONDS") {
			let reference = reference.parse::<f64>().unwrap_or_else(|error| panic!("upstream FA seconds are malformed: {error}"));
			assert!(seconds >= reference * 0.7 && seconds <= reference * 1.3, "FA time {seconds} is outside 30% of the upstream {reference} seconds");
		}
		return;
	}
	if std::env::var_os("RECIPE_FA_CPU_REFERENCE").is_some() {
		println!("FA_EVIDENCE {}", evidence.encode());
		return;
	}

	let baseline = reference(test, bits);
	let precise = reference(test, 64);
	let output_error = maximum_error(&evidence.outputs, &precise.outputs);
	let baseline_output_error = maximum_error(&baseline.outputs, &precise.outputs);
	let gradient_error = maximum_error(&evidence.moments, &precise.moments);
	let baseline_gradient_error = maximum_error(&baseline.moments, &precise.moments);
	println!("FA_ERROR head={head_width} output={output_error} output_baseline={baseline_output_error} gradient={gradient_error} gradient_baseline={baseline_gradient_error}");
	assert!(output_error <= baseline_output_error * 2.0, "FA output error {output_error} exceeds twice baseline error {baseline_output_error}");
	assert!(gradient_error <= baseline_gradient_error * 2.0, "FA gradient error {gradient_error} exceeds twice baseline error {baseline_gradient_error}");
}

#[test]
fn causal_sequence_head_64_forward_and_backward() {
	compare(64, 16, "causal_sequence_head_64_forward_and_backward")
}

#[test]
fn causal_sequence_head_128_forward_and_backward() {
	compare(128, 16, "causal_sequence_head_128_forward_and_backward")
}

#[test]
fn causal_sequence_head_256_forward_and_backward() {
	compare(256, 16, "causal_sequence_head_256_forward_and_backward")
}

#[test]
fn causal_sequence_fp32_forward_and_backward() {
	compare(64, 32, "causal_sequence_fp32_forward_and_backward")
}
