use recipe::*;
use std::path::Path;

const EPOCHS: usize = 1;
const HIDDEN_WIDTH: usize = 8;
const OUTPUT_WIDTH: usize = 1;
const PRECISION_BITS: u8 = 32;
const SEED: usize = 17;

fn write_evidence(path: &Path, values: &[f64]) {
	let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
	for value in values {
		bytes.extend_from_slice(&value.to_le_bytes());
	}
	std::fs::write(path, bytes).expect("cannot write lifecycle evidence");
}

fn main() {
	let phase = std::env::var("RECIPE_WINDOWS_LIFECYCLE_PHASE").expect("RECIPE_WINDOWS_LIFECYCLE_PHASE is absent");
	let bundle = std::env::var_os("RECIPE_WINDOWS_LIFECYCLE_BUNDLE").expect("RECIPE_WINDOWS_LIFECYCLE_BUNDLE is absent");
	let evidence = std::env::var_os("RECIPE_WINDOWS_LIFECYCLE_EVIDENCE").expect("RECIPE_WINDOWS_LIFECYCLE_EVIDENCE is absent");
	let bundle = Path::new(&bundle);
	let evidence = Path::new(&evidence);

	match phase.as_str() {
		"train" | "resume" => {
			let data = recipe.data("data/numeric/single_csv.csv").target("target");
			let model = recipe.model().layer(HIDDEN_WIDTH).relu().layer(OUTPUT_WIDTH).loss(mse);
			let training = recipe.train().seed(SEED).epochs(EPOCHS).fp(PRECISION_BITS).save(bundle);
			let report = if phase == "resume" { training.resume(bundle).run(&model, &data) } else { training.run(&model, &data) };
			let mut values = Vec::with_capacity(report.predictions().len() + 2);
			values.push(report.initial_loss());
			values.push(report.final_loss());
			values.extend_from_slice(report.predictions());
			write_evidence(evidence, &values);
			println!(
				"phase={phase} initial_loss={} initial_bits={:016x} final_loss={} final_bits={:016x} predictions={}",
				report.initial_loss(),
				report.initial_loss().to_bits(),
				report.final_loss(),
				report.final_loss().to_bits(),
				report.predictions().len()
			);
		}
		"infer" => {
			let output = recipe.infer(bundle, &[1.25]);
			write_evidence(evidence, &output);
			println!("phase=infer values={} bits={:x?}", output.len(), output.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
		}
		_ => panic!("unknown lifecycle phase: {phase}"),
	}
}
