//! Public lifecycle coverage for every added input-target arity fixture.
use recipe::*;
use std::{collections::{BTreeMap, BTreeSet}, env, fs, path::{Path, PathBuf}};

struct Case {
	label: String,
	path: PathBuf,
	inputs: Vec<String>,
	targets: Vec<String>,
	training_rows: usize,
	evaluation_rows: usize,
	output_width: usize,
}

fn cases() -> Vec<Case> {
	let root = PathBuf::from(env::var("RECIPE_ARITY_ROOT").unwrap_or_else(|_| ".".to_owned()));
	let manifest = PathBuf::from(env::var("RECIPE_ARITY_MANIFEST").unwrap_or_else(|_| root.join(".github/runtime/arity.tsv").to_string_lossy().into_owned()));
	let mut cases = fs::read_to_string(&manifest)
		.unwrap_or_else(|error| panic!("cannot read {}: {error}", manifest.display()))
		.lines()
		.skip(1)
		.filter(|line| !line.is_empty())
		.map(|line| {
			let fields = line.split('\t').collect::<Vec<_>>();
			assert_eq!(fields.len(), 6, "arity manifest row has six fields: {line}");
			let path = root.join(fields[0]);
			assert!(path.exists(), "arity source is absent: {}", path.display());
			Case {
				label: fields[0].to_owned(),
				path,
				inputs: fields[1].split(',').map(str::to_owned).collect(),
				targets: fields[2].split(',').map(str::to_owned).collect(),
				training_rows: fields[3].parse().unwrap(),
				evaluation_rows: fields[4].parse().unwrap(),
				output_width: fields[5].parse().unwrap(),
			}
		})
		.collect::<Vec<_>>();
	assert_eq!(cases.len(), 216, "the arity manifest must name three added arities for all 72 situations");
	assert!(cases.windows(2).all(|pair| pair[0].label < pair[1].label), "arity manifest paths must be unique and sorted");
	let mut coverage = BTreeMap::<(String, String), BTreeSet<(bool, bool)>>::new();
	for case in &cases {
		let mut name = Path::new(&case.label).file_name().and_then(|value| value.to_str()).expect("arity source name is UTF-8").to_owned();
		while let Some((stem, extension)) = name.rsplit_once('.') {
			if extension.is_empty() || !extension.chars().all(|value| value.is_ascii_alphanumeric()) { break }
			name = stem.to_owned();
		}
		let (base, arity) = [
			("_two_inputs_two_targets", (true, true)),
			("_two_views_two_targets", (true, true)),
			("_two_targets", (false, true)),
			("_two_inputs", (true, false)),
			("_two_views", (true, false)),
		]
		.into_iter()
		.find_map(|(suffix, arity)| name.strip_suffix(suffix).map(|base| (base.to_owned(), arity)))
		.unwrap_or_else(|| panic!("arity source lacks a recognized suffix: {}", case.label));
		let family = case.label.split('/').nth(1).expect("arity source has a family").to_owned();
		assert!(coverage.entry((family, base)).or_default().insert(arity), "arity source duplicates a base arity: {}", case.label);
	}
	let expected = BTreeSet::from([(true, false), (false, true), (true, true)]);
	assert_eq!(coverage.len(), 72, "the arity manifest must cover exactly 72 base situations");
	for ((family, base), arities) in coverage { assert_eq!(arities, expected, "{family}/{base} does not contain all three added arities"); }
	if let Ok(selected) = env::var("RECIPE_ARITY_CASES") {
		let selected = selected.split(',').map(str::trim).collect::<Vec<_>>();
		cases.retain(|case| selected.iter().any(|value| case.label.contains(value)));
	}
	cases
}

fn declaration(case: &Case) -> Data {
	if case.label.contains("/split_files_") || case.label.contains("/chronological_splits_") {
		let training = case.path.join("train.csv");
		let evaluation = case.path.join("test.csv");
		assert!(training.exists() && evaluation.exists(), "{} lacks explicit training or evaluation data", case.label);
		recipe
			.data(training.to_string_lossy().into_owned())
			.include(case.inputs.clone())
			.target(case.targets.clone())
			.test(evaluation.to_string_lossy().into_owned())
	} else {
		recipe
			.data(case.path.to_string_lossy().into_owned())
			.include(case.inputs.clone())
			.target(case.targets.clone())
			.split(0.8)
	}
}

#[derive(Clone)]
struct BundleInfo {
	schema: Vec<(String, String)>,
	features: Vec<String>,
	targets: Vec<String>,
	inputs: Vec<String>,
	outputs: Vec<String>,
	shape: [usize; 4],
	training_rows: usize,
}

fn unhex(value: &str) -> String {
	let bytes = (0..value.len()).step_by(2).map(|index| u8::from_str_radix(&value[index..index + 2], 16).unwrap()).collect::<Vec<_>>();
	String::from_utf8(bytes).unwrap()
}

fn bundle(path: &Path) -> BundleInfo {
	let document = fs::read_to_string(path).unwrap();
	let mut schema = Vec::new();
	let mut features = Vec::new();
	let mut targets = Vec::new();
	let mut inputs = Vec::new();
	let mut outputs = Vec::new();
	let mut shape = None;
	let mut training_rows = None;
	let mut in_schema = false;
	let mut in_graph = false;
	for line in document.lines().map(str::trim).filter(|line| !line.is_empty()) {
		if line == "schema" { in_schema = true; continue }
		if line == "graph" { in_schema = false; in_graph = true; continue }
		let (kind, value) = line.split_once(' ').unwrap_or((line, ""));
		if in_schema {
			schema.push((kind.to_owned(), value.to_owned()));
			if kind == "feature" { features.push(value.split_once(' ').map_or(value, |(_, name)| name).to_owned()); }
			if kind == "target" { targets.push(value.to_owned()); }
		} else if in_graph {
			match kind {
				"in" => inputs.push(unhex(value)),
				"out" => outputs.push(unhex(value)),
				"shape" if shape.is_none() => {
					let values = value.split_whitespace().map(|value| value.parse::<usize>().unwrap()).collect::<Vec<_>>();
					shape = Some(values.try_into().expect("shape has four values"));
				}
				"training_rows" if training_rows.is_none() => training_rows = Some(value.parse().unwrap()),
				_ => {}
			}
		}
	}
	BundleInfo {
		schema,
		features,
		targets,
		inputs,
		outputs,
		shape: shape.expect("bundle shape is absent"),
		training_rows: training_rows.expect("bundle training rows are absent"),
	}
}

fn belongs(actual: &str, expected: &str) -> bool {
	actual == expected || actual.starts_with(&format!("{expected}.")) || actual.ends_with(&format!(".{expected}")) || actual.contains(&format!(".{expected}."))
}

fn ordered_groups(role: &str, actual: &[String], expected: &[String]) {
	assert!(!actual.is_empty(), "{role} schema is empty");
	let mut groups = Vec::new();
	for value in actual {
		let matches = expected.iter().enumerate().filter(|(_, name)| belongs(value, name)).map(|(index, _)| index).collect::<Vec<_>>();
		assert_eq!(matches.len(), 1, "{role} field {value:?} must match exactly one declaration in {expected:?}");
		if groups.last() != matches.first() { groups.push(matches[0]); }
	}
	assert_eq!(groups, (0..expected.len()).collect::<Vec<_>>(), "{role} schema {actual:?} does not exactly follow {expected:?}");
}

fn bits(values: &[f64]) -> Vec<u64> { values.iter().map(|value| value.to_bits()).collect() }

fn digest(mut hash: u64, bytes: &[u8]) -> u64 {
	for byte in bytes { hash = (hash ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211); }
	hash
}

fn run(case: &Case, index: usize, work: &Path) -> u64 {
	let data = declaration(case);
	let model = recipe.model().layer(4).gelu().layer(1).loss(mse);
	let saved_path = work.join(format!("case-{index:03}.ogdl"));
	let cold = recipe.train().fp(32).seed(234).epochs(1).stop(0.0).save(&saved_path).run(&model, &data);
	assert!(cold.initial_loss().is_finite() && cold.final_loss().is_finite(), "{} has a nonfinite cold loss", case.label);
	let saved = bundle(&saved_path);
	ordered_groups("input", &saved.features, &case.inputs);
	ordered_groups("target", &saved.targets, &case.targets);
	ordered_groups("output", &saved.outputs, &case.targets);
	assert_eq!(saved.training_rows, case.training_rows, "{} training row count changed", case.label);
	assert_eq!(saved.inputs.len(), saved.shape[0] * saved.shape[1], "{} input width differs from its shape", case.label);
	assert_eq!(saved.outputs.len(), saved.shape[2] * saved.shape[3], "{} output width differs from its shape", case.label);
	assert_eq!(saved.outputs.len(), case.output_width, "{} output width changed", case.label);
	assert_eq!(cold.predictions().len(), case.evaluation_rows * case.output_width, "{} cold evaluation lost or duplicated sample boundaries", case.label);

	let resumed = recipe.train().fp(32).seed(234).epochs(1).stop(0.0).resume(&saved_path).save(&saved_path).run(&model, &data);
	assert!(resumed.initial_loss().is_finite() && resumed.final_loss().is_finite(), "{} has a nonfinite resumed loss", case.label);
	let after = bundle(&saved_path);
	assert_eq!(after.training_rows, case.training_rows, "{} resumed training row count changed", case.label);
	assert_eq!(saved.schema, after.schema, "{} resume changed the complete input-target schema", case.label);
	assert_eq!(saved.features, after.features, "{} resume changed feature order", case.label);
	assert_eq!(saved.targets, after.targets, "{} resume changed target order", case.label);
	assert_eq!(saved.inputs, after.inputs, "{} resume changed encoded input order", case.label);
	assert_eq!(saved.outputs, after.outputs, "{} resume changed output order", case.label);
	assert_eq!(resumed.predictions().len(), case.evaluation_rows * case.output_width, "{} resumed evaluation lost or duplicated sample boundaries", case.label);

	let input = vec![0.0; saved.inputs.len()];
	let first = recipe.infer(&saved_path, &input);
	let second = recipe.infer(&saved_path, &input);
	assert_eq!(first.len(), case.output_width, "{} inference output width changed", case.label);
	assert!(first.iter().all(|value| value.is_finite()), "{} inference produced a nonfinite value", case.label);
	assert_eq!(bits(&first), bits(&second), "{} repeated inference changed output bits", case.label);
	let mut hash = digest(1_469_598_103_934_665_603, case.label.as_bytes());
	for (kind, value) in &saved.schema { hash = digest(digest(hash, kind.as_bytes()), value.as_bytes()); }
	for value in saved.inputs.iter().chain(&saved.outputs) { hash = digest(hash, value.as_bytes()); }
	for value in [case.training_rows, case.evaluation_rows, case.output_width] { hash = digest(hash, &value.to_le_bytes()); }
	fs::remove_file(saved_path).unwrap();
	hash
}

fn message(payload: Box<dyn std::any::Any + Send>) -> String {
	match payload.downcast::<String>() {
		Ok(message) => *message,
		Err(payload) => payload.downcast::<&str>().map_or_else(|_| "non-string panic".to_owned(), |message| (*message).to_owned()),
	}
}

fn main() {
	let cases = cases();
	let work = PathBuf::from(env::var("RECIPE_ARITY_WORK").expect("RECIPE_ARITY_WORK is required"));
	fs::create_dir_all(&work).unwrap();
	let hook = std::panic::take_hook();
	std::panic::set_hook(Box::new(|_| {}));
	let mut failures = Vec::new();
	let mut summary = 1_469_598_103_934_665_603;
	for (index, case) in cases.iter().enumerate() {
		match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(case, index, &work))) {
			Ok(hash) => summary = digest(summary, &hash.to_le_bytes()),
			Err(payload) => {
				let failure = message(payload);
				println!("arity FAIL {}: {failure}", case.label);
				failures.push(failure);
			}
		}
	}
	std::panic::set_hook(hook);
	println!("arity lifecycle cases={} passed={} failed={} schema_digest={summary:016x}", cases.len(), cases.len() - failures.len(), failures.len());
	assert!(failures.is_empty(), "{} arity lifecycle cases failed", failures.len());
}
