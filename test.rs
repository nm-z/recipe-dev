use recipe::*;
use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const DATA_MODES: usize = 4;
const AUTOREGRESSIVE_DATA_MODES: usize = 4;
const MODEL_OPERATIONS: usize = 22;
const ACTIVATIONS: usize = 16;
const NORMALIZATIONS: usize = 2;
const QUANTIZATIONS: &[(u16, u8, u16)] = &[
	(0, 4, 0), (0, 4, 1), (0, 5, 0), (0, 5, 1), (0, 8, 0), (0, 8, 1),
	(0, 2, 3), (0, 6, 3), (0, 8, 3),
	(0, 3, 3), (0, 3, 4), (0, 3, 5), (0, 3, 6),
	(0, 4, 3), (0, 4, 4), (0, 4, 5), (0, 5, 3), (0, 5, 4), (0, 5, 5),
	(0, 4, 2),
	(1, 1, 3), (1, 1, 4),
	(1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 2, 4),
	(1, 3, 1), (1, 3, 2), (1, 3, 3), (1, 3, 4),
	(1, 4, 2), (1, 4, 5),
];
const LOSSES: usize = 7;
const ARITHMETICS: usize = 10;
const STOP_MODES: usize = 3;
const LEARNING_RATES: usize = 2;
const LIFECYCLES: usize = 2;

struct DataCase {
	path: String,
	test: Option<String>,
	mode: usize,
	autoregressive: bool,
}

#[derive(Clone, Copy)]
struct Stage {
	operation: usize,
	activation: usize,
	normalization: usize,
	quantization: usize,
}

#[derive(Clone, Copy)]
struct TrainCase {
	arithmetic: usize,
	stop: usize,
	rate: usize,
}

struct Failure {
	phase: &'static str,
	message: String,
}

fn checked_product(values: &[u64]) -> u64 {
	values.iter().copied().try_fold(1_u64, u64::checked_mul).expect("composition count exceeds u64")
}

fn take(value: &mut u64, radix: usize) -> usize {
	let selected = *value % radix as u64;
	*value /= radix as u64;
	selected as usize
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
	while right != 0 {
		(left, right) = (right, left % right);
	}
	left
}

fn mix(mut value: u64) -> u64 {
	value ^= value >> 30;
	value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
	value ^= value >> 27;
	value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
	value ^ value >> 31
}

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
	for byte in bytes {
		hash ^= u64::from(*byte);
		hash = hash.wrapping_mul(1_099_511_628_211);
	}
	hash
}

fn content_hash(path: &Path) -> u64 {
	fn visit(path: &Path, hash: &mut u64) {
		if path.is_dir() {
			let mut entries = std::fs::read_dir(path)
				.unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
				.map(|entry| entry.expect("cannot read data entry").path())
				.filter(|path| !matches!(path.extension().and_then(|value| value.to_str()), Some("py" | "pyi" | "ipynb")))
				.collect::<Vec<_>>();
			entries.sort();
			for entry in entries { visit(&entry, hash) }
		} else {
			*hash = hash_bytes(*hash, path.to_string_lossy().as_bytes());
			*hash = hash_bytes(*hash, &std::fs::read(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display())));
		}
	}
	let mut hash = 1_469_598_103_934_665_603;
	visit(path, &mut hash);
	hash
}

fn base() -> String {
	let commit = Command::new("git").args(["rev-parse", "HEAD"]).output().expect("cannot read composition commit");
	let status = Command::new("git").args(["status", "--porcelain", "--untracked-files=no"]).output().expect("cannot read composition tree state");
	format!("commit={} tracked_tree={}", String::from_utf8_lossy(&commit.stdout).trim(), if status.stdout.is_empty() { "clean" } else { "modified" })
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
	match payload.downcast::<String>() {
		Ok(message) => *message,
		Err(payload) => payload.downcast::<&str>().map_or_else(|_| "non-string panic".to_owned(), |message| (*message).to_owned()),
	}
}

fn seed() -> u64 {
	std::env::var("RECIPE_COMPOSITION_SEED")
		.ok()
		.map(|value| value.parse().expect("RECIPE_COMPOSITION_SEED must be an unsigned integer"))
		.unwrap_or_else(|| {
			SystemTime::now()
				.duration_since(UNIX_EPOCH)
				.expect("system clock precedes the Unix epoch")
				.as_nanos() as u64
		})
}

fn start_cursor(total: u64) -> u64 {
	std::env::var("RECIPE_COMPOSITION_CURSOR")
		.ok()
		.map(|value| value.parse().expect("RECIPE_COMPOSITION_CURSOR must be an unsigned integer"))
		.unwrap_or(0)
		.min(total)
}

fn end_cursor(start: u64, total: u64) -> u64 {
	std::env::var("RECIPE_COMPOSITION_COUNT")
		.ok()
		.map(|value| value.parse::<u64>().expect("RECIPE_COMPOSITION_COUNT must be an unsigned integer"))
		.map_or(total, |count| start.saturating_add(count).min(total))
}

fn datasets() -> Vec<DataCase> {
	let mut families = std::fs::read_dir("data")
		.expect("cannot read data")
		.map(|entry| entry.expect("cannot read a data family").path())
		.filter(|path| path.is_dir())
		.collect::<Vec<_>>();
	families.sort();
	let mut cases = Vec::new();
	for family in families {
		let mut representations = std::fs::read_dir(&family)
			.unwrap_or_else(|error| panic!("cannot read {}: {error}", family.display()))
			.map(|entry| entry.expect("cannot read a data representation").path())
			.filter(|path| {
				!matches!(path.extension().and_then(|value| value.to_str()), Some("py" | "pyi" | "ipynb"))
			})
			.collect::<Vec<_>>();
		representations.sort();
		for path in representations {
			let autoregressive = path.ends_with("autoregressive_lines.txt");
			let modes = if autoregressive { AUTOREGRESSIVE_DATA_MODES } else { DATA_MODES };
			let path = path.to_string_lossy().into_owned();
			cases.extend((0..modes).map(|mode| DataCase { path: path.clone(), test: None, mode, autoregressive }));
		}
	}
	for family in ["numeric", "temporal"] {
		let directory = format!("data/{family}/{}", if family == "numeric" { "split_files" } else { "chronological_splits" });
		let (path, test) = (format!("{directory}/train.csv"), format!("{directory}/test.csv"));
		cases.extend((4..7).map(|mode| DataCase { path: path.clone(), test: Some(test.clone()), mode, autoregressive: false }));
	}
	assert!(!cases.is_empty(), "data defines no composition cases");
	cases
}

fn data(case: &DataCase) -> (Data, String) {
	let (mut data, mut source) = if case.autoregressive {
		(recipe.data(auto).set(case.path.clone()), format!("recipe.data(auto).set({:?})", case.path))
	} else {
		(recipe.data(case.path.clone()).target("target"), format!("recipe.data({:?}).target(\"target\")", case.path))
	};
	match case.mode {
		0 => {}
		1 => { data = data.norm(z_score); source.push_str(".norm(z_score)") }
		2 => { data = data.split(0.8); source.push_str(".split(0.8)") }
		3 => { data = data.broadcast(); source.push_str(".broadcast()") }
		4 => { let test = case.test.clone().expect("test source is absent"); data = data.test(test.clone()); source.push_str(&format!(".test({test:?})")) }
		5 => { let test = case.test.clone().expect("test source is absent"); data = data.test(test.clone()).split(0.8); source.push_str(&format!(".test({test:?}).split(0.8)")) }
		6 => { let test = case.test.clone().expect("test source is absent"); data = data.norm(z_score).broadcast().test(test.clone()).split(0.8); source.push_str(&format!(".norm(z_score).broadcast().test({test:?}).split(0.8)")) }
		_ => unreachable!(),
	}
	(data, source)
}

fn stage(mut value: u64) -> Stage {
	Stage {
		operation: take(&mut value, MODEL_OPERATIONS),
		activation: take(&mut value, ACTIVATIONS),
		normalization: take(&mut value, NORMALIZATIONS),
		quantization: take(&mut value, QUANTIZATIONS.len() + 1),
	}
}

fn operation(model: Model, operation: usize) -> (Model, &'static str) {
	match operation {
		0 => (model.layer(4), ".layer(4)"),
		1 => (model.layer(8), ".layer(8)"),
		2 => (model.conv(8, 1), ".conv(8, 1)"),
		3 => (model.conv(8, 3), ".conv(8, 3)"),
		4 => (model.pool(2), ".pool(2)"),
		5 => (model.kmeans(2), ".kmeans(2)"),
		6 => (model.knn(3), ".knn(3)"),
		7 => (model.svm(), ".svm()"),
		8 => (model.forest(2), ".forest(2)"),
		9 => (model.bayes(), ".bayes()"),
		10 => (model.cbst(), ".cbst()"),
		11 => (model.xgbst(), ".xgbst()"),
		12 => (model.lgbm(), ".lgbm()"),
		13 => (model.attn(1), ".attn(1)"),
		14 => (model.rnn(8), ".rnn(8)"),
		15 => (model.gru(8), ".gru(8)"),
		16 => (model.lstm(8), ".lstm(8)"),
		17 => (model.layer(8).res([layer(8), layer(8)]), ".layer(8)|.res([layer(8), layer(8)])"),
		18 => (model.layer(8).res([conv(8, 1), conv(8, 1)]), ".layer(8)|.res([conv(8, 1), conv(8, 1)])"),
		19 => (model.layer(8).res([conv(8, 1), relu(), layer(8)]), ".layer(8)|.res([conv(8, 1), relu(), layer(8)])"),
		20 => (model.layer(8).moe(2, 1, 8, Activation::Silu, Scoring::Softmax, true, true), ".layer(8)|.moe(2, 1, 8, Activation::Silu, Scoring::Softmax, true, true)"),
		21 => (model.perc(8), ".perc(8)"),
		_ => unreachable!(),
	}
}

fn activation(model: Model, activation: usize) -> (Model, &'static str) {
	match activation {
		0 => (model, ""),
		1 => (model.cos(), ".cos()"),
		2 => (model.exp(), ".exp()"),
		3 => (model.log(), ".log()"),
		4 => (model.ln(), ".ln()"),
		5 => (model.huber(), ".huber()"),
		6 => (model.tan(), ".tan()"),
		7 => (model.relu(), ".relu()"),
		8 => (model.leak(), ".leak()"),
		9 => (model.sigmoid(), ".sigmoid()"),
		10 => (model.tanh(), ".tanh()"),
		11 => (model.selu(), ".selu()"),
		12 => (model.gelu(), ".gelu()"),
		13 => (model.silu(), ".silu()"),
		14 => (model.elu(), ".elu()"),
		15 => (model.prelu(), ".prelu()"),
		_ => unreachable!(),
	}
}

fn quantization(model: Model, quantization: usize) -> (Model, String) {
	if quantization == 0 {
		return (model, String::new());
	}
	let (family, bits, variant) = QUANTIZATIONS[quantization - 1];
	let source = match (family, variant) {
		(0, 0 | 1) => format!(".qi({bits}).{variant}"),
		(0, 2) => format!(".qi({bits}).nf"),
		(0, 3) => format!(".qi({bits}).k"),
		(0, 4) => format!(".qi({bits}).k.s"),
		(0, 5) => format!(".qi({bits}).k.m"),
		(0, 6) => format!(".qi({bits}).k.l"),
		(1, 1) => format!(".iq({bits}).xxs"),
		(1, 2) => format!(".iq({bits}).xs"),
		(1, 3) => format!(".iq({bits}).s"),
		(1, 4) => format!(".iq({bits}).m"),
		(1, 5) => format!(".iq({bits}).nl"),
		_ => unreachable!(),
	};
	(model.quantize(family, bits, variant), source)
}

fn apply(model: Model, stage: Stage) -> (Model, String) {
	let (model, operation) = operation(model, stage.operation);
	let (model, activation) = activation(model, stage.activation);
	let (model, normalization) = if stage.normalization == 0 { (model, "") } else { (model.norm(batch), ".norm(batch)") };
	let (model, quantization) = quantization(model, stage.quantization);
	(model, format!("{operation}{activation}{normalization}{quantization}"))
}

fn model(mut ordinal: u64) -> (Model, String) {
	let stages = checked_product(&[MODEL_OPERATIONS as u64, ACTIVATIONS as u64, NORMALIZATIONS as u64, (QUANTIZATIONS.len() + 1) as u64]);
	let one = stages;
	let mut model = recipe.model();
	let mut description = String::from("recipe.model()");
	if ordinal < one {
		let selected = stage(ordinal);
		let (next, source) = apply(model, selected);
		model = next;
		description.push('|');
		description.push_str(&source);
	} else {
		ordinal -= one;
		for selected in [stage(ordinal % stages), stage(ordinal / stages)] {
			let (next, source) = apply(model, selected);
			model = next;
			description.push('|');
			description.push_str(&source);
		}
	}
	(model, description)
}

fn loss(model: Model, loss: usize) -> (Model, &'static str) {
	match loss {
		0 => (model.loss(mse), ".loss(mse)"),
		1 => (model.loss(rmse), ".loss(rmse)"),
		2 => (model.loss(huber), ".loss(huber)"),
		3 => (model.loss(mae), ".loss(mae)"),
		4 => (model.loss(bce), ".loss(bce)"),
		5 => (model.loss(ce), ".loss(ce)"),
		6 => (model.loss(focal), ".loss(focal)"),
		_ => unreachable!(),
	}
}

fn train(case: TrainCase, seed: usize) -> (Train, String) {
	let train = recipe
		.train()
		.optimizer(adamw)
		.lr([0.001, 0.0001][case.rate])
		.seed(seed)
		.epochs(1)
		.log(all);
	let mut source = format!("recipe.train().optimizer(adamw).lr({}).seed({seed}).epochs(1).log(all)", [0.001, 0.0001][case.rate]);
	let train = match case.stop {
		0 => train,
		1 => { source.push_str(".stop(0.0)"); train.stop(0.0) }
		2 => { source.push_str(".stop(0.8)"); train.stop(0.8) }
		_ => unreachable!(),
	};
	let (train, arithmetic) = match case.arithmetic {
		0 => (train.fp(8), ".fp(8)"),
		1 => (train.fp(16), ".fp(16)"),
		2 => (train.fp(32), ".fp(32)"),
		3 => (train.fp(64), ".fp(64)"),
		4 => (train.int(1), ".int(1)"),
		5 => (train.int(4), ".int(4)"),
		6 => (train.int(8), ".int(8)"),
		7 => (train.bf(16), ".bf(16)"),
		8 => (train.tf(32), ".tf(32)"),
		9 => (train.f(6, 9), ".f(6, 9)"),
		_ => unreachable!(),
	};
	source.push_str(arithmetic);
	(train, source)
}

fn input_width(path: &Path) -> usize {
	let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
	let mut shape = text
		.lines()
		.find_map(|line| line.trim().strip_prefix("shape "))
		.expect("saved model has no input shape")
		.split_whitespace();
	let channels = shape.next().expect("saved model has no input channels").parse::<usize>().expect("saved input channels are invalid");
	let length = shape.next().expect("saved model has no input length").parse::<usize>().expect("saved input length is invalid");
	channels.checked_mul(length).expect("saved input width overflows")
}

fn readable_chain(source: &str) -> String {
	let mut source = source.replace(").", ")\n\t\t.").replace("].", "]\n\t\t.");
	for suffix in ["k.s", "k.m", "k.l", "xxs", "xs", "nf", "nl", "k", "s", "m", "0", "1"] {
		source = source.replace(&format!(")\n\t\t.{suffix}."), &format!(").{suffix}\n\t\t."));
	}
	source
}

fn readable_model(source: &str) -> String {
	source.replace('|', "\n\t\t")
}

fn reproduction(case: u64, data_case: &DataCase, model_ordinal: u64, loss_ordinal: usize, train_case: TrainCase, lifecycle: usize, phase: &str) -> String {
	let (_, data) = data(data_case);
	let (model, mut model_source) = model(model_ordinal);
	let (_, loss) = loss(model, loss_ordinal);
	model_source.push('|');
	model_source.push_str(loss);
	let (_, train) = train(train_case, case as usize);
	let (data, model, train) = (readable_chain(&data), readable_model(&model_source), readable_chain(&train));
	let mut body = format!(r#"use recipe::*;

fn main() {{
	let bundle = "/tmp/recipe-composition-repro.ogdl";
	let data = {data};
	let model = {model};"#);
	if phase == "setup" {
		return format!("{body}\n}}\n");
	}
	body.push_str(&format!("\n\tlet report = {train}\n\t\t.save(bundle)\n\t\t.run(&model, &data);\n\tassert!(report.final_loss().is_finite());"));
	if phase == "training" {
		return format!("{body}\n}}\n");
	}
	if lifecycle == 1 {
		body.push_str(&format!("\n\tlet resumed = {train}\n\t\t.resume(bundle)\n\t\t.save(bundle)\n\t\t.run(&model, &data);\n\tassert!(resumed.final_loss().is_finite());"));
	}
	if phase == "resumed training" {
		return format!("{body}\n}}\n");
	}
	let bundle_path = PathBuf::from(format!("/tmp/recipe-composition-{}.ogdl", std::process::id()));
	let width = input_width(&bundle_path);
	body.push_str(&format!("\n\tlet output = recipe.infer(bundle, &[0.0; {width}]);\n\tassert!(!output.is_empty());\n\tassert!(output.iter().all(|value| value.is_finite()));"));
	format!("{body}\n}}\n")
}

fn execute(case: u64, data_case: &DataCase, model_ordinal: u64, loss_ordinal: usize, train_case: TrainCase, lifecycle: usize, phase: &Cell<&'static str>) {
	let bundle = PathBuf::from(format!("/tmp/recipe-composition-{}.ogdl", std::process::id()));
	if bundle.exists() {
		std::fs::remove_file(&bundle).unwrap_or_else(|error| panic!("cannot remove {}: {error}", bundle.display()));
	}
	let (data, _) = data(data_case);
	let (model, _) = model(model_ordinal);
	let (model, _) = loss(model, loss_ordinal);
	// Recipe has no checkpoints: save and resume use the same file.
	phase.set("training");
	let (training, _) = train(train_case, case as usize);
	let report = training.save(&bundle).run(&model, &data);
	assert!(report.final_loss().is_finite(), "composition {case} produced a nonfinite training loss");
	if lifecycle == 1 {
		phase.set("resumed training");
		let (training, _) = train(train_case, case as usize);
		let report = training.resume(&bundle).save(&bundle).run(&model, &data);
		assert!(report.final_loss().is_finite(), "composition {case} produced a nonfinite resumed loss");
	}
	phase.set("inference");
	let input = vec![0.0; input_width(&bundle)];
	let output = recipe.infer(&bundle, &input);
	assert!(!output.is_empty(), "composition {case} inference produced no values");
	assert!(output.iter().all(|value| value.is_finite()), "composition {case} inference produced a nonfinite value");
	std::fs::remove_file(&bundle).unwrap_or_else(|error| panic!("cannot remove {}: {error}", bundle.display()));
}

fn attempt(case: u64, data_case: &DataCase, model_ordinal: u64, loss_ordinal: usize, train_case: TrainCase, lifecycle: usize) -> std::result::Result<(), Failure> {
	let phase = Cell::new("setup");
	let hook = std::panic::take_hook();
	std::panic::set_hook(Box::new(|_| {}));
	let result = catch_unwind(AssertUnwindSafe(|| execute(case, data_case, model_ordinal, loss_ordinal, train_case, lifecycle, &phase)))
		.map_err(|payload| Failure { phase: phase.get(), message: panic_message(payload) });
	std::panic::set_hook(hook);
	result
}

fn emit_failure(seed: u64, cursor: u64, next_cursor: u64, step: u64, case: u64, data_case: &DataCase, description: &str, loss_ordinal: usize, train_case: TrainCase, lifecycle: usize, failure: &Failure, replay: &Failure, source: &str) {
	let mut fingerprint = hash_bytes(1_469_598_103_934_665_603, failure.phase.as_bytes());
	fingerprint = hash_bytes(fingerprint, failure.message.as_bytes());
	std::fs::write("/tmp/recipe-composition-repro.rs", source).expect("cannot write composition reproduction");
	eprintln!("RECIPE FAILURE BEGIN");
	eprintln!("id={fingerprint:016x}");
	eprintln!("base={}", base());
	eprintln!("cursor=seed:{seed} cursor:{cursor} next:{next_cursor} step:{step} composition:{case}");
	eprintln!("data=path:{:?} hash:{:016x}", data_case.path, content_hash(Path::new(&data_case.path)));
	eprintln!("configuration=data_mode:{} model:{} loss:{} arithmetic:{} stop:{} rate:{} lifecycle:{}", data_case.mode, description, loss_ordinal, train_case.arithmetic, train_case.stop, train_case.rate, lifecycle);
	eprintln!("expected=training, optional resume, and inference produce finite numerical results through the public Recipe API");
	eprintln!("observed=phase:{} message:{}", failure.phase, failure.message);
	eprintln!("output=phase:{} message:{}", failure.phase, failure.message);
	eprintln!("replay=phase:{} message:{} stable:{}", replay.phase, replay.message, failure.phase == replay.phase && failure.message == replay.message);
	eprintln!("command=cargo run --bin recipe -- /tmp/recipe-composition-repro.rs");
	eprintln!("reproduction:\n```rust\n{source}```");
	eprintln!("RECIPE FAILURE END");
}

fn main() {
	let datasets = datasets();
	let stages = checked_product(&[MODEL_OPERATIONS as u64, ACTIVATIONS as u64, NORMALIZATIONS as u64, (QUANTIZATIONS.len() + 1) as u64]);
	let models = stages.checked_add(stages.checked_mul(stages).expect("two-stage model count overflows")).expect("model count overflows");
	let total = checked_product(&[
		datasets.len() as u64,
		models,
		LOSSES as u64,
		ARITHMETICS as u64,
		STOP_MODES as u64,
		LEARNING_RATES as u64,
		LIFECYCLES as u64,
	]);
	let seed = seed();
	let offset = mix(seed) % total;
	let mut step = mix(seed ^ 0x9e37_79b9_7f4a_7c15) % total;
	while gcd(step, total) != 1 {
		step = (step + 1) % total;
	}
	let start = start_cursor(total);
	let end = end_cursor(start, total);
	eprintln!("composition space={total} seed={seed} cursor={start} end={end} step={step}");
	for cursor in start..end {
		let mut ordinal = ((offset as u128 + cursor as u128 * step as u128) % total as u128) as u64;
		let data_ordinal = take(&mut ordinal, datasets.len());
		let model_ordinal = ordinal % models;
		ordinal /= models;
		let loss_ordinal = take(&mut ordinal, LOSSES);
		let train_case = TrainCase {
			arithmetic: take(&mut ordinal, ARITHMETICS),
			stop: take(&mut ordinal, STOP_MODES),
			rate: take(&mut ordinal, LEARNING_RATES),
		};
		let lifecycle = take(&mut ordinal, LIFECYCLES);
		assert_eq!(ordinal, 0, "composition decoder left unused state");
		let case = ((offset as u128 + cursor as u128 * step as u128) % total as u128) as u64;
		let (selected_model, mut description) = model(model_ordinal);
		let (_, selected_loss) = loss(selected_model, loss_ordinal);
		description.push_str(selected_loss);
		description = description.replace('|', "");
		eprintln!("composition {case}: data={} mode={} model={} loss={} arithmetic={} stop={} rate={} lifecycle={}", datasets[data_ordinal].path, datasets[data_ordinal].mode, description, loss_ordinal, train_case.arithmetic, train_case.stop, train_case.rate, lifecycle);
		if let Err(failure) = attempt(case, &datasets[data_ordinal], model_ordinal, loss_ordinal, train_case, lifecycle) {
			let replay = attempt(case, &datasets[data_ordinal], model_ordinal, loss_ordinal, train_case, lifecycle)
				.err()
				.unwrap_or(Failure { phase: "replay", message: "the exact replay passed".to_owned() });
			let source = reproduction(case, &datasets[data_ordinal], model_ordinal, loss_ordinal, train_case, lifecycle, failure.phase);
			emit_failure(seed, cursor, cursor + 1, step, case, &datasets[data_ordinal], &description, loss_ordinal, train_case, lifecycle, &failure, &replay, &source);
		}
	}
	eprintln!("composition cursor={end}");
}
