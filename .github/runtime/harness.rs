use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// Every numeric source the loader accepts with a "target" column, plus the README data options.
const DATASETS: [&str; 15] = ["sample_subfolders", "sample_subfolders", "sample_subfolders", "sample_subfolders_two_inputs", "sample_subfolders_two_targets", "sample_subfolders_two_inputs_two_targets", "single_csv_two_targets.csv", "single_csv.csv", "sharded_csv", "split_files", "compressed_csv.csv.gz", "arrays_npz.npz", "arrays_hdf5.h5", "records_jsonl.jsonl", "samples_sqlite.sqlite"];
const DATA_OPTIONS: [&str; 4] = ["", "", ".norm(z_score)", ".split(0.8)"];

fn dataset(cursor: u64) -> String {
	format!("data/numeric/{}", pick(cursor, 7, &DATASETS))
}

// Two-target sources name their columns; every other source has one "target" column.
fn target(cursor: u64) -> &'static str {
	if pick(cursor, 7, &DATASETS).contains("two_targets") { r#"["power", "efficiency"]"# } else { "\"target\"" }
}

struct Failure {
	phase: String,
	message: String,
	output: String,
}

fn number(name: &str, default: u64) -> u64 {
	env(name).unwrap_or_else(|| default.to_string()).parse().unwrap_or_else(|_| panic!("{name} is not an unsigned integer"))
}

fn env(name: &str) -> Option<String> {
	std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn runner() -> PathBuf {
	env("RECIPE_COMPOSITION_RUNNER").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("target/debug/recipe"))
}

fn reproduction() -> PathBuf {
	env("RECIPE_COMPOSITION_REPRO").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("/tmp/recipe-repro-{}.rs", std::process::id())))
}

fn phase_file(path: &Path) -> PathBuf {
	path.with_extension("phase")
}

const WIDTHS: [usize; 5] = [4, 8, 16, 24, 32];
const ACTIVATIONS: [&str; 10] = ["", ".relu()", ".gelu()", ".silu()", ".tanh()", ".sigmoid()", ".selu()", ".elu()", ".leak()", ".prelu()"];
// Constructs whose defect is filed and unfixed are skipped so trials reach the rest of the grammar:
// norm(layer) #852, attention on NVIDIA #854 and its bf16 literal #855, ensemble width mismatch #853,
// moe or ensemble members that change the length #859. Set false once those fixes merge.
const AVOID_FILED: bool = true;
const NORMS: [&str; 7] = ["", "", "", ".norm(batch)", ".norm(layer)", ".norm(rms)", ".norm(l2)"];
const NORMS_FILED: [&str; 6] = ["", "", "", ".norm(batch)", ".norm(rms)", ".norm(l2)"];
const QUANTS: [&str; 7] = ["", "", "", ".qi(8).q0", ".qi(4).q1", ".qi(5).q0", ".qi(4).nf"];
// The model builder spells the same quantization suffixes as tuple fields.
const MODEL_QUANTS: [&str; 10] = ["", "", "", ".qi(8).0", ".qi(4).1", ".qi(5).0", ".qi(4).nf", ".qi(2).k", ".qi(3).k.s", ".iq(2).xxs"];
const LOSSES: [&str; 6] = ["mse", "rmse", "huber", "mae", "bce", "focal"];
const PRECISIONS: [&str; 7] = [".fp(32)", ".fp(32)", ".fp(16)", ".bf(16)", ".fp(64)", ".tf(32)", ".int(8)"];
const PRECISIONS_FILED: [&str; 6] = [".fp(32)", ".fp(32)", ".fp(16)", ".fp(64)", ".tf(32)", ".int(8)"];
const ATTENTION_OPTIONS: [&str; 6] = ["", ".kv(1)", ".qk(rms)", ".qk(l2)", ".gate()", ".rope(neox, {width}, 10000.0)"];
// Estimators take no activation, normalization or quantization.
const ESTIMATORS: [&str; 7] = ["svm()", "bayes()", "cbst(8)", "xgbst(8)", "lgbm(8)", "kmeans(4)", "knn(3)"];

// Every choice hashes the cursor with its own salt, so neighbouring cursors share nothing.
fn mix(cursor: u64, salt: u64) -> u64 {
	let mut value = cursor.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(salt);
	value ^= value >> 30;
	value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
	value ^= value >> 27;
	value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
	value ^ (value >> 31)
}

fn pick<'a>(cursor: u64, salt: u64, options: &[&'a str]) -> &'a str {
	options[(mix(cursor, salt) % options.len() as u64) as usize]
}

// One README block in grammar order: operation, activation, normalization, quantization.
// `member` restricts the operation to length-preserving ones and `width` pins its width for branch members.
fn block(cursor: u64, salt: u64, quants: &[&str], member: bool, width: Option<usize>) -> String {
	let bits = mix(cursor, salt);
	let width = width.unwrap_or(WIDTHS[(bits % 5) as usize]);
	let operation = match (bits >> 3) % if member { 5 } else { 12 } {
		0 => format!("layer({width})"),
		1 => format!("rnn({width})"),
		2 => format!("gru({width})"),
		3 if member => format!("lstm({width})"),
		4 if member => format!("perc({width})"),
		3 => format!("conv({width}, {})", 2 + (bits >> 8) % 4),
		4 => format!("rnn({width})"),
		5 => format!("gru({width})"),
		6 => format!("lstm({width})"),
		7 => format!("perc({width})"),
		8 if AVOID_FILED => format!("layer({width})"),
		8 => format!("attn({}).width({width}){}", 1 + (bits >> 8) % 4, pick(cursor, salt + 4, &ATTENTION_OPTIONS).replace("{width}", &width.to_string())),
		9 => return pick(cursor, salt + 5, &ESTIMATORS).to_owned(),
		_ => format!("pool({})", 2 + (bits >> 8) % 3),
	};
	let norms: &[&str] = if AVOID_FILED { &NORMS_FILED } else { &NORMS };
	format!("{operation}{}{}{}", pick(cursor, salt + 1, &ACTIVATIONS), pick(cursor, salt + 2, norms), pick(cursor, salt + 3, quants))
}

fn branch(cursor: u64, salt: u64, count: usize, width: Option<usize>) -> String {
	(0..count).map(|index| block(cursor, salt + 10 * index as u64, &QUANTS, AVOID_FILED, width)).collect::<Vec<_>>().join(", ")
}

fn composition(cursor: u64, salt: u64) -> String {
	let bits = mix(cursor, salt);
	let count = 1 + (bits % 4) as usize;
	let shared = AVOID_FILED.then_some(WIDTHS[((bits >> 12) % 5) as usize]);
	match (bits >> 4) % 4 {
		0 => format!(".res([{}])", branch(cursor, salt + 100, count, None)),
		1 => format!(".moe({}, [{}])", 1 + (bits >> 8) as usize % count, branch(cursor, salt + 100, count, None)),
		2 => format!(".ensemble([{}])", branch(cursor, salt + 100, count, shared)),
		_ => format!(".res([{}]).scale({})", branch(cursor, salt + 100, count, None), ["0.5", "1.5", "2.0"][((bits >> 8) % 3) as usize]),
	}
}

fn model(cursor: u64) -> String {
	let bits = mix(cursor, 1);
	let mut text = "recipe.model()".to_owned();
	for index in 0..1 + bits % 2 {
		text.push('.');
		text.push_str(&block(cursor, 200 + 10 * index, &MODEL_QUANTS, false, None));
	}
	if (bits >> 8) % 4 != 0 {
		text.push_str(&composition(cursor, 300));
	}
	if (bits >> 12) % 3 == 0 {
		text.push('.');
		text.push_str(&block(cursor, 400, &MODEL_QUANTS, false, None));
	}
	format!("{text}.layer(1).loss({})", pick(cursor, 5, &LOSSES))
}

fn source(cursor: u64, seed: u64) -> String {
	let model = model(cursor);
	let precision = pick(cursor, 6, if AVOID_FILED { &PRECISIONS_FILED[..] } else { &PRECISIONS[..] });
	let data = dataset(cursor);
	let options = pick(cursor, 8, &DATA_OPTIONS);
	let target = target(cursor);
	format!(
		r#"use recipe::*;
use std::path::{{Path, PathBuf}};

fn phase(value: &str) {{
    if let Some(path) = std::env::var_os("RECIPE_COMPOSITION_PHASE_PATH") {{
        std::fs::write(path, value).expect("cannot write phase");
    }}
}}

fn width(path: &Path) -> usize {{
    let text = std::fs::read_to_string(path).expect("cannot read saved model");
	let mut shape = text.lines().find_map(|line| line.trim().strip_prefix("shape ")).expect("saved model has no shape").split_whitespace();
    shape.next().expect("saved model has no channels").parse::<usize>().expect("saved channels are invalid") * shape.next().expect("saved model has no length").parse::<usize>().expect("saved length is invalid")
}}

fn main() {{
    let root = std::env::var_os("RECIPE_TRIAL_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let bundle = root.join(format!("recipe-model-{{}}.ogdl", std::process::id()));
    phase("setup");
    let data = recipe.data("{data}").target({target}){options};
    let model = {model};
    phase("training");
    let report = recipe.train().optimizer(adamw).lr(0.001).seed({seed}).epochs(20).log(all){precision}.save(&bundle).run(&model, &data);
    assert!(report.final_loss().is_finite());
    phase("inference");
    let output = recipe.infer(&bundle, &vec![0.0; width(&bundle)]);
    assert!(!output.is_empty() && output.iter().all(|value| value.is_finite()));
    std::fs::remove_file(bundle).expect("cannot remove saved model");
}}
"#
	)
}

fn run(path: &Path) -> Result<(), Failure> {
	let output = Command::new(runner()).arg(path).env("RECIPE_COMPOSITION_PHASE_PATH", phase_file(path)).output().map_err(|error| Failure { phase: "harness".to_owned(), message: error.to_string(), output: error.to_string() })?;
	if output.status.success() {
		return Ok(());
	}
	let output = diagnostic(&output);
	let phase = std::fs::read_to_string(phase_file(path)).unwrap_or_else(|_| "compilation".to_owned());
	// A panic message follows its "panicked at" line; the trailing backtrace note is never the failure.
	let lines = output.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>();
	let message = lines
		.iter()
		.position(|line| line.contains("panicked at"))
		.and_then(|index| lines.get(index + 1))
		.or_else(|| lines.iter().rev().find(|line| !line.starts_with("note:")))
		.map_or("candidate failed", |line| line);
	Err(Failure { phase, message: normalize(message), output })
}

// Panic messages embed per-process temp paths; keep each file name's shape and line, drop the digits that change per run.
fn normalize(message: &str) -> String {
	message
		.split(' ')
		.map(|token| {
			if !token.starts_with('/') {
				return token.to_owned();
			}
			let name = token.rsplit('/').next().unwrap_or(token);
			let (file, rest) = name.split_once(':').unwrap_or((name, ""));
			let file = file.chars().filter(|character| !character.is_ascii_digit()).collect::<String>();
			if rest.is_empty() { file } else { format!("{file}:{rest}") }
		})
		.collect::<Vec<_>>()
		.join(" ")
}

fn diagnostic(output: &Output) -> String {
	let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
	let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
	match (stderr.is_empty(), stdout.is_empty()) {
		(false, false) => format!("{stderr}\n{stdout}"),
		(false, true) => stderr,
		(true, false) => stdout,
		(true, true) => format!("candidate exited with {}", output.status),
	}
}

fn hash(mut value: u64, text: &str) -> u64 {
	for byte in text.bytes() {
		value ^= u64::from(byte);
		value = value.wrapping_mul(1_099_511_628_211);
	}
	value
}

fn base() -> String {
	let commit = Command::new("git").args(["rev-parse", "HEAD"]).output().expect("cannot read Recipe commit");
	let source = Command::new("git").args(["hash-object", "--no-filters", "recipe.rs"]).output().expect("cannot read Recipe source");
	format!("commit={} recipe_source={}", String::from_utf8_lossy(&commit.stdout).trim(), String::from_utf8_lossy(&source.stdout).trim())
}

fn emit(cursor: u64, source: &str, path: &Path, failure: &Failure, replay: &Failure, seed: u64) {
	let id = hash(hash(1_469_598_103_934_665_603, &failure.phase), &failure.message);
	eprintln!("RECIPE FAILURE BEGIN");
	eprintln!("id={id:016x}");
	eprintln!("base={}", base());
	eprintln!("cursor=cursor:{cursor} next:{} composition:{cursor}", cursor + 1);
	eprintln!("data=path:{}", dataset(cursor));
	eprintln!("configuration=cursor:{cursor} seed:{seed}");
	eprintln!("expected=the generated public Recipe composition trains and infers with finite values");
	eprintln!("observed=phase:{} message:{}", failure.phase, failure.message);
	eprintln!("output=phase:{}\n{}", failure.phase, failure.output);
	eprintln!("replay=phase:{} message:{} stable:{}", replay.phase, replay.message, failure.phase == replay.phase && failure.message == replay.message);
	eprintln!("command=target/debug/recipe test.rs");
	eprintln!("reproduction:\n```rust\n{source}```");
	eprintln!("RECIPE FAILURE END");
	let _ = path;
}

fn main() {
	let start = number("RECIPE_COMPOSITION_CURSOR", 0);
	let end = start.saturating_add(number("RECIPE_COMPOSITION_COUNT", 1));
	let seed = number("RECIPE_COMPOSITION_REPLAY_SEED", 17);
	for cursor in start..end {
		let source = source(cursor, seed);
		let path = reproduction();
		std::fs::write(&path, &source).expect("cannot write reproduction");
		eprintln!("composition {cursor}: kind=generated body={}", model(cursor));
		if let Err(failure) = run(&path) {
			let replay = run(&path).err().unwrap_or(Failure { phase: "replay".to_owned(), message: "replay passed".to_owned(), output: "replay passed".to_owned() });
			emit(cursor, &source, &path, &failure, &replay, seed);
		}
	}
	eprintln!("composition cursor={end}");
}
