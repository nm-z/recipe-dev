use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

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
// Every README precision: fp(8|16|32|64), int(1|4|8), bf(16), tf(32) and two computed formats
// f(exp, mantissa), the e5m2 and e4m3 layouts of an 8-bit float. fp(32) keeps three entries so
// a third of the trials stay at the reference precision.
const PRECISIONS: [&str; 13] = [".fp(32)", ".fp(32)", ".fp(32)", ".fp(16)", ".bf(16)", ".fp(64)", ".tf(32)", ".int(8)", ".fp(8)", ".int(4)", ".int(1)", ".f(5, 2)", ".f(4, 3)"];
const PRECISIONS_FILED: [&str; 12] = [".fp(32)", ".fp(32)", ".fp(32)", ".fp(16)", ".fp(64)", ".tf(32)", ".int(8)", ".fp(8)", ".int(4)", ".int(1)", ".f(5, 2)", ".f(4, 3)"];
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
// `sequential` says whether a sequence still reaches this block, see `model`.
fn block(cursor: u64, salt: u64, quants: &[&str], member: bool, width: Option<usize>, sequential: bool) -> String {
	let bits = mix(cursor, salt);
	let width = width.unwrap_or(WIDTHS[(bits % 5) as usize]);
	let operation = match (bits >> 3) % if member { 5 } else { 12 } {
		0 => format!("layer({width})"),
		1 => format!("rnn({width})"),
		2 => format!("gru({width})"),
		3 if member => format!("lstm({width})"),
		4 if member => format!("perc({width})"),
		3 if !sequential => format!("layer({width})"),
		3 => format!("conv({width}, {})", 2 + (bits >> 8) % 4),
		4 => format!("rnn({width})"),
		5 => format!("gru({width})"),
		6 => format!("lstm({width})"),
		7 => format!("perc({width})"),
		8 if AVOID_FILED => format!("layer({width})"),
		8 => format!("attn({}).width({width}){}", 1 + (bits >> 8) % 4, pick(cursor, salt + 4, &ATTENTION_OPTIONS).replace("{width}", &width.to_string())),
		9 => return pick(cursor, salt + 5, &ESTIMATORS).to_owned(),
		_ if !sequential => format!("layer({width})"),
		_ => format!("pool({})", 2 + (bits >> 8) % 3),
	};
	let norms: &[&str] = if AVOID_FILED { &NORMS_FILED } else { &NORMS };
	format!("{operation}{}{}{}", pick(cursor, salt + 1, &ACTIVATIONS), pick(cursor, salt + 2, norms), pick(cursor, salt + 3, quants))
}

fn branch(cursor: u64, salt: u64, count: usize, width: Option<usize>) -> String {
	(0..count).map(|index| block(cursor, salt + 10 * index as u64, &QUANTS, AVOID_FILED, width, false)).collect::<Vec<_>>().join(", ")
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
	// Only sample_subfolders holds a sequence per sample (33-line scans, 477 positions); every
	// other source, the two-input and two-target folders included, is one value or one row per
	// sample. An estimator leaves one value per sample as well. A convolution or pool after
	// either only asks Recipe for a kernel longer than the sequence (#214).
	let mut sequential = dataset(cursor) == "data/numeric/sample_subfolders";
	let mut push = |text: &mut String, salt: u64| {
		let block = block(cursor, salt, &MODEL_QUANTS, false, None, sequential);
		sequential &= !ESTIMATORS.iter().any(|estimator| block.starts_with(estimator));
		text.push('.');
		text.push_str(&block);
	};
	for index in 0..1 + bits % 2 {
		push(&mut text, 200 + 10 * index);
	}
	if (bits >> 8) % 4 != 0 {
		text.push_str(&composition(cursor, 300));
	}
	if (bits >> 12) % 3 == 0 {
		push(&mut text, 400);
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
	let harness = |error: std::io::Error| Failure { phase: "harness".to_owned(), message: error.to_string(), output: error.to_string() };
	let timeout_seconds = number("RECIPE_COMPOSITION_TIMEOUT_SECONDS", 180);
	assert!(timeout_seconds > 0, "RECIPE_COMPOSITION_TIMEOUT_SECONDS must be greater than zero");
	let mut child = Command::new(runner())
		.arg(path)
		.env("RECIPE_COMPOSITION_PHASE_PATH", phase_file(path))
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.map_err(harness)?;
	let mut stdout = child.stdout.take().expect("candidate stdout is absent");
	let mut stderr = child.stderr.take().expect("candidate stderr is absent");
	// The candidate's stderr is copied to `<reproduction>.output` as it arrives, so a watchdog that
	// ends the trial can read how far it got (epoch lines) before the kill.
	let progress = path.with_extension("output");
	let stderr_thread = std::thread::spawn(move || {
		let mut bytes = Vec::new();
		let mut file = std::fs::File::create(progress).ok();
		let mut buffer = [0_u8; 4096];
		while let Ok(count) = stderr.read(&mut buffer) {
			if count == 0 {
				break;
			}
			bytes.extend_from_slice(&buffer[..count]);
			if let Some(file) = file.as_mut() {
				let _ = file.write_all(&buffer[..count]);
			}
		}
		bytes
	});
	let stdout_thread = std::thread::spawn(move || {
		let mut bytes = Vec::new();
		let _ = stdout.read_to_end(&mut bytes);
		bytes
	});
	let started = Instant::now();
	let mut timed_out = false;
	let status = loop {
		if let Some(status) = child.try_wait().map_err(harness)? {
			break status;
		}
		if started.elapsed() >= Duration::from_secs(timeout_seconds) {
			timed_out = true;
			match child.kill() {
				Ok(()) => break child.wait().map_err(harness)?,
				Err(error) => match child.try_wait().map_err(harness)? {
					Some(status) => break status,
					None => return Err(harness(error)),
				},
			}
		}
		std::thread::sleep(Duration::from_millis(100));
	};
	let output = Output { status, stdout: stdout_thread.join().unwrap_or_default(), stderr: stderr_thread.join().unwrap_or_default() };
	if output.status.success() && !timed_out {
		return Ok(());
	}
	let output_status = output.status;
	let output = diagnostic(&output);
	let phase = std::fs::read_to_string(phase_file(path)).unwrap_or_else(|_| "compilation".to_owned());
	// A panic message follows its "panicked at" line; the trailing backtrace note is never the failure.
	let lines = output.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>();
	// A candidate the kernel killed left no panic line; its last line is a progress line that
	// names no failure, so the signal is the message. A candidate that exits with a code and no
	// panic line names that code and the last line it printed (#878 ended in inference this way).
	#[cfg(unix)]
	let signal = std::os::unix::process::ExitStatusExt::signal(&output_status).map(|signal| format!("terminated by signal {signal}"));
	#[cfg(not(unix))]
	let signal = None::<String>;
	let last = lines.iter().rev().find(|line| !line.starts_with("note:")).map(|line| (*line).to_owned());
	let message = if timed_out {
		format!("candidate exceeded {timeout_seconds}s during {phase}")
	} else {
		lines
			.iter()
			.position(|line| line.contains("panicked at"))
			.and_then(|index| lines.get(index + 1))
			.map(|line| (*line).to_owned())
			.or(signal)
			.or_else(|| output_status.code().map(|code| match &last {
				Some(last) => format!("exited with code {code} without a failure line, after: {last}"),
				None => format!("exited with code {code} without output"),
			}))
			.or(last)
			.unwrap_or_else(|| "candidate failed".to_owned())
	};
	Err(Failure { phase, message: normalize(&message), output })
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
	// A replayed source names its own data; the generated one is the cursor's.
	let data = source.lines().find_map(|line| line.split_once("recipe.data(\"").and_then(|(_, rest)| rest.split_once('"')).map(|(path, _)| path.to_owned())).unwrap_or_else(|| dataset(cursor));
	eprintln!("data=path:{data}");
	eprintln!("configuration=cursor:{cursor} seed:{seed}");
	eprintln!("expected=the generated public Recipe composition trains and infers with finite values");
	eprintln!("observed=phase:{} message:{}", failure.phase, failure.message);
	eprintln!("output=phase:{}\n{}", failure.phase, failure.output);
	eprintln!("replay=phase:{} message:{} stable:{}", replay.phase, replay.message, failure.phase == replay.phase && failure.message == replay.message);
	eprintln!("command=target/release/recipe harness.rs");
	eprintln!("reproduction:\n```rust\n{source}```");
	eprintln!("RECIPE FAILURE END");
	let _ = path;
}

fn main() {
	let start = number("RECIPE_COMPOSITION_CURSOR", 0);
	let end = start.saturating_add(number("RECIPE_COMPOSITION_COUNT", 1));
	let seed = number("RECIPE_COMPOSITION_REPLAY_SEED", 17);
	// RECIPE_COMPOSITION_SOURCE names a reproduction to run as it is, in place of the generated
	// one: a failure recorded against an earlier Recipe source is checked again at the current one.
	let given = env("RECIPE_COMPOSITION_SOURCE").map(|path| std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("cannot read {path}: {error}")));
	for cursor in start..end {
		let (source, kind) = match &given {
			Some(source) => (source.clone(), "replayed"),
			None => (source(cursor, seed), "generated"),
		};
		let path = reproduction();
		std::fs::write(&path, &source).expect("cannot write reproduction");
		let body = source.lines().find_map(|line| line.trim().strip_prefix("let model = ")).map_or_else(|| model(cursor), |line| line.trim_end_matches(';').to_owned());
		eprintln!("composition {cursor}: kind={kind} body={body}");
		if let Err(failure) = run(&path) {
			let replay = run(&path).err().unwrap_or(Failure { phase: "replay".to_owned(), message: "replay passed".to_owned(), output: "replay passed".to_owned() });
			emit(cursor, &source, &path, &failure, &replay, seed);
		}
	}
	eprintln!("composition cursor={end}");
}
