//! The fail-closed aggregate behind the `recipe/runtime-gate` check.
//!
//! It is deliberately a standalone program with no dependencies: it is compiled
//! with `rustc` directly so that the aggregate cannot be affected by a failure
//! in the crate it is judging.
//!
//! Run it as `gate <evidence-directory> <results-json> <candidate-sha> <run-id> <run-attempt>`.
//! `--self-check` instead runs the decision table below, which covers every
//! state the gate must reject.
use std::collections::BTreeMap;
use std::path::Path;

/// The six cells are hardcoded. A cell that stops reporting must fail the gate
/// rather than shrink the expected set.
const EXPECTED: [&str; 6] = ["recipe/linux-cpu", "recipe/linux-gpu", "recipe/windows-cpu", "recipe/windows-gpu", "recipe/macos-cpu", "recipe/macos-gpu"];
/// The cells whose evidence must additionally show GPU execution.
const GPU_CELLS: [&str; 3] = ["recipe/linux-gpu", "recipe/windows-gpu", "recipe/macos-gpu"];

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
	Pass,
	Fail(String),
}

struct Evidence {
	cell: String,
	commit: String,
	run_id: String,
	run_attempt: String,
	backend: String,
	device: String,
	executed: u64,
	failed: u64,
	gpu_execution: bool,
}

/// Extracts `"key": <value>` from a flat JSON document. The producers are the
/// workflow and the suite, both of which emit a known shape, so this stays a
/// scanner rather than a general parser.
fn field(document: &str, key: &str) -> Option<String> {
	let needle = format!("\"{key}\"");
	let start = document.find(&needle)? + needle.len();
	let rest = document[start..].trim_start();
	let rest = rest.strip_prefix(':')?.trim_start();
	if let Some(text) = rest.strip_prefix('"') {
		let mut value = String::new();
		let mut characters = text.chars();
		while let Some(character) = characters.next() {
			match character {
				'\\' => value.push(characters.next()?),
				'"' => return Some(value),
				other => value.push(other),
			}
		}
		return None;
	}
	let end = rest.find([',', '}', ']', '\n']).unwrap_or(rest.len());
	Some(rest[..end].trim().to_owned())
}

fn number(document: &str, key: &str) -> Option<u64> {
	field(document, key)?.parse().ok()
}

fn parse_evidence(cell_document: &str, suite_document: &str) -> Result<Evidence, String> {
	let missing = |key: &str| format!("evidence is missing {key}");
	Ok(Evidence {
		cell: field(cell_document, "cell").ok_or_else(|| missing("cell"))?,
		commit: field(cell_document, "commit").ok_or_else(|| missing("commit"))?,
		run_id: field(cell_document, "run_id").ok_or_else(|| missing("run_id"))?,
		run_attempt: field(cell_document, "run_attempt").ok_or_else(|| missing("run_attempt"))?,
		backend: field(cell_document, "backend").ok_or_else(|| missing("backend"))?,
		device: field(cell_document, "device").ok_or_else(|| missing("device"))?,
		executed: number(suite_document, "executed").ok_or_else(|| missing("executed"))?,
		failed: number(suite_document, "failed").ok_or_else(|| missing("failed"))?,
		gpu_execution: field(cell_document, "gpu_execution").is_some_and(|value| value == "true"),
	})
}

/// The whole decision. Every rejected state returns `Verdict::Fail`.
fn decide(results: &BTreeMap<String, String>, evidence: &BTreeMap<String, Result<Evidence, String>>, candidate_sha: &str, run_id: &str, run_attempt: &str) -> Verdict {
	let mut problems = Vec::new();
	for cell in EXPECTED {
		let Some(result) = results.get(cell) else {
			problems.push(format!("{cell}: no job result was reported"));
			continue;
		};
		// Only an exact success counts. failure, cancelled, skipped and neutral
		// all land here.
		if result != "success" {
			problems.push(format!("{cell}: job result is {result:?}, expected \"success\""));
			continue;
		}
		let found = match evidence.get(cell) {
			Some(Ok(found)) => found,
			Some(Err(error)) => {
				problems.push(format!("{cell}: evidence did not validate: {error}"));
				continue;
			}
			None => {
				problems.push(format!("{cell}: evidence artifact is missing"));
				continue;
			}
		};
		if found.cell != cell {
			problems.push(format!("{cell}: evidence declares cell {:?}", found.cell));
		}
		if found.commit != candidate_sha {
			problems.push(format!("{cell}: evidence commit {} is not the candidate {candidate_sha}", found.commit));
		}
		if found.run_id != run_id || found.run_attempt != run_attempt {
			problems.push(format!("{cell}: evidence is from run {}/{} not {run_id}/{run_attempt}", found.run_id, found.run_attempt));
		}
		if found.executed == 0 {
			problems.push(format!("{cell}: evidence records zero executed checks"));
		}
		if found.failed != 0 {
			problems.push(format!("{cell}: evidence records {} failed checks", found.failed));
		}
		if GPU_CELLS.contains(&cell) {
			if !found.gpu_execution {
				problems.push(format!("{cell}: evidence does not show GPU execution"));
			}
			if found.backend == "cpu" || found.device == "cpu" {
				problems.push(format!("{cell}: GPU cell reports backend {:?} device {:?}", found.backend, found.device));
			}
		}
	}
	for reported in results.keys() {
		if !EXPECTED.contains(&reported.as_str()) {
			problems.push(format!("{reported}: unexpected cell in the result set"));
		}
	}
	if problems.is_empty() { Verdict::Pass } else { Verdict::Fail(problems.join("; ")) }
}

fn read_results(document: &str) -> BTreeMap<String, String> {
	// The workflow writes {"recipe/linux-cpu":"success",...}.
	let mut results = BTreeMap::new();
	for cell in EXPECTED {
		if let Some(value) = field(document, cell) {
			results.insert(cell.to_owned(), value);
		}
	}
	results
}

fn main() {
	let arguments: Vec<String> = std::env::args().skip(1).collect();
	if arguments.first().is_some_and(|value| value == "--self-check") {
		self_check();
		return;
	}
	let [directory, results_path, candidate_sha, run_id, run_attempt] = <[String; 5]>::try_from(arguments).unwrap_or_else(|_| {
		eprintln!("usage: gate <evidence-directory> <results-json> <candidate-sha> <run-id> <run-attempt>");
		std::process::exit(2)
	});
	let results_document = std::fs::read_to_string(&results_path).unwrap_or_else(|error| panic!("cannot read {results_path}: {error}"));
	let results = read_results(&results_document);
	let mut evidence = BTreeMap::new();
	for cell in EXPECTED {
		let slug = cell.replace('/', "-");
		let cell_directory = Path::new(&directory).join(&slug);
		let cell_document = std::fs::read_to_string(cell_directory.join("cell.json"));
		let suite_document = std::fs::read_to_string(cell_directory.join("suite.json"));
		match (cell_document, suite_document) {
			(Ok(cell_document), Ok(suite_document)) => {
				evidence.insert(cell.to_owned(), parse_evidence(&cell_document, &suite_document));
			}
			(Err(error), _) => {
				evidence.insert(cell.to_owned(), Err(format!("cell.json is unreadable: {error}")));
			}
			(_, Err(error)) => {
				evidence.insert(cell.to_owned(), Err(format!("suite.json is unreadable: {error}")));
			}
		}
	}
	println!("gate candidate={candidate_sha} run={run_id}/{run_attempt}");
	for cell in EXPECTED {
		let summary = match evidence.get(cell) {
			Some(Ok(found)) => format!("commit={} executed={} failed={} backend={} device={}", found.commit, found.executed, found.failed, found.backend, found.device),
			Some(Err(error)) => format!("invalid ({error})"),
			None => "missing".to_owned(),
		};
		println!("  {cell}: result={} evidence={summary}", results.get(cell).map_or("<missing>", String::as_str));
	}
	match decide(&results, &evidence, &candidate_sha, &run_id, &run_attempt) {
		Verdict::Pass => println!("GATE PASS: all {} cells succeeded for {candidate_sha}", EXPECTED.len()),
		Verdict::Fail(reason) => {
			eprintln!("GATE FAIL: {reason}");
			std::process::exit(1)
		}
	}
}

// ---------------------------------------------------------------------------
// Decision-table self-check. Every rejected state the gate must catch has a
// case here; `--self-check` runs them and exits non-zero on any regression.
// ---------------------------------------------------------------------------

fn sample(cell: &str, gpu: bool) -> Evidence {
	Evidence {
		cell: cell.to_owned(),
		commit: "cafe1234".to_owned(),
		run_id: "42".to_owned(),
		run_attempt: "1".to_owned(),
		backend: if gpu { "nvidia".to_owned() } else { "cpu".to_owned() },
		device: if gpu { "nv0".to_owned() } else { "cpu".to_owned() },
		executed: 8,
		failed: 0,
		gpu_execution: gpu,
	}
}

fn healthy() -> (BTreeMap<String, String>, BTreeMap<String, Result<Evidence, String>>) {
	let mut results = BTreeMap::new();
	let mut evidence = BTreeMap::new();
	for cell in EXPECTED {
		results.insert(cell.to_owned(), "success".to_owned());
		evidence.insert(cell.to_owned(), Ok(sample(cell, GPU_CELLS.contains(&cell))));
	}
	(results, evidence)
}

fn self_check() {
	let mut failures: Vec<String> = Vec::new();
	let cases: Vec<(&str, Box<dyn Fn(&mut BTreeMap<String, String>, &mut BTreeMap<String, Result<Evidence, String>>)>, bool)> = vec![
		("all six succeed", Box::new(|_: &mut _, _: &mut _| {}), true),
		("a cell failed", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			results.insert("recipe/linux-cpu".to_owned(), "failure".to_owned());
		}), false),
		("a cell was cancelled", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			results.insert("recipe/macos-gpu".to_owned(), "cancelled".to_owned());
		}), false),
		("a cell was skipped", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			results.insert("recipe/windows-gpu".to_owned(), "skipped".to_owned());
		}), false),
		("a cell was neutral", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			results.insert("recipe/linux-gpu".to_owned(), "neutral".to_owned());
		}), false),
		("a cell timed out", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			results.insert("recipe/linux-gpu".to_owned(), "timed_out".to_owned());
		}), false),
		("a cell is missing entirely", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			results.remove("recipe/macos-cpu");
		}), false),
		("an unexpected cell appeared", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			results.insert("recipe/freebsd-cpu".to_owned(), "success".to_owned());
		}), false),
		("evidence artifact is missing", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			evidence.remove("recipe/windows-cpu");
		}), false),
		("evidence is malformed", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			evidence.insert("recipe/windows-cpu".to_owned(), Err("truncated".to_owned()));
		}), false),
		("evidence is for another commit", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			let mut found = sample("recipe/linux-cpu", false);
			found.commit = "deadbeef".to_owned();
			evidence.insert("recipe/linux-cpu".to_owned(), Ok(found));
		}), false),
		("evidence is from another run", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			let mut found = sample("recipe/linux-cpu", false);
			found.run_id = "41".to_owned();
			evidence.insert("recipe/linux-cpu".to_owned(), Ok(found));
		}), false),
		("evidence is from another attempt", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			let mut found = sample("recipe/linux-cpu", false);
			found.run_attempt = "2".to_owned();
			evidence.insert("recipe/linux-cpu".to_owned(), Ok(found));
		}), false),
		("evidence is mislabelled", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			evidence.insert("recipe/linux-cpu".to_owned(), Ok(sample("recipe/macos-cpu", false)));
		}), false),
		("zero checks executed", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			let mut found = sample("recipe/linux-cpu", false);
			found.executed = 0;
			evidence.insert("recipe/linux-cpu".to_owned(), Ok(found));
		}), false),
		("a check failed inside the suite", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			let mut found = sample("recipe/linux-cpu", false);
			found.failed = 1;
			evidence.insert("recipe/linux-cpu".to_owned(), Ok(found));
		}), false),
		("a GPU cell fell back to CPU", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			evidence.insert("recipe/linux-gpu".to_owned(), Ok(sample("recipe/linux-gpu", false)));
		}), false),
		("a GPU cell claims no GPU execution", Box::new(|_: &mut _, evidence: &mut BTreeMap<String, Result<Evidence, String>>| {
			let mut found = sample("recipe/macos-gpu", true);
			found.gpu_execution = false;
			evidence.insert("recipe/macos-gpu".to_owned(), Ok(found));
		}), false),
		("every cell failed", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| {
			for cell in EXPECTED {
				results.insert(cell.to_owned(), "failure".to_owned());
			}
		}), false),
		("no results at all", Box::new(|results: &mut BTreeMap<String, String>, _: &mut _| results.clear()), false),
	];

	for (name, mutate, expected_pass) in cases {
		let (mut results, mut evidence) = healthy();
		mutate(&mut results, &mut evidence);
		let verdict = decide(&results, &evidence, "cafe1234", "42", "1");
		let passed = verdict == Verdict::Pass;
		let outcome = match &verdict {
			Verdict::Pass => "pass".to_owned(),
			Verdict::Fail(reason) => format!("fail: {reason}"),
		};
		if passed == expected_pass {
			println!("self-check {name}: OK ({outcome})");
		} else {
			println!("self-check {name}: REGRESSED ({outcome})");
			failures.push(name.to_owned());
		}
	}

	if failures.is_empty() {
		println!("SELF-CHECK PASS");
	} else {
		eprintln!("SELF-CHECK FAIL: {}", failures.join(", "));
		std::process::exit(1)
	}
}
