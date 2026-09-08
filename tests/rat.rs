//! Public RAT evaluator contract.
//!
//! The evaluator is deliberately a small executable rather than a Rust test
//! double. This keeps the test on the same process boundary as a caller's
//! evaluator and exercises the public `Train::rat` path.

use recipe::*;
use std::{
	fs,
	path::{Path, PathBuf},
	process::Command,
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

struct Fixture {
	root: PathBuf,
	data: PathBuf,
	evaluator: PathBuf,
	calls: PathBuf,
}

impl Fixture {
	fn new(case: &str) -> Self {
		let root = std::env::temp_dir().join(format!("recipe-rat-{case}-{}", std::process::id()));
		if root.exists() {
			fs::remove_dir_all(&root).unwrap_or_else(|error| panic!("cannot remove {}: {error}", root.display()));
		}
		fs::create_dir_all(&root).unwrap_or_else(|error| panic!("cannot create {}: {error}", root.display()));
		let data = root.join("samples.tsv");
		fs::write(&data, "feature\talpha\tbeta\n1\t0.25\t0.75\n").unwrap_or_else(|error| panic!("cannot write {}: {error}", data.display()));
		let calls = root.join("calls");
		let evaluator = root.join("evaluate");
		let body = match case {
			"valid" => format!("line=$(cat)\nprintf '%s\\n' \"$line\" >> '{}'\nprintf 'diagnostic\\n' >&2\nprintf '0\\n'", shell_quote(&calls),),
			"malformed" => "cat >/dev/null\nprintf 'malformed diagnostic\\n' >&2\nprintf 'not-a-reward\\n'".to_owned(),
			"range" => "cat >/dev/null\nprintf 'range diagnostic\\n' >&2\nprintf '1.1\\n'".to_owned(),
			"exit" => "cat >/dev/null\nprintf 'exit diagnostic\\n' >&2\nexit 7".to_owned(),
			other => panic!("unknown RAT evaluator test case {other:?}"),
		};
		let script = format!("#!/bin/sh\nset -eu\n{body}\n");
		fs::write(&evaluator, script).unwrap_or_else(|error| panic!("cannot write {}: {error}", evaluator.display()));
		#[cfg(unix)]
		{
			let mut permissions = fs::metadata(&evaluator).unwrap_or_else(|error| panic!("cannot inspect {}: {error}", evaluator.display())).permissions();
			permissions.set_mode(0o700);
			fs::set_permissions(&evaluator, permissions).unwrap_or_else(|error| panic!("cannot make {} executable: {error}", evaluator.display()));
		}
		Self { root, data, evaluator, calls }
	}

	fn run(&self) -> TrainingReport {
		let evaluator = recipe.model().layer(1);
		let proposal = recipe.model().layer(2).loss(&evaluator);
		let data = recipe.data(self.data.to_string_lossy().into_owned()).target(["alpha", "beta"]);
		recipe.train().rat(self.evaluator.to_string_lossy().into_owned()).fp(64).seed(17).epochs(1).save(self.root.join("proposal.ogdl")).run(&proposal, &data)
	}
}

impl Drop for Fixture {
	fn drop(&mut self) {
		let _ = fs::remove_dir_all(&self.root);
	}
}

fn shell_quote(path: &Path) -> String {
	format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

fn run_child(case: &str) -> std::process::Output {
	Command::new(std::env::current_exe().unwrap_or_else(|error| panic!("cannot locate RAT test binary: {error}")))
		.env("RECIPE_FORCE_CPU", "1")
		.env_remove("RECIPE_DEVICE")
		.env("RECIPE_RAT_CASE", case)
		.arg("--exact")
		.arg("rat_evaluator_contract")
		.arg("--nocapture")
		.output()
		.unwrap_or_else(|error| panic!("cannot run RAT test case {case:?}: {error}"))
}

#[test]
fn rat_evaluator_contract() {
	if let Ok(case) = std::env::var("RECIPE_RAT_CASE") {
		let fixture = Fixture::new(&case);
		if case == "valid" {
			let report = fixture.run();
			assert!(report.final_loss().is_finite(), "valid reward produced a nonfinite loss");
			assert_eq!(report.predictions().len(), 2, "the report did not return the proposal");
			let artifact = fixture.root.join("proposal.ogdl");
			let artifact_size = fs::metadata(&artifact).unwrap_or_else(|error| panic!("cannot inspect {}: {error}", artifact.display())).len();
			assert!(artifact_size != 0, "valid reward did not save a model artifact");
			assert_eq!(recipe.infer(&artifact, &[1.0]).len(), 2, "the saved proposal has the wrong output width");
			let saved = fs::read_to_string(&artifact).unwrap_or_else(|error| panic!("cannot read {}: {error}", artifact.display()));
			assert!(saved.contains("out 616c706861") && saved.contains("out 62657461"), "the saved proposal lost its declared output names");
			let calls = fs::read_to_string(&fixture.calls).unwrap_or_else(|error| panic!("cannot read {}: {error}", fixture.calls.display()));
			let lines = calls.lines().collect::<Vec<_>>();
			assert!(!lines.is_empty(), "the evaluator received no proposal");
			for line in lines {
				let (alpha, beta) = line.split_once(',').unwrap_or_else(|| panic!("proposal line has no comma: {line:?}"));
				assert!(alpha.strip_prefix("alpha=").and_then(|value| value.parse::<f64>().ok()).is_some(), "alpha field is malformed: {line:?}");
				assert!(beta.strip_prefix("beta=").and_then(|value| value.parse::<f64>().ok()).is_some(), "beta field is malformed: {line:?}");
			}
		} else {
			fixture.run();
		}
		return;
	}

	let valid = run_child("valid");
	assert!(valid.status.success(), "valid evaluator failed:\n{}\n{}", String::from_utf8_lossy(&valid.stdout), String::from_utf8_lossy(&valid.stderr));
	assert!(String::from_utf8_lossy(&valid.stderr).contains("diagnostic"), "evaluator stderr was not passed through");
	for case in ["malformed", "range", "exit"] {
		let output = run_child(case);
		assert!(!output.status.success(), "{case} evaluator unexpectedly succeeded:\n{}", String::from_utf8_lossy(&output.stdout));
		assert!(!output.stderr.is_empty(), "{case} evaluator failure had no diagnostic");
	}
}
