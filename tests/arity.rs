//! Every input-target arity in each represented data situation.
//!
//! A data family holds one fixture per representation, and a representation that
//! covers arity names its variants by suffix: `_two_inputs` (or `_two_views` for
//! the image views), `_two_targets`, and both together. The base fixture is the
//! one input, one target case. This walks the data tree rather than listing
//! cases, so a representation that gains a variant is covered by adding the
//! fixture alone, and a representation that loses one fails loudly instead of
//! silently dropping an arity.

use recipe::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The target columns each family names when a fixture carries two of them. One
/// target is always called `target`.
const FAMILIES: &[(&str, [&str; 2])] = &[
	("numeric", ["power", "efficiency"]),
	("ordinal", ["score", "rank"]),
	("temporal", ["load", "cost"]),
	("text", ["label", "length"]),
	("categoric", ["label", "weight"]),
	("image", ["class", "brightness"]),
];

/// The suffixes that widen a representation, longest first so the combined form
/// is recognized before either half.
const SUFFIXES: &[(&str, bool, bool)] =
	&[("_two_inputs_two_targets", true, true), ("_two_views_two_targets", true, true), ("_two_targets", false, true), ("_two_inputs", true, false), ("_two_views", true, false)];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Arity {
	inputs: bool,
	targets: bool,
}

impl Arity {
	fn label(self) -> &'static str {
		match (self.inputs, self.targets) {
			(false, false) => "1 input  1 target ",
			(true, false) => "2 inputs 1 target ",
			(false, true) => "1 input  2 targets",
			(true, true) => "2 inputs 2 targets",
		}
	}
}

/// The representation a fixture belongs to and the arity it carries. A fixture
/// whose name carries no arity suffix is that representation's base case.
fn classify(name: &str) -> (String, Arity) {
	for (suffix, inputs, targets) in SUFFIXES {
		if let Some(base) = name.strip_suffix(suffix) {
			return (base.to_owned(), Arity { inputs: *inputs, targets: *targets });
		}
	}
	(name.to_owned(), Arity { inputs: false, targets: false })
}

/// The fixture name without its extension, so `single_csv_two_targets.csv` and
/// the directory `sharded_csv_two_targets` classify alike. A name such as
/// `compressed_csv.csv.gz` keeps every extension off.
fn bare(path: &Path) -> String {
	let mut name = path.file_name().and_then(|value| value.to_str()).unwrap_or_default().to_owned();
	while let Some((stem, extension)) = name.rsplit_once('.') {
		if extension.is_empty() || !extension.chars().all(|value| value.is_ascii_alphanumeric()) {
			break;
		}
		name = stem.to_owned();
	}
	name
}

fn cases() -> Vec<(&'static str, String, Arity, PathBuf, [&'static str; 2])> {
	let mut cases = Vec::new();
	for (family, pair) in FAMILIES {
		let directory = PathBuf::from("data").join(family);
		let mut entries = std::fs::read_dir(&directory)
			.unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()))
			.map(|entry| entry.expect("cannot read a data entry").path())
			.collect::<Vec<_>>();
		entries.sort();
		// A representation is covered here only when it names at least one widened
		// variant; anything else is a single-arity fixture the combo suite owns.
		let mut representations = BTreeMap::<String, Vec<(Arity, PathBuf)>>::new();
		for path in entries {
			let (base, arity) = classify(&bare(&path));
			representations.entry(base).or_default().push((arity, path));
		}
		for (base, mut found) in representations {
			if found.iter().all(|(arity, _)| *arity == Arity { inputs: false, targets: false }) {
				continue;
			}
			found.sort_by_key(|(arity, _)| *arity);
			for (arity, path) in found {
				cases.push((*family, base.clone(), arity, path, *pair));
			}
		}
	}
	assert!(!cases.is_empty(), "the data tree names no arity variants");
	cases
}

/// The message a panic carried, so one refusing case reports itself instead of
/// ending the walk: this suite exists to show the whole matrix, and a case that
/// cannot even load is exactly what it is meant to surface.
fn caught(payload: Box<dyn std::any::Any + Send>) -> String {
	match payload.downcast::<String>() {
		Ok(message) => *message,
		Err(payload) => payload.downcast::<&str>().map_or_else(|_| "non-string panic".to_owned(), |message| (*message).to_owned()),
	}
}

/// Trains, saves and reads back one fixture, answering the failure to report.
fn run(arity: Arity, path: &Path, pair: [&'static str; 2]) -> Option<String> {
	let source = path.to_string_lossy().into_owned();
	let data = if arity.targets { recipe.data(source).target(pair) } else { recipe.data(source).target(["target"]) };
	let model = recipe.model().layer(4).gelu().layer(1).loss(mse);
	let bundle = std::env::temp_dir().join(format!("recipe-arity-{}-{}.ogdl", std::process::id(), path.display().to_string().replace(['/', '\\', '.'], "-")));
	let report = recipe.train().epochs(2).save(&bundle).run(&model, &data);
	let outputs = if arity.targets { 2 } else { 1 };
	let predictions = report.predictions();
	let failure = if !report.final_loss().is_finite() {
		Some(format!("final loss is {}", report.final_loss()))
	} else if predictions.is_empty() {
		Some("no predictions".to_owned())
	} else if predictions.len() % outputs != 0 {
		Some(format!("{} predictions are not a whole number of {outputs}-target rows", predictions.len()))
	} else if let Some(value) = predictions.iter().find(|value| !value.is_finite()) {
		Some(format!("prediction {value} is not finite"))
	} else if !bundle.exists() {
		Some("the saved bundle is absent".to_owned())
	} else {
		None
	};
	let _ = std::fs::remove_file(&bundle);
	failure
}

#[test]
fn every_arity_in_every_representation() {
	let cases = cases();
	let mut summary = String::from("\ninput-target arity coverage\n");
	let mut failures = Vec::new();
	// A refusing case is reported with the rest rather than ending the walk.
	let hook = std::panic::take_hook();
	std::panic::set_hook(Box::new(|_| {}));
	let outcomes = cases
		.iter()
		.map(|(_, _, arity, path, pair)| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(*arity, path, *pair))).unwrap_or_else(|payload| Some(caught(payload))))
		.collect::<Vec<_>>();
	std::panic::set_hook(hook);
	for ((family, base, arity, _, _), outcome) in cases.iter().zip(outcomes) {
		match outcome {
			None => {
				let _ = writeln!(summary, "  {family:<10} {base:<34} {}  ok", arity.label());
			}
			Some(failure) => {
				let _ = writeln!(summary, "  {family:<10} {base:<34} {}  FAILED: {failure}", arity.label());
				failures.push(format!("{family}/{base} {}: {failure}", arity.label()));
			}
		}
	}
	let _ = writeln!(summary, "  {} cases", cases.len());
	print!("{summary}");
	assert!(failures.is_empty(), "{} of {} arity cases failed:\n{}", failures.len(), cases.len(), failures.join("\n"));
}
