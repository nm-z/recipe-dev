#!/usr/bin/env bash
# Verifies the file's canonical model on token pairs from README.md. Vocabulary
# targets retain its pretrained output head. MSE is measured over one-hot targets.
# Usage: RECIPE_TRAIN_GGUF=/path/model.gguf .github/runtime/train-check.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
recipe=${RECIPE_BIN:-$root/target/release/recipe}
gguf=${RECIPE_TRAIN_GGUF:?RECIPE_TRAIN_GGUF must name a GGUF file}
work=${RECIPE_TRAIN_WORK:-$(mktemp -d /home/nate/codex/recipe-train-XXXXXX)}
mkdir -p "$work"
export RECIPE_TRAIN_GGUF="$gguf" RECIPE_TRAIN_WORK="$work" RECIPE_TRAIN_ROOT="$root"

script="$work/train.rs"
cat > "$script" <<'RECIPE'
use recipe::*;
use recipe::log::{epoch, loss};
use std::io::Write;

fn main() {
	let path = std::env::var("RECIPE_TRAIN_GGUF").unwrap();
	let work = std::env::var("RECIPE_TRAIN_WORK").unwrap();
	let file = recipe.gguf(&path);
	let bound = file.model();
	let model = bound.model();
	let vocabulary = bound.vocabulary();
	let root = std::env::var("RECIPE_TRAIN_ROOT").unwrap();
	let text = std::fs::read_to_string(format!("{root}/README.md")).unwrap();
	let ids = file.tokenizer().encode(&text);
	let pairs = ids.windows(2).take(16).collect::<Vec<_>>();
	assert_eq!(pairs.len(), 16);
	let columns = (0..vocabulary).map(|id| format!("next_{id}")).collect::<Vec<_>>();
	let table = format!("{work}/rows.csv");
	let mut rows = std::fs::File::create(&table).unwrap();
	writeln!(rows, "token,{}", columns.join(",")).unwrap();
	let mut reference = Vec::new();
	let mut expected_loss = 0.0;
	for pair in &pairs {
		write!(rows, "{}", pair[0]).unwrap();
		for id in 0..vocabulary {
			write!(rows, ",{}", usize::from(id == pair[1] as usize)).unwrap();
		}
		writeln!(rows).unwrap();
		let logits = bound.infer(&pair[..1]);
		assert_eq!(logits.len(), vocabulary);
		for (id, value) in logits.iter().enumerate() {
			expected_loss += (value - f64::from(u8::from(id == pair[1] as usize))).powi(2);
		}
		reference.extend(logits);
	}
	drop(rows);
	expected_loss /= reference.len() as f64;
	let data = recipe.data([path.clone(), table]).target(columns);
	for line in model.binding(&data).unwrap().listing() {
		println!("binding {line}");
	}
	let trained = recipe.train().lr(0.0001).stop(0.0).epochs(4).log([epoch, loss]).run(model, &data);
	assert_eq!(trained.rows, pairs.len());
	assert_eq!(trained.loss.len(), 4);
	assert_eq!(trained.initial_predictions().len(), reference.len(), "training must retain the vocabulary head");
	let mut initial = trained.initial_predictions().chunks_exact(vocabulary).collect::<Vec<_>>();
	let mut reference = reference.chunks_exact(vocabulary).collect::<Vec<_>>();
	let order = |left: &&[f64], right: &&[f64]| left.iter().zip(*right).map(|(a, b)| a.total_cmp(b)).find(|value| !value.is_eq()).unwrap_or(std::cmp::Ordering::Equal);
	initial.sort_by(order);
	reference.sort_by(order);
	let maximum = initial.iter().flat_map(|row| row.iter()).zip(reference.iter().flat_map(|row| row.iter()))
		.map(|(left, right)| (left - right).abs()).fold(0.0_f64, f64::max);
	assert!(maximum < 0.001, "pretrained logit mismatch {maximum}");
	assert!((trained.itl.loss - expected_loss).abs() < 0.001 * expected_loss.max(1.0), "initial loss differs from inference");
	assert!(trained.fnl.loss.is_finite() && trained.fnl.loss < trained.itl.loss);
	assert!(trained.fnl.loss < trained.loss[0][0], "loss must fall after the first epoch");
	assert!(trained.initial_predictions().iter().zip(trained.predictions()).any(|(left, right)| left.to_bits() != right.to_bits()),
		"the update must change predictions");
	println!("pretrained MSE {expected_loss:.9}, initial MSE {:.9}, first-epoch MSE {:.9}, final MSE {:.9}, maximum logit error {maximum:.9}",
		trained.itl.loss, trained.loss[0][0], trained.fnl.loss);
	let numeric = recipe.data(format!("{root}/.github/runtime/data/linear.csv")).target("target");
	let integer = recipe.model().layer(32).fp(32).layer(1).int(8);
	let fallback = recipe.train().lr(0.001).stop(0.0).epochs(4).run(&integer, &numeric);
	assert!(fallback.fnl.loss.is_finite() && fallback.fnl.loss < fallback.itl.loss);
	println!("integer-checkpoint default: initial MSE {:.9}, final MSE {:.9}", fallback.itl.loss, fallback.fnl.loss);
	println!("train-check passed: pretrained vocabulary logits agree, updates change predictions, and loss decreases");
}
RECIPE

for run in 1 2; do
	if ! (cd "$work" && RECIPE_FORCE_CPU=1 "$recipe" run "$script" --device cpu --cfg stored) > "$work/run-$run.log" 2>&1; then
		tail -n 20 "$work/run-$run.log" >&2
		exit 1
	fi
	tail -n 3 "$work/run-$run.log"
	grep -q '^train-check passed:' "$work/run-$run.log"
done
