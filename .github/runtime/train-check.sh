#!/usr/bin/env bash
# Trains the Qwen3-0.6B definition twice on a fixed 16-row table. Each run must
# exit 0 and end with a final loss below its first-epoch loss.
#
# Usage: RECIPE_TRAIN_GGUF=/path/Qwen3-0.6B-Q8_0.gguf .github/runtime/train-check.sh
# Set RECIPE_BIN to choose the binary. On a shared host, run this under the
# model-run lock; the script itself does not take it.
set -euo pipefail

recipe=${RECIPE_BIN:-./target/release/recipe}
gguf=${RECIPE_TRAIN_GGUF:?RECIPE_TRAIN_GGUF must name the Qwen3-0.6B Q8_0 GGUF file}
epochs=4
work=$(mktemp -d "${TMPDIR:-/tmp}/recipe-train-XXXXXX")
trap 'rm -rf "$work"' EXIT

table="$work/rows.csv"
{
	echo "token,next"
	for ((i = 0; i < 16; i++)); do
		echo "$((1000 + i * 7)),$((1000 + (i + 1) * 7))"
	done
} > "$table"

script="$work/qwen3-train.rs"
cat > "$script" <<'RECIPE'
use recipe::*;
use recipe::log::{epoch, loss, time};

fn main() {
	let gguf = std::env::var("RECIPE_TRAIN_GGUF").expect("RECIPE_TRAIN_GGUF names the Qwen3 GGUF file");
	let table = std::env::var("RECIPE_TRAIN_TABLE").expect("RECIPE_TRAIN_TABLE names the CSV dataset");
	let data = recipe.data(gguf).set(table).target("next");
	let mut model = recipe.model().embed(tokenizer.ggml.tokens, qwen3.embedding_length);
	for _ in 0..qwen3.block_count {
		model = model
			.res([
				norm(rms),
				attn(qwen3.attention.head_count).kv(qwen3.attention.head_count_kv).width(qwen3.attention.key_length)
					.qk(rms).rope(neox, qwen3.attention.key_length, qwen3.rope.freq_base),
			])
			.res([
				norm(rms),
				layer(qwen3.feed_forward_length).silu() * layer(qwen3.feed_forward_length),
				layer(qwen3.embedding_length),
			]);
	}
	let model = model.norm(rms).layer(tokenizer.ggml.tokens);
	let trained = recipe.train().lr(0.001).epochs(4).log([time, epoch, loss]).run(&model, &data);
	println!("final loss {}", trained.fnl.loss);
}
RECIPE

# Prints the loss after each epoch (stderr), then the final loss (stdout), for one run.
losses() {
	cat "$1.err" "$1.out" | sed 's/\x1b\[[0-9;]*m//g' | awk '
		/ epoch / { print $NF }
		/^final loss / { final = $3 }
		END { if (final != "") print final }'
}

check_run() {
	local name=$1 status=0 log="$work/$1"
	RECIPE_TRAIN_GGUF="$gguf" RECIPE_TRAIN_TABLE="$table" "$recipe" run "$script" --device cpu > "$log.out" 2> "$log.err" || status=$?
	if [ "$status" -ne 0 ]; then
		echo "$name: exit $status" >&2
		tail -n 20 "$log.err" >&2
		return 1
	fi
	mapfile -t values < <(losses "$log")
	local count=$(( ${#values[@]} - 1 ))
	if [ "$count" -ne "$epochs" ]; then
		echo "$name: expected $epochs epoch losses, found $count" >&2
		head -c 1500 "$log.out" >&2
		tail -c 1200 "$log.err" >&2
		return 1
	fi
	local first=${values[0]} final=${values[$count]}
	echo "$name: first-epoch loss $first, final loss $final"
	awk '/^(bound [0-9]+ weighted|target projection) /' "$log.err"
	if ! awk -v final="$final" -v first="$first" 'BEGIN { exit !(final + 0 < first + 0) }'; then
		echo "$name: final loss $final is not below first-epoch loss $first" >&2
		return 1
	fi
}

check_run run-1
check_run run-2
echo "train-check passed: both runs exited 0 with a final loss below the first-epoch loss"
