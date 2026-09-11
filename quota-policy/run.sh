#!/usr/bin/env bash
set -uo pipefail
run_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$run_dir" || exit 1
export RECIPE_DEVICE=amd0
export RECIPE_BINARY="$run_dir/recipe-worker"
export RECIPE_QUOTA_EVALUATOR_BIN="$run_dir/quota-evaluator"
export RECIPE_QUOTA_EVALUATOR_SETUP="$run_dir/quota-setup.sql"
export RECIPE_QUOTA_DEVICES=amd0.cpu.archy:nv0.nv1.nv2.nv3.nv4.nv5.nv6.nv7.cpu
export QUOTA_RUST_METRICS="$run_dir/workers.tsv"
export RECIPE_QUOTA_METRICS_LOG="$run_dir/evaluations.tsv"
export QUOTA_BENCHMARK_CORPUS=/home/nate/Desktop/recipe-dev/corpus-clean.tsv
export QUOTA_BENCHMARK_SAVE="$run_dir/quota-model.ogdl"
date -u +%FT%TZ > started-at.txt
"$run_dir/quota-policy" full 1 "$run_dir/evaluate-quota" > training.stdout 2> training.stderr
result=$?
printf '%s\n' "$result" > exit-status.txt
date -u +%FT%TZ > finished-at.txt
exit "$result"
