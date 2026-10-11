#!/bin/bash
# usage: sweep.sh <worktree> <tag> <device>
W=$1; TAG=$2; DEV=$3
cd "$W" || exit 1
CASES=(
  "4 16,16,16,16,16,16,16,16"
  "1 16,16,16,16,16,16,16,16"
  "3 8,24,8,24,8,24,8"
  "3 8,24,8,24,8,24,8,24"
  "3 16,16,16,16,16,16,16"
  "1 16,16,16,16,16,16"
  "3 8,24,8,24,8,24"
  "2 8,24,8,24,8,24,8"
  "3 16,16,16,16,16,16,16,16"
  "1 16"
  "1 16,8"
  "2 16,8"
  "3 16,16,16"
  "8 16,16,16,16,16,16,16,16"
)
: > "sweep-$TAG.txt"
for case in "${CASES[@]}"; do
  set -- $case
  rm -rf "$W/home-$TAG"; mkdir -p "$W/home-$TAG"
  START=$(date +%s.%N)
  ( while true; do echo "$(date +%s.%N) $(ps -eo pid,comm,args | rg -v 'rg ' | rg "recipe-native.*(home-$TAG)" | awk '{print $1":"$2}' | tr '\n' ' ')"; sleep 0.5; done ) > "sampler-$TAG.txt" 2> "sampler-$TAG.err" < "$W/sweep.sh" &
  SP=$!
  TOPK=$1 WIDTHS=$2 HOME="$W/home-$TAG" RECIPE_DEVICE=$DEV timeout 900 "$W/target/release/recipe" repro.rs > "run-$TAG.out" 2> "run-$TAG.err"
  CODE=$?
  kill $SP; wait $SP 2> "sampler-$TAG.err"
  COMPILE=$(awk -v s="$START" 'NF>1 { for (i=2;i<=NF;i++) { key=$i; if (!(key in first)) first[key]=$1; last[key]=$1 } } END { t=0; for (k in first) t+=last[k]-first[k]; printf "%.1f", t }' "sampler-$TAG.txt")
  echo "exit=$CODE compile_seconds=$COMPILE $(rg -o 'RESULT.*' "run-$TAG.err" || tail -1 "run-$TAG.err")" >> "sweep-$TAG.txt"
done
echo done >> "sweep-$TAG.txt"
