#!/bin/bash
# usage: probe.sh <worktree> <tag> <device> [EXPERTS] [TOPK]
W=$1; TAG=$2; DEV=$3
export EXPERTS=${4:-8} TOPK=${5:-4}
cd "$W" || exit 1
rm -rf "$W/home-$TAG/.cache"; mkdir -p "$W/home-$TAG"
(
  while true; do
    echo "$(date +%s.%N) $(ps -eo pid,comm,args | rg -v 'rg ' | rg "recipe-native.*(home-$TAG)" | awk '{print $1":"$2}' | tr '\n' ' ')"
    sleep 0.5
  done
) > "sampler-$TAG.txt" 2> "sampler-$TAG.err" < "$W/probe.sh" &
SP=$!
START=$(date +%s.%N)
HOME="$W/home-$TAG" RECIPE_DEBUG=1 RECIPE_DEVICE=$DEV timeout 900 "$W/target/release/recipe" repro.rs > "run-$TAG.out" 2> "run-$TAG.err"
echo "exit=$?"
END=$(date +%s.%N)
kill $SP; wait $SP 2> "sampler-$TAG.err"
rg -n RESULT "run-$TAG.err"
echo "total=$(echo "$END - $START" | bc)"
awk -v s="$START" 'NF>1 { for (i=2;i<=NF;i++) { key=$i; if (!(key in first)) first[key]=$1; last[key]=$1 } } END { for (k in first) printf "%s start=+%.1f end=+%.1f dur=%.1f\n", k, first[k]-s, last[k]-s, last[k]-first[k] }' "sampler-$TAG.txt" | sort -t+ -k2 -n
cp recipe.log "recipe-$TAG.log"
