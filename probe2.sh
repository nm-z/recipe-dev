#!/bin/bash
# usage: probe2.sh <worktree> <tag> <device> <script> ; samples compilers and the script's own CPU state each 0.5 s
W=$1; TAG=$2; DEV=$3; SCRIPT=$4
cd "$W" || exit 1
rm -rf "$W/home-$TAG"; mkdir -p "$W/home-$TAG"
( while true; do echo "$(date +%s.%N) $(ps -eo pid,comm,stat,pcpu,args | rg -v 'rg ' | rg "recipe-native.*(home-$TAG)|recipe-script-[0-9]+ $W/target" | awk '{print $2":"$3}' | tr '\n' ' ')"; sleep 0.5; done ) > "sampler-$TAG.txt" 2> "sampler-$TAG.err" < "$W/probe2.sh" &
SP=$!
START=$(date +%s.%N)
HOME="$W/home-$TAG" RECIPE_DEBUG=1 RECIPE_DEVICE=$DEV timeout 900 "$W/target/release/recipe" "$SCRIPT" > "run-$TAG.out" 2> "run-$TAG.err"
echo "exit=$?"
END=$(date +%s.%N)
kill $SP; wait $SP 2> "sampler-$TAG.err"
rg -o 'RESULT.*' "run-$TAG.err" || tail -3 "run-$TAG.err"
echo "total=$(echo "$END - $START" | bc)"
awk -v s="$START" 'NF>1 { for (i=2;i<=NF;i++) { key=$i; if (!(key in first)) first[key]=$1; last[key]=$1 } } END { for (k in first) printf "%s start=+%.1f end=+%.1f dur=%.1f\n", k, first[k]-s, last[k]-s, last[k]-first[k] }' "sampler-$TAG.txt" | sort -t+ -k2 -n
cp recipe.log "recipe-$TAG.log"
