#!/bin/bash
set -u
cd "$(dirname "$0")"
exec 9>/home/nate/codex/flash-1092-die1.lock
flock 9
export CUDA_VISIBLE_DEVICES=GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28
export NITER=${NITER:-100}
export LD_LIBRARY_PATH=/home/nate/llama-cuda118-master-bin:${LD_LIBRARY_PATH:-}
tag=${RUN_TAG:-initial}
binary=${BENCH_BINARY:-./bench}
status=0
printf 'start %s pid=%s\n' "$(date -Is)" "$$" > "run-state-$tag.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,power.limit,clocks.sm,clocks.mem,utilization.gpu --format=csv > gpu-state.txt
: > "results-$tag.csv"
: > "run-errors-$tag.txt"
while IFS=$'\t' read -r t m k file shard start bytes tensor; do
	[[ $t == type ]] && continue
	[[ -n ${ONLY_TYPE:-} && $t != "$ONLY_TYPE" ]] && continue
	[[ -n ${ONLY_TYPES:-} && ! $t =~ ^(${ONLY_TYPES})$ ]] && continue
	printf '%s %s\n' "$(date -Is)" "$tensor" >> "run-state-$tag.txt"
	sample=$file
	[[ ${EXPERTS:-1} != 1 ]] && sample="${file%.bin}-e${EXPERTS}.bin"
	"$binary" "$t" "$m" "$k" "samples/$sample" >> "results-$tag.csv" 2>>"run-errors-$tag.txt" || status=1
done < shapes.tsv
printf 'end %s status=%s\n' "$(date -Is)" "$status" >> "run-state-$tag.txt"
exit "$status"
