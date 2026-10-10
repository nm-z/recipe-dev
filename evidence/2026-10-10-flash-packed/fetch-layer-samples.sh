#!/bin/bash
set -euo pipefail
cd /home/nate/work/cx-fl-kern/evidence/2026-10-10-flash-packed
mkdir -p samples/layer
for layer in 0 2; do
	for op in gate up down; do
		read -r shard type k m offset < <(awk -F'\t' -v name="blk.$layer.ffn_${op}_exps.weight" '$2==name{print $1,$3,$4,$5,$8}' inventory.tsv)
		block=256;size=74
		[[ $type == 18 ]] && size=98
		[[ $type == 20 ]] && { block=32;size=18; }
		bytes=$((k*m/block*size*3));start=$((offset+31840));end=$((start+bytes-1))
		path="samples/layer/l${layer}-${op}-e3.bin"
		[[ -f $path && $(stat -c %s "$path") == "$bytes" ]] && continue
		url="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-0000${shard}-of-00003.gguf?layer-range=$start-$end"
		curl -fLsS --retry 2 --range "$start-$end" -D "$path.http" "$url" -o "$path.partial"
		[[ $(stat -c %s "$path.partial") == "$bytes" ]]
		rg -iq "content-range: bytes $start-$end/" "$path.http"
		mv "$path.partial" "$path"
	done
done
