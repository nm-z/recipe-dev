#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for experts in 10 16; do
	while IFS=$'\t' read -r type m k file shard start bytes tensor; do
		[[ $type == 17 || $type == 18 || $type == 20 ]] || continue
		end=$((start+bytes*experts-1))
		name="${file%.bin}-e${experts}.bin"
		url="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-0000${shard}-of-00003.gguf?expert-range=$start-$end"
		[[ -f samples/$name && $(stat -c %s "samples/$name") == $((bytes*experts)) ]] && continue
		curl -fLsS --retry 2 --range "$start-$end" -D "samples/$name.http" "$url" -o "samples/$name.partial"
		[[ $(stat -c %s "samples/$name.partial") == $((bytes*experts)) ]]
		rg -iq "content-range: bytes $start-$end/" "samples/$name.http"
		mv "samples/$name.partial" "samples/$name"
	done < shapes.tsv
done
