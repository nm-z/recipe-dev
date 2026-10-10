#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p samples
while IFS=$'\t' read -r type m k file shard start bytes tensor; do
	[[ $type == type ]] && continue
	[[ -f samples/$file && $(stat -c %s "samples/$file") == "$bytes" ]] && continue
	end=$((start+bytes-1))
	url="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-0000${shard}-of-00003.gguf?packed-range=${start}-${end}"
	curl -fLsS --retry 2 --range "$start-$end" -D "samples/$file.http" "$url" -o "samples/$file.partial"
	[[ $(stat -c %s "samples/$file.partial") == "$bytes" ]]
	rg -iq "content-range: bytes $start-$end/" "samples/$file.http"
	mv "samples/$file.partial" "samples/$file"
	printf '%s %s\n' "$file" "$tensor"
done < shapes.tsv
