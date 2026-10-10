#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
revision=766911a6b7369840a91dbcd95f9f997acaab6cd6
mkdir -p headers
for shard in 1 2 3; do
	url="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/$revision/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-0000${shard}-of-00003.gguf?header-range=0-16777215"
	curl -fLsS --retry 2 --range 0-16777215 -D "headers/$shard.http" "$url" -o "headers/$shard.partial"
	rg -iq '^HTTP/.* 206' "headers/$shard.http"
	rg -iq '^content-range: bytes 0-' "headers/$shard.http"
done
./gguf-header headers/{1,2,3}.partial > inventory.tsv 2> metadata.txt
printf '%s\n' "$revision" > revision.txt
