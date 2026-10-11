#!/bin/bash
# Property check for edited model definitions. Each seed applies one to three
# random drop, swap, reorder, or re-parameterize edits to a Qwen3 chat definition
# with sed and awk, then runs the result on CPU. Any non-zero exit fails the check.
#
# Environment:
#   RECIPE_BIN            recipe binary (default: target/release/recipe)
#   RECIPE_EDIT_GGUF      GGUF the definitions read (default: the Qwen3-0.6B Q8_0 file)
#   EDIT_PROPERTY_SEEDS   seeds to run (default: 1 through 12)
#   EDIT_PROPERTY_DIR     scratch directory for definitions and logs
#   EDIT_PROPERTY_TIMEOUT seconds allowed for each run (default: 900)
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
bin=${RECIPE_BIN:-$root/target/release/recipe}
gguf=${RECIPE_EDIT_GGUF:-/home/nate/models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf}
seeds=${EDIT_PROPERTY_SEEDS:-$(seq 1 12)}
dir=${EDIT_PROPERTY_DIR:-$(mktemp -d "${HOME}/.cache/edit-property.XXXXXX")}
limit=${EDIT_PROPERTY_TIMEOUT:-900}
T=$'\t'
mkdir -p "$dir"

# The unedited definition. Its loop body holds an attention residual, then a feed-forward residual.
base() {
cat <<'DEFINITION'
use recipe::*;
use recipe::infer::{cached, input, out, pp, tg, time};

const GGUF: &str = "@GGUF@";

#[rustfmt::skip]
fn main() {
	let data = recipe.data(GGUF);

	let mut model = recipe.model()
		.no(bias)
		.embed(tokenizer.ggml.tokens, qwen3.embedding_length);

	for _ in 0..qwen3.block_count {
		model = model
			.res([
				norm(rms),
				attn(qwen3.attention.head_count).kv(qwen3.attention.head_count_kv)
					.width(qwen3.attention.key_length)
					.qk(rms)
					.rope(neox, qwen3.attention.key_length, qwen3.rope.freq_base),
			])
			.res([
				norm(rms),
				layer(qwen3.feed_forward_length).silu()
					* layer(qwen3.feed_forward_length),
				layer(qwen3.embedding_length),
			]);
	}

	model = model.norm(rms).layer(tokenizer.ggml.tokens);

	recipe.infer().chat([time, pp, tg, input, out, cached]).run(&model, &data);
}
DEFINITION
}

# Line numbers of the attention residual (L1..E1) and the feed-forward residual (L2..E2).
anchors() {
	L1=$(grep -nxF "$T$T$T.res([" "$1" | sed -n 1p | cut -d: -f1)
	L2=$(grep -nxF "$T$T$T.res([" "$1" | sed -n 2p | cut -d: -f1)
	[ -n "$L1" ] && [ -n "$L2" ] || return 1
	E1=$(awk -v s="$L1" -v t="$T" 'NR > s && index($0, t t t "])") == 1 { print NR; exit }' "$1")
	E2=$(awk -v s="$L2" -v t="$T" 'NR > s && index($0, t t t "])") == 1 { print NR; exit }' "$1")
}

# Each edit writes the edited definition to stdout and fails when its anchor is absent.
edit_drop_residual() {
	anchors "$1" || return 1
	if [ "$2" -eq 1 ]; then
		sed "${L1},${E1}d" "$1"
	else
		sed -e "${E1}s/)\$/);/" -e "${L2},${E2}d" "$1"
	fi
}

edit_swap_residuals() {
	anchors "$1" || return 1
	{
		sed -n "1,$((L1 - 1))p" "$1"
		sed -n "${L2},${E2}p" "$1" | sed '$s/]);$/])/'
		sed -n "${L1},${E1}p" "$1" | sed '$s/])$/]);/'
		sed -n "$((E2 + 1)),\$p" "$1"
	}
}

# The first argument is the residual, 1 for attention and 2 for feed-forward.
edit_layer_norm() {
	anchors "$1" || return 1
	local line=$(( $2 == 1 ? L1 + 1 : L2 + 1 ))
	sed -n "${line}p" "$1" | grep -q 'norm(rms),' || return 1
	sed "${line}s/norm(rms)/norm(layer)/" "$1"
}

edit_drop_norm() {
	anchors "$1" || return 1
	local line=$(( $2 == 1 ? L1 + 1 : L2 + 1 ))
	sed -n "${line}p" "$1" | grep -q 'norm(rms),' || return 1
	sed "${line}d" "$1"
}

edit_activation() {
	sed 's/silu()/gelu()/' "$1"
}

edit_heads() {
	sed 's/attn(qwen3.attention.head_count)/attn(8)/' "$1"
}

edit_layers() {
	sed "s/0\.\.qwen3\.block_count/0..$2/" "$1"
}

edit_width() {
	sed 's/layer(qwen3.feed_forward_length)/layer(1024)/g' "$1"
}

kinds=(drop_residual swap_residuals layer_norm drop_norm activation heads layers width)
layer_counts=(10 14 40)

failed=0
checked=0
for seed in $seeds; do
	RANDOM=$seed
	cur="$dir/seed-$seed.rs"
	base | sed "s|@GGUF@|$gguf|" > "$cur"
	edits=""
	count=$((1 + RANDOM % 3))
	for _ in $(seq 1 "$count"); do
		kind=${kinds[RANDOM % ${#kinds[@]}]}
		arg=$((1 + RANDOM % 2))
		case $kind in
			layers) arg=${layer_counts[RANDOM % ${#layer_counts[@]}]} ;;
		esac
		if "edit_$kind" "$cur" "$arg" > "$dir/next.rs" 2> /dev/null; then
			mv "$dir/next.rs" "$cur"
			edits="$edits $kind:$arg"
		else
			edits="$edits $kind:skipped"
		fi
	done
	checked=$((checked + 1))
	log="$dir/seed-$seed"
	set +e
	(cd "$dir" && timeout "$limit" "$bin" run "$cur" --device cpu --ctx 64 -p hi > "$log.out" 2> "$log.err")
	rc=$?
	set -e
	if [ "$rc" -ne 0 ]; then
		failed=$((failed + 1))
		echo "seed $seed: exit $rc, edits:$edits"
	else
		echo "seed $seed: exit 0, edits:$edits"
	fi
done

echo "edit property: $checked seeds, $failed failed, logs in $dir"
[ "$failed" -eq 0 ]
