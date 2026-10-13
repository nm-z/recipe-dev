#!/usr/bin/env bash
set -euo pipefail

# Run on a machine with an NVIDIA GPU and a real GGUF that fits its VRAM.
# All inference goes through the Recipe CLI and the canonical GGUF model.
runner=$(readlink -f "${RECIPE_RUNNER:?Set RECIPE_RUNNER to the built Recipe CLI}")
model=${1:?Pass a real GGUF path}
device=${RECIPE_RESIDENCY_DEVICE:-nv0}
work=${RECIPE_RESIDENCY_WORK:-/home/nate/codex/recipe-residency}
mkdir -p "$work"
rm -f "$work/crash.pid" "$work/active.pid"
cp --reflink=auto "$model" "$work/model.gguf"
ln -f "$model" "$work/expiry.gguf"
cat > "$work/run.rs" <<'RS'
use recipe::*;
use std::{fs, io::Write, time::{Duration, Instant}};
fn observe(logits: &[f64], residency: &[WeightResidency]) {
	assert!(!logits.is_empty() && logits.iter().all(|value| value.is_finite()));
	let expected = std::env::var("EXPECT").unwrap();
	assert!(!residency.is_empty());
	for resident in residency {
		assert_eq!(resident.outcome, expected);
		assert!(!resident.cache_reused);
		if expected == "reused" {
			assert_eq!((resident.read_bytes, resident.converted_bytes, resident.uploaded_bytes), (0, 0, 0));
			assert_ne!(resident.holder_pid, std::process::id());
		}
		println!("resident {} {} {} {} {} {} {} {:.6} {} {}", resident.outcome, resident.identity, resident.holder_pid, resident.bytes,
			resident.read_bytes, resident.converted_bytes, resident.uploaded_bytes, resident.attach_seconds, resident.kernel_identity, resident.artifact.display());
	}
	let result = std::env::var("RESULT").unwrap();
	let mut output = fs::File::create(result).unwrap();
	for value in logits { output.write_all(&value.to_le_bytes()).unwrap(); }
}
fn main() {
	let path = std::env::var("MODEL").unwrap();
	let file = recipe.gguf(&path);
	if std::env::var("CAPACITY").as_deref() == Ok("1") {
		let _ = file.model().place(1048576, &[]);
		panic!("oversized placement was admitted");
	}
	let mut model = file.model().model().clone();
	if std::env::var("LAYOUT").as_deref() == Ok("fp32") { model = model.fp(32); }
	let scale = std::env::var("SCALE").unwrap_or_else(|_| "1".to_owned()).parse::<f64>().unwrap();
	if scale != 1.0 { model = model.scale(scale).fp(32); }
	let data = recipe.data(path.as_str());
	let started = Instant::now();
	if let Ok(marker) = std::env::var("PAUSE") {
		let placed = recipe.infer().place(&model, &data, 16, &[]);
		observe(&placed.prefill(&[1, 2, 3]), &placed.residency());
		fs::write(marker, std::process::id().to_string()).unwrap();
		std::thread::sleep(Duration::from_secs(240));
		return;
	}
	let report = recipe.infer().prompt_ids([1, 2, 3]).tokens(1).run(&model, &data);
	observe(&report.logits, &report.residency);
	println!("preparation {:.6} compile {:.6} request {:.6}", started.elapsed().as_secs_f64(), report.compile.seconds(), report.time.seconds());
}
RS

run() {
	local label=$1 expected=$2 ttl=$3 scale=${4:-1} selected=${5:-model}
	local args=()
	local context=(--ctx "${RECIPE_RESIDENCY_CONTEXT:-16}")
	if [[ ${RECIPE_RESIDENCY_CONTEXT:-16} == auto ]]; then context=(); fi
	if [[ $ttl != private ]]; then args=(--ttl "$ttl"); fi
	if ! MODEL="$work/$selected.gguf" EXPECT="$expected" SCALE="$scale" RESULT="$work/$label.bin" \
		timeout 120 "$runner" run "$work/run.rs" --device "$device" "${context[@]}" "${args[@]}" > "$work/$label.log" 2>&1; then
		tail -30 "$work/$label.log"
		return 1
	fi
	awk '/^resident / || /^preparation /' "$work/$label.log"
}
identity() { awk '/^resident / { print $3 }' "$work/$1.log"; }
kernel() { awk '/^resident / { print $10; exit }' "$work/$1.log"; }
artifact() { awk '/^resident / { print $11; exit }' "$work/$1.log"; }

run private private private
run cold created 1
run warm reused 1
cmp "$work/private.bin" "$work/cold.bin"
cmp "$work/cold.bin" "$work/warm.bin"
[[ $(identity cold) == "$(identity warm)" ]]
echo 'PASS cold/warm: same allocation, exact logits, zero weight reads/conversions/uploads'

RECIPE_RESIDENCY_CONTEXT=auto run auto-context reused 1
[[ $(identity auto-context) == "$(identity cold)" ]]
cmp "$work/cold.bin" "$work/auto-context.bin"
echo 'PASS automatic context fitting reuses resident weights'

run private-scale private private 1.25
run scale reused 1 1.25
cmp "$work/private-scale.bin" "$work/scale.bin"
[[ $(identity scale) == "$(identity cold)" ]]
[[ $(kernel scale) != "$(kernel cold)" ]]
if cmp -s "$work/cold.bin" "$work/scale.bin"; then echo 'FAIL scale has no effect'; exit 1; fi
echo 'PASS kernel-only recipe change: retained weights, changed logits match private execution'

LAYOUT=fp32 run layout created 1
LAYOUT=fp32 run layout-private private private
LAYOUT=fp32 run layout-warm reused 1
[[ $(identity layout) != "$(identity cold)" ]]
cmp "$work/layout.bin" "$work/layout-private.bin"
cmp "$work/layout.bin" "$work/layout-warm.bin"
echo 'PASS incompatible weight layout: new allocation, exact agreement with private execution'

ptx=$(artifact warm)
cp "$ptx" "$work/original.ptx"
trap 'cp "$work/original.ptx" "$ptx"' EXIT
for revision in {1..10}; do
	printf -v option '1.%02d' "$revision"
	run "option-$revision" reused 1 "$option"
	run "option-private-$revision" private private "$option"
	cmp "$work/option-$revision.bin" "$work/option-private-$revision.bin"
	ptx=$(artifact "option-$revision")
	cp "$ptx" "$work/original.ptx"
	tr -d '\000' < "$work/original.ptx" > "$work/edited.ptx"
	printf '\n// residency revision %s\n\000' "$revision" >> "$work/edited.ptx"
	cp "$work/edited.ptx" "$ptx"
	run "edit-$revision" reused 1 "$option"
	cmp "$work/option-private-$revision.bin" "$work/edit-$revision.bin"
	[[ $(identity "edit-$revision") == "$(identity cold)" ]]
	[[ $(kernel "edit-$revision") != "$(kernel "option-$revision")" ]]
	cp "$work/original.ptx" "$ptx"
done
ptx=$(artifact warm)
cp "$ptx" "$work/original.ptx"
echo 'PASS 10 recipe option and PTX edits: changed digests, same allocation, exact agreement with private execution'

printf 'invalid PTX\000' > "$ptx"
if run invalid reused 1; then echo 'FAIL invalid PTX accepted'; exit 1; fi
cp "$work/original.ptx" "$ptx"
run restored reused 1
cmp "$work/cold.bin" "$work/restored.bin"
echo 'PASS failed module replacement preserves resident weights'

MODEL="$work/model.gguf" EXPECT=reused RESULT="$work/crash.bin" PAUSE="$work/crash.pid" \
	"$runner" run "$work/run.rs" --device "$device" --ctx 16 --ttl 1 > "$work/crash.log" 2>&1 &
client=$!
for attempt in {1..120}; do
	[[ ! -f $work/crash.pid ]] || break
	kill -0 "$client"
	sleep 0.1
done
[[ -f $work/crash.pid ]]
kill -KILL "$(cat "$work/crash.pid")"
wait "$client" || [[ $? == 137 ]]
run recovered reused 1
[[ $(identity recovered) == "$(identity cold)" ]]
cmp "$work/cold.bin" "$work/recovered.bin"
echo 'PASS runner SIGKILL: holder and allocation survive, exact logits after reattachment'

# A request whose context cannot fit must leave the unexpired allocation live.
if MODEL="$work/model.gguf" CAPACITY=1 \
	timeout 120 "$runner" run "$work/run.rs" --device "$device" --ttl 1 > "$work/capacity.log" 2>&1; then
	echo 'FAIL oversized context was admitted'; exit 1
fi
if ! rg -q 'cannot fit|allocation|available|memory' "$work/capacity.log"; then cat "$work/capacity.log"; exit 1; fi
run after-capacity reused 1
[[ $(identity after-capacity) == "$(identity cold)" ]]
echo 'PASS capacity rejection preserves unexpired resident weights'

MODEL="$work/expiry.gguf" EXPECT=created RESULT="$work/active.bin" PAUSE="$work/active.pid" \
	"$runner" run "$work/run.rs" --device "$device" --ctx 16 --ttl 0 > "$work/active.log" 2>&1 &
client=$!
for attempt in {1..120}; do
	[[ ! -f $work/active.pid ]] || break
	kill -0 "$client"
	sleep 0.1
done
[[ -f $work/active.pid ]]
run overlap reused 0 1 expiry
cmp "$work/active.bin" "$work/overlap.bin"
kill -KILL "$(cat "$work/active.pid")"
wait "$client" || [[ $? == 137 ]]
sleep 0.1
run expired created 0 1 expiry
[[ $(identity expired) != "$(identity overlap)" ]]
echo 'PASS active TTL 0 lease and overlapping client: no active unload, idle expiry creates a new allocation'

rm -f "$work/active.pid"
MODEL="$work/expiry.gguf" EXPECT=created RESULT="$work/long-active.bin" PAUSE="$work/active.pid" \
	"$runner" run "$work/run.rs" --device "$device" --ctx 16 --ttl 0 > "$work/long-active.log" 2>&1 &
client=$!
for attempt in {1..600}; do
	[[ ! -f $work/active.pid ]] || break
	kill -0 "$client"
	sleep 0.1
done
[[ -f $work/active.pid ]]
run long-overlap reused 1 1 expiry
sleep 35
sleep 30
kill -KILL "$(cat "$work/active.pid")"
wait "$client" || [[ $? == 137 ]]
run after-long-overlap reused 0 1 expiry
[[ $(identity after-long-overlap) == "$(identity long-overlap)" ]]
echo 'PASS longest overlapping TTL starts after the final active runner exits'

# Change one actual tensor-data byte at the same path, then compare the new
# retained execution with a private execution of those same changed bytes.
length=$(stat -c %s "$work/model.gguf")
old=$(od -An -tu1 -j "$((length - 1))" -N1 "$work/model.gguf")
printf -v byte '\\%03o' "$((old ^ 1))"
printf '%b' "$byte" | dd of="$work/model.gguf" bs=1 seek="$((length - 1))" conv=notrunc status=none
run changed created 1
run changed-private private private
run changed-warm reused 1
[[ $(identity changed) != "$(identity cold)" ]]
cmp "$work/changed.bin" "$work/changed-private.bin"
cmp "$work/changed.bin" "$work/changed-warm.bin"
echo 'PASS changed tensor contents at the same path: new allocation, exact agreement with private execution'

# A holder is shared by same-user runners. Destructive recovery requires the
# operator to reserve the selected GPUs and confirm no other holder clients or
# retained models use them for the duration of this test.
if [[ ${RECIPE_RESIDENCY_EXCLUSIVE_HOLDER:-0} == 1 ]]; then
	for holder in $(awk '/^resident / { print $4 }' "$work/changed.log" | sort -u); do
		[[ $(readlink "/proc/$holder/exe") == "$runner" ]]
		tr '\000' '\n' < "/proc/$holder/cmdline" | rg -qx -- '--weight-holder'
		kill -KILL "$holder"
	done
	sleep 0.1
	run holder-recovered created 1
	[[ $(identity holder-recovered) != "$(identity changed)" ]]
	cmp "$work/changed.bin" "$work/holder-recovered.bin"
	echo 'PASS stale holder recovery: new owner and allocation, exact logits'
else
	echo 'PENDING stale holder recovery: requires RECIPE_RESIDENCY_EXCLUSIVE_HOLDER=1 on exclusively reserved GPUs'
fi

echo 'PASS residency checks'
