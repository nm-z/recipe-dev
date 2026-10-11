#!/bin/bash
# Exercise the public schema CLI and loader. The loopback server implements the
# documented Jev choice protocol; it is a protocol fixture, not a model.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
bin=${RECIPE_BIN:-$root/target/release/recipe}
work=${RECIPE_SCHEMA_WORK:-/home/nate/codex/recipe-schema-proof}
mkdir -p "$work"
rm -f "$work"/request-*.json
server_pid=
trap 'if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi' EXIT

cat > "$work/server.rs" <<'RUST'
use std::{fs, io::{Read, Write}, net::TcpListener, process::{Command, Stdio}};
fn main() {
	let work = std::env::args().nth(1).unwrap();
	// Synthetic PCM files cover the audio loader without an external dataset.
	for index in 0..4 {
		let folder = format!("{work}/audio/{}", if index < 2 { "a" } else { "b" });
		fs::create_dir_all(&folder).unwrap();
		let samples = (0..16).flat_map(|frame| ((index * 16 + frame) as i16 * 128).to_le_bytes()).collect::<Vec<_>>();
		let mut wav = b"RIFF".to_vec();
		wav.extend((36 + samples.len() as u32).to_le_bytes());
		wav.extend(b"WAVEfmt ");
		wav.extend(16u32.to_le_bytes());
		wav.extend(1u16.to_le_bytes());
		wav.extend(1u16.to_le_bytes());
		wav.extend(8000u32.to_le_bytes());
		wav.extend(16000u32.to_le_bytes());
		wav.extend(2u16.to_le_bytes());
		wav.extend(16u16.to_le_bytes());
		wav.extend(b"data");
		wav.extend((samples.len() as u32).to_le_bytes());
		wav.extend(samples);
		fs::write(format!("{folder}/sample-{index}.wav"), wav).unwrap();
	}
	let listener = TcpListener::bind("127.0.0.1:0").unwrap();
	fs::write(format!("{work}/port"), listener.local_addr().unwrap().port().to_string()).unwrap();
	let mut sequence = 0;
	for stream in listener.incoming() {
		let mut stream = stream.unwrap();
		stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
		let mut bytes = Vec::new();
		let mut chunk = [0u8; 4096];
		let (start, size) = loop {
			let count = stream.read(&mut chunk).unwrap();
			assert!(count > 0);
			bytes.extend_from_slice(&chunk[..count]);
			if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
				let header = String::from_utf8(bytes[..end].to_vec()).unwrap();
				assert!(header.starts_with("POST /v1/systemone "));
				let size: usize = header.lines().find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(|value| value.trim().parse().unwrap())).unwrap();
				break (end + 4, size);
			}
		};
		while bytes.len() < start + size {
			let count = stream.read(&mut chunk).unwrap();
			assert!(count > 0);
			bytes.extend_from_slice(&chunk[..count]);
		}
		let body = &bytes[start..start + size];
		fs::write(format!("{work}/request-{sequence}.json"), body).unwrap();
		sequence += 1;
		let mode = fs::read_to_string(format!("{work}/mode")).unwrap_or_default();
		let query = r#"
			if (.questions | length) != 1 or ([.state.metadata | .. | objects | has("snippets") or has("questions")] | any) then error("metadata or question leak") else . end |
			(.questions | keys[0]) as $id | .questions[$id] as $q |
			(if $id == "parse" then .state.metadata.proposal.parse
			elif $id == "samples" then (if .state.metadata.proposal.parse == "image_folder" then .state.metadata.images.total elif .state.metadata.proposal.parse == "audio_folder" then .state.metadata.audio.total else .state.metadata.proposal.samples end | tostring)
			elif $id == "sqlite_table" then .state.metadata.databases.entries[0].tables.entries[0].name
			elif ($id | startswith("table")) then ($id | capture("table(?<t>[0-9]+).feature.(?<c>[0-9]+)")) as $n | .state.metadata.databases.entries[0].tables.entries[$n.t|tonumber].columns[$n.c|tonumber].dtype
			elif $id == "feature.sample" then "audio"
			else .state.metadata.tables.entries[0].columns[($id | split(".")[1] | tonumber)].dtype end) as $suggested |
			(if $mode == "correct" and $id == "feature.0" then "float" elif $mode == "bad-feature" and $id == "feature.0" then "integer" else $suggested end) as $choice |
			{answers:{($id):{type:"choice",choice:$choice,confidence:0.2,probabilities:($q.criteria | with_entries(.value = (if .key == $choice then 1 else 0 end)))}}} |
			if $mode == "offered" then .answers[$id].choice = "unoffered"
			elif $mode == "missing" then .answers = {}
			elif $mode == "extra" then .answers.unasked = .answers[$id]
			elif $mode == "probability" then .answers[$id].probabilities[$choice] = 2
			elif $mode == "distribution" then .answers[$id].probabilities[$choice] = 0.5
			elif $mode == "contradiction" then (.answers[$id].probabilities | keys | map(select(. != $choice))[0]) as $other |
				.answers[$id].probabilities |= with_entries(.value = (if .key == $choice then 0.1 elif .key == $other then 0.9 else 0 end))
			elif $mode == "tie" then (.answers[$id].probabilities | length) as $count | .answers[$id].probabilities |= with_entries(.value = (1 / $count))
			elif $mode == "type" then .answers[$id].type = "noul"
			else . end
		"#;
		let mut child = Command::new("jq").args(["-ce", "--arg", "mode", mode.trim(), query]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
		child.stdin.take().unwrap().write_all(body).unwrap();
		let output = child.wait_with_output().unwrap();
		assert!(output.status.success());
		let status = if mode.trim() == "http" { "503 Service Unavailable" } else { "200 OK" };
		write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", output.stdout.len()).unwrap();
		stream.write_all(&output.stdout).unwrap();
	}
}
RUST
rustc --edition=2024 "$work/server.rs" -o "$work/server"
rm -f "$work/port"
"$work/server" "$work" > "$work/server.log" 2>&1 &
server_pid=$!
for ((attempt=0; attempt<100; attempt++)); do
	[[ -f "$work/port" ]] && break
	kill -0 "$server_pid"
	sleep 0.05
done
endpoint="http://127.0.0.1:$(cat "$work/port")/v1/systemone"
printf 'ok' > "$work/mode"

probe() { "$bin" schema probe "$1" > "$work/proposal.json"; }
check() { "$bin" schema check "$work/proposal.json" "$work/answers.json" "$work/schema.json" --jev "$endpoint" > "$work/decisions.txt"; }
pass() { printf 'PASS %s\n' "$1"; }
reject() {
	rm -f "$work/answers.json" "$work/schema.json"
	if check > "$work/rejected.log" 2>&1; then echo "unexpected acceptance: $1" >&2; exit 1; fi
	[[ ! -e "$work/answers.json" && ! -e "$work/schema.json" ]]
	pass "$1"
}

probe "$root/.github/runtime/data/linear.csv"
cp "$work/proposal.json" "$work/first.json"
probe "$root/.github/runtime/data/linear.csv"
cmp "$work/first.json" "$work/proposal.json"
jq -e '.tables.entries[0] | .rows == 128 and (.columns | all(.snippets | map(.row) == [0,64,127]))' "$work/proposal.json" >/dev/null
check
jq -s -e 'map(.state.answers | length) == [0,1,2,3]' "$work"/request-{0..3}.json >/dev/null
awk '$3 != 1 {exit 1}' "$work/decisions.txt"
"$bin" schema finalize "$work/proposal.json" "$work/answers.json" "$work/repeated.json" > "$work/repeated.txt"
cmp "$work/schema.json" "$work/repeated.json"
cmp "$work/decisions.txt" "$work/repeated.txt"
[[ $(wc -l < "$work/decisions.txt") == 4 ]]
pass 'deterministic probe, round trip, and finalize'

# Aliases must fail before a request or a write, preserving all existing files.
sha256sum "$work/proposal.json" "$work/answers.json" "$work/schema.json" > "$work/before-aliases.sha256"
before_requests=$(printf '%s\n' "$work"/request-*.json | wc -l)
reject_paths() {
	if "$bin" schema check "$work/proposal.json" "$1" "$2" --jev "$endpoint" > "$work/alias.log" 2>&1; then echo 'accepted schema file aliases' >&2; exit 1; fi
	sha256sum -c "$work/before-aliases.sha256" > "$work/alias-hashes.log"
	[[ $(printf '%s\n' "$work"/request-*.json | wc -l) == "$before_requests" ]]
}
reject_paths "$work/answers.json" "$work/./answers.json"
reject_paths "$work/answers.json" "$work/./proposal.json"
reject_paths "$work/./proposal.json" "$work/schema.json"
reject_paths "$work/fresh.json" "$work/./fresh.json"
reject_paths "$work/fresh.json" "$work/FRESH.json"
reject_paths "$work/σ.json" "$work/ς.json"
mkdir -p "$work/alias-directory"
reject_paths "$work/answers.json" "$work/alias-directory/../answers.json"
ln -sf "$work/answers.json" "$work/answer-link.json"
reject_paths "$work/answer-link.json" "$work/answers.json"
ln -sf "$work/proposal.json" "$work/proposal-link.json"
reject_paths "$work/answers.json" "$work/proposal-link.json"
rm -f "$work/proposal-hardlink.json"
ln "$work/proposal.json" "$work/proposal-hardlink.json"
reject_paths "$work/answers.json" "$work/proposal-hardlink.json"
ln -sf "$work/fresh.json" "$work/dangling-link.json"
reject_paths "$work/fresh.json" "$work/dangling-link.json"
[[ ! -e "$work/fresh.json" && ! -e "$work/FRESH.json" && ! -e "$work/σ.json" && ! -e "$work/ς.json" ]]
pass 'reject path, symlink, and hard-link aliases before requests or writes'

printf 'tie' > "$work/mode"
check
jq -e '.answers | all(.probability > 0 and .probability <= 1)' "$work/answers.json" >/dev/null
pass 'accept a highest-probability tie'

printf 'correct' > "$work/mode"
printf '1,2\n3,4\n5,6\n' > "$work/headerless.csv"
probe "$work/headerless.csv"
jq -e '.tables.entries[0].columns[0] | .dtype == "integer" and (.snippets | map(.text) == ["1","3","5"])' "$work/proposal.json" >/dev/null
pass 'headerless snippet order'
check
jq -e '.columns[0].dtype == "float"' "$work/schema.json" >/dev/null
pass 'supported correction'
printf 'ok' > "$work/mode"
for rows in 1 2; do
	head -n "$rows" "$work/headerless.csv" > "$work/short.csv"
	probe "$work/short.csv"
	jq -e --argjson rows "$rows" '.tables.entries[0] | .rows == $rows and (.columns | all(.snippets | map(.row) == [range(0; $rows)]))' "$work/proposal.json" >/dev/null
done
pass 'unique snippets for one- and two-row tables'
probe "$root/data/text/single_csv.csv"
jq -e '.tables.entries[0].columns[0].snippets | length == 3 and all(.text | utf8bytelength <= 64)' "$work/proposal.json" >/dev/null
check
if rg -q 'Sample 00000|engine works under load' "$work"/request-*.json; then echo 'cell value sent to Jev' >&2; exit 1; fi
pass 'bounded snippets excluded from Jev requests'
for mode in offered missing extra probability distribution contradiction type http bad-feature; do
	printf '%s' "$mode" > "$work/mode"
	reject "reject $mode without output files"
done
printf 'ok' > "$work/mode"

# Synthetic edge fixture for UTF-8 truncation, separate from the repository datasets.
awk 'BEGIN {print "input,target"; for(i=0;i<100;i++) printf "é"; print ",1"}' > "$work/unicode.csv"
probe "$work/unicode.csv"
jq -e '.tables.entries[0].columns[0].snippets[0] | (.text | utf8bytelength) == 64 and .bytes == 200' "$work/proposal.json" >/dev/null
: > "$work/empty.csv"
probe "$work/empty.csv"
jq -e '.tables.entries[0] | .rows == 0 and .columns == []' "$work/proposal.json" >/dev/null
pass 'UTF-8 limit and empty table'

probe "$root/data/numeric/samples_sqlite.sqlite"
check
jq -e '.parse == "sqlite_table" and (.columns | length > 0)' "$work/schema.json" >/dev/null
jq -e '.databases.entries[0].tables.entries | all(.rows as $n | .columns | all(.snippets | map(.row) == [0,($n/2|floor),($n-1)]))' "$work/proposal.json" >/dev/null
if rg -q '^feature\.' "$work/decisions.txt"; then echo 'inactive question was answered' >&2; exit 1; fi
pass 'SQLite snippets and active branches'

if "$bin" schema check "$work/proposal.json" "$work/no-answers.json" "$work/no-schema.json" --jev https://opencode.ai/zen/v1/systemone > "$work/endpoint.log" 2>&1; then exit 1; fi
[[ ! -e "$work/no-answers.json" && ! -e "$work/no-schema.json" ]]
pass 'reject nonlocal endpoint'

cat > "$work/load.rs" <<'RUST'
use recipe::*;
fn main() {
	let data = recipe.data(std::env::var("SCHEMA_SOURCE").unwrap()).schema(std::env::var("SCHEMA_FILE").unwrap()).target(std::env::var("SCHEMA_TARGET").unwrap_or("target".to_owned()));
	let model = recipe.model().layer(1).fp(32).loss(mse);
	let report = recipe.train().seed(17).lr(0.000001).epochs(1).run(&model, &data);
	assert!(report.final_loss().is_finite());
	assert_eq!(report.rows, std::env::var("SCHEMA_ROWS").unwrap().parse::<usize>().unwrap());
	let device = report.memory.first().and_then(|line| line.split_whitespace().next()).expect("loader reported no device");
	assert!(device.ends_with(&format!(":{}", std::env::var("RECIPE_DEVICE").unwrap())));
	println!("schema loader device={device} rows={} loss={}", report.rows, report.final_loss());
}
RUST
SCHEMA_SOURCE="$root/data/numeric/samples_sqlite.sqlite" SCHEMA_FILE="$work/schema.json" SCHEMA_ROWS=$(jq .samples "$work/schema.json") "$bin" "$work/load.rs" --device "${RECIPE_SCHEMA_DEVICE:-cpu}" > "$work/load.log" 2>&1
pass 'confirmed SQLite schema loads and trains'
probe "$root/.github/runtime/data/linear.csv"
check
SCHEMA_SOURCE="$root/.github/runtime/data/linear.csv" SCHEMA_FILE="$work/schema.json" SCHEMA_ROWS=$(jq .samples "$work/schema.json") "$bin" "$work/load.rs" --device "${RECIPE_SCHEMA_DEVICE:-cpu}" >> "$work/load.log" 2>&1
pass 'confirmed CSV schema loads and trains'
for source in "$root/data/image/class_subfolders" "$work/audio"; do
	probe "$source"
	check
	SCHEMA_SOURCE="$source" SCHEMA_FILE="$work/schema.json" SCHEMA_ROWS=$(jq .samples "$work/schema.json") SCHEMA_TARGET=class "$bin" "$work/load.rs" --device "${RECIPE_SCHEMA_DEVICE:-cpu}" >> "$work/load.log" 2>&1
	pass "confirmed $(jq -r .parse "$work/schema.json") schema loads and trains"
done
printf 'SCHEMA PASS\n'
