#!/usr/bin/env bash
# Runs the immutable candidate archive on one Camber NVIDIA L4 job.
#
# The control credential stays in this controller process. The worker receives
# only the archive, the worker program, and fixed digest and commit values.
set -euo pipefail

: "${CAMBER_API_KEY:?the Camber control credential is required}"
: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${SNAPSHOT:?SNAPSHOT is required}"
: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"
: "${TRUSTED_RUNTIME:?TRUSTED_RUNTIME is required}"

# A queued job costs nothing but the wait, while a job the controller gives up
# on keeps its queue place and runs the whole worker for nobody; so the queue
# wait is bounded by the room the 60-minute job leaves after the worker's own
# 1500 s, and the execution deadline counts from the first poll that finds the
# job running rather than from submission.
QUEUE_DEADLINE_SECONDS="${QUEUE_DEADLINE_SECONDS:-1500}"
RUN_DEADLINE_SECONDS="${RUN_DEADLINE_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-20}"
WORKER_EXECUTION_TIMEOUT_SECONDS="${WORKER_EXECUTION_TIMEOUT_SECONDS:-1500}"
CANCEL_RETRY_WINDOW_SECONDS="${CANCEL_RETRY_WINDOW_SECONDS:-300}"

case "$WORKER_EXECUTION_TIMEOUT_SECONDS" in
	''|*[!0-9]*) echo "worker execution timeout must be an integer" >&2; exit 1 ;;
esac
case "$CANCEL_RETRY_WINDOW_SECONDS" in
	''|*[!0-9]*) echo "cancel retry window must be an integer" >&2; exit 1 ;;
esac

mkdir -p evidence

camber_bin_dir="${HOME}/.camber/bin"
export PATH="${camber_bin_dir}:${PATH}"
if ! command -v camber >/dev/null 2>&1; then
	curl -fsSL https://cli.cambercloud.com/install-v2.sh | bash
fi
export PATH="${camber_bin_dir}:${PATH}"
command -v camber >/dev/null 2>&1 || { echo "Camber CLI is unavailable" >&2; exit 1; }
camber version

job_help="$(camber job --help)"
case "$job_help" in
	*create*get*logs*) ;;
	*) echo "Camber CLI has no current job create/get/logs surface" >&2; exit 1 ;;
esac
stash_help="$(camber stash cp --help)"
case "$stash_help" in
	*"Copy Stash"*) ;;
	*) echo "Camber CLI has no current Stash copy surface" >&2; exit 1 ;;
esac

echo "== resolving the Camber workspace =="
me_json="$(camber me --output json)"
username="$(printf '%s' "$me_json" | jq -er '.username // .user.username // .user_name')"
case "$username" in
	''|*[!A-Za-z0-9._-]*) echo "Camber returned an invalid workspace name" >&2; exit 1 ;;
esac

echo "== verifying the immutable archive =="
actual_sha256="$(sha256sum "$SNAPSHOT" | awk '{print $1}')"
if [ "$actual_sha256" != "$SNAPSHOT_SHA256" ]; then
	echo "snapshot checksum mismatch before upload: $actual_sha256 != $SNAPSHOT_SHA256" >&2
	exit 1
fi
: "${RECIPE_TRIAL_CURSOR:?RECIPE_TRIAL_CURSOR is required}"
: "${RECIPE_TRIAL_COUNT:?RECIPE_TRIAL_COUNT is required}"
case "$RECIPE_TRIAL_CURSOR$RECIPE_TRIAL_COUNT" in
	''|*[!0-9]*) echo "the trial cursor and count must be integers" >&2; exit 1 ;;
esac
[ -f "$TRUSTED_RUNTIME/harness.rs" ] || { echo "the trial harness is absent" >&2; exit 1; }
tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" harness.rs

request_key="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${CANDIDATE_SHA:0:12}"
stash_root="stash://${username}/recipe-trial/${request_key}"
printf '%s\n' "$stash_root" > camber-stash-root

cat > worker.sh <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

# The controller marks the run directory abandoned when it stops waiting for
# this job (queue deadline, cancellation); a job that starts afterwards has no
# reader, so it returns at once instead of building and running the suite.
if [ -f abandoned ]; then
	echo "the controller abandoned this job before it started: $(cat abandoned)"
	exit 0
fi

: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"
: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${WORKER_EXECUTION_TIMEOUT_SECONDS:?worker execution timeout is required}"
: "${RECIPE_TRIAL_CURSOR:?trial cursor is required}"
: "${RECIPE_TRIAL_COUNT:?trial count is required}"

worker_started="$(date +%s)"
source_root="$(pwd)"
root="${TMPDIR:-/tmp}/recipe-camber-${CANDIDATE_SHA:0:12}-$$"
mkdir -p "$root"
cp "$source_root/recipe-source.tar.gz" "$root/recipe-source.tar.gz"
cp "$source_root/trusted-runtime.tar.gz" "$root/trusted-runtime.tar.gz"
cd "$root"
archive="$root/recipe-source.tar.gz"
archive_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [ "$archive_sha256" != "$SNAPSHOT_SHA256" ]; then
	echo "snapshot checksum mismatch in the Camber worker: $archive_sha256 != $SNAPSHOT_SHA256" >&2
	exit 1
fi

echo "== worker: GPU information =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ! nvidia-smi --query-gpu=name --format=csv,noheader | awk 'BEGIN { IGNORECASE=1 } /L4/ { found=1 } END { exit(found ? 0 : 1) }'; then
	echo "the allocated GPU is not an NVIDIA L4" >&2
	exit 1
fi

echo "== worker: native Ubuntu userspace =="
work="$root/work"
toolchains="$root/toolchains"
rustup_home="$toolchains/rustup"
cargo_home="$toolchains/cargo"
llvm_version="23.1.0"
llvm_archive="LLVM-${llvm_version}-Linux-X64.tar.xz"
llvm_sha256="18da30f77f475688a18f7704d23f9f155ae007ed9922dbed6850a9419d9fec8c"
llvm_root="$toolchains/LLVM-${llvm_version}-Linux-X64"
mkdir -p "$work" "$toolchains"
tar -xzf "$archive" -C "$work"
mkdir -p "$work/.github/runtime"
tar -xzf "$root/trusted-runtime.tar.gz" -C "$work/.github/runtime"

echo "== worker: install user-local Rust =="
export RUSTUP_HOME="$rustup_home"
export CARGO_HOME="$cargo_home"
export PATH="$CARGO_HOME/bin:$PATH"
curl -fsSL https://sh.rustup.rs -o "$toolchains/rustup-init.sh"
sh "$toolchains/rustup-init.sh" -y --profile minimal --default-toolchain stable

echo "== worker: install user-local LLVM =="
curl -fsSL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${llvm_version}/$llvm_archive" -o "$toolchains/$llvm_archive"
printf '%s  %s\n' "$llvm_sha256" "$toolchains/$llvm_archive" | sha256sum -c -
tar -xJf "$toolchains/$llvm_archive" -C "$toolchains"
[ -x "$llvm_root/bin/clang" ] || { echo "user-local clang is absent" >&2; exit 1; }
[ -x "$llvm_root/bin/ld.lld" ] || { echo "user-local ld.lld is absent" >&2; exit 1; }

nvcc_path="$(command -v nvcc || true)"
[ -n "$nvcc_path" ] || { echo "nvcc is absent from the Camber worker" >&2; exit 1; }
nvcc_path="$(readlink -f "$nvcc_path")"
cuda_root="$(cd "$(dirname "$nvcc_path")/.." && pwd -P)"
cuda_device_library="$cuda_root/nvvm/libdevice/libdevice.10.bc"
[ -f "$cuda_device_library" ] || { echo "CUDA libdevice is absent: $cuda_device_library" >&2; exit 1; }
ldconfig -p | grep -q 'libcuda\.so\.1' || { echo "libcuda.so.1 is absent from the Camber worker" >&2; exit 1; }

echo "== worker: patch extracted platform paths =="
candidate_manifest="$work/Cargo.toml"
[ -f "$candidate_manifest" ] || { echo "candidate Cargo.toml is absent" >&2; exit 1; }
sed -i \
	-e "s|cpu-compiler = { linux = \"/usr/bin/clang\"|cpu-compiler = { linux = \"$llvm_root/bin/clang\"|" \
	-e "s|cpu-linker = { linux = \"/usr/bin/ld\"|cpu-linker = { linux = \"$llvm_root/bin/ld.lld\"|" \
	-e 's|cpu-linker-driver = { linux = "ld"|cpu-linker-driver = { linux = "lld"|' \
	-e "s|nvidia-compiler = { linux = \"/usr/bin/clang\"|nvidia-compiler = { linux = \"$llvm_root/bin/clang\"|" \
	-e "s|nvidia-toolkit = { linux = \"/opt/cuda\"|nvidia-toolkit = { linux = \"$cuda_root\"|" \
	"$candidate_manifest"
grep -Fq "cpu-compiler = { linux = \"$llvm_root/bin/clang\"" "$candidate_manifest" || { echo "candidate CPU compiler path was not patched" >&2; exit 1; }
grep -Fq "nvidia-compiler = { linux = \"$llvm_root/bin/clang\"" "$candidate_manifest" || { echo "candidate NVIDIA compiler path was not patched" >&2; exit 1; }
grep -Fq "nvidia-toolkit = { linux = \"$cuda_root\"" "$candidate_manifest" || { echo "candidate CUDA root was not patched" >&2; exit 1; }

cd "$work"
echo "== worker: toolchain =="
rustc --version
cargo --version
"$llvm_root/bin/clang" --version | head -1
echo "CUDA root: $cuda_root"
echo "== worker: build with the NVIDIA backend =="
if ! timeout --signal=TERM --kill-after=30s "${WORKER_EXECUTION_TIMEOUT_SECONDS}s" cargo build --release --lib --bin recipe; then
	echo "the Camber worker build failed or reached its hard timeout" >&2
	exit 1
fi
build_seconds=$(( $(date +%s) - worker_started ))
echo "== worker: build took ${build_seconds}s =="
echo "== worker: run the composition harness on nv0, cursor $RECIPE_TRIAL_CURSOR count $RECIPE_TRIAL_COUNT =="
mkdir -p "$work/evidence" "$work/trial"
cp "$work/.github/runtime/harness.rs" "$work/harness.rs"
trial_started="$(date +%s)"
# The harness writes one composition line per cursor and a failure packet per
# defect on stderr; that stream is the trial's evidence. A nonzero exit is
# recorded, not fatal: the packets it wrote before are still evidence.
timeout --signal=TERM --kill-after=30s "${WORKER_EXECUTION_TIMEOUT_SECONDS}s" env \
	RECIPE_DEVICE=nv0 \
	RECIPE_COMPOSITION_RUNNER="$work/target/release/recipe" \
	RECIPE_COMPOSITION_CURSOR="$RECIPE_TRIAL_CURSOR" \
	RECIPE_COMPOSITION_COUNT="$RECIPE_TRIAL_COUNT" \
	RECIPE_COMPOSITION_REPLAY_SEED=17 \
	RECIPE_COMPOSITION_REPRO="$work/trial/repro.rs" \
	RECIPE_TRIAL_DIRECTORY="$work/trial" \
	"$work/target/release/recipe" harness.rs > "$work/trial/harness.out" 2> "$work/trial/harness.log"
harness_status=$?
trial_seconds=$(( $(date +%s) - trial_started ))
echo "== worker: harness exit $harness_status after ${trial_seconds}s =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mkdir -p "$root/evidence"
# The snapshot is an archive without history, so the harness's base line
# carries no commit; the candidate commit stands in for it.
{
	echo "camber-l4 trial cursor=$RECIPE_TRIAL_CURSOR count=$RECIPE_TRIAL_COUNT seed=17 commit=$CANDIDATE_SHA build_seconds=$build_seconds trial_seconds=$trial_seconds harness_exit=$harness_status"
	awk -v sha="$CANDIDATE_SHA" '{ sub(/^base=commit=[0-9a-f]*/, "base=commit=" sha); print }' "$work/trial/harness.log"
} > "$root/evidence/trial.log"
printf '%s\n' "RECIPE_TRIAL_LOG_BEGIN"
base64 -w0 "$root/evidence/trial.log"
printf '\n%s\n' "RECIPE_TRIAL_LOG_END"
echo "WORKER EXIT 0"
WORKER
chmod +x worker.sh

echo "== uploading the archive and worker to Stash =="
camber stash cp "$SNAPSHOT" "$stash_root/recipe-source.tar.gz"
camber stash cp trusted-runtime.tar.gz "$stash_root/trusted-runtime.tar.gz"
camber stash cp worker.sh "$stash_root/worker.sh"

echo "== jobs of this workflow still queued ahead =="
# Jobs an earlier run abandoned keep their queue place until they start; the
# count says how many of them this job waits behind.
if queued_json="$(camber job list --size 50 --output json 2>/dev/null)"; then
	printf '%s' "$queued_json" | jq -r '[.. | objects | select(has("job_id") and has("mount_dir")) | select((.mount_dir // "") | startswith("recipe-runtime/")) | select(((.job_status // "") | ascii_upcase) as $s | $s == "PENDING" or $s == "QUEUED" or $s == "SUBMITTED" or $s == "RUNNING")] | "\(length) queued or running: \([.[] | "\(.job_id):\(.job_status):\(.mount_dir)"] | join(" "))"' || echo "could not summarize the job list"
else
	echo "could not list jobs"
fi

job_command="SNAPSHOT_SHA256=$SNAPSHOT_SHA256 CANDIDATE_SHA=$CANDIDATE_SHA WORKER_EXECUTION_TIMEOUT_SECONDS=$WORKER_EXECUTION_TIMEOUT_SECONDS RECIPE_TRIAL_CURSOR=$RECIPE_TRIAL_CURSOR RECIPE_TRIAL_COUNT=$RECIPE_TRIAL_COUNT bash worker.sh"
for provider_attempt in 1 2; do
	echo "== creating the Camber L4 job, attempt $provider_attempt of 2 =="
	create_output="$(printf 'y\n' | camber job create \
		--engine base \
		--size xsmall \
		--gpu \
		--num-nodes 1 \
		--path "$stash_root/" \
		--cmd "$job_command" 2>&1)" || {
		printf '%s\n' "$create_output" >&2
		exit 1
	}
	printf '%s\n' "$create_output"
	job_id="$(printf '%s\n' "$create_output" | awk -F: '/Job ID:/ { gsub(/[[:space:]]/, "", $2); print $2; exit }')"
	if [ -z "$job_id" ]; then
		job_id="$(printf '%s\n' "$create_output" | jq -er '.job_id // .id // empty' 2>/dev/null || true)"
	fi
	case "$job_id" in
		''|*[!0-9]*) echo "Camber did not return a numeric job ID" >&2; exit 1 ;;
	esac
	printf '%s\n' "$job_id" > camber-job-id
	echo "submitted Camber job $job_id for $CANDIDATE_SHA"

	echo "== polling the Camber job =="
	started="$(date +%s)"
	running_since=""
	state=""
	while :; do
		now="$(date +%s)"
		elapsed=$((now - started))
		job_json="$(camber job get "$job_id" --output json)"
		state="$(printf '%s' "$job_json" | jq -er '(.job_status // .status // .state // "") | tostring | ascii_upcase' 2>/dev/null || true)"
		echo "  t=${elapsed}s state=${state:-UNKNOWN}"
		case "$state" in
		COMPLETED|SUCCEEDED|SUCCESS|FINISHED|FAILED|ERROR|CANCELLED|CANCELED|TERMINATED)
			break
			;;
		QUEUED|PENDING|SUBMITTED)
			if [ "$elapsed" -ge "$QUEUE_DEADLINE_SECONDS" ]; then
				echo "queue deadline of ${QUEUE_DEADLINE_SECONDS}s exceeded" >&2
				exit 1
			fi
			;;
		*)
			if [ -z "$running_since" ]; then
				running_since="$now"
				echo "  running after ${elapsed}s in the queue"
			fi
			if [ $((now - running_since)) -ge "$RUN_DEADLINE_SECONDS" ]; then
				echo "execution deadline of ${RUN_DEADLINE_SECONDS}s exceeded" >&2
				exit 1
			fi
			;;
		esac
		sleep "$POLL_SECONDS"
	done

	echo "== collecting job logs and worker evidence =="
	logs_returned=false
	if camber job logs "$job_id" > evidence/worker-run.log 2>&1 && [ -s evidence/worker-run.log ]; then
		logs_returned=true
	else
		logs_returned=false
		echo "Camber did not return job logs" >&2
	fi

	case "$state" in
		COMPLETED|SUCCEEDED|SUCCESS|FINISHED)
			break
			;;
		CANCELLED|CANCELED)
			if [ "$logs_returned" = false ] && [ "$provider_attempt" -eq 1 ] \
				&& [ "$elapsed" -le "$CANCEL_RETRY_WINDOW_SECONDS" ]; then
				echo "Camber cancelled the job before returning logs; retrying once" >&2
				continue
			fi
			;;
	esac
	echo "the Camber job reached $state, not a successful terminal state" >&2
	exit 1
done
[ -f evidence/worker-run.log ] || { echo "worker log is absent" >&2; exit 1; }
awk '/^RECIPE_TRIAL_LOG_BEGIN$/{capture=1; next} /^RECIPE_TRIAL_LOG_END$/{capture=0; exit} capture{print}' evidence/worker-run.log | tr -d '\r\n' | base64 -d > "evidence/camber-l4-run-${RECIPE_TRIAL_CURSOR}-$((RECIPE_TRIAL_CURSOR + RECIPE_TRIAL_COUNT - 1)).txt" || {
	echo "the Camber job log did not contain the trial log" >&2
	exit 1
}
grep -q "WORKER EXIT 0" evidence/worker-run.log || { echo "the worker did not report a zero exit status" >&2; exit 1; }
grep -E '^== worker: (build took|harness exit)' evidence/worker-run.log
head -1 "evidence/camber-l4-run-${RECIPE_TRIAL_CURSOR}-$((RECIPE_TRIAL_CURSOR + RECIPE_TRIAL_COUNT - 1)).txt"
grep -c '^RECIPE FAILURE BEGIN$' "evidence/camber-l4-run-${RECIPE_TRIAL_CURSOR}-$((RECIPE_TRIAL_CURSOR + RECIPE_TRIAL_COUNT - 1)).txt" || true
cat > evidence/cell.json <<JSON
{
  "cell": "recipe/camber-trial",
  "commit": "$CANDIDATE_SHA",
  "run_id": "${GITHUB_RUN_ID:-unknown}",
  "run_attempt": "${GITHUB_RUN_ATTEMPT:-unknown}",
  "provider": "camber",
  "camber_job_id": "$job_id",
  "cursor": "$RECIPE_TRIAL_CURSOR",
  "count": "$RECIPE_TRIAL_COUNT",
  "snapshot_sha256": "$SNAPSHOT_SHA256"
}
JSON
cat evidence/cell.json
echo "recipe/linux-gpu completed on $route"
