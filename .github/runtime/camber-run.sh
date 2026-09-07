#!/usr/bin/env bash
# Submits the candidate snapshot to a Camber NVIDIA L4 job, waits for a
# terminal state, retrieves the worker's evidence and process exit status, and
# fails on any error, timeout, missing GPU or missing evidence.
#
# The control credential (CAMBER_API_KEY) stays in this controller job. It is
# never placed in the environment the suite executes in on the worker.
#
# A chatbot or dashboard saying the run passed is not evidence. Only the
# worker's own exit status and the suite's evidence document decide this cell.
set -euo pipefail

: "${CAMBER_API_KEY:?the Camber control credential is required}"
: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${SNAPSHOT:?SNAPSHOT is required}"
: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"

# Finite deadlines. A job that has not reached a terminal state by the queue or
# execution deadline is terminated by the caller's always() step and this cell
# fails.
QUEUE_DEADLINE_SECONDS="${QUEUE_DEADLINE_SECONDS:-900}"
RUN_DEADLINE_SECONDS="${RUN_DEADLINE_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-20}"

mkdir -p evidence

echo "== validating the installed Camber interface =="
# The interface is validated before any command is encoded against it: the
# published "camber job" examples do not necessarily match the installed
# version, and guessing here would produce a false failure.
python3 -m pip install --quiet --disable-pip-version-check camber
camber --version
camber --help > camber-help.txt
echo "installed Camber verbs:"
grep -E '^[[:space:]]+[a-z-]+[[:space:]]' camber-help.txt | head -30
if ! grep -qE '^[[:space:]]+(job|cloud)\b' camber-help.txt; then
	echo "the installed Camber CLI exposes no recognised job interface" >&2
	cat camber-help.txt >&2
	exit 1
fi
# Record the exact submit surface this version offers, so a future change is
# visible in the evidence rather than silently mis-encoded.
camber job submit --help > camber-submit-help.txt || true
head -40 camber-submit-help.txt || true

echo "== staging the immutable snapshot =="
sha="$(sha256sum "$SNAPSHOT" | cut -d' ' -f1)"
if [ "$sha" != "$SNAPSHOT_SHA256" ]; then
	echo "snapshot checksum mismatch before submission: $sha != $SNAPSHOT_SHA256" >&2
	exit 1
fi
cp "$SNAPSHOT" recipe-source.tar.gz

# The worker script. It establishes Arch userspace, verifies the GPU, builds
# Recipe with its NVIDIA backend and runs the shared suite on nv0. It never
# falls back to the CPU: `--device nv0` hard-errors when the device is absent,
# because a build carrying the nvidia cfg does not add a CPU device.
cat > worker.sh <<'WORKER'
set -euo pipefail
echo "== worker: GPU prerequisites =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi 'L4'; then
	echo "the allocated GPU is not an L4" >&2
	exit 1
fi

echo "== worker: Arch userspace =="
# Arch userspace through a real rootfs, not a substituted Ubuntu.
arch_root=/tmp/archroot
mkdir -p "$arch_root"
# The newest dated bootstrap release, discovered rather than hardcoded.
release="$(curl -fsSL https://geo.mirror.pkgbuild.com/iso/ | grep -oE '[0-9]{4}\.[0-9]{2}\.[0-9]{2}/' | sort -r | head -1 | tr -d '/')"
[ -n "$release" ] || { echo "no Arch bootstrap release found" >&2; exit 1; }
echo "using Arch bootstrap release $release"
curl -fsSL "https://geo.mirror.pkgbuild.com/iso/$release/archlinux-bootstrap-x86_64.tar.zst" -o /tmp/arch.tar.zst
tar -I zstd -xf /tmp/arch.tar.zst -C "$arch_root" --strip-components=1
cp /etc/resolv.conf "$arch_root/etc/resolv.conf"
for mount in proc sys dev; do
	mount --rbind "/$mount" "$arch_root/$mount"
done
# The CUDA driver belongs to the host kernel; the chroot gets a copy of the
# user-space driver library and the device nodes above.
for library in $(ldconfig -p | awk '/libcuda\.so/ {print $NF}'); do
	install -D "$library" "$arch_root$library"
done

mkdir -p "$arch_root/work"
tar -xzf /work/recipe-source.tar.gz -C "$arch_root/work"

cat > "$arch_root/root/run.sh" <<'INNER'
set -euo pipefail
pacman-key --init
pacman-key --populate archlinux
pacman -Syu --noconfirm --needed rust clang llvm lld cuda git tar
ldconfig
cd /work
echo "== worker: toolchain =="
rustc --version
cargo --version
clang --version | head -1
echo "== worker: build with the NVIDIA backend =="
cargo build --release --lib --bin recipe
echo "== worker: execute the suite on nv0 =="
mkdir -p /work/evidence /work/gpu-work
RECIPE_SUITE_ROOT=/work/.github/runtime \
RECIPE_SUITE_WORK=/work/gpu-work \
RECIPE_EVIDENCE=/work/evidence/suite.json \
  ./target/release/recipe --device nv0 .github/runtime/suite.rs | tee /work/run.log
INNER

chroot "$arch_root" /bin/bash /root/run.sh
cp -r "$arch_root/work/evidence" /work/evidence
cp "$arch_root/work/run.log" /work/run.log

echo "== worker: assert GPU execution =="
route="$(grep -m1 '^selected route ' /work/run.log | awk '{print $3}')"
device="${route##*:}"
case "$device" in
	nv*) echo "executed on $route" ;;
	*) echo "expected an nv device, got '$device'; CPU fallback is a failure" >&2; exit 1 ;;
esac
if ! grep -q "SUITE PASS" /work/run.log; then
	echo "the suite did not report SUITE PASS" >&2
	exit 1
fi
echo "WORKER EXIT 0"
WORKER

echo "== submitting =="
job_id="$(camber job submit \
	--engine gpu \
	--gpu-type l4 \
	--gpu-count 1 \
	--upload "recipe-source.tar.gz" \
	--upload "worker.sh" \
	--command "bash worker.sh" \
	--format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
echo "$job_id" > camber-job-id
echo "submitted Camber job $job_id for $CANDIDATE_SHA"

echo "== polling =="
started="$(date +%s)"
state=""
while true; do
	elapsed=$(( $(date +%s) - started ))
	state="$(camber job get "$job_id" --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
	echo "  t=${elapsed}s state=$state"
	case "$state" in
		COMPLETED|FAILED|CANCELLED|ERROR)
			break
			;;
		QUEUED|PENDING)
			if [ "$elapsed" -ge "$QUEUE_DEADLINE_SECONDS" ]; then
				echo "queue deadline of ${QUEUE_DEADLINE_SECONDS}s exceeded" >&2
				exit 1
			fi
			;;
	esac
	if [ "$elapsed" -ge "$RUN_DEADLINE_SECONDS" ]; then
		echo "execution deadline of ${RUN_DEADLINE_SECONDS}s exceeded" >&2
		exit 1
	fi
	sleep "$POLL_SECONDS"
done

echo "== retrieving logs and evidence =="
camber job logs "$job_id" > evidence/worker.log || true
tail -50 evidence/worker.log || true
camber job download "$job_id" --path evidence --destination evidence/ || true

if [ "$state" != "COMPLETED" ]; then
	echo "the Camber job reached $state, not COMPLETED" >&2
	exit 1
fi
if ! grep -q "WORKER EXIT 0" evidence/worker.log; then
	echo "the worker did not report a zero exit status" >&2
	exit 1
fi
if [ ! -f evidence/suite.json ]; then
	echo "the worker returned no suite evidence" >&2
	exit 1
fi

gpu_name="$(grep -m1 -oiE 'NVIDIA L4|L4' evidence/worker.log | head -1)"
route="$(grep -m1 '^selected route ' evidence/worker.log | awk '{print $3}')"
device="${route##*:}"
case "$device" in
	nv*) ;;
	*) echo "the retrieved evidence does not show an nv device" >&2; exit 1 ;;
esac

cat > evidence/cell.json <<JSON
{
  "cell": "recipe/linux-gpu",
  "commit": "$CANDIDATE_SHA",
  "run_id": "${GITHUB_RUN_ID:-unknown}",
  "run_attempt": "${GITHUB_RUN_ATTEMPT:-unknown}",
  "provider": "camber",
  "camber_job_id": "$job_id",
  "os": "archlinux rootfs on the Camber L4 worker",
  "arch": "x86_64",
  "backend": "nvidia",
  "device": "$device",
  "gpu_model": "$gpu_name",
  "gpu_execution": true,
  "snapshot_sha256": "$SNAPSHOT_SHA256"
}
JSON
cat evidence/cell.json
echo "recipe/linux-gpu completed on $route"
