#!/usr/bin/env bash
# Runs inside the Linux AMD GPU worker, invoked through managed Run Command
# (RunShellScript, as root). It holds no cloud-management credential. The
# archive URLs are short-lived and read-only.
#
# Run Command returns only the tail of stdout, so command output goes to log
# files and only markers, failures and the suite evidence reach stdout.
#
# Parameters arrive as name=value: Run Command exports named parameters as
# environment variables, and positional name=value arguments are accepted too.
set -euo pipefail

for argument in "$@"; do
	case "$argument" in
		phase=* | candidateSha=* | snapshotSha256=* | runtimeSuiteSha256=* | snapshotUriEncoded=* | runtimeSuiteUriEncoded=* | progressUriEncoded=* | workload=* | trialCursor=* | trialCount=* | trialUriEncoded=*)
			export "${argument%%=*}=${argument#*=}"
			;;
	esac
done
: "${phase:?phase is required}"
: "${candidateSha:?candidateSha is required}"
: "${snapshotSha256:?snapshotSha256 is required}"
: "${runtimeSuiteSha256:?runtimeSuiteSha256 is required}"
: "${snapshotUriEncoded:?snapshotUriEncoded is required}"
: "${runtimeSuiteUriEncoded:?runtimeSuiteUriEncoded is required}"
: "${progressUriEncoded:?progressUriEncoded is required}"
workload="${workload:-suite}"
trialCursor="${trialCursor:-0}"
trialCount="${trialCount:-0}"
trialUriEncoded="${trialUriEncoded:-}"
case "$phase" in preflight | execute) ;; *) echo "GUEST FAILURE: phase must be preflight or execute"; exit 1 ;; esac

export HOME=/root
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export DEBIAN_FRONTEND=noninteractive
export PATH="$CARGO_HOME/bin:/opt/rocm/bin:$PATH"
ROOT=/var/lib/recipe
LOGS="$ROOT/logs"
mkdir -p "$LOGS"
progress_history=""

fail() {
	# stderr: a failure inside a command substitution must still reach the Run Command output.
	echo "GUEST FAILURE: $*" >&2
	exit 1
}

# Runs a command with its output in a log file; on failure the log tail is the evidence.
quiet() {
	local what="$1"
	shift
	local log
	log="$LOGS/$(printf '%s' "$what" | tr -c 'A-Za-z0-9' '-').log"
	if ! "$@" > "$log" 2>&1; then
		tail -n 25 "$log"
		fail "$what failed (log $log)"
	fi
}

decode_uri() {
	local encoded="$1" base64 padding
	[[ "$encoded" =~ ^[A-Za-z0-9_-]+$ ]] || fail "protected URL encoding contains an invalid character"
	base64="$(printf '%s' "$encoded" | tr '_-' '/+')"
	padding=$(( (4 - ${#base64} % 4) % 4 ))
	[ "$padding" -ne 3 ] || fail "protected URL encoding has an invalid length"
	base64+="$(printf '%*s' "$padding" '' | tr ' ' '=')"
	printf '%s' "$base64" | base64 -d
}

report_phase() {
	local uri
	uri="$(decode_uri "$progressUriEncoded")" || return 0
	progress_history+="$candidateSha $phase $1 $(date -u +%Y-%m-%dT%H:%M:%SZ)"$'\n'
	printf '%s' "$progress_history" | curl --silent --show-error --max-time 30 -X PUT -H 'x-ms-blob-type: BlockBlob' --data-binary @- "$uri" > /dev/null 2>&1 || echo "guest progress upload failed at $1"
}

download() {
	local uri="$1" destination="$2" sha256="$3" actual
	curl --silent --show-error --fail --retry 3 --max-time 600 -o "$destination" "$uri" || fail "protected download failed for $destination"
	actual="$(sha256sum "$destination" | cut -d' ' -f1)"
	[ "$actual" = "$sha256" ] || fail "download checksum mismatch for $destination: $actual != $sha256"
}

# The manifest names the ROCm files the build consumes; read them rather than restating them.
manifest_path() {
	sed -n "s/^$1 = { linux = \"\\([^\"]*\\)\" }\$/\\1/p" "$2"
}

confirm_gpu() {
	local attempt
	echo "== guest: GPU and driver =="
	echo "kernel=$(uname -r) os=$(. /etc/os-release && printf '%s' "$PRETTY_NAME")"
	gpu="$(grep -l "^0x1002$" /sys/bus/pci/devices/*/vendor 2> /dev/null | head -n 1)"
	[ -n "$gpu" ] || fail "no AMD PCI function is attached to this worker"
	gpu="$(dirname "$gpu")"
	echo "pci: ${gpu##*/} device=$(cat "$gpu/device") class=$(cat "$gpu/class")"
	echo "kfd=$([ -e /dev/kfd ] && echo present || echo absent) module=$(find "/lib/modules/$(uname -r)" -name 'amdgpu.ko*' 2> /dev/null | head -n 1)"
}

# The Azure image ships no amdgpu module and blocklists it, and the AMD driver
# extension finishes asynchronously, so the guest installs AMD's amdgpu-dkms for
# the running kernel itself (Microsoft's manual V710 procedure) and loads it.
install_driver() {
	local attempt codename
	if [ ! -e /dev/kfd ]; then
		codename="$(. /etc/os-release && printf '%s' "$VERSION_CODENAME")"
		# amdgpu links against DRM helper modules the Azure kernel ships only in modules-extra.
		quiet "kernel headers and modules" apt-get install -y --no-install-recommends "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" dkms gnupg
		install -d -m 0755 /etc/apt/keyrings
		curl --silent --show-error --fail --retry 3 https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/rocm.gpg || fail "the AMD driver repository key download failed"
		printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/%s/ubuntu %s main\n' "${AMDGPU_RELEASE:-31.50}" "$codename" > /etc/apt/sources.list.d/amdgpu.list
		quiet "AMD driver repository refresh" apt-get update
		quiet "amdgpu-dkms build for $(uname -r)" apt-get install -y amdgpu-dkms
		# Explicit modprobe ignores a blocklist; removing it keeps the driver across reboots.
		sed -i '/^blacklist amdgpu/d' /etc/modprobe.d/*.conf
		# The V710 virtual function has stalled its kernel SDMA ring mid-suite ("Fence
		# fallback timer expired on ring sdma1", first dispatch never completes); page
		# table updates by the CPU keep compute dispatch off that ring.
		printf 'options amdgpu vm_update_mode=%s\n' "${AMDGPU_VM_UPDATE_MODE:-3}" > /etc/modprobe.d/recipe-amdgpu.conf
		depmod -a
		if ! modprobe amdgpu > "$LOGS/modprobe.log" 2>&1; then
			tail -n 5 "$LOGS/modprobe.log"
			echo "dmesg: $(dmesg 2> /dev/null | grep -i -E 'amdgpu|unknown symbol' | tail -n 6 | tr '\n' ' ')"
			fail "amdgpu did not load on kernel $(uname -r)"
		fi
	fi
	for attempt in $(seq 1 24); do
		[ -e /dev/kfd ] && break
		if [ "$attempt" -eq 24 ]; then
			echo "dkms: $(dkms status 2> /dev/null | grep -i amdgpu | head -n 2 | tr '\n' ' ')"
			echo "dmesg: $(dmesg 2> /dev/null | grep -i amdgpu | tail -n 3 | tr '\n' ' ')"
			fail "/dev/kfd is absent after loading amdgpu on kernel $(uname -r)"
		fi
		sleep 5
	done
	renders=(/dev/dri/renderD*)
	echo "kfd=present render nodes=${#renders[@]} driver=$(modinfo -F version amdgpu 2> /dev/null) vm_update_mode=$(cat /sys/module/amdgpu/parameters/vm_update_mode 2> /dev/null || echo unknown)"
}

initialize_toolchain() {
	local allow_install="$1" missing="" path llvm_script
	if ! command -v rustc > /dev/null || ! command -v cargo > /dev/null; then
		missing+=" rust"
		if [ "$allow_install" = true ]; then
			download "https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init" "$ROOT/rustup-init" "20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c"
			chmod +x "$ROOT/rustup-init"
			quiet "rustup installation" "$ROOT/rustup-init" -y --profile minimal --default-toolchain stable --no-modify-path
		fi
	fi
	# The CPU backend uses the manifest's /usr/bin/clang and /usr/bin/ld; LLVM 22 matches the Windows guest.
	if ! /usr/bin/clang --version 2> /dev/null | grep 'version 22\.' > /dev/null; then
		missing+=" llvm"
		if [ "$allow_install" = true ]; then
			quiet "base packages" apt-get install -y --no-install-recommends build-essential ca-certificates curl gnupg lsb-release software-properties-common wget
			llvm_script="$ROOT/llvm.sh"
			curl --silent --show-error --fail --retry 3 -o "$llvm_script" https://apt.llvm.org/llvm.sh || fail "the LLVM installer download failed"
			quiet "LLVM 22 installation" bash "$llvm_script" 22
			ln -sf /usr/bin/clang-22 /usr/bin/clang
			ln -sf /usr/bin/llc-22 /usr/bin/llc
		fi
	fi
	if [ ! -x /opt/rocm/lib/llvm/bin/clang ] || [ ! -d /opt/rocm/amdgcn/bitcode ] || ! ldconfig -p | grep 'libhsa-runtime64\.so\.1' > /dev/null; then
		missing+=" rocm"
		if [ "$allow_install" = true ]; then
			# The AMD GPU driver extension installs the kernel driver; the compiler, device
			# libraries and HSA runtime come from AMD's ROCm repository for this release.
			install -d -m 0755 /etc/apt/keyrings
			curl --silent --show-error --fail --retry 3 https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor --yes -o /etc/apt/keyrings/amdrocm.gpg || fail "the ROCm repository key download failed"
			release="$(. /etc/os-release && printf '%s' "${VERSION_ID//./}")"
			cat > /etc/apt/sources.list.d/amdrocm-stable.sources <<-SOURCES
			X-Repo-Id: amdrocm-stable
			Types: deb
			URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu${release}/
			Suites: stable
			Components: main
			Architectures: amd64
			Signed-By: /etc/apt/keyrings/amdrocm.gpg
			Enabled: yes
			SOURCES
			quiet "ROCm repository refresh" apt-get update
			quiet "ROCm installation" apt-get install -y "${ROCM_PACKAGE:-amdrocm10.0-gfx1101}"
			if [ ! -e /opt/rocm ]; then
				path="$(ls -d /opt/rocm-* 2> /dev/null | sort -V | tail -n 1)"
				[ -n "$path" ] || fail "the ROCm installation left no /opt/rocm tree"
				ln -s "$path" /opt/rocm
			fi
			path="$(dirname "$(find /opt/rocm/ -name 'libhsa-runtime64.so.1' -print -quit)")"
			[ "$path" != . ] || fail "the ROCm installation has no libhsa-runtime64.so.1"
			printf '%s\n' "$path" > /etc/ld.so.conf.d/recipe-rocm.conf
			ldconfig
		fi
	fi
	if [ "$allow_install" != true ]; then
		echo "platform ready; install required:${missing:- none}"
		return 0
	fi
	echo "rustc=$(rustc --version) cargo=$(cargo --version)"
	echo "clang=$(/usr/bin/clang --version | head -n 1)"
	echo "rocm clang=$(/opt/rocm/lib/llvm/bin/clang --version 2> /dev/null | head -n 1 || echo absent)"
	echo "hsa runtime=$(ldconfig -p | grep 'libhsa-runtime64\.so\.1' | head -n 1 | sed 's/.*=> //' || echo absent)"
	echo "rocminfo=$(rocminfo 2> /dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1 || echo unavailable)"
}

run_trial() {
	local work="$1" runtime="$2" trial exit_code
	echo "== guest: running the composition harness on amd0, cursor $trialCursor count $trialCount =="
	trial="$work/trial"
	mkdir -p "$trial"
	cp "$runtime/harness.rs" "$work/harness.rs"
	set +e
	RECIPE_DEVICE=amd0 \
	RECIPE_COMPOSITION_CAPABILITY=gfx1101 \
	RECIPE_COMPOSITION_RUNNER="$work/target/release/recipe" \
	RECIPE_COMPOSITION_CURSOR="$trialCursor" \
	RECIPE_COMPOSITION_COUNT="$trialCount" \
	RECIPE_COMPOSITION_REPLAY_SEED=17 \
	RECIPE_COMPOSITION_REPRO="$trial/repro.rs" \
	RECIPE_TRIAL_DIRECTORY="$trial" \
		"$work/target/release/recipe" "$work/harness.rs" > "$trial/harness.out" 2> "$trial/harness.log"
	exit_code=$?
	set -e
	echo "TRIAL EXIT $exit_code compositions=$(grep -c '^composition [0-9]*:' "$trial/harness.log" || true) packets=$(grep -c '^RECIPE FAILURE BEGIN$' "$trial/harness.log" || true)"
	report_phase "trial-ready"
	curl --silent --show-error --fail --max-time 300 -X PUT -H 'x-ms-blob-type: BlockBlob' --data-binary "@$trial/harness.log" "$(decode_uri "$trialUriEncoded")" > /dev/null || fail "the trial log upload failed"
	echo "uploaded the trial log"
}

run_suite() {
	local work="$1" runtime="$2" exit_code route device
	echo "== guest: executing the suite on amd0 =="
	report_phase "suite-start"
	mkdir -p "$work/evidence" "$work/gpu-work"
	# --device amd0 hard-errors when the device is absent or AMD support is not
	# compiled in; there is no silent CPU fallback.
	set +e
	(
		cd "$work"
		RECIPE_SUITE_ROOT="$runtime" \
		RECIPE_SUITE_WORK="$work/gpu-work" \
		RECIPE_EVIDENCE="$work/evidence/suite.json" \
		RECIPE_SUITE_PROGRESS=1 \
		RECIPE_TRACE_PATH="$ROOT/suite-trace-$candidateSha.log" \
			timeout --signal=TERM --kill-after=30s 600s "$work/target/release/recipe" --device amd0 "$runtime/suite.rs"
	) > "$work/run.log" 2>&1
	exit_code=$?
	set -e
	if [ "$exit_code" -ne 0 ]; then
		tail -n 30 "$work/run.log"
		if [ "$exit_code" -eq 124 ]; then
			# A hang says nothing on its own: the trace shows the last dispatch, the kernel log any GPU reset.
			{
				echo "--- last suite trace lines ---"
				tail -n 15 "$ROOT/suite-trace-$candidateSha.log" 2> /dev/null || echo "no suite trace"
				echo "--- last amdgpu kernel messages ---"
				dmesg 2> /dev/null | grep -i -E 'amdgpu|kfd' | tail -n 10 || true
			} >&2
			fail "the runtime suite exceeded 600s after $(grep -c '^check ' "$work/run.log" || true) completed checks"
		fi
		fail "the runtime suite failed with exit code $exit_code"
	fi
	grep -q 'SUITE PASS executed=8' "$work/run.log" || { tail -n 30 "$work/run.log"; fail "the suite did not report eight passing checks"; }
	route="$(sed -n 's/^suite device \([^[:space:]]*\).*/\1/p' "$work/run.log" | head -n 1)"
	[ -n "$route" ] || fail "no suite device: training did not report its device"
	device="${route##*:}"
	case "$device" in amd*) ;; *) fail "expected an amd device, got '$device'; CPU fallback is a failure" ;; esac
	jq -e '.executed == 8 and .failed == 0' "$work/evidence/suite.json" > /dev/null 2>&1 || fail "the suite evidence does not contain eight passing checks"
	report_phase "suite-ready checks=8 exit=$exit_code"
	echo "executed on $route"
	echo "SUITE-EVIDENCE-BEGIN"
	cat "$work/evidence/suite.json"
	echo
	echo "SUITE-EVIDENCE-END"
}

report_phase "gpu-check"
confirm_gpu
report_phase "toolchain-start"
if [ "$phase" = preflight ]; then
	initialize_toolchain false
	report_phase "toolchain-ready"
	echo "PREFLIGHT EXIT 0"
	exit 0
fi
quiet "package index refresh" apt-get update
quiet "archive and JSON tools" apt-get install -y --no-install-recommends ca-certificates curl jq tar gzip
report_phase "driver-start"
install_driver
report_phase "driver-ready"
initialize_toolchain true
report_phase "toolchain-ready"

# One directory per candidate: two commits must never share a mutable tree.
work="$ROOT/$candidateSha"
rm -rf "$work"
mkdir -p "$work"
echo "== guest: verifying the snapshot =="
download "$(decode_uri "$snapshotUriEncoded")" "$ROOT/snapshot.tar.gz" "$snapshotSha256"
tar -xzf "$ROOT/snapshot.tar.gz" -C "$work"
runtime="$ROOT/trusted-runtime"
rm -rf "$runtime"
mkdir -p "$runtime"
download "$(decode_uri "$runtimeSuiteUriEncoded")" "$ROOT/runtime-suite.tar.gz" "$runtimeSuiteSha256"
tar -xzf "$ROOT/runtime-suite.tar.gz" -C "$runtime"
echo "snapshot and trusted runtime verified"
for key in hsa-compiler hsa-device-library; do
	path="$(manifest_path "$key" "$work/Cargo.toml")"
	[ -e "$path" ] && echo "$key=$path" || echo "$key=$path (absent: the build compiles without AMD support)"
done
report_phase "snapshot-ready"

echo "== guest: building with the AMD backend =="
report_phase "build-start"
(cd "$work" && quiet "the native GPU build" cargo build --release --lib --bin recipe)
report_phase "build-ready"

if [ "$workload" = trial ]; then
	report_phase "trial-start"
	run_trial "$work" "$runtime"
else
	run_suite "$work" "$runtime"
fi
echo "GUEST EXIT 0"
