#!/usr/bin/env bash
# Runs inside the Arch Linux NVIDIA EC2 worker as root, over the controller's
# per-run SSH session. It holds no cloud-management credential: the controller
# copies the snapshot and trusted runtime in before each phase.
#
# driver:  refreshes the system and installs the open NVIDIA kernel module,
#          CUDA and the toolchain; the controller reboots the worker afterwards
#          so the module matches the running kernel.
# execute: confirms the GPU, builds Recipe with the NVIDIA backend and runs the
#          suite on nv0.
set -euo pipefail

for argument in "$@"; do
	case "$argument" in
		phase=* | candidateSha=* | snapshotSha256=* | runtimeSuiteSha256=*)
			export "${argument%%=*}=${argument#*=}"
			;;
	esac
done
: "${phase:?phase is required}"
case "$phase" in driver | execute) ;; *) echo "GUEST FAILURE: phase must be driver or execute"; exit 1 ;; esac

ROOT=/var/lib/recipe
INCOMING="$ROOT/incoming"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"
export HOME=/root
export PATH="/opt/cuda/bin:$PATH"

fail() {
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

install_driver() {
	echo "== guest: refreshing the keyring and the system =="
	# First boot initializes the keyring in the background; wait for it.
	local waited=0
	while systemctl is-active --quiet pacman-init.service 2>/dev/null && [ "$waited" -lt 300 ]; do
		sleep 5
		waited=$((waited + 5))
	done
	if [ ! -s /etc/pacman.d/gnupg/pubring.gpg ] && [ ! -s /etc/pacman.d/gnupg/pubring.kbx ]; then
		quiet "keyring initialization" pacman-key --init
		quiet "keyring population" pacman-key --populate archlinux
	fi
	# The module is built for whichever kernel package the image boots. Ask before
	# the upgrade, which removes the running kernel's module directory.
	local kernel
	kernel="$(pacman -Qqo "/usr/lib/modules/$(uname -r)/vmlinuz" 2>/dev/null || true)"
	[ -n "$kernel" ] || kernel="$(pacman -Qq linux linux-lts 2>/dev/null | head -n 1 || true)"
	[ -n "$kernel" ] || fail "cannot tell which kernel package the image boots"
	echo "kernel package=$kernel running=$(uname -r)"
	quiet "keyring refresh" pacman -Sy --noconfirm --needed archlinux-keyring
	quiet "system upgrade" pacman -Su --noconfirm
	echo "== guest: installing the open NVIDIA module, CUDA and the toolchain =="
	quiet "NVIDIA, CUDA and toolchain packages" pacman -S --noconfirm --needed \
		"$kernel-headers" nvidia-open-dkms nvidia-utils cuda \
		base-devel clang llvm lld rust jq tar gzip
	grep -q '^nvidia' <(dkms status 2>/dev/null) || fail "dkms built no NVIDIA module"
	dkms status | grep '^nvidia'
	[ -f /opt/cuda/nvvm/libdevice/libdevice.10.bc ] || fail "CUDA libdevice is absent from /opt/cuda"
	[ -x /opt/cuda/bin/ptxas ] || fail "ptxas is absent from /opt/cuda"
	echo "DRIVER EXIT 0"
}

confirm_gpu() {
	echo "== guest: GPU and driver =="
	lspci -nn 2>/dev/null | grep -i 'nvidia' || true
	command -v nvidia-smi > /dev/null || fail "nvidia-smi is absent"
	local inventory
	inventory="$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1)" || {
		echo "$inventory"
		lsmod | grep -i nvidia || true
		fail "nvidia-smi cannot reach the GPU: the open module did not load"
	}
	echo "$inventory"
	ldconfig -p | grep -q 'libcuda\.so\.1' || fail "libcuda.so.1 is absent"
	echo "== guest: toolchain =="
	rustc --version
	cargo --version
	clang --version | head -n 1
	/opt/cuda/bin/ptxas --version | tail -n 1
}

verify() {
	local file="$1"
	local expected="$2"
	local actual
	actual="$(sha256sum "$file" | cut -d' ' -f1)"
	[ "$actual" = "$expected" ] || fail "$(basename "$file") checksum mismatch: $actual != $expected"
}

run_suite() {
	: "${candidateSha:?candidateSha is required}"
	: "${snapshotSha256:?snapshotSha256 is required}"
	: "${runtimeSuiteSha256:?runtimeSuiteSha256 is required}"
	# One directory per candidate: two commits must never share a mutable tree.
	local work="$ROOT/$candidateSha"
	local runtime="$ROOT/trusted-runtime"
	rm -rf "$work" "$runtime"
	mkdir -p "$work" "$runtime" "$work/evidence" "$work/gpu-work"
	echo "== guest: verifying the snapshot =="
	verify "$INCOMING/snapshot.tar.gz" "$snapshotSha256"
	verify "$INCOMING/trusted-runtime.tar.gz" "$runtimeSuiteSha256"
	tar -xzf "$INCOMING/snapshot.tar.gz" -C "$work"
	tar -xzf "$INCOMING/trusted-runtime.tar.gz" -C "$runtime"
	echo "snapshot and trusted runtime verified"

	echo "== guest: building with the NVIDIA backend =="
	(cd "$work" && quiet "the native GPU build" cargo build --release --lib --bin recipe)

	echo "== guest: executing the suite on nv0 =="
	# --device nv0 hard-errors when the device is absent or NVIDIA support is not
	# compiled in; there is no silent CPU fallback.
	set +e
	(
		cd "$work"
		RECIPE_SUITE_ROOT="$runtime" \
		RECIPE_SUITE_WORK="$work/gpu-work" \
		RECIPE_EVIDENCE="$work/evidence/suite.json" \
		RECIPE_SUITE_PROGRESS=1 \
			timeout --signal=TERM --kill-after=30s 600s "$work/target/release/recipe" --device nv0 "$runtime/suite.rs"
	) > "$work/run.log" 2>&1
	local exit_code=$?
	set -e
	cp "$work/run.log" "$ROOT/run.log"
	if [ "$exit_code" -ne 0 ]; then
		tail -n 30 "$work/run.log"
		[ "$exit_code" -ne 124 ] || fail "the runtime suite exceeded 600s after $(grep -c '^check ' "$work/run.log" || true) completed checks"
		fail "the runtime suite failed with exit code $exit_code"
	fi
	grep -q 'SUITE PASS executed=8' "$work/run.log" || { tail -n 30 "$work/run.log"; fail "the suite did not report eight passing checks"; }
	local route device
	route="$(sed -n 's/^suite device \([^[:space:]]*\).*/\1/p' "$work/run.log" | head -n 1)"
	[ -n "$route" ] || fail "no suite device: training did not report its device"
	device="${route##*:}"
	case "$device" in nv*) ;; *) fail "expected an nv device, got '$device'; CPU fallback is a failure" ;; esac
	jq -e '.executed == 8 and .failed == 0' "$work/evidence/suite.json" > /dev/null 2>&1 || fail "the suite evidence does not contain eight passing checks"
	cp "$work/evidence/suite.json" "$ROOT/suite.json"
	chmod 0644 "$ROOT/suite.json" "$ROOT/run.log"
	echo "executed on $route"
	echo "GUEST EXIT 0"
}

if [ "$phase" = driver ]; then
	install_driver
else
	confirm_gpu
	run_suite
fi
