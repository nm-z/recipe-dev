#!/usr/bin/env bash
# Provisions an isolated native Windows NVIDIA worker, transfers the exact
# candidate snapshot, executes the Recipe GPU suite through managed Run
# Command, and reads the guest process's actual result.
#
# A successful deployment is not a successful run. The only thing that decides
# this cell is the guest command's exit code and the evidence it returns.
#
# No cloud-management credential is ever placed in the guest environment, and
# no inbound RDP or SSH is opened: Run Command reaches the guest through the
# Azure control plane.
set -euo pipefail

: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${SNAPSHOT:?SNAPSHOT is required}"
: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${RUN_ATTEMPT:?RUN_ATTEMPT is required}"
: "${TRUSTED_RUNTIME:?TRUSTED_RUNTIME is required}"
if [[ ! "$RUN_ID" =~ ^[0-9]+$ ]] || [[ ! "$RUN_ATTEMPT" =~ ^[0-9]+$ ]]; then
	echo "RUN_ID and RUN_ATTEMPT must be decimal workflow identifiers" >&2
	exit 2
fi

GROUP="${AZURE_RESOURCE_GROUP:-recipe-ci}"
LOCATION="${AZURE_LOCATION:-eastus}"
# Standard_NC4as_T4_v3 is the smallest T4 shape; the quota request targets it.
SIZE="${AZURE_VM_SIZE:-Standard_NC4as_T4_v3}"
IMAGE="${AZURE_VM_IMAGE:-Win2022AzureEdition}"
# One worker per run: two candidates must never share a mutable working directory.
WORKER="recipe-wgpu-${RUN_ID}-${RUN_ATTEMPT}"
DEADLINE_SECONDS="${AZURE_DEADLINE_SECONDS:-3600}"

mkdir -p evidence

echo "== subscription and credit =="
az account show --query "{name:name, id:id, state:state}" -o json | tee evidence/azure-account.json
# Record what is actually left, so the finite trial window is visible in the
# evidence rather than assumed.
az consumption budget list --query "[].{name:name, amount:amount, timeGrain:timeGrain}" -o json > evidence/azure-budgets.json || echo "no budget data available"

echo "== confirming GPU quota before provisioning =="
family="standardNCASv3_T4Family"
usage="$(az vm list-usage --location "$LOCATION" --query "[?contains(localName, 'NCASv3_T4')].{current:currentValue, limit:limit}" -o json)"
echo "$usage" | tee evidence/azure-quota.json
limit="$(echo "$usage" | python3 -c 'import json,sys; rows=json.load(sys.stdin); print(rows[0]["limit"] if rows else 0)')"
if [ "${limit:-0}" -lt 4 ]; then
	cat > evidence/blocker.json <<JSON
{
  "blocker": "azure-gpu-quota-unavailable",
  "detail": "The $family quota in $LOCATION is $limit cores, which cannot host a $SIZE worker. A new Azure account carries zero GPU quota until a quota increase is approved, and the pay-as-you-go upgrade that makes the request eligible removes the credit spending protection.",
  "resolution": "Obtain explicit billing authorization, upgrade the credit account to pay-as-you-go, then request at least 4 cores of $family in a region that stocks $SIZE."
}
JSON
	cat evidence/blocker.json
	echo "recipe/windows-gpu is blocked: no GPU quota" >&2
	exit 1
fi

list_worker() {
	az vm list \
		--resource-group "$GROUP" \
		--query "[?name=='$WORKER'].name" \
		-o tsv \
		--only-show-errors
}

wait_for_worker_absent() {
	local attempt remaining
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
		if ! remaining="$(list_worker)"; then
			echo "could not read back worker state during cleanup" >&2
			return 1
		fi
		if [ -z "$remaining" ]; then
			return 0
		fi
		sleep 5
	done
	echo "worker $WORKER is still present after deletion" >&2
	return 1
}

cleanup_on_exit() {
	local primary=$?
	local cleanup_status=0
	trap - EXIT
	echo "== releasing the worker =="
	local current
	if ! current="$(list_worker)"; then
		cleanup_status=1
		echo "could not determine whether worker $WORKER exists" >&2
	elif [ -n "$current" ]; then
		if ! az vm delete --resource-group "$GROUP" --name "$WORKER" --yes --force-deletion true --only-show-errors; then
			cleanup_status=1
			echo "worker deletion failed for $WORKER" >&2
		fi
	else
		echo "worker $WORKER was already absent"
	fi
	if ! wait_for_worker_absent; then
		cleanup_status=1
	fi
	if [ "$primary" -ne 0 ]; then
		if [ "$cleanup_status" -ne 0 ]; then
			echo "primary status $primary preserved; cleanup also failed" >&2
		fi
		exit "$primary"
	fi
	if [ "$cleanup_status" -ne 0 ]; then
		exit 1
	fi
	exit 0
}
trap cleanup_on_exit EXIT

echo "== provisioning the isolated worker =="
az group create --name "$GROUP" --location "$LOCATION" --only-show-errors -o none
az vm create \
	--resource-group "$GROUP" \
	--name "$WORKER" \
	--image "$IMAGE" \
	--size "$SIZE" \
	--admin-username recipeci \
	--admin-password "$(python3 -c 'import secrets,string; print("Aa1!" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(28)))')" \
	--public-ip-address "" \
	--nsg-rule NONE \
	--os-disk-delete-option Delete \
	--nic-delete-option Delete \
	--only-show-errors -o json > evidence/azure-vm.json
echo "provisioned $WORKER ($SIZE) in $GROUP/$LOCATION with no public address"

echo "== installing the NVIDIA GPU driver extension =="
az vm extension set \
	--resource-group "$GROUP" \
	--vm-name "$WORKER" \
	--name NvidiaGpuDriverWindows \
	--publisher Microsoft.HpcCompute \
	--version 1.6 \
	--only-show-errors -o none
echo "driver extension installed"

echo "== transferring the immutable snapshot =="
# The snapshot travels as base64 through the Run Command payload, so the guest
# never needs network credentials or a clone of the newest commit.
[ -f "$TRUSTED_RUNTIME/suite.rs" ] || { echo "trusted suite is absent" >&2; exit 1; }
[ -d "$TRUSTED_RUNTIME/data" ] || { echo "trusted suite data is absent" >&2; exit 1; }
tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" suite.rs data
runtime_sha256="$(sha256sum trusted-runtime.tar.gz | cut -d' ' -f1)"
base64 -w0 "$SNAPSHOT" > snapshot.b64
split -b 60000 snapshot.b64 snapshot-chunk-
snapshot_chunks=(snapshot-chunk-*)
base64 -w0 trusted-runtime.tar.gz > runtime-suite.b64
split -b 60000 runtime-suite.b64 runtime-suite-chunk-
runtime_chunks=(runtime-suite-chunk-*)
echo "snapshot split into ${#snapshot_chunks[@]} chunks; trusted runtime split into ${#runtime_chunks[@]} chunks"

az vm run-command invoke \
	--resource-group "$GROUP" --name "$WORKER" \
	--command-id RunPowerShellScript \
	--scripts 'New-Item -ItemType Directory -Force -Path C:\recipe | Out-Null; Remove-Item -Force C:\recipe\snapshot.b64,C:\recipe\runtime-suite.b64 -ErrorAction SilentlyContinue; "staged"' \
	--only-show-errors -o none

for chunk in "${snapshot_chunks[@]}"; do
	az vm run-command invoke \
		--resource-group "$GROUP" --name "$WORKER" \
		--command-id RunPowerShellScript \
		--scripts "Add-Content -Path C:\\recipe\\snapshot.b64 -Value '$(cat "$chunk")' -NoNewline" \
		--only-show-errors -o none
done
for chunk in "${runtime_chunks[@]}"; do
	az vm run-command invoke \
		--resource-group "$GROUP" --name "$WORKER" \
		--command-id RunPowerShellScript \
		--scripts "Add-Content -Path C:\\recipe\\runtime-suite.b64 -Value '$(cat "$chunk")' -NoNewline" \
		--only-show-errors -o none
done

echo "== executing the native Windows GPU suite in the guest =="
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
guest_script="$(python3 - "$script_dir/azure-guest.ps1" <<'PY'
import json, pathlib, sys
script = pathlib.Path(sys.argv[1]).read_text()
print(json.dumps(script))
PY
)"
python3 - "$guest_script" <<'PY' > guest.ps1
import json, sys
sys.stdout.write(json.loads(sys.argv[1]))
PY

started="$(date +%s)"
az vm run-command invoke \
	--resource-group "$GROUP" --name "$WORKER" \
	--command-id RunPowerShellScript \
	--scripts "@guest.ps1" \
	--parameters "candidateSha=$CANDIDATE_SHA" "snapshotSha256=$SNAPSHOT_SHA256" "runtimeSuiteSha256=$runtime_sha256" \
	--only-show-errors -o json > evidence/azure-runcommand.json
elapsed=$(( $(date +%s) - started ))
echo "run command returned after ${elapsed}s"

if [ "$elapsed" -ge "$DEADLINE_SECONDS" ]; then
	echo "the guest execution deadline of ${DEADLINE_SECONDS}s was exceeded" >&2
	exit 1
fi

# An HTTP 200 from Run Command says the control plane accepted the request. The
# guest's own exit status is inside the message body.
python3 - <<'PY' > evidence/guest.log
import json, pathlib
document = json.loads(pathlib.Path("evidence/azure-runcommand.json").read_text())
for entry in document.get("value", []):
	print(f"--- {entry.get('code')} ---")
	print(entry.get("message", ""))
PY
tail -80 evidence/guest.log

if ! grep -q "GUEST EXIT 0" evidence/guest.log; then
	echo "the guest process did not report a zero exit status" >&2
	exit 1
fi

python3 - <<'PY'
import pathlib, re, sys
log = pathlib.Path("evidence/guest.log").read_text()
match = re.search(r"^SUITE-EVIDENCE-BEGIN$(.*?)^SUITE-EVIDENCE-END$", log, re.S | re.M)
if not match:
	sys.exit("the guest returned no suite evidence")
pathlib.Path("evidence/suite.json").write_text(match.group(1).strip())
print("recovered suite evidence")
PY

route="$(grep -m1 '^selected route ' evidence/guest.log | awk '{print $3}')"
device="${route##*:}"
case "$device" in
	nv*) ;;
	*) echo "expected an nv device, got '$device'; CPU fallback is a failure" >&2; exit 1 ;;
esac

gpu_model="$(grep -m1 -oE 'Tesla T4|NVIDIA T4|T4' evidence/guest.log | head -1)"
cat > evidence/cell.json <<JSON
{
  "cell": "recipe/windows-gpu",
  "commit": "$CANDIDATE_SHA",
  "run_id": "$RUN_ID",
  "run_attempt": "$RUN_ATTEMPT",
  "provider": "azure",
  "resource_group": "$GROUP",
  "worker": "$WORKER",
  "vm_size": "$SIZE",
  "location": "$LOCATION",
  "image": "$IMAGE",
  "os": "Windows Server 2022 Datacenter Azure Edition",
  "arch": "x86_64",
  "backend": "nvidia",
  "device": "$device",
  "gpu_model": "$gpu_model",
  "gpu_execution": true,
  "snapshot_sha256": "$SNAPSHOT_SHA256"
}
JSON
cat evidence/cell.json
echo "recipe/windows-gpu completed on $route"
