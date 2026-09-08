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
# Standard_NC4as_T4_v3 is the smallest T4 shape.
SIZE="${AZURE_VM_SIZE:-Standard_NC4as_T4_v3}"
IMAGE="${AZURE_VM_IMAGE:-MicrosoftWindowsServer:WindowsServer:2022-datacenter-azure-edition-core:latest}"
FAMILY="Standard NCASv3_T4 Family"
REQUIRED_CORES=4
# One worker per run: two candidates must never share a mutable working directory.
WORKER="recipe-wgpu-${RUN_ID}-${RUN_ATTEMPT}"
DEADLINE_SECONDS="${AZURE_DEADLINE_SECONDS:-3600}"

mkdir -p evidence

echo "== subscription =="
az account show --query "{name:name, id:id, state:state}" -o json | tee evidence/azure-account.json

echo "== selecting available GPU capacity =="
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
az vm list \
	--query "[?hardwareProfile.vmSize=='$SIZE'].{name:name,resource_group:resourceGroup,location:location,created_at:timeCreated,recipe_owner:tags.\"recipe-owner\",recipe_pool:tags.\"recipe-pool\",recipe_worker:tags.\"recipe-worker\"}" \
	--only-show-errors -o json | tee evidence/azure-gpu-inventory.json

# Fetch the subscription-aware SKU catalog once, then check quota only in
# regions where Azure offers this exact shape to this subscription.
sku="$(az vm list-skus --resource-type virtualMachines --size "$SIZE" --all --query "[?name=='$SIZE']" -o json --only-show-errors)"
mapfile -t supported_locations < <(
	jq -r --arg size "$SIZE" '
		.[] | select(.name == $size) as $sku
		| $sku.locations[] as $location
		| select([
			$sku.restrictions[]?
			| select(.type == "Location")
			| select((.restrictionInfo.locations // []) | index($location))
		] | length == 0)
		| $location
	' <<< "$sku" | sort -u
)
if [ "${#supported_locations[@]}" -eq 0 ]; then
	echo "$SIZE is unavailable to this subscription" >&2
	exit 1
fi

declare -a candidate_locations=()
declare -A location_seen=()
if [ -n "${AZURE_LOCATION:-}" ]; then
	candidate_locations+=("$AZURE_LOCATION")
	location_seen["$AZURE_LOCATION"]=1
else
	# Prefer nearby US regions, then use any other subscription-supported region.
	for location in eastus eastus2 centralus southcentralus northcentralus westus2 westus3; do
		candidate_locations+=("$location")
		location_seen["$location"]=1
	done
	for location in "${supported_locations[@]}"; do
		if [ -z "${location_seen[$location]:-}" ]; then
			candidate_locations+=("$location")
			location_seen["$location"]=1
		fi
	done
fi

read_quota() {
	local location="$1"
	local usage current limit regional_current regional_limit free regional_free
	usage="$(az vm list-usage --location "$location" -o json --only-show-errors)"
	current="$(jq -r --arg family "$FAMILY" '[.[] | select(.name.value == $family)][0].currentValue // 0' <<< "$usage")"
	limit="$(jq -r --arg family "$FAMILY" '[.[] | select(.name.value == $family)][0].limit // 0' <<< "$usage")"
	regional_current="$(jq -r '[.[] | select(.name.value == "cores")][0].currentValue // 0' <<< "$usage")"
	regional_limit="$(jq -r '[.[] | select(.name.value == "cores")][0].limit // 0' <<< "$usage")"
	free=$((limit - current))
	regional_free=$((regional_limit - regional_current))
	jq -n \
		--arg location "$location" \
		--arg family "$FAMILY" \
		--argjson current "$current" \
		--argjson limit "$limit" \
		--argjson free "$free" \
		--argjson regional_current "$regional_current" \
		--argjson regional_limit "$regional_limit" \
		--argjson regional_free "$regional_free" \
		'{location:$location, family:$family, current:$current, limit:$limit, free:$free, regional_current:$regional_current, regional_limit:$regional_limit, regional_free:$regional_free}'
}

: > evidence/azure-capacity.jsonl
LOCATION=""
for location in "${candidate_locations[@]}"; do
	if ! printf '%s\n' "${supported_locations[@]}" | grep -Fqx "$location"; then
		continue
	fi
	quota="$(read_quota "$location")"
	printf '%s\n' "$quota" | tee -a evidence/azure-capacity.jsonl
	free="$(jq -r '.free' <<< "$quota")"
	regional_free="$(jq -r '.regional_free' <<< "$quota")"
	if [ "$free" -ge "$REQUIRED_CORES" ] && [ "$regional_free" -ge "$REQUIRED_CORES" ]; then
		LOCATION="$location"
		printf '%s\n' "$quota" > evidence/azure-quota.json
		break
	fi
done

if [ -z "$LOCATION" ]; then
	cat > evidence/blocker.json <<JSON
{
  "blocker": "azure-gpu-quota-unavailable",
  "detail": "No subscription-supported region has $REQUIRED_CORES unused $FAMILY cores and regional cores for one $SIZE worker.",
  "resolution": "Grant the GPU controller Microsoft.Quota/quotas/write and increase the family quota, or free an existing allocation listed in azure-gpu-inventory.json."
}
JSON
	cat evidence/blocker.json
	echo "recipe/windows-gpu is blocked: no unused GPU quota" >&2
	exit 1
fi
echo "selected $LOCATION for $SIZE"

echo "== resolving the Windows image =="
az vm image show \
	--location "$LOCATION" \
	--urn "$IMAGE" \
	--query '{urn:urn, id:id, architecture:architecture, hyperVGeneration:hyperVGeneration}' \
	--only-show-errors -o json | tee evidence/azure-image.json

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
	--location "$LOCATION" \
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
