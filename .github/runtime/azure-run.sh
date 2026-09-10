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
# Azure control plane. A Standard public IP provides explicit outbound access;
# its default NSG has no inbound allow rule.
set -euo pipefail

: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${SNAPSHOT:?SNAPSHOT is required}"
: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${RUN_ATTEMPT:?RUN_ATTEMPT is required}"
: "${TRUSTED_RUNTIME:?TRUSTED_RUNTIME is required}"
: "${AZURE_STORAGE_ACCOUNT:?AZURE_STORAGE_ACCOUNT is required}"
: "${AZURE_STORAGE_CONTAINER:?AZURE_STORAGE_CONTAINER is required}"
if [[ ! "$RUN_ID" =~ ^[0-9]+$ ]] || [[ ! "$RUN_ATTEMPT" =~ ^[0-9]+$ ]]; then
	echo "RUN_ID and RUN_ATTEMPT must be decimal workflow identifiers" >&2
	exit 2
fi

GROUP="${AZURE_RESOURCE_GROUP:-recipe-ci}"
# Standard_NC4as_T4_v3 is the smallest T4 shape.
SIZE="${AZURE_VM_SIZE:-Standard_NC4as_T4_v3}"
IMAGE="${AZURE_VM_IMAGE:-microsoft-dsvm:dsvm-win-2022:winserver-2022:25.05.10}"
FAMILY="Standard NCASv3_T4 Family"
REQUIRED_CORES=4
# One worker per run: two candidates must never share a mutable working directory.
WORKER="recipe-wgpu-${RUN_ID}-${RUN_ATTEMPT}"
COMPUTER_NAME="rgpu$(printf '%s' "$WORKER" | sha256sum | cut -c1-11)"
TRANSFER_ROOT="runtime/windows/${RUN_ID}-${RUN_ATTEMPT}"
SNAPSHOT_BLOB="$TRANSFER_ROOT/snapshot.tar.gz"
RUNTIME_BLOB="$TRANSFER_ROOT/runtime-suite.tar.gz"
PREFLIGHT_DEADLINE_SECONDS="${AZURE_PREFLIGHT_DEADLINE_SECONDS:-600}"
DEADLINE_SECONDS="${AZURE_DEADLINE_SECONDS:-2700}"
ADMISSION_WAIT_SECONDS="${AZURE_ADMISSION_WAIT_SECONDS:-1800}"
ADMISSION_LEASE_SECONDS=60
ADMISSION_BLOB="runtime/windows/admission.lock"
admission_lease_id=""
admission_renew_pid=""
admission_started="$(date +%s)"

if [[ ! "$ADMISSION_WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
	echo "AZURE_ADMISSION_WAIT_SECONDS must be a nonnegative integer" >&2
	exit 2
fi

release_admission() {
	if [ -n "$admission_renew_pid" ]; then
		kill "$admission_renew_pid" 2>/dev/null || true
		wait "$admission_renew_pid" 2>/dev/null || true
		admission_renew_pid=""
	fi
	if [ -n "$admission_lease_id" ]; then
		az storage blob lease release \
			--auth-mode login \
			--account-name "$AZURE_STORAGE_ACCOUNT" \
			--container-name "$AZURE_STORAGE_CONTAINER" \
			--blob-name "$ADMISSION_BLOB" \
			--lease-id "$admission_lease_id" \
			--only-show-errors -o none || {
			echo "could not release the Azure GPU admission lease; it will expire automatically" >&2
			return 1
		}
		admission_lease_id=""
	fi
}
trap 'release_admission || true' EXIT

mkdir -p evidence

echo "== subscription =="
az account show --query "{name:name, id:id, state:state}" -o json | tee evidence/azure-account.json

ensure_admission_blob() {
	local exists
	if ! exists="$(az storage blob exists \
		--auth-mode login \
		--account-name "$AZURE_STORAGE_ACCOUNT" \
		--container-name "$AZURE_STORAGE_CONTAINER" \
		--name "$ADMISSION_BLOB" \
		--query exists -o tsv --only-show-errors)"; then
		echo "could not inspect the Azure GPU admission blob" >&2
		return 1
	fi
	if [ "$exists" = true ]; then
		return 0
	fi
	if ! az storage blob upload \
		--auth-mode login \
		--account-name "$AZURE_STORAGE_ACCOUNT" \
		--container-name "$AZURE_STORAGE_CONTAINER" \
		--name "$ADMISSION_BLOB" \
		--data "$WORKER" \
		--overwrite false \
		--content-type text/plain \
		--only-show-errors -o none; then
		# Another waiting run may have created it between exists and upload.
		exists="$(az storage blob exists \
			--auth-mode login \
			--account-name "$AZURE_STORAGE_ACCOUNT" \
			--container-name "$AZURE_STORAGE_CONTAINER" \
			--name "$ADMISSION_BLOB" \
			--query exists -o tsv --only-show-errors)" || return 1
		[ "$exists" = true ] || return 1
	fi
}

renew_admission() {
	while sleep 30; do
		renewed=false
		for attempt in 1 2 3; do
			if az storage blob lease renew \
				--auth-mode login \
				--account-name "$AZURE_STORAGE_ACCOUNT" \
				--container-name "$AZURE_STORAGE_CONTAINER" \
				--blob-name "$ADMISSION_BLOB" \
				--lease-id "$admission_lease_id" \
				--only-show-errors -o none; then
				renewed=true
				break
			fi
			sleep 2
		done
		if [ "$renewed" != true ]; then
			echo "Azure GPU admission lease renewal failed; stopping before provisioning can overlap" >&2
			kill -TERM "$$" 2>/dev/null || true
			return 1
		fi
	done
}

acquire_admission() {
	local deadline now lease_error_file lease_error detail
	deadline=$(( $(date +%s) + ADMISSION_WAIT_SECONDS ))
	if ! ensure_admission_blob; then
		cat > evidence/blocker.json <<JSON
{
  "blocker": "azure-gpu-admission-storage-unavailable",
  "detail": "The Azure storage blob used for GPU admission could not be inspected or created.",
  "resolution": "Restore the storage account/container permissions before retrying the Windows GPU check."
}
JSON
		cat evidence/blocker.json
		return 1
	fi
	lease_error_file="evidence/azure-admission-error.log"
	: > "$lease_error_file"
	while :; do
		if admission_lease_id="$(az storage blob lease acquire \
			--auth-mode login \
			--account-name "$AZURE_STORAGE_ACCOUNT" \
			--container-name "$AZURE_STORAGE_CONTAINER" \
			--blob-name "$ADMISSION_BLOB" \
			--lease-duration "$ADMISSION_LEASE_SECONDS" \
			--query leaseId -o tsv --only-show-errors 2>"$lease_error_file")" && [ -n "$admission_lease_id" ]; then
			admission_renew_pid=""
			renew_admission &
			admission_renew_pid=$!
			printf '{"worker":"%s","wait_seconds":%s,"lease_seconds":%s}\n' \
				"$WORKER" "$(( $(date +%s) - admission_started ))" "$ADMISSION_LEASE_SECONDS" \
				> evidence/azure-admission.json
			cat evidence/azure-admission.json
			return 0
		fi
		lease_error="$(sed -n '1,8p' "$lease_error_file")"
		if ! grep --ignore-case --extended-regexp --quiet 'lease.?already|active lease|condition.?not.?met|status.?code.?409|\b409\b' "$lease_error_file"; then
			detail="$(jq -Rs . < "$lease_error_file")"
			cat > evidence/blocker.json <<JSON
{
  "blocker": "azure-gpu-admission-error",
  "detail": $detail,
  "resolution": "Resolve the Azure storage authorization or network error before retrying the Windows GPU check."
}
JSON
			cat evidence/blocker.json
			return 1
		fi
		if [ -n "$lease_error" ]; then
			echo "Azure GPU admission is busy: $lease_error"
		fi
		now="$(date +%s)"
		if [ "$now" -ge "$deadline" ]; then
			cat > evidence/blocker.json <<JSON
{
  "blocker": "azure-gpu-admission-timeout",
  "detail": "Another Windows GPU runtime owns the shared Azure admission lease, and this run waited ${ADMISSION_WAIT_SECONDS} seconds without a slot.",
  "resolution": "Retry this run after the active Windows GPU runtime releases its worker."
}
JSON
			cat evidence/blocker.json
			return 1
		fi
		echo "Azure GPU admission is busy; waiting for the active worker"
		sleep 15
	done
}

echo "== acquiring the Windows GPU admission lease =="
acquire_admission

echo "== selecting available GPU capacity =="
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
inventory="$(az vm list --show-details \
	--query "[?hardwareProfile.vmSize=='$SIZE'].{name:name,resource_group:resourceGroup,location:location,power_state:powerState,created_at:timeCreated,recipe_owner:tags.\"recipe-owner\",recipe_pool:tags.\"recipe-pool\",recipe_worker:tags.\"recipe-worker\"}" \
	--only-show-errors -o json)"
printf '%s\n' "$inventory" | tee evidence/azure-gpu-inventory.json

# A forced workflow cancellation can skip the always() cleanup step. Reclaim a
# prior per-run worker only after GitHub confirms that its owning run completed.
if command -v gh >/dev/null && [ -n "${GH_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
	while IFS=$'\t' read -r prior_worker prior_group; do
		if [[ ! "$prior_worker" =~ ^recipe-wgpu-([0-9]+)-([0-9]+)$ ]]; then
			continue
		fi
		prior_run="${BASH_REMATCH[1]}"
		prior_attempt="${BASH_REMATCH[2]}"
		if ! prior_status="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$prior_run" --jq .status 2>/dev/null)"; then
			echo "could not confirm the owner of prior worker $prior_worker" >&2
			continue
		fi
		if [ "$prior_status" = "completed" ]; then
			echo "== reclaiming terminal-run worker $prior_worker =="
			AZURE_RESOURCE_GROUP="$prior_group" \
			RUN_ID="$prior_run" \
			RUN_ATTEMPT="$prior_attempt" \
				bash "$TRUSTED_RUNTIME/azure-cleanup.sh"
		fi
	done < <(jq -r --arg current "$WORKER" '.[] | select(.name != $current) | [.name, .resource_group] | @tsv' <<< "$inventory")
fi

# Refresh after reclamation, then fail closed if an earlier Recipe worker is
# still present. The lease normally prevents this state; the inventory guard
# also prevents overlap if a hard-canceled controller outlives its lease.
inventory="$(az vm list --show-details \
	--query "[?hardwareProfile.vmSize=='$SIZE'].{name:name,resource_group:resourceGroup,location:location,power_state:powerState,created_at:timeCreated,recipe_owner:tags.\"recipe-owner\",recipe_pool:tags.\"recipe-pool\",recipe_worker:tags.\"recipe-worker\"}" \
	--only-show-errors -o json)"
active_workers="$(jq -r --arg current "$WORKER" '
	.[]
	| select(.name != $current)
	| select((.recipe_owner == "recipe-runtime-ci") or ((.name // "") | startswith("recipe-wgpu-")))
	| select((.power_state // "") != "VM deallocated" and (.power_state // "") != "VM stopped")
	| .name
' <<< "$inventory")"
if [ -n "$active_workers" ]; then
	cat > evidence/blocker.json <<JSON
{
  "blocker": "azure-gpu-worker-active",
  "detail": "A prior Recipe Windows GPU worker is still present: ${active_workers//$'\n'/, }.",
  "resolution": "Wait for the owning run's cleanup to delete the worker, then retry. No second worker was provisioned."
}
JSON
	cat evidence/blocker.json
	exit 1
fi

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
	if ! usage="$(az vm list-usage --location "$location" -o json --only-show-errors)"; then
		return 1
	fi
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
	if ! quota="$(read_quota "$location")"; then
		echo "skipping $location because its compute usage endpoint is unavailable" >&2
		continue
	fi
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
	if ! release_admission; then
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
admin_password="Aa1!$(openssl rand -hex 18)"
az vm create \
	--resource-group "$GROUP" \
	--name "$WORKER" \
	--computer-name "$COMPUTER_NAME" \
	--location "$LOCATION" \
	--image "$IMAGE" \
	--size "$SIZE" \
	--security-type Standard \
	--admin-username recipeci \
	--admin-password "$admin_password" \
	--public-ip-address "$WORKER-ip" \
	--public-ip-address-allocation static \
	--public-ip-sku Standard \
	--nsg-rule NONE \
	--os-disk-delete-option Delete \
	--nic-delete-option Delete \
	--tags \
		"recipe-owner=recipe-runtime-ci" \
		"recipe-worker=$WORKER" \
		"recipe-run-id=$RUN_ID" \
		"recipe-run-attempt=$RUN_ATTEMPT" \
	--only-show-errors -o json > evidence/azure-vm.json
unset admin_password
echo "provisioned $WORKER ($SIZE) in $GROUP/$LOCATION with explicit outbound access and no inbound rule"

echo "== transferring the immutable snapshot =="
# Upload each archive once to the existing private container. The guest gets
# only short-lived read URLs and never receives a cloud-management credential.
[ -f "$TRUSTED_RUNTIME/suite.rs" ] || { echo "trusted suite is absent" >&2; exit 1; }
[ -d "$TRUSTED_RUNTIME/data" ] || { echo "trusted suite data is absent" >&2; exit 1; }
tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" suite.rs data
runtime_sha256="$(sha256sum trusted-runtime.tar.gz | cut -d' ' -f1)"
snapshot_actual="$(sha256sum "$SNAPSHOT" | cut -d' ' -f1)"
[ "$snapshot_actual" = "$SNAPSHOT_SHA256" ] || { echo "snapshot checksum mismatch before upload" >&2; exit 1; }
az storage blob upload \
	--auth-mode login \
	--account-name "$AZURE_STORAGE_ACCOUNT" \
	--container-name "$AZURE_STORAGE_CONTAINER" \
	--name "$SNAPSHOT_BLOB" \
	--file "$SNAPSHOT" \
	--overwrite false \
	--metadata "recipe_worker=$WORKER" \
	--only-show-errors --no-progress -o none
az storage blob upload \
	--auth-mode login \
	--account-name "$AZURE_STORAGE_ACCOUNT" \
	--container-name "$AZURE_STORAGE_CONTAINER" \
	--name "$RUNTIME_BLOB" \
	--file trusted-runtime.tar.gz \
	--overwrite false \
	--metadata "recipe_worker=$WORKER" \
	--only-show-errors --no-progress -o none
sas_start="$(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%SZ)"
sas_expiry="$(date -u -d '2 hours' +%Y-%m-%dT%H:%M:%SZ)"
SNAPSHOT_URI="$(az storage blob generate-sas \
	--auth-mode login --as-user --full-uri --https-only \
	--account-name "$AZURE_STORAGE_ACCOUNT" \
	--container-name "$AZURE_STORAGE_CONTAINER" \
	--name "$SNAPSHOT_BLOB" \
	--permissions r --start "$sas_start" --expiry "$sas_expiry" \
	-o tsv --only-show-errors)"
RUNTIME_URI="$(az storage blob generate-sas \
	--auth-mode login --as-user --full-uri --https-only \
	--account-name "$AZURE_STORAGE_ACCOUNT" \
	--container-name "$AZURE_STORAGE_CONTAINER" \
	--name "$RUNTIME_BLOB" \
	--permissions r --start "$sas_start" --expiry "$sas_expiry" \
	-o tsv --only-show-errors)"
case "$SNAPSHOT_URI" in https://*\?*) ;; *) echo "snapshot read URL generation failed" >&2; exit 1 ;; esac
case "$RUNTIME_URI" in https://*\?*) ;; *) echo "runtime read URL generation failed" >&2; exit 1 ;; esac
if [ "$(curl --silent --fail --max-time 120 "$SNAPSHOT_URI" | sha256sum | cut -d' ' -f1)" != "$SNAPSHOT_SHA256" ]; then
	echo "private snapshot download verification failed" >&2
	exit 1
fi
if [ "$(curl --silent --fail --max-time 120 "$RUNTIME_URI" | sha256sum | cut -d' ' -f1)" != "$runtime_sha256" ]; then
	echo "private runtime download verification failed" >&2
	exit 1
fi
snapshot_uri_encoded="$(printf '%s' "$SNAPSHOT_URI" | base64 -w0 | tr '+/' '-_' | tr -d '=')"
runtime_uri_encoded="$(printf '%s' "$RUNTIME_URI" | base64 -w0 | tr '+/' '-_' | tr -d '=')"
echo "uploaded and verified the private per-run archives"

echo "== executing the native Windows GPU suite in the guest =="
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cp "$script_dir/azure-guest.ps1" guest.ps1

invoke_guest() {
	local phase="$1"
	local deadline="$2"
	local marker="$3"
	local document="evidence/azure-${phase}.json"
	local log="evidence/${phase}.log"
	local started status elapsed
	started="$(date +%s)"
	set +e
	timeout --signal=TERM --kill-after=30s "${deadline}s" \
		az vm run-command invoke \
			--resource-group "$GROUP" --name "$WORKER" \
			--command-id RunPowerShellScript \
			--scripts "@guest.ps1" \
			--parameters "phase=$phase" "candidateSha=$CANDIDATE_SHA" "snapshotSha256=$SNAPSHOT_SHA256" "runtimeSuiteSha256=$runtime_sha256" "snapshotUriEncoded=$snapshot_uri_encoded" "runtimeSuiteUriEncoded=$runtime_uri_encoded" \
			--only-show-errors -o json > "$document"
	status=$?
	set -e
	elapsed=$(( $(date +%s) - started ))
	echo "$phase command returned after ${elapsed}s with controller status $status"
	if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
		echo "$phase exceeded its ${deadline}s controller deadline" >&2
		return 1
	fi
	if [ "$status" -ne 0 ]; then
		echo "$phase control-plane invocation failed with status $status" >&2
		return "$status"
	fi
	jq -r '.value[] | "--- \(.code) ---\n\(.message // "")"' "$document" > "$log"
	tail -80 "$log"
	if ! grep -Fq "$marker" "$log"; then
		echo "$phase did not report '$marker'" >&2
		return 1
	fi
}

echo "== checking the DSVM before the build =="
invoke_guest "preflight" "$PREFLIGHT_DEADLINE_SECONDS" "PREFLIGHT EXIT 0"

echo "== building and executing Recipe on the DSVM =="
invoke_guest "execute" "$DEADLINE_SECONDS" "GUEST EXIT 0"
cp evidence/execute.log evidence/guest.log

awk '
	/^SUITE-EVIDENCE-BEGIN$/ { capture = 1; next }
	/^SUITE-EVIDENCE-END$/ { capture = 0; found = 1; next }
	capture { print }
	END { if (!found) exit 1 }
' evidence/guest.log > evidence/suite.json
[ -s evidence/suite.json ] || { echo "the guest returned no suite evidence" >&2; exit 1; }
echo "recovered suite evidence"

route="$(grep -m1 '^selected route ' evidence/guest.log | awk '{print $3}')"
device="${route##*:}"
case "$device" in
	nv*) ;;
	*) echo "expected an nv device, got '$device'; CPU fallback is a failure" >&2; exit 1 ;;
esac

gpu_model="$(grep -m1 -oE 'Tesla T4|NVIDIA T4|T4' evidence/preflight.log)"
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
  "os": "Windows Server 2022 Data Science Virtual Machine",
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
