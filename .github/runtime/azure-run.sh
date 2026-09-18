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
# The guest runs one workload: the suite (default), or the composition harness over
# RECIPE_TRIAL_COUNT cursors from RECIPE_TRIAL_CURSOR, whose stderr packets are the evidence.
RECIPE_WORKLOAD="${RECIPE_WORKLOAD:-suite}"
AZURE_TRIAL_MAX_COUNT=40
case "$RECIPE_WORKLOAD" in
	suite) ;;
	trial)
		: "${RECIPE_TRIAL_CURSOR:?RECIPE_TRIAL_CURSOR is required}"
		: "${RECIPE_TRIAL_COUNT:?RECIPE_TRIAL_COUNT is required}"
		case "$RECIPE_TRIAL_CURSOR$RECIPE_TRIAL_COUNT" in
			''|*[!0-9]*) echo "the trial cursor and count must be integers" >&2; exit 1 ;;
		esac
		if [ "$RECIPE_TRIAL_COUNT" -lt 1 ] || [ "$RECIPE_TRIAL_COUNT" -gt "$AZURE_TRIAL_MAX_COUNT" ]; then
			echo "the Azure trial count must be between 1 and $AZURE_TRIAL_MAX_COUNT" >&2
			exit 1
		fi
		;;
	*) echo "RECIPE_WORKLOAD must be suite or trial" >&2; exit 1 ;;
esac
TRIAL_BLOB="$TRANSFER_ROOT/trial.txt"
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
	local lease_id
	if [ -n "$admission_renew_pid" ]; then
		kill "$admission_renew_pid" 2>/dev/null || true
		wait "$admission_renew_pid" 2>/dev/null || true
		admission_renew_pid=""
	fi
	if [ -n "$admission_lease_id" ]; then
		# Forget the id before the call: with the renewer gone the lease expires
		# within ADMISSION_LEASE_SECONDS whether or not this release lands, and a
		# second release with a stale id would only fail again.
		lease_id="$admission_lease_id"
		admission_lease_id=""
		az storage blob lease release \
			--auth-mode login \
			--account-name "$AZURE_STORAGE_ACCOUNT" \
			--container-name "$AZURE_STORAGE_CONTAINER" \
			--blob-name "$ADMISSION_BLOB" \
			--lease-id "$lease_id" \
			--only-show-errors -o none || {
			echo "could not release the Azure GPU admission lease; it will expire automatically" >&2
			return 1
		}
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

# Takes the shared lease, waiting until the run-wide admission deadline. The
# caller may release and re-acquire it while it waits for cores, so the wait
# budget is measured from the start of admission, not from this call.
acquire_admission() {
	local deadline now lease_error_file lease_error detail
	deadline=$(( admission_started + ADMISSION_WAIT_SECONDS ))
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
			-o tsv --only-show-errors 2>"$lease_error_file")" && [ -n "$admission_lease_id" ]; then
			admission_renew_pid=""
			renew_admission &
			admission_renew_pid=$!
			echo "holding the admission lease after $(( $(date +%s) - admission_started ))s"
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

echo "== resolving GPU shapes offered to this subscription =="
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
# Fetch the subscription-aware SKU catalog once, then check quota only in
# regions where Azure offers this exact shape to this subscription. This is
# the slowest read (about a minute) and needs no lease.
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
declare -a quota_locations=()
for location in "${candidate_locations[@]}"; do
	if printf '%s\n' "${supported_locations[@]}" | grep -Fqx "$location"; then
		quota_locations+=("$location")
	fi
done
echo "candidate regions: ${quota_locations[*]}"

# A forced workflow cancellation can skip the always() cleanup step, and a
# leaked worker holds GPU quota. Reclaim a prior per-run worker only once its
# owning run is terminal: GitHub reports the run completed, or the worker
# belongs to an earlier attempt of this very run (a new attempt cannot start
# before the previous one finished, while the run's status reports the latest
# attempt). Workers of live runs are expected neighbours now that quota, not
# a global slot, bounds admission. Reclamation is best effort: another
# controller may be reclaiming the same worker, and the quota read decides
# whether this run can proceed.
reclaim_terminal_workers() {
	local inventory prior_worker prior_group prior_run prior_attempt prior_status
	if ! inventory="$(az vm list --show-details \
		--query "[?hardwareProfile.vmSize=='$SIZE'].{name:name,resource_group:resourceGroup,location:location,power_state:powerState,created_at:timeCreated,recipe_owner:tags.\"recipe-owner\",recipe_pool:tags.\"recipe-pool\",recipe_worker:tags.\"recipe-worker\"}" \
		--only-show-errors -o json)"; then
		echo "could not list GPU workers; skipping reclamation" >&2
		return 0
	fi
	printf '%s\n' "$inventory" > evidence/azure-gpu-inventory.json
	while IFS=$'\t' read -r prior_worker prior_group; do
		if [[ ! "$prior_worker" =~ ^recipe-wgpu-([0-9]+)-([0-9]+)$ ]]; then
			continue
		fi
		prior_run="${BASH_REMATCH[1]}"
		prior_attempt="${BASH_REMATCH[2]}"
		if [ "$prior_run" = "$RUN_ID" ] && [ "$prior_attempt" -lt "$RUN_ATTEMPT" ]; then
			prior_status=completed
		elif command -v gh >/dev/null && [ -n "${GH_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
			if ! prior_status="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$prior_run" --jq .status 2>/dev/null)"; then
				echo "could not confirm the owner of prior worker $prior_worker" >&2
				continue
			fi
		else
			continue
		fi
		if [ "$prior_status" = "completed" ]; then
			echo "== reclaiming terminal-run worker $prior_worker =="
			AZURE_RESOURCE_GROUP="$prior_group" \
			RUN_ID="$prior_run" \
			RUN_ATTEMPT="$prior_attempt" \
				bash "$TRUSTED_RUNTIME/azure-cleanup.sh" \
				|| echo "reclamation of $prior_worker did not fully complete; another controller may have removed it" >&2
		fi
	done < <(jq -r --arg current "$WORKER" '.[] | select(.name != $current) | [.name, .resource_group] | @tsv' <<< "$inventory")
}

echo "== reclaiming workers whose runs already completed =="
reclaim_terminal_workers
cat evidence/azure-gpu-inventory.json

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

# One pass over the candidate regions. Sets LOCATION to the first region with
# an unused worker's worth of family and regional cores that has not refused
# this run's allocation, GRANTED_LOCATIONS to every region whose quota could
# ever hold one, and OPEN_LOCATIONS to the granted regions that have not
# refused, so the caller can tell "busy" from "never granted" from "every
# region is out of this shape".
LOCATION=""
GRANTED_LOCATIONS=""
OPEN_LOCATIONS=""
USAGE_UNREAD=0
declare -A refused_locations=()
declare -A quota_refusals=()
select_location() {
	local location quota free regional_free limit regional_limit unread=0
	LOCATION=""
	GRANTED_LOCATIONS=""
	OPEN_LOCATIONS=""
	USAGE_UNREAD=0
	: > evidence/azure-capacity.jsonl
	for location in "${quota_locations[@]}"; do
		if ! quota="$(read_quota "$location")"; then
			echo "skipping $location because its compute usage endpoint is unavailable" >&2
			unread=$((unread + 1))
			continue
		fi
		printf '%s\n' "$quota" >> evidence/azure-capacity.jsonl
		limit="$(jq -r '.limit' <<< "$quota")"
		regional_limit="$(jq -r '.regional_limit' <<< "$quota")"
		if [ "$limit" -lt "$REQUIRED_CORES" ] || [ "$regional_limit" -lt "$REQUIRED_CORES" ]; then
			continue
		fi
		GRANTED_LOCATIONS="${GRANTED_LOCATIONS:+$GRANTED_LOCATIONS }$location"
		if [ -n "${refused_locations[$location]:-}" ]; then
			continue
		fi
		OPEN_LOCATIONS="${OPEN_LOCATIONS:+$OPEN_LOCATIONS }$location"
		free="$(jq -r '.free' <<< "$quota")"
		regional_free="$(jq -r '.regional_free' <<< "$quota")"
		if [ "$free" -ge "$REQUIRED_CORES" ] && [ "$regional_free" -ge "$REQUIRED_CORES" ]; then
			LOCATION="$location"
			printf '%s\n' "$quota" > evidence/azure-quota.json
			return 0
		fi
	done
	# Nothing is free. Once every region has been read, later polls only need
	# the regions that hold a grant.
	USAGE_UNREAD=$unread
	if [ -n "$GRANTED_LOCATIONS" ] && [ "$unread" -eq 0 ]; then
		read -r -a quota_locations <<< "$GRANTED_LOCATIONS"
	fi
	return 1
}

# The persistent resource group keeps the region it was first created in; a
# worker's own --location is what places it, so the group is created only
# when absent and never moved.
ensure_group() {
	local exists
	exists="$(az group exists --name "$GROUP" -o tsv --only-show-errors)"
	if [ "$exists" != true ]; then
		az group create --name "$GROUP" --location "$LOCATION" --only-show-errors -o none
	fi
}

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

# Removes this run's worker and whatever az vm create left beside it (the
# public IP, NSG and NIC are created before the VM object), so a refused
# allocation can be retried under the same name.
purge_worker() {
	local status=0 kind resource resources current
	if ! current="$(list_worker)"; then
		echo "could not determine whether worker $WORKER exists" >&2
		return 1
	fi
	if [ -n "$current" ]; then
		az vm delete --resource-group "$GROUP" --name "$WORKER" --yes --force-deletion true --only-show-errors || status=1
	fi
	wait_for_worker_absent || status=1
	for kind in nic disk public-ip nsg; do
		case "$kind" in
			nic) resources="$(az network nic list --resource-group "$GROUP" --query "[?starts_with(name, '$WORKER')].name" -o tsv --only-show-errors)" ;;
			disk) resources="$(az disk list --resource-group "$GROUP" --query "[?starts_with(name, '$WORKER')].name" -o tsv --only-show-errors)" ;;
			public-ip) resources="$(az network public-ip list --resource-group "$GROUP" --query "[?starts_with(name, '$WORKER')].name" -o tsv --only-show-errors)" ;;
			nsg) resources="$(az network nsg list --resource-group "$GROUP" --query "[?starts_with(name, '$WORKER')].name" -o tsv --only-show-errors)" ;;
		esac || { status=1; continue; }
		while IFS= read -r resource; do
			[ -n "$resource" ] || continue
			echo "removing $kind $resource left by the refused allocation"
			case "$kind" in
				nic) az network nic delete --resource-group "$GROUP" --name "$resource" --only-show-errors ;;
				disk) az disk delete --resource-group "$GROUP" --name "$resource" --yes --only-show-errors ;;
				public-ip) az network public-ip delete --resource-group "$GROUP" --name "$resource" --only-show-errors ;;
				nsg) az network nsg delete --resource-group "$GROUP" --name "$resource" --only-show-errors ;;
			esac || status=1
		done <<< "$resources"
	done
	return "$status"
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

# The shared lease covers only the quota read and the allocation, so two
# controllers cannot both see the same free cores; it is released as soon as
# this run's worker holds its own quota, and also while this run waits for
# cores, so a waiting controller never blocks another one's allocation or
# reclamation. Workers of other runs then execute side by side, bounded by
# the family quota rather than by one global slot.
write_blocker() {
	local blocker="$1" detail="$2" resolution="$3"
	jq -n --arg blocker "$blocker" --arg detail "$detail" --arg resolution "$resolution" \
		'{blocker:$blocker, detail:$detail, resolution:$resolution}' > evidence/blocker.json
	cat evidence/blocker.json
}

admission_deadline=$(( admission_started + ADMISSION_WAIT_SECONDS ))
provision_error_file="evidence/azure-provision-error.log"
: > "$provision_error_file"
lease_wait_seconds=0
allocation_attempts=0
echo "== provisioning the isolated worker =="
while :; do
	acquire_started="$(date +%s)"
	echo "== acquiring the Windows GPU admission lease =="
	acquire_admission
	lease_wait_seconds=$(( lease_wait_seconds + $(date +%s) - acquire_started ))
	if ! select_location; then
		release_admission || true
		if [ -z "$GRANTED_LOCATIONS" ] && [ "$USAGE_UNREAD" -eq 0 ]; then
			write_blocker "azure-gpu-quota-unavailable" \
				"No subscription-supported region grants $REQUIRED_CORES $FAMILY cores and regional cores for one $SIZE worker." \
				"Grant the GPU controller Microsoft.Quota/quotas/write and increase the family quota, or free an existing allocation listed in azure-gpu-inventory.json."
			echo "recipe/windows-gpu is blocked: no GPU quota is granted" >&2
			exit 1
		fi
		if [ -n "$GRANTED_LOCATIONS" ] && [ -z "$OPEN_LOCATIONS" ]; then
			write_blocker "azure-gpu-allocation-refused" \
				"Every granted region ($GRANTED_LOCATIONS) refused the $SIZE allocation for capacity after $allocation_attempts attempts; the last refusal is in azure-provision-error.log." \
				"Retry when a region has $SIZE capacity, or grant the family quota in another region."
			exit 1
		fi
		if [ "$(date +%s)" -ge "$admission_deadline" ]; then
			write_blocker "azure-gpu-quota-busy" \
				"Every open region ($OPEN_LOCATIONS) had its $FAMILY cores in use by other Windows GPU workers for ${ADMISSION_WAIT_SECONDS} seconds." \
				"Raise the $FAMILY quota so more workers fit, or retry after the active workers release their cores."
			exit 1
		fi
		if [ -z "$GRANTED_LOCATIONS" ]; then
			echo "no region's compute usage could be read ($USAGE_UNREAD unavailable); retrying"
		else
			echo "all granted GPU cores in ($OPEN_LOCATIONS) are in use; waiting for a worker to release them"
		fi
		sleep 15
		# A worker whose run ended after the first pass may be holding the cores.
		reclaim_terminal_workers
		continue
	fi
	echo "selected $LOCATION for $SIZE"
	echo "== resolving the Windows image in $LOCATION =="
	az vm image show \
		--location "$LOCATION" \
		--urn "$IMAGE" \
		--query '{urn:urn, id:id, architecture:architecture, hyperVGeneration:hyperVGeneration}' \
		--only-show-errors -o json | tee evidence/azure-image.json
	ensure_group
	allocation_attempts=$((allocation_attempts + 1))
	admin_password="Aa1!$(openssl rand -hex 18)"
	if az vm create \
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
		--only-show-errors -o json > evidence/azure-vm.json 2>"$provision_error_file"; then
		unset admin_password
		break
	fi
	unset admin_password
	cat "$provision_error_file" >&2
	# A capacity refusal excludes the region for this run; a quota refusal
	# means another controller's worker landed between the usage read and the
	# allocation, so the same region is polled again. Compute also reports
	# permanent refusals (image, security type, disallowed properties) as
	# OperationNotAllowed, so only its quota wording is retried. Anything
	# else is a real provisioning failure.
	if grep --ignore-case --extended-regexp --quiet 'AllocationFailed|SkuNotAvailable|ZonalAllocationFailed|OverconstrainedAllocationRequest|Capacity Restrictions' "$provision_error_file"; then
		refused_locations["$LOCATION"]=1
		echo "$LOCATION is out of $SIZE capacity (attempt $allocation_attempts); purging the partial worker and excluding the region" >&2
	elif grep --ignore-case --extended-regexp --quiet 'QuotaExceeded|OperationNotAllowed[^{}]*(quota|exceeding approved)' "$provision_error_file"; then
		quota_refusals["$LOCATION"]=$(( ${quota_refusals[$LOCATION]:-0} + 1 ))
		if [ "${quota_refusals[$LOCATION]}" -ge 3 ]; then
			# The usage read keeps promising cores the allocation cannot get.
			refused_locations["$LOCATION"]=1
			echo "$LOCATION refused the allocation for quota ${quota_refusals[$LOCATION]} times although its usage showed free cores; excluding the region" >&2
		else
			echo "$LOCATION ran out of $FAMILY cores during the allocation (attempt $allocation_attempts); purging the partial worker" >&2
		fi
	else
		release_admission || true
		purge_worker || true
		write_blocker "azure-gpu-provision-failed" \
			"az vm create for $SIZE in $LOCATION failed for a reason other than capacity or quota: $(sed -n '1,12p' "$provision_error_file" | tr -s '[:space:]' ' ' | cut -c1-1500)" \
			"Resolve the provisioning error in azure-provision-error.log before retrying the Windows GPU check."
		exit 1
	fi
	purge_worker || true
	release_admission || true
	if [ "$(date +%s)" -ge "$admission_deadline" ]; then
		write_blocker "azure-gpu-allocation-refused" \
			"Azure refused the $SIZE allocation $allocation_attempts times for capacity or quota within ${ADMISSION_WAIT_SECONDS} seconds; the last refusal is in azure-provision-error.log." \
			"Retry when a region has $SIZE capacity, or grant the family quota in another region."
		exit 1
	fi
	# The usage counters trail the allocation that beat this one.
	sleep 15
done
release_admission || true
printf '{"worker":"%s","location":"%s","lease_wait_seconds":%s,"admission_seconds":%s,"allocation_attempts":%s,"lease_seconds":%s}\n' \
	"$WORKER" "$LOCATION" "$lease_wait_seconds" "$(( $(date +%s) - admission_started ))" "$allocation_attempts" "$ADMISSION_LEASE_SECONDS" \
	> evidence/azure-admission.json
cat evidence/azure-admission.json
echo "provisioned $WORKER ($SIZE) in $GROUP/$LOCATION with explicit outbound access and no inbound rule"

echo "== transferring the immutable snapshot =="
# Upload each archive once to the existing private container. The guest gets
# only short-lived read URLs and never receives a cloud-management credential.
if [ "$RECIPE_WORKLOAD" = trial ]; then
	[ -f "$TRUSTED_RUNTIME/harness.rs" ] || { echo "the trial harness is absent" >&2; exit 1; }
	tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" harness.rs
else
	[ -f "$TRUSTED_RUNTIME/suite.rs" ] || { echo "trusted suite is absent" >&2; exit 1; }
	[ -d "$TRUSTED_RUNTIME/data" ] || { echo "trusted suite data is absent" >&2; exit 1; }
	tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" suite.rs data
fi
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
# A trial returns its harness stderr through the same private container: the guest gets a
# short-lived create/write URL for one blob, since Run Command output is capped.
trial_uri_encoded=""
if [ "$RECIPE_WORKLOAD" = trial ]; then
	TRIAL_URI="$(az storage blob generate-sas \
		--auth-mode login --as-user --full-uri --https-only \
		--account-name "$AZURE_STORAGE_ACCOUNT" \
		--container-name "$AZURE_STORAGE_CONTAINER" \
		--name "$TRIAL_BLOB" \
		--permissions cw --start "$sas_start" --expiry "$sas_expiry" \
		-o tsv --only-show-errors)"
	case "$TRIAL_URI" in https://*\?*) ;; *) echo "trial write URL generation failed" >&2; exit 1 ;; esac
	trial_uri_encoded="$(printf '%s' "$TRIAL_URI" | base64 -w0 | tr '+/' '-_' | tr -d '=')"
fi

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
			--parameters "phase=$phase" "candidateSha=$CANDIDATE_SHA" "snapshotSha256=$SNAPSHOT_SHA256" "runtimeSuiteSha256=$runtime_sha256" "snapshotUriEncoded=$snapshot_uri_encoded" "runtimeSuiteUriEncoded=$runtime_uri_encoded" "workload=$RECIPE_WORKLOAD" "trialCursor=${RECIPE_TRIAL_CURSOR:-0}" "trialCount=${RECIPE_TRIAL_COUNT:-0}" "trialUriEncoded=$trial_uri_encoded" \
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

if [ "$RECIPE_WORKLOAD" = trial ]; then
	# The trial's evidence is the harness stderr the guest uploaded, named the way the issue
	# machine's inbox expects (device prefix before "-run", then the cursor span).
	trial_file="evidence/azure-t4-run-${RECIPE_TRIAL_CURSOR}-$((RECIPE_TRIAL_CURSOR + RECIPE_TRIAL_COUNT - 1)).txt"
	az storage blob download \
		--auth-mode login \
		--account-name "$AZURE_STORAGE_ACCOUNT" \
		--container-name "$AZURE_STORAGE_CONTAINER" \
		--name "$TRIAL_BLOB" \
		--file "$trial_file" \
		--only-show-errors --no-progress -o none
	[ -s "$trial_file" ] || { echo "the guest uploaded no trial log" >&2; exit 1; }
	# The snapshot has no history, so the harness base line carries no commit; the candidate stands in.
	awk -v sha="$CANDIDATE_SHA" '{ sub(/^base=commit=[0-9a-f]*/, "base=commit=" sha); print }' "$trial_file" > "$trial_file.tmp" && mv "$trial_file.tmp" "$trial_file"
	grep -E '^TRIAL EXIT' evidence/guest.log
	echo "compositions: $(grep -c '^composition [0-9]*:' "$trial_file" || true), packets: $(grep -c '^RECIPE FAILURE BEGIN$' "$trial_file" || true), file: $trial_file"
	echo "recipe/windows-gpu trial completed for cursors $RECIPE_TRIAL_CURSOR..$((RECIPE_TRIAL_CURSOR + RECIPE_TRIAL_COUNT - 1))"
	exit 0
fi

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
