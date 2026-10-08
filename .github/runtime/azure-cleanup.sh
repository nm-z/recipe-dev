#!/usr/bin/env bash
# Removes the per-run Windows GPU worker and every billable resource it
# created, then sweeps anything an earlier controller left behind.
#
# The caller runs this in an always() step, so cancellation and timeout also
# release resources. Budget alerts are not a spending cap; this script and the
# expiry watchdog below are what actually stop the meter.
set -uo pipefail

GROUP="${AZURE_RESOURCE_GROUP:-recipe-ci}"
RUN="${RUN_ID:-${GITHUB_RUN_ID:-}}"
ATTEMPT="${RUN_ATTEMPT:-${GITHUB_RUN_ATTEMPT:-}}"
MAX_AGE_HOURS="${AZURE_MAX_AGE_HOURS:-3}"

if [[ ! "$RUN" =~ ^[0-9]+$ ]] || [[ ! "$ATTEMPT" =~ ^[0-9]+$ ]]; then
	echo "RUN_ID/RUN_ATTEMPT or GITHUB_RUN_ID/GITHUB_RUN_ATTEMPT must identify the worker" >&2
	exit 2
fi

WORKER="recipe-wgpu-${RUN}-${ATTEMPT}"
cleanup_status=0

if ! az account show -o none; then
	echo "Azure authentication is unavailable; cleanup cannot verify deletion" >&2
	exit 1
fi

list_vm() {
	local name="$1"
	az vm list \
		--resource-group "$GROUP" \
		--query "[?name=='$name'].name" \
		-o tsv \
		--only-show-errors
}

list_resources() {
	local kind="$1"
	local prefix="$2"
	local query="[?starts_with(name, '$prefix')].name"
	case "$kind" in
		disk) az disk list --resource-group "$GROUP" --query "$query" -o tsv --only-show-errors ;;
		nic) az network nic list --resource-group "$GROUP" --query "$query" -o tsv --only-show-errors ;;
		public-ip) az network public-ip list --resource-group "$GROUP" --query "$query" -o tsv --only-show-errors ;;
		nsg) az network nsg list --resource-group "$GROUP" --query "$query" -o tsv --only-show-errors ;;
		*) echo "unknown Azure resource kind: $kind" >&2; return 2 ;;
	esac
}

delete_resource() {
	local kind="$1"
	local name="$2"
	case "$kind" in
		disk) az disk delete --resource-group "$GROUP" --name "$name" --yes --only-show-errors ;;
		nic) az network nic delete --resource-group "$GROUP" --name "$name" --only-show-errors ;;
		public-ip) az network public-ip delete --resource-group "$GROUP" --name "$name" --only-show-errors ;;
		nsg) az network nsg delete --resource-group "$GROUP" --name "$name" --only-show-errors ;;
		*) echo "unknown Azure resource kind: $kind" >&2; return 2 ;;
	esac
}

wait_for_vm_absent() {
	local name="$1"
	local attempt remaining
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
		if ! remaining="$(list_vm "$name")"; then
			echo "could not read back VM $name" >&2
			return 1
		fi
		if [ -z "$remaining" ]; then
			return 0
		fi
		sleep 5
	done
	echo "VM $name is still present after deletion" >&2
	return 1
}

wait_for_resources_absent() {
	local kind="$1"
	local prefix="$2"
	local attempt remaining
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
		if ! remaining="$(list_resources "$kind" "$prefix")"; then
			echo "could not read back $kind resources for $prefix" >&2
			return 1
		fi
		if [ -z "$remaining" ]; then
			return 0
		fi
		sleep 5
	done
	echo "$kind resources remain for $prefix: $remaining" >&2
	return 1
}

remove_worker() {
	local name="$1"
	local current resource
	local status=0
	if ! current="$(list_vm "$name")"; then
		echo "could not determine whether VM $name exists" >&2
		return 1
	fi
	if [ -n "$current" ]; then
		echo "deleting VM $name from $GROUP"
		if ! az vm delete --resource-group "$GROUP" --name "$name" --yes --force-deletion true --only-show-errors; then
			echo "VM deletion failed for $name" >&2
			status=1
		fi
	else
		echo "VM $name was already absent"
	fi
	if ! wait_for_vm_absent "$name"; then
		status=1
	fi

	for kind in disk nic public-ip nsg; do
		local resources
		if ! resources="$(list_resources "$kind" "$name")"; then
			status=1
			continue
		fi
		while IFS= read -r resource; do
			if [ -z "$resource" ]; then
				continue
			fi
			echo "deleting $kind $resource"
			if ! delete_resource "$kind" "$resource"; then
				echo "could not delete $kind $resource" >&2
				status=1
			fi
		done <<< "$resources"
		if ! wait_for_resources_absent "$kind" "$name"; then
			status=1
		fi
	done
	return "$status"
}

echo "== deleting $WORKER from $GROUP =="
if ! remove_worker "$WORKER"; then
	cleanup_status=1
fi

# The sweeps below act on other runs' leftovers and race other controllers
# doing the same work (a reclaim in azure-run.sh, a neighbouring cleanup step),
# so a resource that vanishes under them is not this run's failure. They
# report through watchdog_status; only this run's own worker, blobs and
# residual check decide the exit code.
watchdog_status=0

# A worker or blob belongs to a terminal run when GitHub reports the run
# completed, or when it names an earlier attempt of a run whose latest attempt
# is newer (a re-run starts only after the previous attempt finished, while
# the run's status reports the latest attempt).
declare -A run_state=()
owner_is_terminal() {
	local run="$1" attempt="$2" state status latest
	if ! command -v gh >/dev/null || [ -z "${GH_TOKEN:-}" ] || [ -z "${GITHUB_REPOSITORY:-}" ]; then
		return 1
	fi
	state="${run_state[$run]:-}"
	if [ -n "$state" ]; then
		latest="${state#* }"
		# A newer attempt than the cached one started during this sweep.
		if [ "$attempt" -gt "$latest" ]; then
			state=""
		fi
	fi
	if [ -z "$state" ]; then
		if ! state="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run" --jq '.status + " " + (.run_attempt|tostring)' 2>/dev/null)"; then
			return 1
		fi
		run_state[$run]="$state"
	fi
	status="${state% *}"
	latest="${state#* }"
	[ "$attempt" -lt "$latest" ] || [ "$status" = completed ]
}

echo "== expiry watchdog =="
cutoff="$(date -u -d "${MAX_AGE_HOURS} hours ago" +%Y-%m-%dT%H:%M:%SZ)"
echo "removing recipe-wgpu-* workers created before $cutoff"
stale="$(az vm list --resource-group "$GROUP" --query "[?starts_with(name,'recipe-wgpu-')].name" -o tsv --only-show-errors)" || {
	echo "could not list workers for the expiry watchdog" >&2
	stale=""
	watchdog_status=1
}
for worker in $stale; do
	if [ "$worker" = "$WORKER" ]; then
		continue
	fi
	if ! created="$(az resource show --resource-group "$GROUP" --name "$worker" --resource-type Microsoft.Compute/virtualMachines --query "systemData.createdAt" -o tsv --only-show-errors)"; then
		echo "could not read creation time for $worker; another controller may have removed it" >&2
		watchdog_status=1
		continue
	fi
	if [ -n "$created" ] && [[ "$created" < "$cutoff" ]]; then
		echo "watchdog removing stale worker $worker created $created"
		if ! remove_worker "$worker"; then
			watchdog_status=1
		fi
	fi
done

echo "== orphan resource watchdog =="
# Workers of several runs coexist, and az vm create makes the public IP, NSG
# and NIC before the VM object exists. A resource without a VM is an orphan
# only once its owning run attempt is terminal; a live run's half-built
# worker is left alone.
for kind in nic disk public-ip nsg; do
	resources="$(list_resources "$kind" "recipe-wgpu-")" || {
		echo "could not list runtime $kind resources" >&2
		watchdog_status=1
		continue
	}
	while IFS= read -r resource; do
		if [[ ! "$resource" =~ ^(recipe-wgpu-([0-9]+)-([0-9]+)) ]]; then
			continue
		fi
		worker="${BASH_REMATCH[1]}"
		owner_run="${BASH_REMATCH[2]}"
		owner_attempt="${BASH_REMATCH[3]}"
		if [ "$worker" = "$WORKER" ]; then
			continue
		fi
		if ! current="$(list_vm "$worker")"; then
			echo "could not determine whether $resource is orphaned" >&2
			watchdog_status=1
			continue
		fi
		if [ -n "$current" ]; then
			continue
		fi
		if ! owner_is_terminal "$owner_run" "$owner_attempt"; then
			echo "leaving $kind $resource to run $owner_run attempt $owner_attempt"
			continue
		fi
		echo "removing orphan $kind $resource"
		if ! delete_resource "$kind" "$resource"; then
			echo "could not remove orphan $kind $resource; another controller may have removed it" >&2
			watchdog_status=1
		fi
	done <<< "$resources"
done

echo "== deleting private transfer blobs =="
if [ -n "${AZURE_STORAGE_ACCOUNT:-}" ] && [ -n "${AZURE_STORAGE_CONTAINER:-}" ]; then
	for blob in \
		"runtime/windows/${RUN}-${ATTEMPT}/snapshot.tar.gz" \
		"runtime/windows/${RUN}-${ATTEMPT}/runtime-suite.tar.gz"; do
		if ! blob_exists="$(az storage blob exists \
			--auth-mode login \
			--account-name "$AZURE_STORAGE_ACCOUNT" \
			--container-name "$AZURE_STORAGE_CONTAINER" \
			--name "$blob" \
			--query exists -o tsv --only-show-errors)"; then
			echo "could not determine whether $blob exists" >&2
			cleanup_status=1
			continue
		fi
		if [ "$blob_exists" = "true" ]; then
			echo "deleting $blob"
			if ! az storage blob delete \
				--auth-mode login \
				--account-name "$AZURE_STORAGE_ACCOUNT" \
				--container-name "$AZURE_STORAGE_CONTAINER" \
				--name "$blob" \
				--only-show-errors -o none; then
				cleanup_status=1
			fi
		fi
	done
else
	echo "storage transfer is not configured"
fi

echo "== terminal transfer watchdog =="
if [ -n "${AZURE_STORAGE_ACCOUNT:-}" ] && [ -n "${AZURE_STORAGE_CONTAINER:-}" ] && command -v gh >/dev/null && [ -n "${GH_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
	blobs="$(az storage blob list \
		--auth-mode login \
		--account-name "$AZURE_STORAGE_ACCOUNT" \
		--container-name "$AZURE_STORAGE_CONTAINER" \
		--prefix "runtime/windows/" \
		--query "[].name" -o tsv --only-show-errors)" || {
		echo "could not list prior runtime transfer blobs" >&2
		blobs=""
		watchdog_status=1
	}
	while IFS= read -r blob; do
		if [[ ! "$blob" =~ ^runtime/windows/([0-9]+)-([0-9]+)/(snapshot\.tar\.gz|runtime-suite\.tar\.gz)$ ]]; then
			continue
		fi
		prior_run="${BASH_REMATCH[1]}"
		prior_attempt="${BASH_REMATCH[2]}"
		if [ "$prior_run" = "$RUN" ] && [ "$prior_attempt" = "$ATTEMPT" ]; then
			continue
		fi
		if ! owner_is_terminal "$prior_run" "$prior_attempt"; then
			continue
		fi
		echo "deleting terminal-run transfer $blob"
		if ! az storage blob delete \
			--auth-mode login \
			--account-name "$AZURE_STORAGE_ACCOUNT" \
			--container-name "$AZURE_STORAGE_CONTAINER" \
			--name "$blob" \
			--only-show-errors -o none; then
			echo "could not delete $blob; another controller may have removed it" >&2
			watchdog_status=1
		fi
	done <<< "$blobs"
else
	echo "terminal transfer recovery is unavailable"
fi

echo "== residual resources for $WORKER =="
residual="$(az resource list --resource-group "$GROUP" --query "[?starts_with(name,'$WORKER')].{name:name, type:type}" -o table --only-show-errors)" || {
	echo "could not read back residual resources for $WORKER" >&2
	residual="unknown"
	cleanup_status=1
}
if [ -n "$residual" ] && [ "$residual" != "unknown" ]; then
	echo "$residual" >&2
	echo "residual resources remain for $WORKER" >&2
	cleanup_status=1
else
	echo "no residual resources remain for $WORKER"
fi

if [ "$watchdog_status" -ne 0 ]; then
	echo "the watchdog sweeps over other runs' leftovers did not fully complete; the next cleanup repeats them" >&2
fi
if [ "$cleanup_status" -ne 0 ]; then
	echo "cleanup failed for $WORKER" >&2
	exit 1
fi
echo "cleanup complete"
