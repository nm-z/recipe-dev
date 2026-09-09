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
<<<<<<< origin/gguf/670
=======
		nsg) az network nsg list --resource-group "$GROUP" --query "$query" -o tsv --only-show-errors ;;
>>>>>>> origin/minimal
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
<<<<<<< origin/gguf/670
=======
		nsg) az network nsg delete --resource-group "$GROUP" --name "$name" --only-show-errors ;;
>>>>>>> origin/minimal
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

<<<<<<< origin/gguf/670
	for kind in disk nic public-ip; do
=======
	for kind in disk nic public-ip nsg; do
>>>>>>> origin/minimal
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

echo "== expiry watchdog =="
cutoff="$(date -u -d "${MAX_AGE_HOURS} hours ago" +%Y-%m-%dT%H:%M:%SZ)"
echo "removing recipe-wgpu-* workers created before $cutoff"
stale="$(az vm list --resource-group "$GROUP" --query "[?starts_with(name,'recipe-wgpu-')].name" -o tsv --only-show-errors)" || {
	echo "could not list workers for the expiry watchdog" >&2
	stale=""
	cleanup_status=1
}
for worker in $stale; do
	if ! created="$(az resource show --resource-group "$GROUP" --name "$worker" --resource-type Microsoft.Compute/virtualMachines --query "systemData.createdAt" -o tsv --only-show-errors)"; then
		echo "could not read creation time for $worker" >&2
		cleanup_status=1
		continue
	fi
	if [ -n "$created" ] && [[ "$created" < "$cutoff" ]]; then
		echo "watchdog removing stale worker $worker created $created"
		if ! remove_worker "$worker"; then
			cleanup_status=1
		fi
	fi
done

<<<<<<< origin/gguf/670
=======
echo "== orphan resource watchdog =="
for kind in nic disk public-ip nsg; do
	resources="$(list_resources "$kind" "recipe-wgpu-")" || {
		echo "could not list runtime $kind resources" >&2
		cleanup_status=1
		continue
	}
	while IFS= read -r resource; do
		if [[ ! "$resource" =~ ^(recipe-wgpu-[0-9]+-[0-9]+) ]]; then
			continue
		fi
		worker="${BASH_REMATCH[1]}"
		if ! current="$(list_vm "$worker")"; then
			echo "could not determine whether $resource is orphaned" >&2
			cleanup_status=1
			continue
		fi
		if [ -z "$current" ]; then
			echo "removing orphan $kind $resource"
			if ! delete_resource "$kind" "$resource"; then
				cleanup_status=1
			fi
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
		cleanup_status=1
	}
	while IFS= read -r blob; do
		if [[ ! "$blob" =~ ^runtime/windows/([0-9]+)-([0-9]+)/(snapshot\.tar\.gz|runtime-suite\.tar\.gz)$ ]]; then
			continue
		fi
		prior_run="${BASH_REMATCH[1]}"
		if ! prior_status="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$prior_run" --jq .status 2>/dev/null)"; then
			echo "could not confirm the owner of prior transfer $blob" >&2
			cleanup_status=1
			continue
		fi
		if [ "$prior_status" = "completed" ]; then
			echo "deleting terminal-run transfer $blob"
			if ! az storage blob delete \
				--auth-mode login \
				--account-name "$AZURE_STORAGE_ACCOUNT" \
				--container-name "$AZURE_STORAGE_CONTAINER" \
				--name "$blob" \
				--only-show-errors -o none; then
				cleanup_status=1
			fi
		fi
	done <<< "$blobs"
else
	echo "terminal transfer recovery is unavailable"
fi

>>>>>>> origin/minimal
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

if [ "$cleanup_status" -ne 0 ]; then
	echo "cleanup failed for $WORKER" >&2
	exit 1
fi
echo "cleanup complete"
