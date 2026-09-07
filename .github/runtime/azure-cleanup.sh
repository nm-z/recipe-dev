#!/usr/bin/env bash
# Removes the per-run Windows GPU worker and every billable resource it
# created, then sweeps anything an earlier controller left behind.
#
# The caller runs this in an always() step, so cancellation and timeout also
# release resources. Budget alerts are not a spending cap; this script and the
# expiry watchdog below are what actually stop the meter.
set -uo pipefail

GROUP="${AZURE_RESOURCE_GROUP:-recipe-ci}"
RUN="${RUN_ID:-unknown}"
ATTEMPT="${RUN_ATTEMPT:-1}"
WORKER="recipe-wgpu-${RUN}-${ATTEMPT}"
# Nothing in this resource group should outlive a workflow run by more than the
# longest legitimate job.
MAX_AGE_HOURS="${AZURE_MAX_AGE_HOURS:-3}"
cleanup_failed=0

if ! az account show -o none; then
	echo "not authenticated to Azure; cleanup cannot verify billable resources" >&2
	exit 1
fi

echo "== deleting $WORKER from $GROUP =="
if az vm show --resource-group "$GROUP" --name "$WORKER" --only-show-errors -o none; then
	if ! az vm delete --resource-group "$GROUP" --name "$WORKER" --yes --force-deletion true --only-show-errors; then
		echo "the worker deletion command failed" >&2
	fi
else
	echo "the worker was already gone"
fi
if az vm show --resource-group "$GROUP" --name "$WORKER" --only-show-errors -o none; then
	echo "the worker still exists after cleanup" >&2
	cleanup_failed=1
fi
if ! remaining_worker="$(az vm list --resource-group "$GROUP" --query "[?name=='${WORKER}'].name" -o tsv --only-show-errors)"; then
	echo "could not verify the worker readback" >&2
	cleanup_failed=1
elif [ -n "$remaining_worker" ]; then
	echo "worker readback still lists $remaining_worker" >&2
	cleanup_failed=1
fi

echo "== removing disks, interfaces and addresses it left behind =="
for kind in disk nic public-ip; do
	if ! names="$(az "$kind" list --resource-group "$GROUP" --query "[?starts_with(name, '${WORKER}')].name" -o tsv --only-show-errors)"; then
		echo "could not list $kind resources" >&2
		cleanup_failed=1
		continue
	fi
	for resource in $names; do
		echo "removing $kind $resource"
		if ! az "$kind" delete --resource-group "$GROUP" --name "$resource" --yes --only-show-errors; then
			if ! az "$kind" delete --resource-group "$GROUP" --name "$resource" --only-show-errors; then
				echo "could not remove $kind $resource" >&2
				cleanup_failed=1
			fi
		fi
	done
	if ! residual="$(az "$kind" list --resource-group "$GROUP" --query "[?starts_with(name, '${WORKER}')].name" -o tsv --only-show-errors)"; then
		echo "could not verify $kind cleanup" >&2
		cleanup_failed=1
	elif [ -n "$residual" ]; then
		echo "residual $kind resources: $residual" >&2
		cleanup_failed=1
	fi
done

echo "== expiry watchdog =="
# Independent of this run: a controller that died without cleaning up must not
# leave a GPU VM running and billing.
if ! cutoff="$(date -u -d "${MAX_AGE_HOURS} hours ago" +%Y-%m-%dT%H:%M:%SZ)"; then
	echo "could not compute the expiry cutoff" >&2
	cleanup_failed=1
	cutoff=""
fi
echo "removing recipe-wgpu-* workers created before $cutoff"
if ! stale="$(az vm list --resource-group "$GROUP" --query "[?starts_with(name,'recipe-wgpu-')].name" -o tsv --only-show-errors)"; then
	echo "could not list worker VMs" >&2
	cleanup_failed=1
	stale=""
fi
for worker in $stale; do
	if ! created="$(az vm show --resource-group "$GROUP" --name "$worker" --query "timeCreated" -o tsv --only-show-errors)"; then
		echo "could not read creation time for $worker" >&2
		cleanup_failed=1
		continue
	fi
	if [ -n "$created" ] && [[ "$created" < "$cutoff" ]]; then
		echo "watchdog removing stale worker $worker created $created"
		if ! az vm delete --resource-group "$GROUP" --name "$worker" --yes --force-deletion true --only-show-errors; then
			echo "watchdog deletion failed for $worker" >&2
		fi
		if az vm show --resource-group "$GROUP" --name "$worker" --only-show-errors -o none; then
			echo "watchdog worker remains: $worker" >&2
			cleanup_failed=1
		fi
	fi
done

echo "== residual billable resources in $GROUP =="
# Anything still listed here is still costing money; it is printed so the run
# log carries the evidence rather than leaving it to be discovered on a bill.
if ! residual="$(az resource list --resource-group "$GROUP" --query "[].{name:name, type:type}" -o tsv --only-show-errors)"; then
	echo "could not list residual resources" >&2
	cleanup_failed=1
elif [ -n "$residual" ]; then
	echo "$residual"
	echo "residual billable resources remain in $GROUP" >&2
	cleanup_failed=1
else
	echo "no residual resources"
fi
if [ "$cleanup_failed" -ne 0 ]; then
	echo "cleanup failed closed"
	exit 1
fi
echo "cleanup complete"
