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

if ! az account show -o none; then
	echo "not authenticated to Azure; nothing to clean up"
	exit 0
fi

echo "== deleting $WORKER from $GROUP =="
az vm delete --resource-group "$GROUP" --name "$WORKER" --yes --force-deletion true --only-show-errors || echo "the worker was already gone"

echo "== removing disks, interfaces and addresses it left behind =="
for kind in disk nic public-ip; do
	names="$(az "$kind" list --resource-group "$GROUP" --query "[?starts_with(name, '${WORKER}')].name" -o tsv --only-show-errors || true)"
	for resource in $names; do
		echo "removing $kind $resource"
		az "$kind" delete --resource-group "$GROUP" --name "$resource" --yes --only-show-errors || \
			az "$kind" delete --resource-group "$GROUP" --name "$resource" --only-show-errors || \
			echo "could not remove $kind $resource"
	done
done

echo "== expiry watchdog =="
# Independent of this run: a controller that died without cleaning up must not
# leave a GPU VM running and billing.
cutoff="$(date -u -d "${MAX_AGE_HOURS} hours ago" +%Y-%m-%dT%H:%M:%SZ)"
echo "removing recipe-wgpu-* workers created before $cutoff"
stale="$(az vm list --resource-group "$GROUP" --query "[?starts_with(name,'recipe-wgpu-')].name" -o tsv --only-show-errors || true)"
for worker in $stale; do
	created="$(az vm show --resource-group "$GROUP" --name "$worker" --query "timeCreated" -o tsv --only-show-errors || true)"
	if [ -n "$created" ] && [[ "$created" < "$cutoff" ]]; then
		echo "watchdog removing stale worker $worker created $created"
		az vm delete --resource-group "$GROUP" --name "$worker" --yes --force-deletion true --only-show-errors || true
	fi
done

echo "== residual billable resources in $GROUP =="
# Anything still listed here is still costing money; it is printed so the run
# log carries the evidence rather than leaving it to be discovered on a bill.
az resource list --resource-group "$GROUP" --query "[].{name:name, type:type}" -o table --only-show-errors || echo "could not list resources"
echo "cleanup complete"
