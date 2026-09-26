#!/usr/bin/env bash
# Terminates the EC2 worker a run created and deletes its security group and
# key pair, found by the recipe-worker tag. Workers older than three hours are
# terminated too, so a run that died before its cleanup is collected later.
set -euo pipefail
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
export AWS_PAGER=""
RUN_ID="${RUN_ID:-${GITHUB_RUN_ID:-manual}}"
RUN_ATTEMPT="${RUN_ATTEMPT:-${GITHUB_RUN_ATTEMPT:-1}}"
WORKER="recipe-lnv-${RUN_ID}-${RUN_ATTEMPT}"

echo "== terminating $WORKER =="
instances="$(aws ec2 describe-instances \
	--filters "Name=tag:recipe-worker,Values=$WORKER" Name=instance-state-name,Values=pending,running,stopping,stopped \
	--query 'Reservations[].Instances[].InstanceId' --output text)"
if [ -n "$instances" ]; then
	# shellcheck disable=SC2086 # one ID per word
	aws ec2 terminate-instances --instance-ids $instances > /dev/null
	# shellcheck disable=SC2086
	aws ec2 wait instance-terminated --instance-ids $instances
	echo "terminated $instances"
else
	echo "no instance remains for $WORKER"
fi

echo "== deleting the security group and key pair =="
groups="$(aws ec2 describe-security-groups --filters "Name=tag:recipe-worker,Values=$WORKER" --query 'SecurityGroups[].GroupId' --output text)"
for group in $groups; do
	# The network interface can outlive the instance by a few seconds.
	for attempt in 1 2 3 4 5 6; do
		if aws ec2 delete-security-group --group-id "$group" 2> /dev/null; then
			echo "deleted security group $group"
			break
		fi
		[ "$attempt" -lt 6 ] || { echo "security group $group is still in use" >&2; exit 1; }
		sleep 10
	done
done
if aws ec2 describe-key-pairs --key-names "$WORKER" > /dev/null 2>&1; then
	aws ec2 delete-key-pair --key-name "$WORKER"
	echo "deleted key pair $WORKER"
fi
rm -f "${RUNNER_TEMP:-/tmp}/$WORKER.key" "${RUNNER_TEMP:-/tmp}/$WORKER.key.pub"

echo "== expiry watchdog =="
cutoff="$(date -u -d '3 hours ago' +%Y-%m-%dT%H:%M:%S)"
stale="$(aws ec2 describe-instances \
	--filters "Name=tag:recipe-worker,Values=recipe-lnv-*" Name=instance-state-name,Values=pending,running,stopping,stopped \
	--query "Reservations[].Instances[?LaunchTime<'$cutoff'].InstanceId" --output text)"
if [ -n "$stale" ]; then
	# shellcheck disable=SC2086
	aws ec2 terminate-instances --instance-ids $stale > /dev/null
	echo "terminated stale workers $stale"
else
	echo "no stale workers"
fi
echo "cleanup complete"
