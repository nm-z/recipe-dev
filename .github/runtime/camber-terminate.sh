#!/usr/bin/env bash
# Terminates a Camber job after normal completion and after cancellation or
# timeout. The caller runs this in an always() step, so a superseded or
# cancelled workflow still releases the GPU allowance instead of leaving a
# billable job running unattended.
set -uo pipefail

job_id="${1:-}"
if [ -z "$job_id" ]; then
	echo "no Camber job id supplied; nothing to terminate"
	exit 0
fi

echo "terminating Camber job $job_id"
if ! camber job stop "$job_id"; then
	if ! camber job cancel "$job_id"; then
		echo "the job was already in a terminal state"
	fi
fi

echo "final state:"
camber job get "$job_id" --format json || echo "could not read the final state"

# Report what this run consumed so the student allowance can be tracked across
# runs rather than discovered when it is exhausted.
echo "remaining allowance:"
camber cloud quota || camber account usage || echo "the installed CLI exposes no quota command"
