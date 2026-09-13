#!/usr/bin/env bash
# Reads the final Camber state and removes the per-run Stash directory after a
# terminal result. The current CLI has no stop or cancel command.
set -u

job_id="${1:-}"
if [ -z "$job_id" ]; then
	echo "no Camber job ID supplied"
	exit 0
fi

camber_bin_dir="${HOME}/.camber/bin"
export PATH="${camber_bin_dir}:${PATH}"
if ! command -v camber >/dev/null 2>&1; then
	curl -fsSL https://cli.cambercloud.com/install-v2.sh | bash || {
		echo "Camber CLI is unavailable; leaving the run directory" >&2
		exit 0
	}
fi
export PATH="${camber_bin_dir}:${PATH}"

final_json=""
if final_json="$(camber job get "$job_id" --output json 2>&1)"; then
	printf '%s\n' "$final_json"
else
	echo "could not read Camber job $job_id" >&2
	printf '%s\n' "$final_json" >&2
	exit 0
fi

state="$(printf '%s' "$final_json" | jq -er '(.job_status // .status // .state // "") | tostring | ascii_upcase' 2>/dev/null || true)"
stash_root=""
if [ -f camber-stash-root ]; then
	stash_root="$(sed -n '1p' camber-stash-root)"
fi

case "$state" in
	COMPLETED|SUCCEEDED|SUCCESS|FINISHED|FAILED|ERROR|CANCELLED|CANCELED|TERMINATED)
		if [ -n "$stash_root" ]; then
			echo "removing the terminal run directory from Stash"
			camber stash rm -rf "$stash_root" || echo "could not remove the Stash run directory" >&2
		fi
		;;
	*)
		echo "Camber job $job_id is still ${state:-unknown}; the current CLI has no stop or cancel command" >&2
		if [ -n "$stash_root" ]; then
			# The worker returns at once when it finds this marker, so a job that
			# is still queued frees its place as soon as it starts. The inputs
			# stay for a job that is already running.
			printf 'run %s attempt %s at %s\n' "${GITHUB_RUN_ID:-manual}" "${GITHUB_RUN_ATTEMPT:-1}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > abandoned
			if camber stash cp abandoned "$stash_root/abandoned"; then
				echo "marked $stash_root abandoned; the worker exits at once when it starts" >&2
			else
				echo "could not mark $stash_root abandoned" >&2
			fi
			echo "leaving $stash_root so an active job keeps its inputs" >&2
		fi
		;;
esac
