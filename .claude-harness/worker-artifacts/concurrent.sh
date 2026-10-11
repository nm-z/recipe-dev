#!/bin/bash
# Independent public-CLI submissions, run concurrently, each logged separately.
set -u
ROOT=/home/nate/.local/share/claude-harness/worktrees/claude-harness-14
A=$ROOT/.claude-harness/worker-artifacts
BIN=$A/target/release/recipe
cd "$ROOT" || exit 1
rm -f "$A"/run-*.log "$A"/times.txt "$A"/smi.csv "$A"/ps.txt
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,used_memory --format=csv -l 1 > "$A/smi.csv" 2>&1 &
SMI=$!
( while :; do date +%s.%N >> "$A/ps.txt"; ps -eo pid,ppid,rss,cmd | rg 'recipe-(script|job)' | rg -v ' rg ' >> "$A/ps.txt"; sleep 1; done ) &
PS=$!
PIDS=""
for i in 1 2 3 4 5 6; do
	(
		s=$(date +%s.%N)
		"$BIN" .claude-harness/worker-artifacts/job.rs > "$A/run-$i.log" 2>&1
		c=$?
		e=$(date +%s.%N)
		echo "run $i start $s end $e exit $c" >> "$A/times.txt"
	) &
	PIDS="$PIDS $!"
done
for p in $PIDS; do wait "$p"; done
kill "$SMI" "$PS" 2>/dev/null
echo DONE
