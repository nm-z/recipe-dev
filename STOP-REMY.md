# Remy stop for engi disk clone

Stopped by the CEO's October 8 order. The goal is paused until engi is back.

- Recipe implementation stops at `d3661cef` on branch `remy/first-order-gpu`; recipe-dev #994 is still open. No toolchain migration was started in this checkout.
- Rates integration lives in `/home/nate/ceo-orders/rates/`. It selects the 26-line recipe knob-RAT program and preserves cached NNLS diagnostics alongside. Its GPU preflight rejects the current ROCm backend. GPU fitting, rates, and fit time are not measured.
- Pacing integration lives in `/home/nate/ceo-orders/codex-pace/`. Data extraction, objective, hourly fit wiring, shared GPU locking, and non-training checks are prepared. Learned outputs remain unset. The last queued output labels 1.1 pp/h as the CEO interim order. Nine same-bucket full-refill candidates support an explicitly labeled prediction assumption; no banked redemption was performed.
- Both GPU fits require #994 to land and a compliant backend to be installed in the configured checkout. Pacing then requires fresh measured GPU evidence from the rates publication.
- `rates-fit.timer`, `rates-fit.service`, `codex-pace.timer`, and `codex-pace.service` are stopped. No task build or fitting command remains running.

`remy-stop-source.tar.gz` preserves the authored rates and pacing scripts, their systemd units, and the tracked core source/handoff/status files. It contains no rollout corpus, account-response archive, credentials, private quota data, or generated weights. The standalone core repository has no configured remote; its source snapshot is included here so the work is preserved on this branch's configured origin.

After engi returns, inspect the restored files and #994 before resuming. Restore the runtime scripts to their documented directories, reload the user units if needed, and resume only within the applicable GPU/toolchain rules. Do not call the reset-credit consume API; this job publishes predictions for the CEO to act on.
