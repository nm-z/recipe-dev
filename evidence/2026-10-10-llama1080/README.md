# Llama CPU logit comparison

Verification sources for #1080, using the #1044 RCPREF01 harness. The runtime and default precision correction belongs to #1058.

`rcpref_compare.c` reads the full logits that Recipe and llama.cpp record. It computes each side's argmax directly, reports the mean, nearest-rank p99, and maximum of the per-position error ratio in each bin, and can reject a positive trend. The final verdict includes every selected check.

The error ratio at a position is the maximum absolute logit difference divided by the reference vocabulary's maximum-minus-minimum logit range. The slope is the least-squares slope of that ratio against position, in percentage points per 1,000 steps.

## Build

Run on Archy, after acquiring a build slot:

```bash
flock --close /home/nate/work/.build-slot-1.lock \
	nice -n 19 cc -std=c11 -O2 -Wall -Wextra -Werror \
	evidence/2026-10-10-llama1080/rcpref_compare.c -lm \
	-o /home/nate/codex/cx-1080-tools/rcpref-compare
```

The output directory must exist. This tool reads the recorded public observations.

## Compare

```bash
flock --close /home/nate/work/.build-slot-1.lock \
	nice -n 19 /home/nate/codex/cx-1080-tools/rcpref-compare \
	llama.rcpref recipe.rcpref --tsv per-position.tsv --limit 0.10 --bin 256 --max-slope 0
```

`--max-slope 0` requires a nonpositive trend. Review the bin means, p99, maxima, and slope to assess flatness. Omitting the option retains the bound and argmax checks. A selected growth failure produces a final `verdict FAIL` and exit status 1 even when the 10% bound passes.

## Model acceptance

Use the canonical `/home/nate/codex/1044-jev/llama3.template.rs` definition through `recipe run`, with the default precision profile, CPU selection, and the raw prompt `The capital of France is`. Record the existing public `report.input_ids` observation and compare it to llama.cpp's tokenization. The expected IDs are `[128000, 791, 6864, 315, 9822, 374]`.

Build Recipe on Archy into `/home/nate/work/target-cx-recipe1080` with the release profile, `CARGO_BUILD_JOBS=4`, `nice -n 19`, `CUDA_VISIBLE_DEVICES=`, and `.build-slot-1.lock`. Set `RECIPE_FORCE_CPU=1` and select `--device cpu` for execution. The reference is llama.cpp b10705, commit `2578138397d7b422bb0e160efdd429976c55fb55`, with f32 KV and Recipe's winners as the teacher-forced input.

Run Q8_0 for 4,096 steps and Q4_K_M for 1,024 steps. Check the live prompt IDs, position counts, actual argmax agreement, every position's 10% bound, and the reported mean, p99, maximum, and slope. Keep raw RCPREF01 files outside this directory. Each Q8_0 file occupies 4,202,741,784 bytes; retain only the small comparison reports here.

Final #1080 acceptance requires the #1058 shared correction. Shared-target Cargo results are invalid. No Llama model acceptance result is recorded in this directory yet.
