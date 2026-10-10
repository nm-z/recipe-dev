# Packed Flash-Next matvec helpers

Root `packed.ptx` contains device functions and tables, with no launch entries. `recipe::PackedMatvec::ptx()` returns the headerless fragment for the native NVIDIA compiler. `PackedMatvec::SHARED_BYTES` is 40960. The weight holder and forward integration remain with cx-flash.

## Device interface

All pointers are u64. Dimensions, selectors, and the receipt are u32. The shared pointer is a generic address obtained from `cvta.shared`, aligned to 16 bytes.

```text
packed_prepare(x_f32, packed_x, scales, k, source_columns, active_source_columns, cta_index, cta_count) -> void
packed_matvec(type, capacity, active_columns, kind, row_lanes, position_mask,
	weights, packed_x, scales, output_f32, k, m, cta_index, cta_count, shared) -> u32
packed_matvec_wide(same parameters) -> u32
```

`packed_matvec` dispatches CTA sizes 64/128/256/512, plus 1024 for capacities 1/2. The wide function accepts 1024 threads and capacities 1/2. Row lanes are 8/16. Capacity is 1/2/4/8. Every thread in the CTA calls together. Cooperating CTAs use consistent arguments and distinct logical ranks.

- For one expert per CTA, pass cta_index=0 and cta_count=1.
- For a matrix shared by the grid, pass the CTA's logical rank and the cooperating CTA count.
- Kind 0 is XMAD; kind 1 is FP32 magic/FADD; kind 2 is a signed-codebook XMAD candidate for IQ2_XS/IQ3_XXS/IQ4_NL only.
- Types are ggml IDs 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K, 17=IQ2_XS, 18=IQ3_XXS, and 20=IQ4_NL.
- Each source column has k/2 adjacent signed-int16 pairs in u32 words and k/32 float2 records `{half-rounded q8 scale, scale*sum(q8)}`.
- Mask 0 uses dense source columns. Otherwise, the c-th active output column reads the c-th set bit's original source column. The caller supplies every referenced source column. Popcount(mask) must equal active_columns.
- Output is compact column-major, m floats per column. Inactive columns remain untouched. Gate/up outputs are compact; reprepare their compact activation product and use mask 0 for down. Apply routing coefficients at combine.
- The caller provides a grid completion barrier after preparation and after matrix writes. The helper performs CTA barriers only. Inputs and output must not overlap.

The receipt is 1 for a valid call and 0 for invalid type, dimensions, capacity, mask, CTA size, or logical CTA rank/count. A zero-active-column call reads no matrix and writes no output. Typed `packed_g_*`/`packed_h_*` functions are also exported for AOT selection; their parameters are `(weights, packed_x, scales, out, k, m, active, mask, cta_index, cta_count, shared)` and return void.

## Build and benchmark

Run on archy. The sole root Makefile uses nvcc 12.9, PTX 7.4, and ptxas 11.4 for sm_52. Build files use disk storage under `/home/nate/codex/cx-fl-kern-build`.

```bash
make all
make inventory samples
make benchmark
```

Set EXPERTS=10 or 16 and ONLY_TYPES='17|18|20' for one expert per CTA, using the corresponding real expert payload slices. Set ROW_LANES=8/16 to probe mappings. The harness rotates at least 32 MiB of weight bundles for both packed and stock paths. Stock uses ggml_mul_mat for dense matrices and ggml_mul_mat_id for expert bundles. Weights remain resident across every candidate for each matrix.

The pinned revision is 766911a6b7369840a91dbcd95f9f997acaab6cd6. All three HTTP 206 headers contain 1224 tensors. Q2_K and Q3_K are absent. F32/BF16 are present but are outside this quantized kernel track. Token embeddings and the large PLE table are lookups and are excluded from the matvec inventory.

## Evidence and open gates

`results-expert*.csv` records every current helper candidate on 19 real matrix shapes and batches 1/2/4/8. All arithmetic checks pass against independent stock CPU decoding with f64 sums on spaced rows. Every output is checked against stock CUDA. `winners.tsv` records the fastest measured preparation-plus-helper path and a conservative stock rate across mappings.

Performance acceptance is not passed. The 85% target is 123.25 GB/s. The current callable schedule remains below stock on Q8_0, several Q6_K shapes, and the IQ expert bundles. The earlier standalone Q4_K/Q5_K ports reached 126-138 GB/s; those row-grid results are not the persistent helper's performance. The 1024-thread helpers also spill, as recorded in `build-handoff.txt`.

Stock IQ2_XS/IQ3_XXS differs from the independently decoded algebra beyond the original 3e-6 float-noise threshold. Both errors are reported. The full-model 10% final-logit gate and 60/80 tok/s checkpoints are not measured by this harness.

The signed codebooks add 1605632 bytes per module, plus the canonical tables. They are format dictionaries; tensor weights remain in their original GGUF layout. Follow-up reductions of spills and codebook traffic are separate, unmeasured work after this stable handoff. The resident full-model load now occupies die 1, so no additional standalone GPU suite is queued from this snapshot.

The shipped library contains only 43 unique measured winner functions. Unlisted kind/CTA/row-lane configurations return 0. Use winners.tsv for the AOT choice; the full candidate generator and benchmark source remain in the evidence directory. The 75,037-line, 2.4 MiB library replaces the 1.1-million-line artifact.
