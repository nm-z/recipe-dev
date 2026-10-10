# Packed Flash-Next matvec helpers

Root `packed.ptx` contains device functions and tables, with no launch entries. `recipe::PackedMatvec::ptx()` returns the headerless fragment for the native NVIDIA compiler. `PackedMatvec::SHARED_BYTES` is 40960. The weight holder and forward integration remain with cx-flash.

## Device interface

All pointers are u64. Dimensions, selectors, and the receipt are u32. The module declares `.extern .shared .align 16 .b8 scratch[]`; the caller launches with 40960 dynamic shared bytes. The PTX interface has no shared-pointer parameter.

```text
packed_prepare(x_f32, packed_x, scales, k, source_columns, active_source_columns, cta_index, cta_count) -> void
packed_matvec(type, capacity, active_columns, kind, row_lanes, position_mask,
	weights, packed_x, scales, output_f32, k, m, cta_index, cta_count) -> u32
packed_matvec_wide(same parameters) -> u32
```

`packed_matvec` dispatches only the measured configurations in `winners.tsv`, `fixed-cta.tsv`, and `cta-256.tsv`. Every format and capacity includes both 256-thread and 512-thread configurations for a persistent launch. The wide function accepts 1024 threads and capacities 1/2. Row lanes are 8/16. Capacity is 1/2/4/8. Every thread in the CTA calls together. Cooperating CTAs use consistent arguments and distinct logical ranks.

- Split expert rows across the assigned logical CTA group; pass its rank as cta_index and its size as cta_count. Allocate the die's 16 CTAs across selected experts in proportion to weight bytes.
- For a matrix shared by the grid, pass the CTA's logical rank and the cooperating CTA count.
- Kind 0 is XMAD; kind 1 is FP32 magic/FADD; kind 2 is a signed-codebook XMAD candidate for IQ2_XS/IQ3_XXS/IQ4_NL only.
- Types are ggml IDs 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K, 17=IQ2_XS, 18=IQ3_XXS, and 20=IQ4_NL.
- Each source column has k/2 adjacent signed-int16 pairs in u32 words and k/32 float2 records `{half-rounded q8 scale, scale*sum(q8)}`.
- Mask 0 uses dense source columns. Otherwise, the c-th active output column reads the c-th set bit's original source column. The caller supplies every referenced source column. Popcount(mask) must equal active_columns.
- Output is compact column-major, m floats per column. Inactive columns remain untouched. Gate/up outputs are compact; reprepare their compact activation product and use mask 0 for down. Apply routing coefficients at combine.
- The caller provides a grid completion barrier after preparation and after matrix writes. The helper performs CTA barriers only. Inputs and output must not overlap.

The receipt is 1 for a valid call and 0 for invalid type, dimensions, capacity, mask, CTA size, or logical CTA rank/count. A zero-active-column call reads no matrix and writes no output. Typed `packed_g_*`/`packed_h_*` functions are also exported for AOT selection; their parameters are `(weights, packed_x, scales, out, k, m, active, mask, cta_index, cta_count)` and return void.

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

The shipped winners use the referenced canonical and signed format tables; unreferenced codebooks are removed. Tensor weights remain in their original GGUF layout. Follow-up codebook and spill reductions remain outside this snapshot until their measured configurations are selected.

The shipped library contains the measured winner functions and the fastest measured 512-thread configuration for every shape and capacity. Unlisted kind/CTA/row-lane configurations return 0. Use winners.tsv for the unrestricted AOT choice, or fixed-cta.tsv for a 512-thread persistent launch and cta-256.tsv for a 256-thread launch; the full candidate generator and benchmark source remain in the evidence directory. The selected library replaces the 1.1-million-line artifact.

## Fused expert gate/up

The measured fused helpers cover IQ2_XS and IQ3_XXS gate/up matrices with k=2560 and m=640, capacities 1/2, and CTA sizes 256/512. Select the measured tuple from `fused-winners.tsv`.

```text
packed_gate_up_T_N_W_L(gate:u64, up:u64, packed_x:u64, scales:u64,
	product_f32:u64, active_columns:u32, position_mask:u32,
	cta_index:u32, cta_count:u32) -> u32
```

`T` is the GGML type, `N` is capacity, `W` is the number of warps, and `L` is row lanes. The helper reads a shared input tile once for both matrices and produces SiLU(gate) * up in compact columns, each with 640 floats. It uses the same mask and logical-group conventions as matvec and leaves inactive outputs untouched. All CTA threads call together, with 40960 dynamic shared bytes.

The helper copies canonical packed-byte grids and sign masks to shared memory once per CTA. XOR/add applies byte signs without carry because every grid magnitude is nonzero; the checked IQ2 and IQ3 ranges are 8..43 and 4..62. PRMT expands signed bytes to int16 pairs; XMAD.S16.S16 accumulates integers. IQ2 applies its two 16-value scale factors after integer sums; IQ3 applies the 32-value factor after its integer sum.

After the group's product writes finish, the caller supplies its completion barrier, prepares the compact product with k=640, supplies another completion barrier, and calls IQ4_NL down. Router coefficients belong to combine. The helper has no grid or peer protocol.

The composed full-active 1/2-column probe passes on real gate/up/down weights, with product error at most 3.38e-7 and checked down error at most 9.24e-8. Current selected-source masked, zero-active, and invalid-mask GPU checks wait for the resident model's lock; those cases are not declared GPU-validated. Whole-model rate and final logits remain integration gates.
