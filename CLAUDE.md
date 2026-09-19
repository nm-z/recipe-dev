# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

```go
amd-nv-cpu.ll (6265)
├── 1 target triple amdgcn-amd-amdhsa
├── 2 ; NUMERIC BEGIN placeholder region (build.rs replaces per precision)
│   ├── 3 declares sqrt/fabs/floor f64; @recipe.add/sub/mul/div/neg on double
│   ├── 9 @recipe.oeq/oge/ogt/ole/olt/one/ord fcmp
│   ├── 16 @recipe.from.u1/u32/s32, to.u32/to.s32, from.f32/from.f16, to.f16, abs, floor, sqrt
│   └── 27 comment: transcendentals bind to build-emitted defs; @recipe.exp/tanh/cos/sin/log → @recipe.math.*; @recipe.set.format no-op; ; NUMERIC END
├── 38 declares workitem.id.x, s.barrier, __ockl_steadyctr_u64; `; RECIPE_WAVE_HELPERS` and `; RECIPE_BLOCK_HELPERS` splice markers; llvm.trap; @contraction_tile external addrspace(3) [0 x double]
├── 43 @contraction_input(input, row.base, position, term, span, length, conv): channel=term/span, window=term%span (conv offset), load input[row.base + channel*length + position + offset]
├── 51 @contraction_delta(delta, output, index, relu): load delta, gate by output>0 when relu
├── 66 @contraction_delta_vector16: same over <16 x double> with per-element relu loop
├── 92 @reduce_rows(source, target, rows, columns, stride, offsets, threads)
│   ├── 92 tid = group*block+lid; parameter loop strided by threads over columns
│   └── 111 row loop sums rows in ascending order via recipe.add; store target[offset+parameter]
├── 132 comment: B tile is k-major rows of tile.n terms, one layout contract
├── 135 index functions: vector_a_index = k*tile.m+m; matrix_a_index = m*tile.k+k; vector_b_index = k*tile.n+n; matrix_b_index = n*tile.k+k
├── 159 @contraction_stage_column16(values, base, r, column, stride): store 16 values down rows r..r+15 into LDS tile
├── 179 @contraction_zero_edges(m.count, n.count, k.count, lid, block, tile.m/n/k)
│   ├── 179 count missing A cols, B cols, A k-rows, B k-rows; loop p from lid stride block
│   ├── 196 classify p into a / b / a.k / b.k region and compute tile index through a_index/b_index
│   └── 236 store 0.0 at index; exit
├── 245 @contraction_a_fragment(k, m.base): load <REGISTER_M x double> at a_index; @contraction_b_fragment(k, n.base): load <REGISTER_N x double> after A region
├── 261 @contraction_stage_a_fragment(fragment, k, m): FRAGMENT_K consecutive k for one m via a_index
├── 279 @contraction_stage_a_columns(fragment, k, m): FRAGMENT_K consecutive m at one k
├── 297 @contraction_stage_b_terms(fragment, k, n): consecutive k for one channel, k-major placement through b_index
├── 320 @contraction_stage_b_fragment(fragment, k, n): consecutive n at one k
├── 340 @contraction_stage_delta_a_fragment(delta, output, relu, k, m): relu-gated delta into A tile
├── 362 @contraction_output_lanes = m.lanes*n.lanes; vector_output_m/n: lane = lid % / ÷ m.lanes, register → local m/n
├── 383 matrix_output_m: wave*16 + (register*2 % 16) + lane/16; matrix_output_n: (register/8)*16 + lane%16
├── 403 vector_store_lane = %store; matrix_store_lane = true; output_register_valid = true
├── 415 comment: K walk in RECIPE_CHUNK_K chunks, ascending fold order; single k lane folds locally, multi-lane exchanges partials through the tile
├── 429 @contraction_vector_accumulate(sums, lane.active, lane.store, lid, lane.k, k.lanes, output.lane/lanes, m/n bases, m/n/k counts, tile.m/n/k)
│   ├── 435 chunk.sums alloca, chunks = ceil(k.count/CHUNK_K), single.sum.slot, local path when k.lanes==1 or chunks<=LOCAL_CHUNKS; owner = lane.k==0
│   ├── 451 local.k.begin/loop: load sums, prefetch A/B fragments for k+1, widen A once per k
│   ├── 468 local.product.loop: REGISTER_COUNT madd (a[i%M] * decode(b[i/M])) fully unrolled (!llvm.loop !0); local.store writes sums back
│   ├── 489 shared.chunk.loop: this lanes chunks from lane.k stride k.lanes; zero its chunk slot
│   ├── 506 k.begin/k.loop: chunk k range clamp, prefetch fragments, widen A
│   ├── 526 register.loop: per register.n broadcast decoded b, madd.vector into chunk slot
│   ├── 549 chunk.finish; chunk.done local barrier; publish.loop copies every chunk slot into tile at (chunk*output.lanes + output.lane)*REGISTER_COUNT
│   ├── 583 publish.done barrier; owner lane fold.loop sums chunks in ascending order into private sums
│   └── 612 exit
├── 615 @contraction_bias_accumulate(sums, destination, enable, first, last, lid, block, n.base, n.count, r.count, out.channels, window, tiles, store.offset)
│   ├── 621 channel loop from lid stride block; slot sums[REGISTER_N]; seed zero on first
│   ├── 629 r.loop: sum decoded staged B tile rows via b_index in ascending r
│   └── 637 sum.store; on last encode into destination[store.offset + out.channels*window + n.base + channel]
├── 647 @contraction_widen_m(<M x double>) → <M x RECIPE_STATE> via recipe.decode per lane
├── 666 ; RECIPE_WMMA catalog line: gfx11-f16/bf16 call intrinsics; gfx11-int8 (iu8 v4i32) and gfx11-int4 (pack.i4.word/pack.i4 + iu4 v2i32) definitions; gfx12-f16/bf16 (two 8-wide halves) and gfx12-int8 (two iu8 v2i32) definitions
├── 669 declare @recipe.wmma(<16 x double>, <16 x double>, <8 x STATE>) placeholder
├── 670 @contraction_matrix_accumulate(same signature as vector)
│   ├── 674 wave/lane → m = wave*16 + lane%16; four n columns lane%16 + 0/16/32/48 with validity masks; K rounded to 16
│   ├── 697 matrix.k.loop: load <16 x double> A row and four B rows through a_index/b_index (zero when invalid), four @recipe.wmma calls
│   ├── 736 matrix.k.done step 16
│   └── 739 matrix.store.loop: add the 4x8 accumulators into sums[register + 0/8/16/24]
├── 776 @recipe.model.decode(matrix, index, node) placeholder (model compiler emits one arm per packed node); @recipe.model.weight(weights, index, decode): load or decode
├── 783 @contraction_forward_gemv_body(input, weights, output, activation, rows, in/out channels+lengths, out.begin/span, kernel, has.bias, relu, transpose, reverse, accumulate, tiles, threads, weight.base, decode)
│   ├── 787 lid/group/block, groups; terms = in.channels; position = out.begin; tiles over out.channels by block
│   ├── 803 job.loop over channel tiles (group stride groups); n.count tail; lane owns one channel
│   ├── 816 sum.loop over k: weight dense load or @recipe.model.decode when decode≠0; input[k*in.length + position]; decode both, state mul/add
│   ├── 849 sum.done: bias at out.channels*terms + channel (dense/packed/zero), add when has.bias
│   └── 873 encode, relu gate, store output[channel*out.length + position]; job.done stride groups
├── 890 @contraction_forward_gemv_wave_body(same args): one output row per wave, wave reduction without LDS
│   ├── 895 lid/group/block/width/waves/wave/lane; terms, out.end, jobs = ceil(out.channels/waves)
│   ├── 912 plane selectors: @recipe.model.plane.rows / plane.base / plane.kind(0|1) (1 = Q4_K, 2 = Q6_K); q4/q6 selectors from either plane; @recipe.int8.dots & @recipe.model.int.activations → int8.dots; alignment 256 (K-quants) / 32 (@recipe.model.block32, block32.stride)
│   ├── 939 comment: int(n) sum on int8-dot backend rounds activations to int8; every other sum stages the column exactly; exact = !int8.dots, q8.available, stage.available
│   ├── 945 tile.bytes/capacity/blocks/chunk span; stage.tile and q8.shared aliases of @contraction_tile; q8 groups of 32-blocks; q8 constants
│   ├── 964 position.loop over out.begin..out.end; chunk.loop over K chunks (first/last flags, pitch = terms/4, k/b32 block offsets); q8 vs stage vs plain dispatch
│   ├── 986 stage.entry/loop: each lane loads input column values, stores quad-interleaved into tile; barrier
│   ├── 1017 q8.entry: per wave group of blocks, q8.extreme.loop finds |max| per lane, q8.max.loop wave-partner butterfly max
│   ├── 1080 q8.max.done: d = |max|/127 (guard zero), inverse; q8.code.loop: roundeven(x*inverse) clamp ±127, lane 0 stores d f32 at block*36, lanes store bytes at +4+lane
│   ├── 1136 q8.group.done stride waves; q8.done barrier
│   ├── 1142 job.loop: channel = job*waves + wave (active mask); row plane: second plane when channel ≥ plane.rows, row.kind → stride 144/210, row.base = plane.base + row.local × blocks × stride; row.q4.on / row.q6.on
│   ├── 1169 branch exact vs int8 paths; exact.q4.loop: slice from lane stride width, block = slice/16 + chunk blocks, byte offset row.base + block×144, @recipe.q4k.exact, masked sum
│   ├── 1194 exact.q6.loop: same shape, block bytes 210, @recipe.q6k.exact
│   ├── 1216 exact.b32.loop: blocks of 32 (slice/2, slice%2), channel row × b32.stride, @recipe.block32.exact(kind)
│   ├── 1241 q4.sum.loop (int8 path): row.base + block×144, q8 block at block*288, @recipe.q4k.slice
│   ├── 1264 q6.sum.loop: same with 210-byte blocks, @recipe.q6k.slice
│   ├── 1289 b32.sum.loop: q8 block at block*36, @recipe.block32.slice(kind)
│   ├── 1317 sum.loop (dense/decoded fallback): k from lane stride width, weight load/decode, input load, state mul/add, masked by channel.active
│   ├── 1349 sum.done phi over seven paths; reduce.loop wave butterfly add via @recipe.wave.partner (partner*4)
│   ├── 1365 reduce.done: lane 0 owner; bias (dense/packed/zero) added only on chunk.first
│   ├── 1394 output index; comment: later chunk adds the partial earlier chunks left; prior added unless chunk.first; relu only on chunk.last; store
│   └── 1411 job.done stride groups; chunk.done barrier, next chunk; position.done; exit
├── 1425 @contraction_forward_body(...) RECIPE_CONTRACTION_BODY dispatcher
│   ├── 1428 fast.* = out.span==1 & kernel==0 & rows==1 & !reverse & !accumulate & !relu & !transpose; wave.available = waves>0 & width>1
│   ├── 1447 comment: int sums use wave body for the whole span; blocked = q4k/q6k aligned 256 or block32 aligned 32; span.ok = one | blocked
│   └── 1470 dispatch: wave.fast → gemv_wave_body; fast.f → gemv_body; else gemm_body
├── 1483 @contraction_forward_gemm_body(...)
│   ├── 1486 comments: packed node decode selector; sums in arithmetic type for the whole K, rounded once at store; sums alloca, ids, widened dims, span/terms, m.total = rows*out.span
│   ├── 1495 clamp m/n/k tiles, m.tiles/n.tiles, jobs; job.loop with SWIZZLE_M grouping of m tiles per n tile
│   ├── 1499 m/n base and counts; m.lanes/n.lanes per REGISTER_M/N
│   ├── 1501 comment: lane owns one output position, leftover lanes share K chunks; lanes/k.lanes/output.lane/lane.k, lane.store via @contraction_store_lane; output m/n base
│   ├── 1518 sum.init.loop zero sums; tile.loop over K tiles (k.count tail)
│   └── 1525 A vector staging shape: span==1 & in.length==1 & k.count%FRAGMENT_K==0 & !(reverse&relu); B vector: fragment full & !transpose & dense
│   ├── 1536 load.loop over a.count + b.count tile elements stride block; load.a.step: m/k → row, position (out.span, out.begin), term
│   ├── 1542 load.a.vector: <FRAGMENT_K> load + stage_a_fragment; load.a.scalar: @contraction_input, relu gate on reverse (activation buffer)
│   ├── 1557 load.b.step: channel/term, direct vs transpose index, tile index; load.b.vector: <FRAGMENT_K> + stage_b_terms; scalar dense load or model.decode
│   ├── 1578 load.store into tile; load.done edge detection (logical partial or schedule tile wider than operand); comment on uninitialised LDS
│   ├── 1594 load.zero → @contraction_zero_edges; load.ready barrier; @contraction_product_accumulate; accumulate.done barrier; next K tile
│   ├── 1604 store.loop over REGISTER_COUNT: output m/n via @contraction_output_m/n, validity, method store lane
│   ├── 1607 store: output index (row, position, channel), bias index dense/packed decode
│   └── 1621 encode sum, add bias, relu on forward only, accumulate onto prior, store; job.done; exit
├── 1624 @pool_forward_body(input, output, context, p, from, to, size, channels, store.index): max pool window [start, min(start+size, length)), track argmax, store max and optional index to context
├── 1648 @rope_body(input, output, p, channels, length, head.width, dims, rotated, base, yarn.mscale/factor/context/low/high, reverse)
│   ├── 1652 decompose p into channel/position/local head channel; active when channel<rotated and local<dims; decode value and yarn params
│   ├── 1663 rotate: partner index (±half*length), frequency = exp(-(2i/dims)*log base)
│   ├── 1674 yarn: ramp clamp((i-low)/(high-low)), blend extrapolated/interpolated by factor when factor>1
│   ├── 1688 angle = position*frequency; cos/sin; sign flip for reverse and upper half; rotated = (v*cos + other*sin)*mscale
│   └── 1695 finish: reverse accumulates into output, forward assigns
├── 1700 hyper-connection stream bodies comment (lanes copies of channels per row)
├── 1704 @expand_forward_body: output[p] = input[row, channel%channels, position]
├── 1711 @expand_reverse_body: adjoint[p] += sum over lanes of delta in lane order
├── 1722 @read_forward_body(stream, gate, output, ..., gated): gate-weighted mean over lanes × 1/lanes
├── 1736 @read_reverse_body: stream.adjoint += gate*dh/lanes; gate.adjoint += stream*dh/lanes when gated
├── 1754 @outer_forward_body: output[p] = gate[row, lane, position] * branch[row, channel, position]
├── 1765 @outer_reverse_branch_body: adjoint += sum over lanes gate*delta
├── 1779 @outer_reverse_gate_body: gate adjoint += sum over channels branch*delta
├── 1792 @fold_forward_body(groups, width, length): output = sum of width channels in group; @fold_reverse_body: input adjoint += group delta
├── 1811 @dconv_forward_body(channels, length, kernel, dilation, decode): causal depthwise conv, tap j reads t-(kernel-1-j)*dilation, weights via @recipe.model.weight
├── 1825 @dconv_reverse_input_body: adjoint += tap * delta at t+(kernel-1-j)*dilation while in range
├── 1838 @dconv_reverse_weight_body(rows, ..., offset): tap p sums input*delta over rows and positions, writes gradient[offset+p]
├── 1850 @softplus(x): log1p(exp(-|x|)) + max(x,0)
├── 1856 comment: delta_step — S ← decay*S + write*k'(v − kS), o = qS when store; ascending sums so position update is chunk-independent
├── 1860 @delta_step(input, gates, output, context, q/k/v/o/a/b/work bases, kwidth, vwidth, length, time, decay.scale, store) #3
│   ├── 1863 decay gate at a.base+time
│   ├── 1866 decay = exp(-softplus(a)*decay.scale); write = sigmoid(b); column.loop over vwidth
│   ├── 1874 read.loop: readout = Σ_i k_i S[i][col]; error = v − readout; write.error
│   ├── 1890 write.loop: S[i][col] = decay*S + k_i*write.error; output sum += q_i*S_new
│   └── 1909 write.store o[col] when %store; column.done; exit
├── 1915 comment: delta_forward_body walks chunks, commits carried state at each chunk start
├── 1918 @delta_forward_body(input, gates, weights, output, context, p, kheads, kwidth, vheads, vwidth, length, chunk, chunks, pairs, entries, decode) #3
│   ├── 1920 row/head from p; state = kwidth*vwidth; commit.count when entries≠0; q/k/v/o/a/b bases from stream layout (GQA khead = head/group)
│   ├── 1934 entry.base / work.base regions in context; decay.scale = exp(@recipe.model.weight(head))
│   ├── 1938 zero.loop clears work state; chunk.loop: commit.loop copies work state to chunk entry
│   └── 1953 time.loop calls @delta_step(store=true) per position; chunk.done; exit
├── 1963 comment: delta_back — previous state index, adjoint carried backward, two head vectors (readout error, key adjoint weight), returns decay-scale contribution
├── 1968 @delta_back(input, gates, context, delta, input.adjoint, gate.adjoint, bases, previous, adjoint.base, vector.base, kwidth, vwidth, length, time, decay.scale) #3
│   ├── 1971 decay/write gates recomputed; seed.row/seed.column: adjoint[i][j] += q_i * delta_o[j]
│   ├── 1998 column.loop/column.row: readout = Σ k_i S_prev, weight = Σ k_i adjoint; store error and weight vectors; value adjoint += write*weight; write.gradient += error*weight
│   ├── 2041 row.loop/row.column: decay.part += S*adj, key.direct += error*adj, key.readout += weight*S, recompute S_next for query.part += delta*S_next, adjoint ← decay*adj − k*write*weight
│   ├── 2089 row.store: key adjoint += write*(key.direct − key.readout); query adjoint += query.part; decay.gradient accumulates
│   └── 2101 gates.entry: decay input gradient via sigmoid slope, write input gradient via write*(1−write); return −decay.gradient*softplus*decay.factor
├── 2117 comment: delta_reverse_body replays each chunk forward from committed entry then walks backward
├── 2121 @delta_reverse_body(input, gates, weights, context, delta, input.adjoint, gate.adjoint, p, kheads, kwidth, vheads, vwidth, length, chunk, chunks, pairs) #3
│   ├── 2124 row/khead from p; stream bases; context regions: entries, work, replay, adjoint, vector, partial
│   ├── 2145 comment + head.loop over the value heads sharing this key head (group); per-head bases; decay.scale
│   ├── 2162 zero.loop clears adjoint; chunk.loop from last chunk: restore.loop copies entry → work
│   ├── 2180 replay.loop: save work state to replay slot then @delta_step(store=false)
│   ├── 2196 backward.loop: @delta_back per position from the end, decay.sum accumulates
│   └── 2205 chunk.done; store partial; head.done; exit
├── 2210 @delta_reverse_decay_body(context, gradient, p, rows, heads, partials, offset): sum per-row partials in row order → gradient[offset+p]
├── 2222 mixture routing comment: scores/weights [expert*length + t]; top-k with ties to lower expert; weights hold marks until last pass
├── 2229 @topk_score(score, maximum, sigmoid): exp(score−max) or logistic
├── 2232 @topk_forward_body(scores, weights, p, experts, length, top, scoring, renormalize)
│   ├── 2236 flags sigmoid/renorm/every/divide; clear.loop zeroes weights row
│   ├── 2243 select.loop × top: scan.loop picks best unmarked (strict greater), select.mark writes 1.0
│   ├── 2254 max.loop over members (marked or every) for softmax shift; sum.loop total of topk_score over members
│   └── 2269 write.loop: marked → score/denominator, else 0
├── 2276 @topk_reverse_body(scores, weights, delta, adjoint, p, experts, length, scoring, renormalize)
│   ├── 2283 max.loop over members (weight≠0 or every); sum.loop total + inner = Σ weight*delta
│   └── 2303 write.loop: slope (sigmoid: s(1−s), softmax: exp) / denominator × (own delta − inner); blocked when renorm and unselected; adjoint +=
├── 2318 mixture dispatch comment: gate/up [expert][hidden][channel], down [expert][channel][hidden]; bucket (position, slot) pairs after per-expert counts
├── 2324 @moe_bucket_body(routing, context, p, pairs, length, experts, top)
│   ├── 2327 base.loop: count selections of lower experts across all pairs → bucket start
│   ├── 2339 fill.loop over pairs: if routed to expert p, slot = number of lower selected experts, write i*top+slot at start+cursor
│   └── 2357 done: store count at context[p]
├── 2358 @expert_in_forward_body(input, routing, weights, output, p, channels, length, hidden, experts, top, decode)
│   ├── 2363 decompose p → row, slot, f, position; find.loop: (slot+1)-th selected expert
│   └── 2374 sum.loop: Σ_k weight[expert][f][k] (model.weight) * input[row][k][position]; store
├── 2386 @expert_in_reverse_input_body(routing, weights, delta, adjoint, p, ...): expert.loop over selected experts; hidden.loop Σ_f delta[slot,f] * W[e][f][channel]; adjoint +=
├── 2414 @expert_in_reverse_weight_body(input, delta, context, gradient, p, ..., offset): expert/f/channel from p; bucket start/count; loop packed pairs: Σ delta[row][slot*hidden+f][pos] * input[row][channel][pos]; gradient[offset+p]
├── 2434 @moe_bucket_start(context, expert, experts): experts + Σ counts of lower experts
├── 2442 @expert_out_forward_body(values, routing, weights, output, p, ...): expert.loop over selected; hidden.loop Σ_f W_down[e][channel][f] * values[slot,f]; times routing weight; store
├── 2469 @expert_out_reverse_values_body(routing, weights, delta, adjoint, p, ...): find slot's expert; Σ_o delta[o] * W_down[e][o][f]; times routed weight; adjoint +=
├── 2501 @expert_out_reverse_weight_body(values, routing, delta, context, gradient, p, ..., offset): bucket loop Σ routed*delta[channel]*values[slot,f]; gradient[offset+p]
├── 2525 @expert_out_reverse_routing_body(values, routing, weights, delta, adjoint, p, ...): per selected expert Σ_o delta[o] * Σ_f W[e][o][f]*values[slot,f] → routing adjoint[e]
├── 2560 @sigmoid(x) = 1/(1+exp(−x))
├── 2563 @attention_tile_dot(left, right, width, left.base, right.base): Σ over channels of decoded tile rows (LDS), state madd chain
├── 2591 @attention_tile_score(query, key, width, key.base, scale): dot / scale, encoded to double
├── 2598 @attention_step_score_store(context, index, value): store RECIPE_STATE at context[index]
├── 2603 @attention_step_score_load(context, index) → RECIPE_STATE
├── 2608 comment: whole-grid inference attention for one query; ordinary context's statistics planes hold packed states per score, K/V context holds the history in cache type
├── 2611 @attention_step_key_dot(query LDS, kv.context, key, kv.head, width, length): 16-way unrolled channel loop
│   ├── 2615 sixteen phi partial sums; channel.0..15, active masks, safe indices
│   ├── 2684 q.N loads from LDS query, masked; key channel/base/index → RECIPE_KV loads from kv.context, @recipe.kv.to.state, masked
│   ├── 2860 sixteen madd chains; channel += 16
│   └── 2878 done: pairwise tree add of 16 partials → sum
├── 2896 @attention_forward_step_body(input, output, context, kv.context, from, heads, channels, position, kv.heads, threads) #3
│   ├── 2900 lid/group/block/global.id; length = from/channels, width = channels/heads, kv.group, kv.plane, reached = position+1; wave/lane/waves/global.wave; scale = sqrt(width)
│   ├── 2922 LDS tile aliases: maximum/denominator global slots after the wave slots
│   ├── 2928 cache.loop: every thread grid-strides kv channels, encodes new K (input[from + ch*length + position]) and V (after kv.plane) into kv.context via @recipe.kv.encode
│   ├── 2955 cache.done grid_barrier; score.tile.loop (group = head, tiles of block keys ≤ reached)
│   ├── 2964 score.query.copy.loop: stage this head's query row into LDS as RECIPE_STATE; barrier
│   ├── 2983 score.key.compute: @attention_step_key_dot / scale → @attention_step_score_store at context[head*length + key]; barrier; advance
│   ├── 3001 score.done grid_barrier; maximum.loop per lane over scores, maximum.wave.loop butterfly, lane 0 writes wave slot
│   ├── 3040 maximum.group.loop lid 0 folds wave slots → maximum.global; barrier
│   ├── 3063 denominator.loop Σ exp(score − max); denominator.wave.loop butterfly; wave slot; group fold → denominator.global
│   ├── 3119 probability.loop: score ← exp(score−max)/denominator; probability.done grid_barrier
│   ├── 3137 output.channel.loop (global.wave stride global.waves): output.key.loop 4 keys per lane per step (stride 4*wave.width), probability × @recipe.kv.to.state(V), four madd chains
│   ├── 3220 output.wave.prepare tree add; output.wave.loop butterfly; lane 0 stores @recipe.model.from.state at output[channel*length + position]
│   └── 3251 output.stats.owner; exit
├── 3256 comment: attention_selected — query keeps the block holding this key; each query owns a row of block scores + admission flags
├── 3258 @attention_selected(context, score.row, blocks, select.block, query, key)
│   └── 3262 flag at score.row + query*2*blocks + blocks + key/select.block; > 0.5
├── 3272 comment: indexer planes arena — index.heads query heads × index.width channels, then one raw key plane; reference pools a block by mean, then key norm and rotary at block position
├── 3277 @attention_index_mean(indexer, context, key.origin, representative.start, query, block.index, select.block, index.width, length, d)
│   ├── 3280 own block (query's own) → prefix.loop sums keys start..query over dim d; count = query+1−start
│   └── 3295 cached: read representative[block][d]; mean = total/filled
├── 3321 comment: attention_index_normalized — trained score uses deferred key scale; unscored returns cached mean
├── 3325 @attention_index_normalized(..., index.heads, index.width, length, mode, epsilon, pooled, d)
│   ├── 3328 !pooled → plain mean; sum.loop Σ mean_d² over index.width
│   ├── 3346 mode 2 (rms): sqrt(sum/width + eps); else l2: max(sqrt(sum), eps)
│   └── 3365 unit = mean/deviation; mode 2 multiplies key scale from key.weights[index.heads*width + d]
├── 3388 comment + @attention_index_rotated(..., dims, base, epsilon, pooled, d): normalized value; inside dims → partner (±half), frequency = exp(−(2i/dims) log base), position = block.index*select.block, cos/sin; lower: cos·v − sin·partner, upper: cos·v + sin·partner
├── 3443 @attention_index_representative: wrapper over attention_index_rotated
├── 3451 comment: attention_index_body — running sum of indexer keys of one block extended by the forward window begin..end
├── 3455 @attention_index_body(indexer, context, p, begin, end, rows, from, heads, channels, kv.heads, value.heads, index.heads, index.width, select.block, gate, epsilon, index.mode, index.dims, index.pooled, index.base)
│   ├── 3458 length, index channels, row stride, blocks = ceil(length/select.block), representative region after two statistics planes
│   ├── 3471 row/block from p; representative.start; window [first, stop); clear when block starts inside the window
│   ├── 3491 clear.loop zeroes the block's representative row
│   └── 3502 key.loop × dim.loop: representative[d] += indexer key value
├── 3531 comment: attention_select_body — every indexer query head scores every causal block representative, heads add, threshold = keep-th best
├── 3535 @attention_select_body(indexer, key.weights, context, p, keep, rows, from, heads, channels, kv.heads, value.heads, index.heads, index.width, select.block, gate, epsilon, index.mode, index.dims, index.pooled, index.base)
│   ├── 3538 layout: blocks, score.stride = 2*blocks, representative base/total, score.base; row/query from p; count = query/select.block + 1
│   ├── 3569 clear.loop zeroes this query's block scores
│   ├── 3580 head.loop over index.heads
│   ├── 3584 head.prepare / score.loop over blocks ≤ count: score.dim.loop Σ_d query[d] × @attention_index_representative(block, d)
│   ├── 3613 score.store: relu the head score, add into block score (in state)
│   ├── 3630 threshold.prepare; rank.loop per block: count blocks strictly greater (ties to earlier index); admit when ahead < keep; store flag 1.0/0.0 after the scores
│   └── 3670 select.exit
├── 3674 @attention_tile_products(output, output.row, delta.base, product.base, query.base, query.count, head.start, head.width, length, lid, block): per query Σ_channels delta(LDS) × output(global) → LDS product
├── 3721 @attention_tile_derivatives(context, shared offsets, query/key bases and counts, tile.n, head.job, length, statistics.denominator.base, head.width, scale, lid, block, score.row, blocks, select.block, select)
│   ├── 3732 pair.loop over query×key tile pairs; causal key ≤ query; optional @attention_selected
│   ├── 3748 complete: score = dot/scale, dp = dot(delta, value), probability = exp(score − max)/denominator, derivative = probability × (dp − product)
│   └── 3774 store probability and derivative tiles (zero when invalid)
├── 3792 @attention_forward_body(input, weights, output, context, kv.context, carry, rows, from, heads, channels, query.begin, query.span, tiles, threads, kv.heads, value.heads, index.heads, index.width, select.block, gate, epsilon, index.mode, index.dims, index.pooled, index.base) #3
│   ├── 3798 ids, length, head.width, scale = sqrt(head.width), kv/value groups and planes, index planes, gate plane, row.stride, blocks/score.stride for selection, query.tiles, head.jobs, statistics planes, representative/score bases
│   ├── 3837 64-bit mirrors of the same layout; jobs = head.jobs × query.tiles
│   ├── 3848 LDS layout: query values, key values, value values, scores, probabilities, accumulator, maximum, denominator, rescale
│   ├── 3860 job.loop: query.tile/head/row from job; query base/count/last; head starts (kv.head, value.head); score.row
│   ├── 3888 query.stage.loop: stage query tile into LDS, zero accumulator
│   └── 3909 statistics.init.loop: maximum = −inf, denominator = 0
│   ├── 3922 query.stage.done barrier; key.tile.loop over key tiles up to query.last; key.count tail
│   ├── 3937 select: tile.scan over queries × blocks in the tile — skip the whole key tile unless some query keeps a causal block in it (@attention_selected)
│   ├── 3968 key.stage.loop: key/value channel indices; past keys with carry read kv.context (@recipe.kv.decode) else input, and carry stores new K/V (@recipe.kv.encode); stage K and V tiles into LDS
│   ├── 4031 key.stage.done barrier; score.loop over query×key pairs: causal + optional selection, @attention_tile_score else −inf, store score tile
│   ├── 4070 softmax.loop per query: running max over tile keys
│   ├── 4094 probability.prepare: comment — unscored query keeps max at −inf, center on zero; old.rescale = exp(old.max − new.max); denominator rescaled
│   ├── 4109 probability.loop: p = exp(score − max) stored, denominator += p; softmax.store max/denominator/rescale
│   ├── 4141 accumulate.loop per (query, channel): old × rescale + Σ_key p × V
│   ├── 4180 accumulate.done barrier; key.tile.advance
│   ├── 4186 output.loop: attention = accumulator/denominator; channel 0 lane writes maximum and denominator statistics to context (two planes)
│   └── 4215 output.value.store: optional gate sigmoid(input gate plane) × attention; store; job.finish; exit
├── 4245 @attention_cache_body(input, kv.context, rows, from, heads, kv.heads, value.heads, length, threads) #3
│   ├── 4248 head.width, kv/value planes, row.stride, total = rows × kv.planes; cache.loop from lid
│   └── 4272 encode input[row*row.stride + from + within] into kv.context[row*kv.planes + within]; stride threads
├── 4281 @attention_forward_matrix_body(input, weights, output, context, kv.context, carry, rows, from, heads, channels, tiles, threads, kv.heads, value.heads, index.*, select.block, gate, epsilon, ...) #3 (WMMA path)
│   ├── 4287 ids, length, head.width, scale; carry → @attention_cache_body + barrier
│   ├── 4301 head.jobs, statistics rows; LDS: q/k/v planes of length×head.width then p (length×length)
│   ├── 4314 job.loop per head.job: stage 3 planes (Q,K,V) per channel; 16-wide vector rows via @contraction_stage_column16, scalar tail
│   ├── 4381 stage.done barrier; wave/lane geometry; 16×16 score tiles jobs = tiles²
│   ├── 4392 score.job.loop per wave: width loop of @recipe.wmma over Q and K fragments (16 channels per step)
│   ├── 4427 score.store.loop: 8 accumulators → (query = tile + 2*out + lane.half, key), scale, causal else −inf, store into p plane
│   ├── 4456 score.done barrier; softmax.loop per query: max, exp − max with denominator, normalize in place
│   ├── 4509 softmax.store: max and denominator into context statistics planes
│   └── 4519 softmax.done barrier; @attention_matrix_product(mode 3) P·V into output; barrier; next head.job
├── 4530 @attention_matrix_product(previous, mode, left.base, right.base, row.base, from, head.start, length, head.width, scale, lid, block)
│   ├── 4534 mode: 0 dq / 2 dv / 3 forward; direct (dq, forward) vs transposed left; unscaled (dv, forward); 16×16 tiles over length × head.width
│   ├── 4567 k.loop by 16: fragment.loop gathers 16 left (direct or transposed) and right values with validity masks, @recipe.wmma
│   └── 4608 store.loop: 8 outputs → (m, n), optional /scale, plane = mode*from unless forward, store into previous[row + channel*length + m]
├── 4643 @attention_reverse_matrix_body(input, output, context, delta, previous, rows, from, heads, channels, tiles, threads, kv/value/index params) #3 (WMMA backward)
│   ├── 4649 ids, length, head.width, scale; LDS planes q, k, v, do, p (length²), ds (length²), d (per query)
│   ├── 4672 job.loop per head.job; stage 4 planes per channel (Q, K, V from input row of 3 planes; plane 3 = delta of output), 16-vector via stage_column16 or scalar
│   ├── 4762 stage.done barrier; d.loop: d[query] = Σ_channels dO × O
│   ├── 4796 d.done barrier; 16×16 score tile jobs per wave: wmma Q·K (score) and dO·V (dp) accumulators
│   ├── 4850 score.store: probability = exp(score/scale − max)/denominator (causal else 0); ds = p × (dp − d); store p and ds tiles
│   └── 4901 score.done barrier; @attention_matrix_product mode 0 (ds·K → dq), mode 1 (dsᵀ·Q → dk), mode 2 (pᵀ·dO → dv); barrier; next job
├── 4913 @attention_reverse_body(input, output, context, delta, previous, rows, from, heads, channels, tiles, threads, kv.heads, value.heads, index.*, select.block, gate, epsilon, ...) #3
│   ├── 4919 lid/group/block
│   ├── 4922 layout: length, head.width, scale, kv/value groups and planes, index planes, gate plane, row.stride, selection blocks, statistics rows, representative/score bases; 64-bit mirrors
│   ├── 4962 tile counts; dq.jobs = head.jobs × query.tiles; dq LDS layout: query, delta, gradient, key, value, probability, derivative, product
│   ├── 4980 dq.job.loop: query tile/head/row; bases (kv.head, value.head, input/output/score rows)
│   ├── 5003 dq.query.stage.loop: stage Q and dO tiles, zero gradient; barrier; @attention_tile_products (D = Σ dO·O); barrier
│   ├── 5038 dq.gate.loop (gate): factor = sigmoid(gate input); gate gradient = dO × O × (1 − factor) → previous; delta ← dO × factor in LDS; barrier
│   ├── 5076 dq.key.tile.loop up to query.last; select scan skips tiles no query keeps
│   ├── 5117 dq.key.stage.loop: stage K and V tiles from input; barrier; @attention_tile_derivatives (p, ds tiles); barrier
│   ├── 5159 dq.accumulate.loop: gradient[q, c] += Σ_key ds × K / scale; barrier; next key tile
│   ├── 5201 dq.store.loop: write query gradient tile into previous (input row plane 0); barrier; next dq job
│   ├── 5231 dq.exit: dkv.jobs = rows × (kv.heads + value.heads) × key.tiles; dkv LDS layout: key, value, key gradient, value gradient, query, delta, probability, derivative, product
│   ├── 5244 dkv.job.loop: key tile / kv.job; comment: key jobs own a key head + query group, value jobs a value head + query group
│   ├── 5252 dkv.job.prepare: fixed head (key head or value head), key tile base/count, head starts, rows, head.base/count = the query heads in this group
│   ├── 5276 dkv.zero.loop clears key and value gradient tiles; barrier
│   ├── 5292 dkv.head.loop over the query heads of the group; comment: tiles restage per query head (key job reads that head's value head and vice versa); barrier
│   ├── 5309 dkv.key.stage.loop: stage K (this head's key head) and V (this head's value head) tiles; barrier
│   ├── 5343 dkv.query.tile.loop from key.base to length; select scan skips unkept tiles
│   ├── 5384 dkv.query.stage.loop: stage Q and dO tiles; barrier; @attention_tile_products; barrier
│   ├── 5417 dkv.gate.loop (gate): delta ← dO × sigmoid(gate); barrier
│   ├── 5444 @attention_tile_derivatives (p, ds); barrier; dkv.accumulate.loop per (key, channel): dK += Σ_q ds × Q / scale, dV += Σ_q p × dO
│   ├── 5502 dkv.accumulate.done barrier; next query tile; next head
│   ├── 5511 dkv.store.begin barrier; dkv.store.loop: key job writes dK into previous key plane, value job writes dV into value plane
│   └── 5552 dkv.store.done barrier; dkv.job.finish; exit
├── 5561 @scan_forward_body(input, weights, output, context, rows, in.channels, length, out.channels, time.begin, time.span, gates, has.bias, tiles, threads, weight.base, decode, coded, cell.activation) #3 (rnn/gru/lstm)
│   ├── 5563 tid = workitem.id (grid-wide); time.limit; matrix spans: input matrix, state matrix, bias; gate.stride; gate.batch = rows × out.elements
│   ├── 5572 precompute.loop over gates: dense weight pointer or packed base; context slice per gate
│   ├── 5582 precompute.step calls @contraction_forward_body per gate (input projection over time.begin..span into context slice); precompute.done grid_barrier
│   ├── 5588 row.loop (row = tid stride threads); time.loop time.begin..limit; gate.loop; hidden.loop
│   ├── 5597 input.load: precomputed input sum from context; state.sum.loop over previous hidden (t−1, zero at t=0); GRU gate 2 multiplies by reset gate
│   ├── 5617 state weight dense/packed; product/add; gate.activate: bias dense/packed; builtin activation tanh (rnn or last gate) else sigmoid; coded cell.activation 1 relu / 2 tanh / 3 sigmoid
│   ├── 5657 gate.store into context[gate][row][hidden][time]
│   ├── 5663 output.loop: load gate0..gate3 (indices selected by rnn/gru/lstm), previous output; GRU h = z·h_prev + (1−z)·candidate
│   ├── 5688 LSTM cell = f·c_prev + i·g stored at context[gates][...], h = o·tanh(c); select rnn (gate0) / gru / lstm; store
│   └── 5702 output.done; time.next; row.next stride threads; exit
├── 5704 @contraction_reverse_body(input, weights, output, delta, previous, gradient, write.input, has.bias, relu, matrix.gradient, rows, in/out channels+lengths, kernel, offset, gradient tiles, previous tiles, threads) RECIPE_CONTRACTION_BODY
│   ├── 5708 sums and bias.sums allocas; ids; dims; window = in.channels × span; gradient.r.total = rows × out.length; gradient.values (matrix + bias)
│   ├── 5718 comment: split-K scratch rows padded (RECIPE_SCRATCH_ROW_MASK/CLEAR) so rows never share a word; gradient.scratch at RECIPE_GRADIENT_SCRATCH_BASE; tiles clamp; jobs
│   ├── 5728 comment: K partitions per SPLIT_SPAN capped at K_PARTITIONS, backend-independent; splits, partition size/extra; direct (1 split) writes gradient, else scratch; tasks = jobs × splits
│   ├── 5748 gradient.job.loop per task: job/split; store row offset; comment + partition bounds [p*q + min(p,r), ...); SWIZZLE_M grouping; m/n bases and counts; lanes
│   ├── 5773 comment: lane owns one output position, leftovers share K; output/k lanes, method store, bias owner (m.base == 0, lid < n.count)
│   ├── 5796 gradient.sum.init.loop; gradient.tile.loop over r partitions (first/last tile flags)
│   ├── 5806 gradient.load.generic.entry: A vector when span==1 & in.length==1 & m.count%FRAGMENT_K==0; B vector when out.length==1 & n.count%FRAGMENT_K==0
│   ├── 5822 gradient.load.loop: A = input via stage_a_columns (vector) or @contraction_input; B = delta (relu-gated by output) via delta_vector16 + stage_b_fragment or @contraction_delta; store to tile
│   ├── 5854 edge zeroing (@contraction_zero_edges); barrier; @contraction_bias_accumulate; @contraction_product_accumulate; barrier; next r tile
│   ├── 5877 gradient.store.loop: valid registers encode sums into gradient.destination[store.offset + filter*window + term]
│   ├── 5887 gradient.finish: when split, grid_barrier + @reduce_rows(scratch → gradient at offset)
│   ├── 5893 previous.check (write.input): previous.m.total = rows × in.length, previous.r.total = out.channels × span; tiles, jobs
│   ├── 5901 previous.job.loop: SWIZZLE_M grouping, m/n bases and counts, lanes, k lanes, method store, output bases
│   └── 5920 previous.sum.init.loop; previous.tile.loop
│   ├── 5922 previous.tile.loop over r (out.channels × span): A vector when span==1 & out.length==1 & r.count%FRAGMENT_K==0; B vector when span==1 & n.count%FRAGMENT_K==0
│   ├── 5930 previous.load.loop: A = relu-gated delta at (row, filter, position − kernel) valid only inside out.length, via stage_delta_a_fragment or @contraction_delta; B = weights[filter*window + channel*span + kernel] via stage_b_fragment or scalar
│   ├── 5969 edge zeroing; barrier; @contraction_product_accumulate; barrier; next r tile
│   ├── 5985 previous.store.loop: valid registers add encoded sums onto previous[row*in.elements + channel*in.length + position]
│   └── 5991 previous.job.done; exit
├── 5992 @scan_reverse_body(input, weights, output, context, delta, previous, gradient, write.input, rows, in.channels, length, out.channels, gates, has.bias, parameters, offset, gradient tiles, previous tiles, threads, coded, cell.activation) #3
│   ├── 5997 tid; sizes: batch, gate strides (input matrix, +state matrix, +bias), delta.base = (gates+1)×batch, gate2.batch, row.gradient.base = (2×gates+1)×batch; rnn/gru/lstm flags; unsupported → trap
│   ├── 6008 row.loop (tid stride threads): clear.gradient.loop zeroes the row's parameter partials; clear.state.loop zeroes dh and dc scratch
│   ├── 6027 time.loop backwards from length−1; scan.mode dispatch lstm / rnn / gru
│   ├── 6034 rnn.delta.loop: dh = dy + future; derivative tanh (or coded relu/tanh/sigmoid); gate delta stored at delta.base
│   ├── 6056 gru.delta.loop: dz = dh·(h_prev − n)·z·(1−z); dn = dh·(1−z)·(1−n²); stored
│   ├── 6084 gru.reset.loop: dr[source] = (Σ_target W_candidate_state[source][target] · dn[target]) · h_prev · r · (1−r)
│   ├── 6120 gate.delta.loop (lstm): dh, dc = dc_future + dh·o·(1−tanh²c); do, di, df, dg with sigmoid/tanh slopes; dc_prev = dc·f stored; four gate deltas stored
│   ├── 6161 parameter.loop: per state-weight/bias parameter add source (h_prev, reset-gated for GRU candidate, or 1 for bias) × gate delta into the row gradient partial
│   ├── 6196 hidden.gradient.loop: dh_prev[channel] = Σ_{gate,hidden} W_state × gate delta (reset-gated for GRU candidate) + GRU direct z·dh; stored to dh scratch
│   ├── 6242 time.done; row.done stride threads; reduce.entry grid_barrier + @reduce_rows(row partials → gradient at offset)
│   ├── 6246 projection.entry grid_barrier; projection.loop per gate: @contraction_reverse_body for the input projection (weights slice, gate delta slice, gradient at offset + gate stride); grid_barrier
│   └── 6261 invalid: llvm.trap; exit; attributes #0 (flat-work-group-size RECIPE_WORKGROUP_SIZE), #1 alwaysinline, #3 noinline
└── 6262 comment: fully unroll the product loop; !0 / !1 llvm.loop.unroll.full metadata


recipe.rs (27343)
├── 1 crate doc: one model graph after probing a compiled discrete GPU backend; allow non_upper_case_globals
├── 4 mod program_ir — compile-time lowering of scalar, predictor, route and normalization pieces to straight-line LLVM SSA
│   ├── 16 enum ScalarOpcode (Add, Constant, Parameter, Subtract, Multiply, Divide, Absolute, Exp, Log, Sin, Cos, Tanh, Greater, StraightThrough, Select) + from_i32
│   ├── 59 enum PredictorOpcode (Feature, Row, Constant, Load, Store, Duplicate, Add, Subtract, Multiply, Divide, Greater, Choose, Nearest, Affine, Gaussian) + from_i32
│   ├── 101 enum EmitError (WrongWidth, InvalidOpcode, InvalidOperand, InvalidReference, StackUnderflow, StackDepth, LocalIndex) + Display
│   ├── 125 type LiteralFn; struct ScalarContext { value_type, suffix, pointer_type, alignment, first, second, weights, decode, prefix, literal }; ScalarForward; ScalarReverse; ScalarInstruction
│   ├── 160 integer() operand check; binary() emits `@recipe.<op><suffix>`; predicate() emits fcmp + from.u1; parse_scalar() triples; scalar_operand() (−1 first, −2 second, n≥0 value)
│   ├── 200 emit_scalar_forward(code, context)
│   │   ├── 205 per instruction: Constant literal; Parameter load from weights or @recipe.model.decode when decode≠0
│   │   ├── 245 StraightThrough passes left; Select = ogt(cond, 0) ? value : 0; binary ops add/sub/mul/div/ogt
│   │   └── 266 unary abs/exp/log/sin/cos/tanh; result = last value
│   ├── 287 add_adjoint() / negate() helpers with sequence counter
│   ├── 300 emit_scalar_reverse(code, context, incoming)
│   │   ├── 305 replay forward value names; parameter_for map; adjoint arrays seeded with incoming on the last instruction; add_operand routes to first/second/slot
│   │   ├── 360 reverse walk: Add, StraightThrough (right), Subtract (negated right), Multiply, Divide (−adj·l/r²)
│   │   ├── 405 Absolute (sign select), Exp (adj·value), Log (adj/left), Sin (adj·cos), Cos (−adj·sin), Tanh (adj·(1−v²)), Select (gated right)
│   │   └── 482 fold parameter adjoints into BTreeMap; return ScalarReverse
│   ├── 493 struct PredictorContext { value_type, suffix, pointer_type, alignment, input, row, features, weights, context, parameters, prefix, literal }; PredictorForward; parse_predictor() pairs
│   ├── 522 emit_predictor_forward(code, locals, context): compile-time stack, locals as SSA values
│   │   ├── 545 Feature: bounds check, load input[row*features + feature]
│   │   ├── 579 Row (from.u32 of row), Constant, Load/Store locals (compile-time), Duplicate, Add/Subtract/Multiply/Divide, Greater, Choose (one(cond,0) ? yes : no)
│   │   ├── 631 Nearest(count, exclude self when negative): KD-tree walk over context index (node fields minimum/first/second/dimension/parent), bound pruning by far-branch box distance
│   │   ├── 666 leaf scan: per candidate distance with early prune, insertion into count-slot sorted (distance, row) carry chain; done: mean of targets at rows*features + r
│   │   ├── 689 Affine: Σ_j (x_j − mean_j)·scale_j·weight_j over three feature planes
│   │   ├── 739 Gaussian: per class base + Σ (x−mean)²·scale, argmax label; stack must end at depth 1
│   │   └── 810 return PredictorForward
│   ├── 816 enum NormalizeMode (Batch, Layer, Rms, Evaluation, L2) + per_row(); struct NormalizeContext { value_type, suffix, pointer_type, alignment, source_value, context, rows, channels, length, width, span, weight, mode, prefix }; NormalizeFragment; GroupShape; GroupIndex
│   ├── 892 emit_group_index(code, prefix, shape, element): row/local/position/channel; per-row modes group span channels into heads of width, `inside` predicate when span < channels; group = row*plane + head*length + position
│   ├── 931 emit_normalize(context, element): mean/scale from arena (mean[group], scale[groups+group]), (x − mean)·scale, optional per-channel weight (weight_column), pass-through outside the span
│   ├── 1001 weight_column(); struct NormalizeReverseContext { ..., state_type, state_zero, source, weight }; NormalizeReverseFragment
│   ├── 1048 emit_normalize_reverse_stats(context, delta, output): per group (tid stride threads) accumulate Σdelta and Σdelta·output in state type over items (batch: rows×length, per-row: width)
│   │   ├── 1124 weighted node: delta × weight and re-normalized source; Rms/L2 skip the delta sum; projection = delta·output
│   │   └── 1183 store: sum mean and projected mean (L2 divides by one) at arena planes 2 and 3 in model type; group stride threads
│   └── 1224 emit_normalize_reverse(context, element, delta, output): loads scale/sum/projected; Evaluation → delta·scale; weighted re-normalizes; contribution = scale·(delta − sum − output·projected); pass-through outside span
├── 1328 struct ScheduleCandidate { node, direction, limits, current, extent }; ScheduleGroup { reference, candidates, unmeasured, best }; assign_candidate() → forward/gradient/previous tile
├── 1353 comment: schedule RAT shared proposer; SCHEDULE_RAT_STATE/ACTION = 8; schedule_axis() log2/16; schedule_candidate_features(); schedule_candidate_state()
├── 1390 struct ScheduleRat { proposer: Graph, gpu, samples, seconds }
│   ├── 1399 new(): synthetic Prepared, model layer(surrogate_width).tanh().layer(8) compiled as proposer
│   ├── 1421 observe(candidate, seconds); can_propose() ≥ 2; proposal_launches() = 2×surrogate_epochs + 1
│   ├── 1440 propose(groups, config): normalize seconds to [0,1]; fit_knn(1) scorer + held-out R²; fit_surrogate; compose proposer + predictor + surrogate with StraightThrough; copy optimizer moments; freeze scorer
│   └── 1494 train composition surrogate_epochs on NativeTape; capture back into proposer; inference on group states; nearest unmeasured candidate by feature distance
├── 1523 struct ScheduleMeasurement { median, low, high, repeatable, state }
├── 1530 measure_schedule(tape, snapshot, assignment, rate, config): warmups + measurements of restore→apply schedule→advance→full_epoch, checks epoch_state repeatability, median/low/high; always restores snapshot
├── 1572 tune_contraction_schedule(tape, rate, config, budget)
│   ├── 1575 early outs (no candidates / no epoch / no contractions / budget < launches); state ratio; device identity; schedule cache hit applies and returns
│   ├── 1595 build ScheduleGroups per node × direction from schedule_candidates(); empty → store heuristic in cache
│   ├── 1616 snapshot, ScheduleRat, baseline measurement (nonrepeatable → keep heuristic, no cache); observe baseline per group
│   ├── 1637 loop while unmeasured: exploration round (random pick) then one proposer round when budget allows; measure each candidate; exact = same epoch state; stable = high < baseline.low and median ≤ baseline×(1−min improvement); observe exact ones; track best per group
│   └── 1694 combine winners, measure combined, accept only if stable; restore snapshot; apply selected; store cache; trace
├── 1720 use program_ir::{PredictorOpcode, ScalarOpcode}; struct NativeLayout { values, contexts, attention_kv, ... }
│   ├── 1731 NativeLayout fields: adjoints, schedule (per-node schedule word offset or MAX), *_bytes, clocks (traced per-node device clock), precisions, input/output_precision, weights, spans, casts, cast_adjoints
├── 1764 enum BackendTarget { Cpu{target}, Amd{architecture}, Nvidia{architecture} }: backend(), artifact_extension() (dll / hsaco / cubin-or-ptx), validate() (CPU identity vs RECIPE_CPU_TARGET, exact gfx/sm)
├── 1809 cpu_identity_field(); LLVM_OPAQUE_POINTER_DEFAULT_MAJOR = 15; APPLE_CLANG_BROKEN_LICM_PROMOTION_PREFIX; cpu_llvm_major(); cpu_compiler_version()
├── 1829 cpu_identity(target) parses `target=;compiler=;cpu=;features=` with canonical sorted ±features
├── 1852 native_cpu_target(): clang -march=native -### query → target-cpu and sorted target-features → BackendTarget::Cpu identity
├── 1876 struct NativeArtifact { backend, layout, precision, artifact bytes, path, storage: StorageImage, training }; StorageImage { bytes, segments (node, StoredBytes) } + is_empty
├── 1903 struct NativePrecision { model, state, acc, source, model_type, state_type, epoch_layout }; symbol names recipe_model_forward/epoch/load/thread; ABI layouts NATIVE_FORWARD_LAYOUT "888844444", epoch fp64/fp32, load "8844"; native_epoch_layout()
├── 1930 macro native_precisions!: base(model) table and new(model, acc) (fp64 acc over float base → `-acc64` sibling via acc64_source); table: -f, default(fp64), -f32, -f16(half), -f8(i8), -bf16(i16), -tf32(float), -int8/-int4/-int1(i8)
├── 1980 NATIVE_SCHEDULE_WORDS = 9; align(); encode_floats(); packed_weight() (inference + node.packed → stored weight)
├── 2006 widest_precision(); native_weight_arena(): per-node offsets aligned to max(element, 8), packed nodes keep stored byte length
├── 2035 reads_beyond_window() (Attention/Delta/Dconv/Pool/Predictor, conv contraction); window_signatures() (whole/pool/last/shift chain); retained_outputs() (graph output, beyond-window operands, scan, differing second window); last_uses()
├── 2106 impl NativeLayout::for_graph(graph, rows, precision, inference)
│   ├── 2115 value slots: inference reuses released same-size slots for transient outputs (not when tracing); casts for operands in another arithmetic (+ cast adjoints in training)
│   ├── 2167 context offset per node via node_context(); inference attention K/V region; adjoints (zero in inference, releases operand slots at last use)
│   └── 2197 schedule words for Contraction/Scan; precisions, weights, spans, input/output precision, clocks when tracing; totals
├── 2220 struct NodePlan { node, value, context, attention_kv, adjoint, stored, requantize, weight_offset, packed } + decode(index) selector; struct RecurBodyLayout { base, cell, cell_values, values, value_stride, adjoints, adjoint_stride, temporary_gradient(+len), cell_delta, per-node offsets }
├── 2260 enum NativeMatrix { Gfx11, Gfx12 } + key(); struct NativeModelIr { graph, layout, precision, variants, weight_bytes, rows, schedule, plans, storage_bytes, inference }
├── 2293 impl NativeModelIr::from_graph(graph, rows, precision, schedule, inference): layout, weight arena, run precision from profile.acc, collect NativeVariants (suffix from source + `_kv<key>` when cache type differs)
│   ├── 2309 per-node validation (source/second ranges, dense parameter range, program range, stored weight count); requantize source or arena weight adds to storage_bytes; NodePlan per node
│   ├── 2355 node_source() (template key via NativePrecision::new), variant() suffix lookup, gradient_base() (weight offset in node elements), precisions() list, node_precision(), storage() → StorageImage segments
├── 2402 matrix_capable() (-f16/-bf16/-int8/-int4); struct NativeVariant { precision, kv, suffix }; kv_key() f16/f32/bf16; variant_suffix() (`default` → `_f64`)
├── 2429 rename_symbol(text, name, renamed) whole-symbol `@name`; parameter_list_end()
├── 2470 link_variant(module, text, suffix): collect defined functions and globals (+recipe.model.decode), strip shared plumbing (recipe.cpu.*, grid_barrier, global_id, workgroup.size.x, wavefront.width), rename the rest with suffix, drop attributes/metadata/target lines and declarations the module already has
├── 2545 template_path(mapping, suffix); backend_template(backend, precision, matrix, kv): `-kv<key>` suffix, RECIPE_{CPU,AMD,NV}_IR mapping, gfx matrix key, RECIPE_CONTRACTION_BODY → #3 on GPUs / #1 on CPU, custom `f(exp,man)` bakes exp/man constants and narrows recipe.round to half/float when the format matches
├── 2595 pointer_type(); definition_span(ir, name) brace matching; strip_definition(); prune_internal_definitions() (drop `define internal` bodies referenced once, iterate to fixpoint); barrier(backend); ptr_gep()
├── 2664 mod quantized — backend-neutral dequantizers over trait QuantOps
│   ├── 2667 enum QuantIntOp (Add..Xor), QuantValueOp; trait QuantOps { Int, Value; index, integer, int, equal, less, select_int, sign_extend, load(bits, offset), half, float, half_bits, table, signed_table, value_table, number, literal, value, select_value, signed }
│   ├── 2712 quant_int(), quant_bits(value, shift, width), quant_parity_sign() (lane 7 sign is parity of the other seven)
│   ├── 2733 enum IqPacking { S, Xs, Xxs }; IqLayout { man, exp, sign, packing, table }; Iq1Layout { medium }; ScalarLayout { sign, exp, man, variant }; Iq4Layout { xs, table }
│   ├── 2778 dequant_iq(quant, layout): per packing (IQ2_S, IQ2_XS, IQ2/IQ3_XXS, IQ3_S) grid code, factor code, sign bits; table word → mantissa 2m+1, exponent (odd factor 2f+1 or f+0.5), × scale × multiplier, signed
│   ├── 2898 dequant_iq1(quant, layout): IQ1_M (medium) vs IQ1_S grid/scale/factor/delta bit
│   │   └── 2954 table word → mantissa (2m+1)−1 signed, ±0.125 delta, exponent 2f+1, × scale
│   ├── 2968 dequant_q45k(quant, man 4|5): nibble (+ high bit for Q5_K), 6-bit scale/min pairs (low 4 groups vs packed high), d·scale·code − dmin·min
│   ├── 3024 dequant_q6k: ql nibble | qh pair<<4 − 32, int8 scale at 192+, d at 208
│   ├── 3063 dequant_scalar(layout): Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 (header 2 or 4, variant 1 adds minimum, variant 0 subtracts 2^(man−sign))
│   ├── 3116 dequant_q2k: 4-bit scale/min metadata per 16, 2-bit codes, d at 80, dmin at 82
│   ├── 3149 dequant_q3k: 6-bit scale from low nibbles + high pairs − 32, 2-bit codes − 4 when high-mask bit clear, d at 108
│   ├── 3190 dequant_q8k: int8 code × f32 scale at 0
│   ├── 3200 dequant_iq4(layout): IQ4_XS (6-bit per-block exponent from low nibble + high bits, −32) or IQ4_NL; signed level table
│   ├── 3241 dequant_nf4(block, table, scales): 4-bit code → value table × per-block scale table
│   ├── 3254 struct HostQuantOps { bytes, index } impl QuantOps on u64/f64 (host-side reference decode)
│   └── 3337 struct NativeQuantOps { globals, ir, backend, precision, suffix, next } impl QuantOps emitting LLVM (`%quant.N`), loads from `%block`, tables as `@recipe_model_<name>` constants, state helpers under the variant suffix
├── 3479 use quantized::*; impl NativeModelIr (the model compiler: emits the forward, epoch and load kernels)
│   ├── 3484 emit_schedule_words(backend, index, prefix, first, count, ir): inference bakes the tile words as constants
│   │   └── 3491 training loads the i32 words from the context arena schedule slot
│   ├── 3506 emit_fixed_primitives(backend, matrix, reverse, training) — one match arm per (reverse, Primitive)
│   │   ├── 3508 plan order (reverse walks backwards); recur_body plans skipped; emit_pointers; traced per-node clock store by tid 0; variant suffix, matrix capability, gradient base, node window, casts
│   │   ├── 3538 forward Contraction: schedule words 0..3, @contraction_forward_body(source, weights, value, ..., begin, span, kernel, bias, relu, tiles, decode)
│   │   ├── 3562 forward Gather: per element token id → @recipe_model_quantized_<layout> row decode into value
│   │   ├── 3582 forward TopK ([1,length] positions) → @topk_forward_body; Expand → @expand_forward_body
│   │   ├── 3614 Rope (forward and reverse share): literals base/mscale/factor/context/fast/slow, whole-row loop or window loop → @rope_body(reverse flag)
│   │   ├── 3641 forward ExpertIn, Read (gated when second ≥ 0), Dconv
│   │   ├── 3686 forward Scan: generic recurrent body → @recipe_recur_scan_forward_<index> (dense weights only), else @scan_forward_body with schedule words, coded cell activation
│   │   ├── 3700 forward Lookup: host-staged [row][position][channel] rows copied into the value plane; Fold; Last (position = source end − 1); Outer
│   │   ├── 3771 forward Delta: [heads,1] pairs walked whole, entries = chunks in training / 0 in inference → @delta_forward_body
│   │   ├── 3795 forward ExpertOut; Pool (argmax indices stored in training)
│   │   ├── 3833 forward Attention: matrix body when full-length, all heads equal; indexer geometry + selectors; blocks ≠ 0 → touched-block @attention_index_body then @attention_select_body per query
│   │   ├── 3872 extended (begin, span) args for the tiled body; K/V pointer and carry flag; fast_attention gate (inference, rows 1, AMD, state ≤ 2×model, LDS ≥ 2k, no blocks, no gate, K/V present, value heads == kv heads); step tile m/n selected at runtime when span == 1
│   │   ├── 3903 normal call vs branch to @attention_forward_step_body when span == 1; barrier
│   │   ├── 3914 forward Elementwise: program_ir::emit_scalar_forward over the node's 3-word program, first/second loads, store
│   │   ├── 3972 forward Predictor: program_ir::emit_predictor_forward (locals, features, weights, context), row = element / output elements
│   │   ├── 4019 reverse Contraction: schedule words 3..9, @contraction_reverse_body(write_input = kernel > 1, bias, relu, matrix gradient, offset = gradient base); kernel ≤ 1 composes the input adjoint with a transposed @contraction_forward_body (accumulate when the source has later readers)
│   │   ├── 4033 reverse Gather: none (frozen packed table); reverse Expand → @expand_reverse_body; reverse TopK → @topk_reverse_body
│   │   ├── 4068 reverse Dconv: @dconv_reverse_input_body over outputs, then @dconv_reverse_weight_body over [channels, kernel] taps (rows 1) into %gradient at gradient base
│   │   ├── 4102 reverse Lookup: none; reverse Fold → @fold_reverse_body; reverse Last: add delta into source adjoint at the last position
│   │   ├── 4141 reverse ExpertIn: @expert_in_reverse_input_body, emit_expert_buckets, then @expert_in_reverse_weight_body over [1, parameters]
│   │   ├── 4171 reverse Read → @read_reverse_body (gate adjoint = second adjoint); reverse Outer → branch body + gate body when gated
│   │   ├── 4210 reverse Delta: @delta_reverse_body over key heads (value heads sharing a key head in one thread), then @delta_reverse_decay_body per value head
│   │   ├── 4245 reverse ExpertOut: values body, routing body, buckets, weight body
│   │   ├── 4288 reverse Pool: scatter delta to the stored argmax index; reverse Attention: @attention_reverse_matrix_body or @attention_reverse_body (index admission is a hard gate, no extra backward)
│   │   ├── 4316 reverse Scan: generic body → @recipe_recur_body_reverse_<index> then @scan_reverse_body with cell delta; else plain @scan_reverse_body
│   │   ├── 4334 reverse Predictor: barrier only
│   │   ├── 4337 reverse Elementwise: forward replay + emit_scalar_reverse; adjoints accumulated (combined when second == −2); trainable scalars go through emit_partitioned_loop (NATIVE_SCALAR_PARTITIONS scratch rows) then @reduce_rows into %gradient
│   │   ├── 4477 forward Normalize: emit_normalize_stats when training or per-row (not Evaluation); per-element program_ir::emit_normalize with optional weight
│   │   ├── 4527 reverse Normalize: emit_normalize_reverse_stats (unless Evaluation); per-element emit_normalize_reverse accumulated into source adjoint
│   │   ├── 4600 normalize weight gradient: partitioned per-channel sums of normalized × delta into scratch after 4×groups statistics, folded by @reduce_rows
│   │   └── 4669 reverse pass ends each node with emit_cast_adjoints
│   ├── 4684 cell_activation(node): coded stage activations packed 4 bits per stage
│   ├── 4699 recurrent_stage_metadata(node): cell width, stage widths, saved channel offsets, weight offsets, max width
│   ├── 4731 recurrent_body_layout(index, node): body range (argument 3/4), per-node value tape offsets and stride, cell tape, adjoint tape, temporary gradient, cell delta
│   ├── 4764 emit_recurrent_stage_metadata(): @recipe_recur_<i>_widths/saved/weights constants
│   ├── 4779 emit_recurrent_body_forward(backend, index, node, layout): @recipe_recur_body_forward_<i>
│   │   ├── 4794 copy cell into the position's cell tape; per body node source/second/weights bases
│   │   ├── 4849 body Contraction: per row × channel × term dot with bias and relu, into the node's value tape
│   │   ├── 4915 body Elementwise: program_ir::emit_scalar_forward per element
│   │   └── 4955 final: copy last body output into the scan output at (channel, time)
│   ├── 4990 emit_recurrent_scan_forward(backend, index, node, layout): @recipe_recur_scan_forward_<i> — time loop; per row/channel input + previous-state dot, bias, cell activation (0 linear / 1 relu / 2 tanh / 3 sigmoid) into cell; call body forward per position
│   ├── 5127 emit_recurrent_body_reverse(backend, index, node, layout): @recipe_recur_body_reverse_<i> — owner lane only; clear temporary gradient; time loop from the end
│   │   ├── 5172 per-position value/adjoint/cell bases; per body node bases
│   │   ├── 5209 clear cell delta for the position; clear every body node's adjoint slice; seed the final body node's adjoint from the scan output delta
│   │   ├── 5273 reverse the body nodes in reverse order: Contraction — relu-gated adjoint, source adjoint (cell delta or body tape) and temporary weight/bias gradients accumulated per row
│   │   ├── 5377 Elementwise — emit_scalar_forward + emit_scalar_reverse per element, destinations into cell delta or body adjoint tapes, parameter adjoints into the temporary gradient
│   │   └── 5461 time.done; owner merges temporary gradients into %gradient at each body node's offset; sync barrier
│   ├── 5505 emit_recurrent_body_functions(backend): scan forward + body forward + body reverse per generic recurrent node
│   ├── 5515 stage_base(node): after (2×gates+1)×rows states, per-row gradients and 2×rows×channels scratch
│   ├── 5526 emit_normalize_stats(backend, index, node, pointers, mode, window)
│   │   ├── 5527 setup: groups/items per mode, per-row modes walk the window's groups one wave per group (wave lane/id/count)
│   │   ├── 5602 emit_index closure (batch vs per-row element addressing); wave_reduce closure (butterfly via @recipe.wave.partner)
│   │   ├── 5643 mean loop (skipped as zero for Rms/L2), optional wave reduce + broadcast; variance loop of centered squares
│   │   ├── 5672 scale: L2 = 1/max(norm, eps), else 1/sqrt(mean square + eps)
│   │   └── 5694 store mean and scale into context (owner lane per wave in per-row modes); group advance; done
│   ├── 5735 emit_node_window(index, node, ir): begin/end from source window; Predictor whole, Pool divides by size, Last = [0,1), conv shifts by kernel−1; span
│   ├── 5771 emit_convert(ir, prefix, from, to, raw): model → state (source variant), fpext/fptrunc between float/double states, state → model (target variant)
│   ├── 5798 emit_casts(backend, index, reverse, window, pointers, ir): forward converts each foreign-arithmetic operand into the cast slot over the node's window (whole operand when lengths differ); reverse redirects the operand adjoint pointers to the cast adjoint slots
│   ├── 5844 emit_cast_adjoints(backend, index, ir): convert the own-type adjoint contribution back and add into the operand's real adjoint (state add)
│   ├── 5872 emit_pointers(backend, index, plan, reverse, ir): %n<i>.source/second/value/context/attention.keys/delta/weights/source.adjoint/second.adjoint GEPs over values/contexts/adjoints/weights arenas (%samples / %input_adjoint for the model input)
│   ├── 5922 emit_native_quantization(backend, format, native, precision, suffix): @recipe_model_quantized_<name><suffix>(matrix, row, column, columns) block address then NativeQuantOps decode
│   ├── 5938 emit_native_nf4(backend, index, stored): per-node NF4 decoder with its own codebook/scale tables
│   ├── 5956 emit_quantized_decoders(backend): per precision suffix, per stored/requantize segment format: table definitions once, one decoder per codec (NF4 per node)
│   ├── 6001 block_dot_fits(backend, plan): packed contraction with model ≥ 2 bytes; AMD int blocks need the Q8 scratch (36 bytes/32) in the tile with float state; others need a 256-value chunk of the column
│   ├── 6025 emit_int_activation_support(backend): @recipe.model.int.activations switch (inference AMD int blocks that fit)
│   ├── 6039 emit_tile_bytes(): @recipe.tile.bytes = shared_values × widest element
│   ├── 6047 block_planes(backend, plan): one or two Q4_K (1) / Q6_K (2) planes with row-aligned counts; emit_plane_support(): @recipe.model.plane.kind/rows/base switches
│   ├── 6094 emit_q4k_support / emit_q6k_support: per-node switches; emit_block32_support: @recipe.model.block32 kind (IQ4_NL 1, Q4_0 2, Q4_1 3, Q8_0 4) and .stride (18/20/34)
│   ├── 6149 emit_weight_decode → @recipe.model.decode (packed nodes); emit_source_decode → @recipe.model.source.decode (#3, requantize sources or unpacked stored weights)
│   ├── 6159 emit_decode_switch(backend, name, select): per precision suffix a switch over nodes; multi-segment weights dispatch by index range into per-format decoders; byte/element totals validated
│   ├── 6216 requantize_function(plan): device quantizer (32-value blocks only) + `recipe.requantize.<name>[.packed]<suffix>`; emit_requantize_call(); emit_requantize_functions() (dedup by name)
│   ├── 6249 emit_requantize(backend, plan): grid-stride blocks; 32 values through source decode → float; signed extreme, min, max
│   │   ├── 6282 IQ4_NL: inverse of extreme, code by IQ4_MID thresholds, least-squares scale over the table levels
│   │   ├── 6304 Q8_0: scale = |ext|/127, round half away, clamp ±127/−128; Q4_0: scale = ext/−8, +8.5 clamp 0..15; Q4_1: (max−min)/15 with +0.5
│   │   └── 6330 packed: write half scale (+ half minimum for Q4_1) and 16 nibble bytes or 32 bytes into the target block; unpacked: store decoded values in the node's type
│   ├── 6361 emit_model_load(backend): requantize functions + @recipe_model_load(weights, storage, threads, node): one node per dispatch — requantize call or grid-stride @recipe.model.source.decode expansion
│   ├── 6399 emit(backend, matrix, loss) — the whole module
│   │   ├── 6400 substitute schedule constants (RECIPE_WORKGROUP_SIZE, REGISTER_M/N/COUNT, FRAGMENT_K, CHUNK_K/VALUES/BIAS_VALUES, SCRATCH_ROW_MASK/CLEAR, GRADIENT_SCRATCH_BASE); run template; link each variant template under its suffix
│   │   ├── 6425 append quantized decoders, weight/source decode, q4k/int-activation/tile-bytes/plane/q6k/block32 support, model load, recurrent metadata and body functions
│   │   ├── 6448 @recipe_model_inference_forward_body (#1) and, with a loss, @recipe_model_training_forward_body (#3)
│   │   ├── 6461 inference on GPUs: @recipe_model_step(#4, flat-work-group-size 32..512 on AMD) with llvm.assume on begin/rows, end = begin+1
│   │   ├── 6467 @recipe_model_forward: training flag dispatch or inference only
│   │   ├── 6474 with loss: @recipe_model_epoch(samples, targets, weights, frozen, moments, variances, gradient, metrics, input_adjoint, values, contexts, adjoints, rows, threads, rate, beta1, beta2, powers, epsilon, decay, run.gradient, run.optimizer): clear gradient/adjoints/input adjoint, training forward, loss + seed, reverse, AdamW
│   │   └── 6501 prune_internal_definitions; CPU appends the module suffix
│   ├── 6508 emit_loss_and_seed(...): thread 0 sums the loss (mse/rmse normalizer, per-item /count for others) into %metrics; all threads seed the output adjoint with emit_loss_gradient
│   ├── 6562 emit_adamw(pointer): per parameterized unpacked node, per-parameter loop honoring the frozen mask; moments/variances in the run state type converted to the node state; bias-corrected update with decoupled decay
│   └── 6635 emit_clear_bytes(backend, base, bytes, label, from): grid-stride byte clear
├── 6645 struct ModelPointers { source, second, value, context, attention_kv, delta, weights, source_adjoint, second_adjoint }; type_literal()
├── 6667 struct IndexerGeometry { mode, dims, pooled, base, key_weights } + none(); impl NativeModelIr::indexer_geometry(index): walk the second-source chain for the side Normalize (mode, pooled, key scale weights) and Rope (dims, base)
├── 6723 attention_value_heads(); attention_selectors(node, precision, ...) → `kv, values, index_heads, index_width, block, gate, epsilon, index_mode, index_dims, pooled, index_base`
├── 6742 native_literal(precision, ty, value); normalize_mode() 0..4; normalize_width(); normalize_span(); normalize_groups() (batch/eval per channel; per-row rows×length×heads)
├── 6793 alignment(ty) (comment: half at align 1 split NVPTX loads); loss_threshold() from RECIPE_HUBER_THRESHOLD; append_binary()
├── 6811 emit_loss_value(): mse/rmse scaled square, huber, mae, bce (clamped sigmoid), focal
├── 6850 emit_loss_gradient(): mse 2d/n, rmse d/(n·loss), huber clamp, mae sign, bce (p−t)/n, focal chain
├── 6907 token_id(value, vocabulary)
├── 6912 graph_positions(graph) (gather counts elements, else input length); struct NodeWindow { begin, span }; integer_argument(); accumulate_owned() (single-writer adjoint add)
├── 6941 NATIVE_SCALAR_PARTITIONS = 4096; struct PartitionedLoop { count, partitions, columns, value_type, suffix, pointer_type, scratch, zero, gradients }
├── 6961 emit_partitioned_loop(ir, index, name, shape, body): partitions × contiguous runs [t·q+min(t,r), …) summed in ascending order into scratch rows (zeroed rows when no fixed sums; every column written otherwise)
├── 7010 emit_fixed_loop(ir, index, name, rows, shape, window, body): compile-time rows, i64 element walk over rows × channels × window span
├── 7029 emit_runtime_window_loop(): same with %rows from the launch (shared artifacts across training/holdout/inference)
├── 7047 emit_row_loop(ir, index, name, per_row, body): rows × per_row elements from the launch
├── 7062 struct DeltaShape { heads, key_heads, chunks, partials, arguments }; delta_extent(node) (key heads/width default to heads/width); delta_shape(node, rows) (pairs, state, chunks, spans, partials offset, argument string)
├── 7102 emit_expert_buckets(ir, backend, index, rows, node, pointers, v): @moe_bucket_body over [experts,1]
├── 7122 NATIVE_ARTIFACT_SERIAL; struct NativeTemporaryFiles + Drop (kept when tracing); remote_native_artifact(target, bytes) → private 0600 file under ~/.cache/recipe/native/remote
├── 7170 home_directory(); native_artifact_directory(key) = ~/.cache/recipe/native/<key>; native_artifact_key(target, ir): FNV over version tag (cpu-v5 / native-v3), target requirement, CPU compiler identity, RECIPE_NATIVE_CONFIGURATION, IR text
├── 7211 native_command(command, role, key): run compiler, capture diagnostics, keep failed.ll beside the cache on failure
├── 7243 struct KernelResources { name, registers, scalars, occupancy }; kernel_resources(diagnostic) parses -Rpass-analysis=kernel-resource-usage remarks
├── 7284 native_cpu_compiler(); native_cpu_compiler_identity() (`<path>@<clang version>`); native_cpu_setting(name) from RECIPE_CPU_* env; native_entry(backend) → (kernel linkage, thread id call)
├── 7320 native_amd_compiler(); native_nvidia_compiler(); native_nvidia_codegen(); NVIDIA_DRIVER_VERSION; native_nvidia_assembler() (ptxas only when its release ≤ driver); native_amd_library(name); native_nvidia_device_library(); native_nvidia_ptx_version(); cpu_unsupported_feature()
├── 7380 compile_native_artifact(target, source, output, key)
│   ├── 7382 CPU: clang -target/-target-cpu/-target-feature, opaque pointers on old LLVM, Apple clang 14 LICM workaround, -x ir -O2 -shared with configured linker; reject ignored features
│   ├── 7416 AMD: clang amdgcn -mcpu -O3 -nogpulib, kernel-resource-usage remarks, `-amdgpu-long-branch-factor=0` (llvm/llvm-project#224205), link ocml/ockl/abi/finite/math + oclc_isa_version bitcode → KernelResources
│   └── 7437 NVIDIA: clang nvptx64 -march (+ ptx feature); with llc: emit bitcode then llc → PTX; ptxas → cubin when usable, else NUL-terminated PTX
├── 7482 compile_model(target, graph, precision, loss, rows, schedule) → NativeArtifact: validate, NativeModelIr::from_graph, gfx11/gfx12 matrix when schedule.matrix, emit
│   ├── 7492 artifact key + ~/.cache/recipe/native/<key>/artifact.<ext>; cache hit reads; miss writes .ll, compiles, requires nonzero kernel occupancy, renames into place
│   └── 7530 NativeArtifact { backend, layout, precision, artifact, path, storage, training }
├── 7534 struct FloatLayout { sign, exp, man } (runtime copy): new, bias, bits, pack (saturating to max finite instead of inf), unpack; power(exponent)
├── 7611 struct FloatFormat { arithmetic, storage }: FP8/FP16/FP32/FP64/BF16/TF32, computed(exp, man) over fp64 storage, native(), bytes, pack/unpack through the arithmetic layout
├── 7640 struct IntFormat { bits }: INT1/INT4/INT8, bytes, pack (round ties even, clamp, mask), unpack (sign extend)
├── 7660 mod gguf — GGUF container reader (mapped shards, metadata, tensors)
│   ├── 7670 MAGIC/VERSION 3/DEFAULT_ALIGNMENT 32; enum GgufValue (U8..F64, Bool, String, Array) + integer()/text()/float()
│   ├── 7725 struct GgufTensor { name, shape, kind, offset, bytes, shard }: elements(), expert(index) slice of [k,n,experts], blocked(), rows(start,count), slice() (block-aligned byte view)
│   ├── 7776 layout(kind): GGML type id → (name, block, stride, StorageFormat) for F32/F16/I8..F64/BF16 and q4_0..iq1m
│   ├── 7817 struct Mapping (mmap on unix, read elsewhere) + Drop munmap; struct Reader { bytes, at, depth } with take/u32/u64/string/value (arrays ≤ 64 deep)
│   ├── 7905 struct Shard { mapping, data }; pub struct Gguf { shards, metadata, tensors, quantization }
│   │   ├── 7920 open(path): first shard + split.count siblings; shard(path, index): magic/version, metadata pairs, alignment, split.no check, tensor descriptors with block-multiple checks, data offset, bounds
│   │   ├── 7980 metadata(), value(), tensors(), tensor(name), required(), integer_at(), float_at(), integers_at(), indices_at(); data(tensor) bytes; tokenizer(); values(tensor) decode; row(tensor, index)
│   │   ├── 8043 stored(tensor) → StoredWeight view of mapped block bytes; embedding_stored() (F32/F16 tables through the gather path)
│   │   └── 8063 bound(plan) → Vec<BoundNode>: all-blocked planes join as mapped runs with per-format segments; token_embd F32/F16 → embedding_stored; otherwise decoded values
│   ├── 8122 embedding_format(tensor); block_format(tensor); decode(tensor, data, count) (raw types or StorageFormat::decompress)
│   └── 8153 sibling(path, index, count) for `<prefix>-NNNNN-of-NNNNN.gguf`
├── 8161 pub use gguf::{Gguf, GgufTensor, GgufValue}
├── 8162 mod tokenizer — byte-level BPE from GGUF metadata (token table, ranks, pre-tokenizer family, added tokens, special ids, chat template)
│   ├── 8171 enum Family (pre-tokenizer alternation)
│   ├── 8177 char classes letter/number/space/newline/other; run(); contraction() ('s|'t|'re|'ve|'m|'ll|'d); trailing_space()
│   ├── 8216 impl Family: named(pre) (gpt-2 group, llama3 group, qwen2 group); word(chars, at) implements each family's pre-tokenizer regex by hand; split(text)
│   ├── 8288 byte_map() GPT-2 byte→char; enum Ranks { Merges, Scores }; ranked(scores, vocabulary)
│   ├── 8332 pub struct Tokenizer { tokens, ids, ranks, added, is_added, bytes, byte_of, family, template, add_bos, add_eos, bos, eos, pad }
│   │   ├── 8350 from_gguf(model): tokenizer.ggml.model == gpt2, family, tokens, merges or scores, byte ids, added tokens (types 3|4, longest first), chat template, special ids/flags
│   │   ├── 8412 bos()/eos()/pad()/vocabulary()/token(id); encode(text): added tokens matched whole first, BOS/EOS framing
│   │   ├── 8448 merge(left, right) lowest rank; encode_plain(): per word byte symbols, llama3 whole-word shortcut, greedy min-rank merges; byte_of_token(); decode(ids) rejoins bytes
│   │   └── 8501 chat(messages, generation) / prompt(): renders tokenizer.chat_template with messages, add_generation_prompt, bos/eos tokens; adds_bos()
│   ├── 8528 enum Piece { Text, Write, Tag }; push_text(); parse(template): `{{ }}`, `{% %}`, `{# #}` with `-` trimming and block-line stripping
│   ├── 8585 enum Value { Undefined, Null, Bool, Int, Text, List, Map, Namespace(Rc<RefCell>) }: truth(), text(), json(), integer(), equals(), contains(), field(), index(), slice(start,end,step), items(), length()
│   ├── 8748 struct Scope { frames }: get(name), assign(target, value) (namespace fields or innermost frame)
│   ├── 8782 enum Token { Name, Str, Int, Op }; OPERATORS; lex(source) (strings with escapes, ints, names, `=` vs `==`)
│   ├── 8841 evaluate(source, scope, live) (dead branches parse without reading); struct Parser { tokens, at, scope }
│   │   ├── 8857 peek/eat_op/eat_name/expect_op/name; expression (inline if/else), or, and, not
│   │   ├── 8925 comparison: == != < <= > >= in / not in / is [not] string|defined|undefined|none|number|boolean|sequence|mapping|true|false
│   │   ├── 8976 concat (~), additive (+ text/list/int, −), multiplicative (* // / %), unary −
│   │   ├── 9031 postfix: .field / .method(args), [index] and [start:end:step] slices, |filter(args)
│   │   └── 9066 primary: strings, ints, parenthesis, list literals, true/false/none, namespace(k=v), range(a[,b]), scope names; arguments() (keyword values kept)
│   ├── 9138 method(value, name, args): startswith/endswith/strip/lstrip/rstrip/lower/upper/replace/split, items/get/keys/values
│   ├── 9166 filter(value, name, args): length/count, trim, string, tojson, lower, upper, capitalize, int, list, first, last, reverse, join, default/d, safe/e/escape
│   ├── 9192 loop_value(index, length); assignment(source) splits `set a = b`
│   ├── 9213 render(pieces, at, out, scope, emit, stop): if/elif/else/endif, for … in … (loop frame, empty body parse), set (inline or block capture), generation tags
│   └── 9286 pub use tokenizer::Tokenizer
├── 9287 mod ngram — host-gathered n-gram / per-layer embeddings from a mapped GGUF table
│   ├── 9297 ABSENT = u32::MAX; pub struct RowHash { ngram, per_order, multipliers, seeds, ... }
│   ├── 9312 RowHash { offsets, vocabularies, ends, absent, image }: heads(), rows(), validate(), rows_at(ids, position) (context cut at end ids, multiply-xor fold or seeded fold per head), words()/from_words() (graph words), text()/parse() (saved-model fields)
│   ├── 9424 table_row(table, width, index): raw f64 rows or block decompress of one row
│   ├── 9438 pub struct Ngram<'a> { model, table, taps, hash, layer, kernel, width, rows }
│   │   ├── 9450 new(model): `ngram.*` seeded form (heads, equal ranges, seeds) or `<arch>.ple.*` reference form (multipliers, offsets, vocab sizes, eos); table shape checks; conv taps
│   │   ├── 9519 heads(), width(), layer(), kernel(), hash(), bytes(); placed() (one or two selected devices); placement(); table()
│   │   ├── 9569 rows(ids, position); block() → PleBlock; gather(); lookup(ids); inject(ids) (taps across positions)
│   │   └── 9610 infer(path, input, ids) / decode(): split saved graph at layer, head on first device, add injected vector on host, tail on last device
│   └── 9634 impl Gguf::ngram(); pub use ngram::{Ngram, RowHash}
├── 9643 mod bundle — the saved-model text format (`recipe-native-model` header)
│   ├── 9646 hex/unhex/text/untext/bool_value; escape/unescape/split_escaped (one backslash per nesting level)
│   ├── 9711 residual_text(); product_branch_text()/product_branch() (older records without quantization/exclusions); residual() (legacy layer/conv/activation steps or full block text); value_at(); scoring()
│   ├── 9769 activation_text()/activation() (code 16 = scale with f64 bits); operation_text(): layer/conv/pool/estimator/attn(heads, keys, dims, base, indexer, gate, width, tokens, score norm, layout, yarn, v=)/rnn/gru/lstm/recur/residual/ensemble/product/moe/hyper/perc/embed/dconv/delta/ple/norm/glu/identity/last/moe_blocks
│   ├── 9861 estimator(name, param) table (kmeans, knn, svm, forest, bayes, cbst, xgbst, lgbm)
│   ├── 9875 operation(value): parse each record; attn handles 10/11/15-field legacy layouts and `v=` value heads
│   ├── 9912 attn record tail: score normalization/dims, rotary layout, optional yarn quadruple, `v=` value heads → AttentionBlock
│   ├── 9952 rnn/gru/lstm/recur/identity/last/residual/ensemble/product/moe_blocks/moe (7-field new form or legacy blocks)/perc/embed/hyper (hex blocks)/dconv (dilation default 1)/delta (optional extents and activations)/ple (RowHash::parse)/norm/glu
│   ├── 10029 normalization_text()/normalization() (0 none, 1 batch, 2 layer, 3 rms, 4 l2); block_text() 12 `|` fields (operation, activation, norm, quantization, profile, qk, frozen, 0, precision, kv, blck, acc); block() accepts 6/8/9/10/11/12 fields
│   ├── 10077 precision_token()/precision_from_token() `family.bits.exp.man.storage`; model_text(); model(blocks, loss, quantization, epsilon, exclusions, precision)
│   ├── 10106 struct StoredGraph { graph, model, precision, inputs, outputs, norm_mean, norm_scale, target_min, target_span, bn_stats, artifact }; struct SemanticGraph { model, precision, input, output, inputs, outputs, tensors, predictors, frozen, state, norm stats, target scale, bn_stats, artifact }
│   ├── 10140 raw_weight(values); semantic_graph(stored): per-node tensors (stored or raw), predictor programs; same_model(); values()/value()/precision() parsers
│   ├── 10212 struct ModelParts; struct SemanticBuilder + finish() (shape/schema/frozen/predictor consistency checks; embed and ple tables own tensors without frozen spans)
│   ├── 10295 stored_weight(format, count, codebook, encoded) decode; load_semantic(path): `.ogdl` line reader — schema section, then per `graph`: model/block/arithmetic/in/out/shape/tensor/predictor/frozen/moments/variances/best_loss/epoch/training_rows/trained_samples/norm_mean/norm_scale/target_min/target_span/bn_stats/artifact
│   ├── 10419 save_semantic(path, schema, graphs) (refresh_storage then semantic_graph); save_semantic_graphs(): writes the document and publishes atomically through a create_new temporary sibling
│   ├── 10492 temporary write + sync + rename, cleanup on failure; join(); same_structure()
│   ├── 10520 artifact_key(model, schema, precision, graph, target): FNV over header, schema, target, precision, model text, per-node spans → `recipe-native-<hash>`
│   ├── 10540 restore(path, schema, graphs, identities): save when absent; on structural match reject resume if evaluation samples were trained or preprocessing differs, then copy saved tensors/state/frozen into the graphs; mismatch prompts `overwrite? Y/n` on stderr/stdin
│   └── 10599 run_infer(path, input, forward); infer_graphs(): chain graphs by named inputs/outputs, normalize inputs, logistic target rescale
├── 10630 std imports (unix/windows OsStrExt, collections, ffi, io, sync, time)
├── 10653 pub static recipe: Recipe; RUN, INTERRUPTED, INTERRUPT_CHECKPOINTED, DEBUG_LOG (recipe.log in CARGO_MANIFEST_DIR), SIGINT/INTERRUPTED_EXIT = 130, SIGNAL
├── 10662 record_interrupt() (writes to fd 2), interrupt() handler (unix signal / windows console), register_interrupt(); TRACE, tracing(), trace(message) appends to recipe.log
├── 10707 pub struct RecipeError(String) + From<fmt::Error>, new(), Display, Error; pub type Result<T>; type Ptr; enum Backend { Cpu, Amd, Nvidia }
├── 10732 pub struct Data { sources, tests, autoregressive, target, features, normalize, split, split_supplied, prepared: OnceLock, file: Option<Gguf> }; enum FeatureSelection { All, Include, Exclude }; struct Auto / const auto; CHAR_IDS [100 chars]
├── 10762 enum RopeLayout { Neox } + code(); trait RopeSelector; struct Neox / const neox
├── 10787 block constructors: layer(width), conv(filters, kernel), norm(selector), pool(size); struct attn(heads) + From<attn> and qk/width/kv/rope/index/gate; rnn/gru/lstm(width); recur([blocks]); perc(width); kmeans/knn/svm/bayes/cbst/xgbst/lgbm; res([..]); ensemble([..]); moe(top_k, [..])
├── 10883 type FitFn / ValidateFn; struct Estimator { fit, validate, param, name } (PartialEq by param+name)
├── 10905 struct Indexer { heads, width, block, keep, tokens, score } + NONE, admitted(); struct AttentionBlock { heads, width, keys, values, rope, yarn, index, gate } + new(heads)
├── 10954 struct DeltaBlock { heads, kernel, key_heads, key_width, value_width, output, conv_activation, output_activation } + new(), extent(channels)
├── 10995 struct PleBlock { heads, width, rows, kernel, dilation, hash } + table()
├── 11010 enum Operation { Layer, Conv, Pool, Estimator, Attention, Rnn, Gru, Lstm, Recur, Residual, Ensemble, Product, Moe(experts, top_k, hidden, activation, scoring, renormalize, shared), MoeBlocks, Perceptron, Embed, Hyper(lanes, rank, blocks), Dconv, Delta, Ple, Norm, Glu, Identity, Last }
├── 11047 pub enum Activation { Linear, Cos, Exp, Log, Ln, Huber, Tan, Relu, Leak, Sigmoid, Tanh, Selu, Gelu, Silu, Elu, Prelu, Scale(bits) } + code()
├── 11093 enum Scoring { Softmax, Sigmoid }; enum BlockNormalization { Batch, Layer, Rms, L2 } + mode() (0/1/2/4; 3 = evaluation); trait NormalizationSelector for Batch/Rms/L2/closure returning layer
├── 11148 macro slots! + pub mod atv (linear/cos/exp/log/ln/huber/tan/relu/leak/sigmoid/tanh/selu/gelu/silu/elu/prelu activation-only blocks); fp_format(bits); macro precision_methods! (f(exp,man), fp, int, bf, tf)
├── 11200 pub struct Block { operation, activation, normalization, qk, quantization, profile, frozen, precision, blck_precision, kv_precision, acc, suffix }; enum Suffix { Fresh, Blck, Kv, End } + for_operation(); PartialEq; struct ProductBranch { blocks, quantization, exclusions }
├── 11269 impl Block: of(), act(), norm(), acc(32|64), qk(), attention() helper, width(), kv() (suffix Kv), rope(layout, dims, base), yarn(factor, context, fast, slow), index(heads, width, block, keep), gate(), quantize(family, bits, variant), profile(), block_activations!, qi()/iq(), arithmetic(format) by suffix, scale(factor)
├── 11391 impl Quantized for Block; precision_methods! for Block, Qk<Block>, attn
├── 11403 pub struct Model { inner: Arc<ModelData>, frozen: Frozen }; pub struct ModelData { blocks, loss, downstream, quantization, precision, epsilon, pending_frozen, exclusions }; Deref; wrap(), edit()
├── 11444 pub struct Frozen + model(); macro qualified_blocks! (frozen.layer/conv/rnn/gru/lstm/perc/dconv/delta/attn/glu/res/recur/ensemble/moe/hyper); pub struct FrozenBlock / static frozen; qualified_parts! + qualify()
├── 11495 trait Exclusion; struct Bias / const bias (mask 1); macro operation_methods!
├── 11509 impl Model
│   ├── 11510 push(operation) (frozen qualifier, model quantization/profile), suffix(), no(option), activate(activation) (opens an Identity step when the last is complete)
│   ├── 11559 layer(width), embed(vocabulary, width), operation_methods! conv/pool/kmeans/knn/svm/bayes/cbst/xgbst/lgbm/rnn/gru/lstm/perc/last/dconv/delta; attn(heads); res/recur/ensemble/moe; gguf_moe()
│   ├── 11604 attention() and delta_block() modifier helpers; dilate(steps); kv(heads); head(width) / width()
│   ├── 11652 delta modifiers keys(count, width)/values(width)/out(width); rope(layout, dims, base), yarn(factor, context, fast, slow), index(heads, width, block, keep), budget(tokens), score(normalization, dims) (rms/l2 only), indexer() helper, gate()
│   ├── 11711 hyper(lanes, rank, branch) (Operation::Hyper), ple(table), norm(normalization) (leading norm opens an Operation::Norm), glu(hidden, activation), qk(normalization)
│   ├── 11746 loss(ModelLoss::Function|Model) sets loss + downstream evaluator, epsilon(value), quantize(family, bits, variant) (family<<12 | variant<<8 | bits, block or model default), qi(bits) / iq(bits)
│   └── 11787 arithmetic(Compute) (panics without a preceding block), acc(bits), description(metrics) (joined operation.activation.qk-.norm.quant names per block)
├── 11832 fn quantization(code) → Q/IQ name; fp16(f32)→u16 / unfp16(u16)→f32 round-to-nearest; put_half/half; qround (magic 12582912.0); positive_max
├── 11886 fn qkx2 (weighted min/scale search over range steps, mad or squared error, writes codes) → (scale, -minimum)
├── 11910 fn q3 (3-bit signed fit with 5 refinement passes, +4 offset); fn qx (levels fit with ±0.9 inverse sweep); fn k_scale(metadata, block) (K-quant 6-bit scale/min unpack)
├── 11957 const IQ4 (16 i8 levels); const IQ3_XXS [u16; 256]; const IQ3_S [u16; 512] grid tables
├── 11985 const IQ2_XXS [u16; 256]; const IQ2_XS [u16; 512]
├── 12016 const IQ1 [u16; 2048]
├── 12091 const IQ2_S [u16; 1024]
├── 12128 const IQ_NEIGHBOR_SHELLS = 3; struct IqNeighbors { exact, candidates: OnceLock per key }; struct IqGrid { points, bits, lanes, shells, neighbors }
├── 12140 impl IqGrid: new(), code(index, lane), key(levels), distance(point, key), neighbors() (exact-key index built once)
│   └── 12172 candidates(key): nearest `shells` distance shells, collects every grid point on them
├── 12203 static IQ3_XXS/IQ3_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ1 _GRID; fn iq_nearest(grid, levels, values, weights, scale) (exact hit or weighted-error best candidate)
├── 12236 fn iq1_level, iq1_nearest (levels -1 + 0.125·shift), iq1_shift(medium, pattern, group)
├── 12269 fn iq1(values, importance, medium): IQ1_S/IQ1_M encoder (256-value chunks, sorted split search per block, grid snap, 3-bit block scales + fp16 super scale ×1.125/1.1125)
├── 12277 fn qp_scale(values, weights, nmax): positive-level scale fit (±0.4 inverse sweep, 5 refinement passes)
├── 12338 fn iq2_xxs(values, importance): IQ2_XXS encoder (sign parity flips, qp_scale seed, ±0.6 sweep, 7-bit sign words, 4-bit block scales)
├── 12346 fn iq2_16(values, importance, xs): IQ2_XS / IQ2_S encoder (16-value blocks, grid membership retry, packed index/sign bytes, scale ×0.9875 for S)
├── 12354 fn iq3_xxs(values): IQ3_XXS encoder (4-lane groups, ±3.0 sweep, scale ×1.0125)
├── 12369 fn iq3_s(values): IQ3_S encoder (9-bit indices with high-bit bytes, sign bytes, paired 4-bit scales, ×1.033)
├── 12379 enum DeviceQuantizer { Iq4Nl, Q4_0, Q4_1, Q8_0 }; fn device_quantizer(codec)
├── 12397 const IQ4_MID (15 midpoints); iq4_code(value), iq4_code_search, iq4_fit(values, tries, codes) → scale
├── 12428 pub(crate) struct StorageFormat(u16); enum NativeDequant { F16, F32, Nf4, Scalar(ScalarLayout), Q2K, Q3K, Q45K(man), Q6K, Q8K, Iq4, Iq1, Iq }
├── 12447 impl NativeDequant: decode<Q: QuantOps>() dispatch into quantized::dequant_*; table() → NativeQuantTable
├── 12483 enum NativeQuantTable { Unsigned, Signed }: name(), definition() emits `@recipe_model_{name}` constant arrays
├── 12510 enum Quantizer (Raw/Scalar/Q2K/Q3K/Q45K/Q6K/Q8K/Nf4/Iq4Nl/Iq4Xs/Iq2Xxs/Iq2/Iq1/Iq3Xxs/Iq3S); struct Quantization { codec, family, bits, variants, block, stride, name, quantizer, native }; macro quantizations! → enum StorageCodec + const QUANTIZATIONS
├── 12550 quantizations! table: F16, F32, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 (32-blocks), NF4, Q2K/Q3K/Q4K/Q5K/Q6K/Q8K (256-blocks), IQ4NL/IQ4XS, IQ3XXS, IQ2XXS/IQ2XS/IQ2S, IQ1S/IQ1M, IQ3S with codes, strides, layouts and tables
├── 12577 fn nf4_codebook(codebook, count, bytes); impl StorageCodec: quantization(), dequantize(data, codebook, count) (NF4 path or decode_blocks)
├── 12601 pub(crate) struct StorageSpec { codec, block, stride }; enum StoredSegment { Owned, Mapped(Arc<gguf::Mapping>, at, len), Absent(len) } length() / Deref
├── 12636 pub(crate) struct StoredBytes(Arc<Vec<StoredSegment>>): mapped(), joined(parts), absent(length), absent_runs(), len(), runs()
│   └── 12675 slice(at, length) gathered across runs; to_vec() (the only mapped copy); From<Vec<u8>>
├── 12707 pub(crate) struct StoredWeight { format, count, bytes, codebook, arithmetic, segments }; format_segments()
├── 12732 struct BoundNode { names, elements, weight }; enum BoundWeight { Stored(StoredWeight), Values(Vec<f64>) }
├── 12751 impl StorageFormat: valid(), named(name), spec() → StorageSpec, encode(arithmetic, importance, config) → StoredWeight
│   └── 12772 unavailable() (lists every GGML format), selection() (S/M/L style variants), tensor(role, more, output) picks the per-tensor format of a mixed K/IQ selection
├── 12814 trait Integer { compress, decompress, bits }; fn decode_blocks (one contiguous block run per CPU worker via parallel_map); fn encode_blocks; fn block_values
├── 12865 impl Integer for StorageFormat: bits(); compress() Scalar arm (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 32-blocks: half scale/min, nibble packing, Q5 high bits, Q8_1 sum)
│   ├── 12923 Q2K arm: 16 qkx2 fits, 4-bit scale/min codes, 2-bit code packing, trailing halves
│   ├── 12957 Q3K arm: q3 fits, 6-bit scales packed into 12 bytes, high-bit mask + 2-bit low planes
│   ├── 13005 Q45K arm: 8 qkx2 fits over 32, 6-bit scale/min metadata (k_scale), nibbles + Q5K high bits
│   ├── 13056 Q6K arm: qx 32-level fits, i8 block scales, low nibble / high 2-bit split
│   ├── 13095 Q8K arm (f32 scale, i8 codes, per-16 sums); Nf4 arm (NF4 16-entry table, block scale codebook, nibbles)
│   ├── 13141 Iq4Nl arm (iq4_fit, tries -1); Iq4Xs arm (8 iq4_fit, 6-bit block scales split low/high)
│   └── 13187 importance-gated IQ2_XXS / IQ2_XS / IQ1_S/M arms, IQ3_XXS, IQ2_S, IQ3_S arms, unavailable() fallback; decompress() → codec.dequantize
├── 13224 pub trait Quantized { quantize }; pub struct Qi<T>(.0, .1, QiSuffix { nf, k: Qk }); Qk<T> { model, s, m, l }; Iq<T> { xxs, xs, s, m, nl }; Deref impls
├── 13262 fn qi_of (bits 2/3/4/5/6/8), iq_of (bits 1..4); impl Quantized for Model; precision_methods!(Model), (Qk<Model>); Qk<Model>/Qk<Block>::arithmetic
├── 13289 impl Estimator::name; impl Operation: name() table (layer/conv/pool/attn/rnn/gru/lstm/recur/residual/product/ensemble/identity/last/moe/perc/embed/hyper/dconv/delta/ple/norm/glu), weighted()
├── 13337 impl Activation::name (linear/cos/exp/log/ln/huber/tan/relu/leak/sigmoid/tanh/selu/gelu/silu/elu/prelu/scale); impl BlockNormalization::name (bnorm/lnorm/rms/l2)
├── 13370 macro activations! → Model::cos/exp/log/ln/huber/tan/relu/leak/sigmoid/tanh/selu/gelu/silu/elu/prelu; impl Model::scale(factor) (Activation::Scale bits)
├── 13399 impl Mul for Model → Block of Operation::Product(ProductBranch, ProductBranch) (epsilon must match); From<Model> for Block (single block or Residual); impl Mul for Block
├── 13430 pub struct Recipe, Adamw, LossFunction(u8); pub enum ModelLoss<'a> { Function, Model } + From impls; pub struct Metric(u8), ZScore; type Normalization / Norm / Loss
├── 13456 consts adamw, mse/rmse/huber/mae/bce/ce(=bce)/focal; metrics Run/Loss/R2/Time/Epoch/blck/tile/Score/Window/Choices; const all [9] / dev [10]; trait IntoMetrics (Metric, [Metric; N])
├── 13502 consts z_score/batch/rms/l2 with marker structs Batch/Rms/L2; impl LossFunction: name(), value(prediction, target, threshold) (mse/rmse/huber/mae/bce/focal)
├── 13543 impl Recipe: data(sources) (a lone .gguf source opens the script file), model() (epsilon from the open GGUF or the Cargo default), train() defaults
├── 13567 fn infer_ids(path, sequences, device): bundle::load_semantic, materialize_saved_graph, NativeTape with TapeInput::Ids, inference forward, target_span rescale
├── 13597 impl Recipe: gguf(path) → Gguf::open, predict(path, input) via bundle::run_infer, infer_ids(path, sequences)
├── 13625 impl Gguf: contract(name, input, width), expert(name, index, input, width), infer(blocks, plan, input, channels), decode(...), decode_stream(..., emit), plan() → Binding, named(name)
├── 13667 pub struct Binding { nodes: Vec<Vec<Plane>> }; pub(crate) enum Plane { Mapped(GgufTensor), Owned { name, values } } elements()/name()/mapped(); impl Binding node(planes) / named(model, name)
├── 13721 fn infer_gguf (bound_graph → tape forward → predictions); fn decode_gguf (prompt into samples, decode_steps with forward_window strides of tile.m, last_logits)
├── 13768 fn bound_graph / bound_graph_on (Shape from channels, config.quantization from the file, Prepared with bound plan and zero target width, compile)
├── 13801 enum RopePairs { Halves, Neighbours } (RopeSelector → always Neox); struct Architecture { names, rope, delta_activation }; const ARCHITECTURES (llama, gemma3, qwen2/qwen3/moe, qwen35/qwen3next silu·silu, qwen4exp silu·sigmoid)
├── 13836 pub struct Bound { file, model, plan, blocks, tensors, vocabulary }; Gguf::model() → Builder::build; impl Bound blocks()/tensors()/vocabulary()/model()/plan(), infer(ids), place(positions, split), decode(positions, split, prompt, sampler, stop, budget)
│   └── 13892 decode() places then decodes; serve(positions, split, address, requests)
├── 13900 struct Builder<'a> { file, architecture, rope, delta_activation, plan, consumed }; struct Dimensions { width, heads, kv, head, rope_dims, rope_base, interval, delta, feed_forward, experts, hyper, indexer, compression }; struct DeltaDims; struct ExpertDims { count, used, hidden, scoring, renormalize }
├── 13943 impl Builder: build(file): architecture row lookup, dimensions, token_embd shape check, rms epsilon, embed block keeps the tensor's own format, optional Ngram ple, per layer attn|delta then ffn|experts branches, output_norm or head-mixer gates, tied output, unread-tensor error → Bound
│   ├── 14011 key()/integer()/integer_or()/present(); dimensions(): embedding_length, head counts, key/value length, rope dims/base, full_attention_interval → DeltaDims from ssm.*, expert_count → ExpertDims (gating func, weights_norm), feed_forward_length, hyper_connection count/rank, indexer head_count/key_length/top_k, compress_ratios
│   ├── 14076 tensor(name, role), optional(name), mapped(planes), slot(planes), whole(name, role), projection(name, role, inputs, outputs), scale(name, role, width, groups, order) (permuted Owned plane), head_order(width, dims), head_rows(tensor, base, order)
│   ├── 14138 attention(branch, layer, dims): attn_q (gated when 2× wide), attn_k/attn_v, per-head row views in head order, gate rows, qk(rms) scales, rope, indexer q/k planes + budget + score scales, attn_output
│   ├── 14209 delta(branch, layer, dims): ssm_alpha/ssm_beta gate planes (+ dt bias, zero beta bias), attn_qkv, ssm_conv1d taps, ssm_a → ln(-a) Owned plane, ssm_norm scales, attn_gate, ssm_out, activation pair from the row
│   ├── 14259 feed_forward (ffn_gate/up/down → glu silu); experts (ffn_gate_inp router, *_exps [k, n, experts] tables, optional shared-expert gate + shexp projections → gguf_moe)
│   ├── 14302 ple(layer, ngram, dims): host table, ple_key, ple_norm_key/query, ple_value, ple_norm_conv, ple_conv1d taps
│   └── 14328 open(layer, part, dims) (hyper mixer hc_{part}_norm/down/up/inject, else attn_norm / ffn_norm / post_attention_norm pre-normalization); close(model, branch, dims) → model.hyper(lanes, rank, branch)
├── 14371 static SCRIPT_FILE: OnceLock<(PathBuf, Gguf)>; fn script_file(); fn open_script_file(path) (one model per process); impl Gguf::rms_epsilon()
├── 14391 pub struct ArchitectureKeys { embedding_length, block_count, feed_forward_length, context_length, vocab_size, final_logit_softcapping, attention: AttentionKeys, rope: RopeKeys { freq_base, dimension_count, scaling: RopeScalingKeys } }; ArchitectureKeys::load(file, prefix) with conventional defaults
├── 14457 pub struct Namespace { prefix, keys: OnceLock<ArchitectureKeys> } (Deref loads from the script file); macro namespaces! → statics gemma3 llama qwen2 qwen3 phi3 deepseek2 glm4 granite
├── 14471 pub struct TokenizerKeys { ggml: GgmlKeys { tokens, model, pre, bos_token_id, eos_token_id, add_bos_token }, chat_template }; TokenizerNamespace Deref (leaked statics from tokenizer.*); pub static tokenizer
├── 14512 trait Width (usize, &[String]) extent(); trait Root sqrt(); const chat = Metric(14), debug = Metric(15); fn arm_trace(metrics) sets TRACE
├── 14547 pub struct Infer { log, tokens }; Recipe::infer() (32 tokens); impl Infer log(metrics), tokens(count), run(model, data)
│   └── 14576 try_run: with_last_projection, conventional_plan, context_length ceiling, chat-template prompt + double-BOS trim, stop_ids, fitting_context, greedy decode_gguf streaming the reply to stdout, prefill/step timing and tok/s
├── 14625 fn stop_ids(coder) (eos plus the id the template closes an assistant turn with, via a sentinel render); fn with_last_projection(model) inserts Operation::Last before the vocabulary layer
├── 14659 fn fitting_context(file, model, plan, device, ceiling): free bytes less RECIPE_PLACEMENT_LAUNCH_RESERVE_BYTES, part_bytes at ceiling and half → per-position slope, largest power of two under the estimate, steps down by 1/16 until it fits
├── 14700 fn conventional_plan(file, model): binds token_embd, then per Residual a block index (attends → attn, else ffn)
│   ├── 14732 step arms: Attention → attention_planes, Product → ffn_gate/ffn_up (activated branch is the gate), Glu → gate/up/down, Layer → ffn_down; normalization → attn_norm / post_attention_norm / ffn_norm / post_ffw_norm; residual norm → output_norm
│   └── 14779 trailing Layer(vocabulary) → output.weight or the tied embedding; Identity/Last carry no normalization; unknown operations error
├── 14795 impl Builder: norm_scale(name, width); attention_planes(layer, attention, normalized, width) (gate presence must match the block, head-ordered q/k rows, value, gate rows, qk scales, attn_output)
├── 14849 pub struct Sampler { temperature, top_k, top_p, min_p, penalty, window, state }; builders temperature()/top_k()/top_p()/min_p()/repeat(penalty, window)/seed()
│   └── 14891 sample(logits, previous): repetition penalty over the window, greedy at temperature ≤ 0, sort, top-k, top-p + min-p mass cut, temperature softmax, LCG draw
├── 14930 pub struct Generation { ids, logits, prefill_seconds, step_seconds }; impl Recipe: sampler() defaults, decode(path, prompt, sampler, stop, budget), serve(path, address, requests), place(path, split), place_primary(path)
├── 14964 fn request_field / request_number / request_ids (query-string parsing); fn try_serve(placed, address, requests) (TcpListener, 400 on error)
├── 14987 fn serve_decode(placed, stream): reads the request head, ids/stop/budget/temperature/top_k/top_p/min_p/repeat/seed fields into a Sampler
│   └── 15012 min_p/penalty/seed fields, chunked `200 OK` header, try_decode streaming each id as its own chunk, terminating chunk
├── 15033 fn trace_logits(step, logits) (first/last three, sum, argmax as llama-eval-callback prints); fn decode_steps(tape, samples, prompt, sampler, stop, budget, emit, logits): prefill then one id per step, timings, stop ids
├── 15079 enum PlacedSource { Saved(Vec<bundle::SemanticGraph>), Bound(Shape) }; pub struct Placed { source, split, tapes: Vec<Vec<NativeTape>>, resident, moved }; fn saved_statistics(nodes, rows)
├── 15106 fn part_bytes(part, precision) (input + weight arena + values/contexts arenas + load scratch); fn stored_first_values(weight, span, stride, block); fn storage_scratch_bytes(graph); fn cuts_connection(graph, start)
├── 15145 fn measured_split(graph, precision, devices): block starts, greedy fill against each device's free bytes less RECIPE_PLACEMENT_LAUNCH_RESERVE_BYTES, never cutting a residual or input connection
├── 15180 fn graph_part(graph, start, end): rebased sources/second/offsets, no Predictor nodes, sliced parameters/frozen/stored/requantize, input from the previous node's output
├── 15233 fn split_graph(graph, split) → parts; fn window_runs(shape, begin, end) (whole arena or one run per channel)
├── 15253 fn place_ranges(graph, split, devices, precision, bn_stats): measured or explicit split validated against free memory, one range_tape per part, resident and hop bytes, bn statistics fully consumed
├── 15283 fn place_model(path, split, devices) (bundle::load_semantic, materialize_saved_graph per graph, place_ranges); fn place_bound(model, positions, split, devices) at Compute::FP64
├── 15313 impl Placed: infer(input), decode(prompt, sampler, stop, budget), serve(address, requests), split(), resident_bytes(), moved_bytes()
│   ├── 15347 try_decode(prompt, sampler, stop, budget, emit): sequence from the saved graph or bound shape, one range set, run_window + last_logits per step, prefill/step seconds, stop ids
│   ├── 15397 last_logits(predictions, begin, end) on the output range; run_window(samples, begin, end) (bundle::infer_graphs per saved graph, or the bound graph directly)
│   └── 15420 forward_window(tapes, samples, begin, end): reset_sequence at position 0, write_tokens on every range, input runs into the first, forward each range and hop only the window's output runs to the next
├── 15447 struct Shape { channels, length } elements(); fn select_last_logits(predictions, output, position)
├── 15463 enum Primitive (Contraction 0, Pool 2, Attention 4, Scan 5, Elementwise 6, Normalize 8, Predictor 9, Gather 10, Rope 11, Expand 12, Read 13, Outer 14, TopK 15, ExpertIn 16, Dconv 17, Delta 18, ExpertOut 19, Lookup 20, Fold 21, Last 22)
├── 15490 struct ScalarProgram(Vec<f64>): op(opcode, left, right), constant(), select(), choose() (selection, never a multiply blend), unary()
├── 15516 impl Node: weights() (Gather/Lookup span their tables), table(), identity(index) description string
├── 15560 struct Node { op, source, second, input, output, offset, parameters, argument: [f64; 9], program_offset, program_count, precision, block fields, packed, ... }
│   └── 15572 block_index, block_kind, frozen, packed, int_bits, precision, acc, kv_precision; struct TrainingState { moments, variances, best_loss, trained_samples, epoch, training_rows }
├── 15599 struct Graph { nodes, parameters, frozen, programs, stored, requantize, input, output, source, state, block_index, block_kind, lanes, rank, block_frozen, block_precision, block_acc, block_blck_precision, block_kv_precision, profile: Precisions, bound: VecDeque<BoundNode>, bound_values, bias, epsilon }
├── 15639 impl Graph: new(shape, epsilon), refresh_storage(config); fn encode_graph_storage(graph, config) (mapped weights stay bytes, tables re-drawn raw, importance from the variance slice, format.encode per node)
├── 15704 fn sequential_operation(operation) (recurses into residual/ensemble/moe/product/hyper parts); fn compile(model, data, targets, rows, gpu, config, initialize): storage validity, sequence vs feature shape, Graph::new + profile + bound plan + bias exclusion
│   ├── 15729 lower_block per block with block_index/kind/frozen, precision trace, lower_collapse when lanes are open, output projection (lower_conv) with model/run quantization, leftover plan entries error, bound_values written into spans, output tensor format
│   └── 15772 initialize_graph + target-mean output bias, frozen mask after initialization, encode_graph_storage
├── 15787 fn materialize_saved_graph(saved, samples, gpu, config): Prepared from the saved shapes and predictors, compile, input/output/weight-count checks, saved tensors into parameter spans and stored slots, frozen + state restored
├── 15826 fn split_at_block(graph, block) → (head, tail) with sources rebased across the boundary
├── 15858 fn range_tape(graph, samples, tokens, gpu, precision, bn_stats, statistics) (injects the part's bn stats); fn forward_part; fn append_graph(graph, part) → new source (rebases sources, offsets, program offsets)
├── 15896 fn lower_block(graph, block, total, data, targets, rows, gpu, config): collapse open lanes unless Hyper/Ple, push block qualifiers, dispatch Layer/Perceptron→lower_project, Conv, Pool, Embed, Dconv, Delta, Ple, Attention, Rnn/Gru/Lstm→lower_scan(1/3/4), Residual, Recur, Ensemble, Product, MoeBlocks, Moe→lower_gguf_moe, Hyper, Norm, Glu, Last, Identity, Estimator
│   └── 15942 lower_activation, lower_normalize, block or run storage format (S/M/L tensor roles by depth; `more` band) + requantize_bound, Compute::Int sums → Q8_0/Q4_0 packed with int_bits (kept when already narrower), batch narrowing, restore outer qualifiers
├── 16010 fn requantize_bound(graph, index, format, config): device-encodable formats → StoredBytes::absent + requantize source; else host decode per span and encode once, cached in REQUANTIZED by (format, count, source runs)
├── 16070 static REQUANTIZED cache; fn push_node(graph, op, output, parameters, argument, second): precision/acc/kv from the block's suffixes or the profile, Node construction, plan entry pop (Stored stays packed, Values deferred to bound_values, tables bind only packed)
│   └── 16132 span reserve for unbound nodes, pushes node/stored/requantize; fn push_program(graph, second, initial, program) (Elementwise node carrying a ScalarProgram); fn push_predictor(graph, program) (frozen table, locals/stack arguments)
├── 16167 fn lower_activation(graph, activation, config): fuses ReLU into a preceding contraction's argument[1], else builds a ScalarProgram: Cos, Exp, Log/Ln (signed log1p), Huber (activation[7]), Tan, Relu (select)
│   └── 16218 Leak (activation[0]) / Prelu (Parameter slope) / Elu / Selu (masked exp, alphas activation[2..3], scale [4]), Sigmoid / Silu via tanh, Tanh, Gelu (constants [5], [6]), Scale(factor); Prelu initial from activation[1]
├── 16283 fn contraction_bias(graph, matrix, width, has_bias) (bias row inferred from the bound view sizes); fn contraction_arguments(kernel, has_bias); fn lower_project(graph, channels); fn lower_contraction(graph, channels, has_bias)
├── 16330 fn lower_flatten_project(graph, target); fn lower_conv(graph, filters, kernel); fn target_means(targets, width, rows) (finite rows only); fn output_bias_offset(graph) (walks the single-source chain to the output contraction)
├── 16379 fn lower_pool(graph, size); fn lower_last(graph) (Primitive::Last to length 1); fn lower_embed(graph, vocabulary, width) (Gather, first block only, no parameters); fn lower_dconv(graph, kernel, dilation)
├── 16408 fn lower_ple(graph, ple, config): hash validation, Lookup node with hash words as program, key projection + grouped rms, stream rms, key·query product folded per lane
│   └── 16448 gate program sigmoid(sign(s)·sqrt(max(|s|, 1e-6))), value projection broadcast by Outer over lanes, rms + dconv(kernel, dilation) + silu, both terms added into the stream
├── 16481 fn yarn_parameters(factor, context, dims, base, fast, slow) → (mscale, low, high) correction range
├── 16499 fn lower_delta(graph, delta, config): 2·heads gate projection, qkv projection + dconv(kernel) (+ conv activation), L2 over the query/key span, Delta node (RECIPE_DELTA_CHUNK), rms over values, gate projection × output activation, output projection
├── 16538 fn lower_attention(graph, attention, qk): head partition repair with a printed suggestion, QKV (+gate plane) projection, qk normalization over query/key heads, Rope node [dims, base, width, rotated, mscale, factor, context, low, high]
│   └── 16596 indexer side path: own projection, scored rms (trailing key scales) or L2, indexer Rope, Attention node [heads, keys, gate, block, admitted, index heads, index width, epsilon, values], closing projection to the input width
├── 16640 fn lower_normalize(graph, normalization, width, span); fn lower_normalize_parameters (Normalize node [mode, epsilon, width, span], scales initialized to 1.0); fn attention_blocks(node)
├── 16668 fn reset(graph, source, shape); fn program(graph, first, second, shape, initial, program); fn binary(graph, first, second, shape, opcode); fn constant(graph, source, shape, value); fn activation(graph, source, shape, value, config)
├── 16695 fn lower_gated(graph, gate, up, shape, activation, config) (act(gate) × up); fn expert(graph, source, shape, block, ...) lowers one block on a source; fn project_moe_shape; scalar helpers maximum(), one_minus(), greater_than()
├── 16739 fn rank_mask(graph, scores, selected, shape, top_k) (pairwise Greater with tie-break by index); fn select(graph, branches, scores, shape, top_k, config) (max-centered exp, rank mask, normalized routed sum)
├── 16786 fn lower_experts(graph, source, input, routing, experts, top_k, hidden, activation, config) (ExpertIn gate and up tables, gated product, ExpertOut); fn lower_glu(graph, hidden, activation, config) (gate/up projections, product, down)
├── 16819 fn lower_gguf_moe(graph, experts, top_k, hidden, activation, scoring, renormalize, shared, config): router projection, TopK node [top_k, scoring, renormalize], lower_experts, shared expert under a bias-free sigmoid [width] gate added to the dispatch
├── 16849 fn lower_moe_blocks(graph, top_k, experts, ...) (each expert block adapted to the first's shape, one router projection per expert, select); fn lower_scan(graph, channels, gates) (Scan node, gates × (input + state [+ bias]))
├── 16885 fn recur_activation(activation) → 0..3; fn lower_recur(graph, parts, ...): leading layer width, optional identity activation stage, program record [width, activation], Scan node, body graph lowered as "recur_body" and appended, scan arguments [1]/[4] = body count, [3] = body start, [5] = 1
├── 16957 fn estimator_count(block); fn first_estimator(block); fn lower_ensemble(graph, members, ...) (members summed from one source, scaled by 1/n)
├── 16997 fn lower_residual(graph, parts, precision, skip, ...): lowers the parts, adapts the skip path by conv (shorter) or projection (channels), Add program in the residual's own precision under block_kind "residual"
├── 17033 fn lower_product(graph, left, right, ...) (both branches from one source with their own bias exclusions and frozen flags, elementwise Multiply)
├── 17060 fn lower_hyper(graph, lanes, rank, blocks, ...): Expand to lanes on first use, lower_gates, Read node under the read gate, inner blocks with lanes closed, Outer under the write gate, Add into the stream
├── 17091 fn lower_gates(graph, lanes, rank, write, config): rms over lanes, down → /lanes → silu → up → sigmoid read gate, inject → /lanes → sigmoid → ×2 write gate; fn lower_scale(graph, factor); fn lower_collapse(graph, config) (Read to the lane mean, lanes = 0)
├── 17138 fn lower_estimator(graph, estimator, data, targets, rows, gpu, config): restored predictor programs or a fresh fit per target channel (graph_inputs, estimator.fit), frozen surrogate graph (fit_surrogate or untrained)
│   └── 17203 push_predictor, append_graph(surrogate), StraightThrough program joining real and surrogate, place_estimator_channel and accumulate across channels
├── 17226 fn place_estimator_channel(graph, channel, width) (frozen one-hot route, no bias); fn next_weight(state, scale) LCG; fn initialize_graph(graph, config)
│   └── 17252 fan-in span per op (Gather width, ExpertIn input, ExpertOut hidden), tables drawn once into stored, uniform ±initial/√fan_in for unfrozen spans, zero biases, identity Dconv taps, Scan gate biases zero (LSTM forget gate 1.0)
├── 17309 fn arguments(first, second) → [f64; 9]; fn checked_add / checked_mul / require / logistic
├── 17324 struct Tile { m, n, k }; struct NativeSchedule { matrix, block, tile, register_m/n/count, fragment_k, chunk_k, chunk_values, chunk_bias_values, scratch_base, shared_values, contractions, attention }; struct NativeContractionTiles { forward, gradient, previous, gradient_shape, parameters }; struct NativeContractionShapes
├── 17363 enum MultiDevice { Local, Forced, Auto }; enum PrecisionKind { Sum, Embed, Attn, Rope, Atvn, Norm, Res } blck(); pub(crate) struct Precisions { sum, embed, attn, rope, kv, atvn, norm, res, acc } (Default fp16 everywhere, acc fp32)
├── 17410 impl Precisions: of(kind), load() (RECIPE_CONFIG or RECIPE_DEFAULT_CONFIG looked up in RECIPE_PRECISION_PROFILES; key = value entries; acc must be fp32/fp64)
├── 17454 fn precision_named(name) (fp8/fp16/fp32/fp64/bf16/tf32/int8/int4/int1); fn precision_kind(op, block_kind) (residual Elementwise → Res, Elementwise/Expand/Read/Last/Fold/TopK → Atvn, Normalize → Norm, Attention, Rope, Gather/Lookup → Embed, else Sum)
├── 17481 struct Config { multi_device, kmeans/svm/tree/forest/bayes/boost/catboost/xgboost/lightgbm knobs, quantization_block, surrogate_epochs/width/rate, schedule_candidates/measurements/budget/warmups/minimum_improvement, initial, beta1, beta2, epsilon, decay, progress_refresh_hz, random_seed, activation: [f64; 8], precision, profile, quantization }
├── 17527 impl Config::load() from the RECIPE_* env! values (RECIPE_MULTI_DEVICE, estimator knobs, RECIPE_SCHEDULE_*, RECIPE_ADAMW_*, activation constants RECIPE_LEAK_SLOPE…RECIPE_HUBER_THRESHOLD, Precisions::load)
├── 17587 fn default_epsilon() (RECIPE_NORMALIZATION_EPSILON); parsers number() (finite, positive), fraction() (≤ 1), natural() (nonzero), count()
├── 17607 fn stored_graph(graph, model, data, scale, precision, target) → bundle::StoredGraph (input names, target outputs from the schema, norm stats, target scale, artifact key)
├── 17626 struct HostLookup { context, precision, hash, table, width, length }; struct NativeTape { program, precision, values, contexts, context_resets, lookups, tokens, adjoints, batch_normalizations, samples, input_adjoint, targets, weights, frozen, moments, variances, gradient, metrics, best_loss, rows, parameters, step, input, output, nodes, capacity, positions, vocabulary, reached }
├── 17671 macro ptrs!; enum ForwardMode { Inference, Training }; enum EpochOperation { Full, Gradient, Optimizer } gradient()/optimizer(); enum TapeInput<'a> { Values, Ids } len()/label()
├── 17717 fn native_context_regions(graph, layout, weights, precision) (a NearestIndex per predictor node running a nearest program, built from its own weight span)
├── 17740 impl NativeTape::new(graph, samples, tokens, targets, gpu, precision, loss): batch/target checks, gpu.native_program, optimizer/adjoint buffers only when training, saved bn statistics and best_loss, ids vs values upload (token_id bounds), upload_weights
│   └── 17789 model load: each stored node written into one scratch sized by the largest source, launch_model_load + synchronize, traced comparison of the first written values against the host decode
│   ├── 17812 load trace tail; zeroed values/contexts arenas; embedding table runs written into the gather's context; a HostLookup (hash words, table, width, length) per Lookup node
│   ├── 17854 context_resets over every mutable context span (gather tables and evaluation-mode normalizations persist), nearest-index regions, tape trace line, write_contraction_schedule, NativeTape construction, stage_lookups(0, positions)
│   ├── 17917 write_schedule(); apply_contraction_schedule(contractions) (re-derives the dominant tile); snapshot_state() → NativeTrainingState (every buffer downloaded); restore_state(state); epoch_state(objective) → NativeEpochState
│   ├── 17985 forward(mode); token(value) → id; stage_lookups(begin, end) (rows every lookup's hash addresses gathered on the host into its staging context)
│   ├── 18018 forward_window(begin, end, mode) / forward_window_with_samples(samples, ...): reached marker, stage_lookups, single-position step dispatch thread count on AMD/NVIDIA, launch_forward, per-node clocks trace when the layout has a clocks slot
│   ├── 18053 evaluate(graph, samples, first) (upload weights, bounded chunks of `capacity` rows, arenas cleared per chunk, inference forward); input_runs(begin, end)
│   ├── 18091 write_samples(first, values) (i32 ids or input-precision floats); write_tokens(first, values); output_window(begin, end) (per-node window derivation: Predictor whole, Pool /size, Last → 0..1, kernel lag)
│   ├── 18134 reset_sequence() (tokens, samples, values, mutable contexts); resident_bytes(); inject_bn_stats(stats); extract_bn_stats(); predictions()
│   ├── 18170 last_logits(predictions, begin, end); output(first, count) (download from the last value arena; traced per-node value heads once or on a nonfinite output; finite check)
│   ├── 18220 predictions_at(node, output); epoch_launch(rate, config, operation) (AdamW constants packed in the state precision, 23-argument call, launch_epoch with gradient/optimizer flags)
│   ├── 18270 objective() (metrics[0]); full_epoch(rate, config); gradient_launch; optimizer_launch; advance(); weight_segments(); weights()
│   └── 18307 download_gradient(); upload_gradient(gradient); upload_weights(weights); optimizer_state(); capture(graph) (weights, moments, variances, epoch, best_loss); tile(); schedule() extents string
├── 18372 schedule() tail; NativeTape::device_label(); fn device_label(gpu) (host:name); fn observe_loss(best_loss, loss, tolerance) → checkpoint trigger over the four-slot loss state
├── 18401 struct TransferCost { latency, bandwidth } seconds(bytes); struct Link { to_host, from_host, work, overhead }; fn measure_link(gpu, config) (RECIPE_TOPOLOGY_PROBE_BYTES latency and bandwidth probes both ways, then calibrate)
├── 18454 static LINKS: OnceLock<Result<Vec<Link>>>; struct Transfer { from, to, bytes, cost } seconds(); struct Placement { shares, gradient_to_host, gradient_to_primary, weights_to_host, weights_from_host, loss, predicted: [f64; 4] } movements()/seconds()
├── 18486 fn gradient_work(graph, rows) (2mnk per contraction tile + elementwise + loss reduction); fn optimizer_work(graph); fn calibrate(gpu, config) (surrogate model on synthetic rows, minimum full vs optimizer-only epoch timings → work/s and dispatch overhead)
├── 18559 fn route_counts(route, links, rows, policy) (work-proportional row shares, one row minimum, largest remainders)
├── 18585 fn plan_route(route, links, graph, rows, bytes, loss, policy) → (counts, Placement) with predicted [computation (slowest shard + optimizer), transfers, synchronization, persistent-state movement]
├── 18613 fn select_route(gpus, graph, rows, precision, loss, config): measured links printed, candidates by policy (Local [0], Forced all, Auto every subset), fastest device leads each route, lowest predicted epoch wins
├── 18654 struct DeviceTape { shards, placement }; new(graph, samples, targets, gpus, precision, loss, config) (batch normalization forces one device, select_route, one NativeTape per contiguous row shard)
│   ├── 18690 forward(), predictions(), evaluate(), inject_bn_stats(), extract_bn_stats() (training forward first), advance(), step(), best_loss(), tile(), tune(rate, epochs, config) (launch budget split across shards → tune_contraction_schedule), schedule()
│   ├── 18758 epoch(rate, tolerance, config): one shard → full_epoch; else scoped threads run gradient_launch + download_gradient per shard, share-weighted loss (rmse recombined as a root), aggregate gradient to the primary, optimizer_launch, weights broadcast, observe_loss
│   └── 18810 weights(); capture(graph); print_devices(graph) (gather/lookup movement lines, per-shard rows and share, every movement with latency/bandwidth/ms)
├── 18849 enum CheckpointStatus { Saved, Kept }; fn checkpoint(path, schema, stored, tape) (keeps a saved bundle with a better best loss); fn structural(value); fn graph_rows_buffer(shape, rows, element); fn attention_kv_bytes(node, rows, precision)
├── 18885 fn embedding_row(node) → (Quantization, row bytes) (whole blocks per row, no NF4); fn arena_weight(node, stored) (tables excluded); enum History { None, Positions(n), Sequence }; struct Carried { history, values } NONE / any()
├── 18933 fn carried(node, rows) → Carried per op (Attention: softmax statistics + indexer representatives over the sequence; Scan: 2·gates+1 states, one position back; Delta: chunk entry/live/replay state spans; Dconv (kernel−1)·dilation tail; kernel lag; Pool/Lookup whole sequence; batch Normalize mean+variance); fn carried_state(nodes, rows)
├── 18989 const NEAREST_LEAF_ROWS = 16, NEAREST_NODE_FIELDS = 5; fn nearest_rows(parameters, features); fn nearest_index_shape(rows, features); fn nearest_layout(node, programs) → (features, rows, bytes) for predictors running a Nearest opcode
├── 19006 fn recurrent_body_elements(graph, node, rows) (scan state arena, cell scratch, value and adjoint tapes per position, body gradient tape, cell adjoint)
├── 19043 fn node_context(graph, node, rows, precision, inference): per-op context bytes — Elementwise scalar partials (NATIVE_SCALAR_PARTITIONS), Predictor workspace, Attention statistics/representatives/scores/derivatives, Scan spans/gradients/scratch, Delta pairs, Pool u64 context, Gather table bytes, Lookup staging, ExpertIn/Out u32 buckets, Normalize statistics + partials
├── 19119 fn narrow(value, role) → i32; struct Buffer { runtime, pointer, bytes }; const ZERO_FILL_BYTES = 64 MiB; impl Buffer: upload<T>(), zeroed() (bounded host blocks), upload_float(), write_runs(offset, stored), reserve()
│   └── 19161 upload_weights(runtime, graph, precision, inference) (packed runs written from their mappings, load-kernel nodes skipped, other spans encoded in node precision); write_float_bytes; write_bytes; clear; clear_range; download<T>; download_range; download_float; download_float_bytes; Drop frees
├── 19238 struct Kernel { object, shared, element, kernarg, private, layout }; struct Dispatch { kernel, geometry }; type NativeForward / NativeModelLoad / NativeCpuThread / NativeEpochF64/F32/F16/F8; enum NativeCpuEpoch; struct NativeCpuProgram { _library, thread, forward, epoch, model_load }
├── 19279 #[cfg(amd)] struct HsaReader / HsaExecutable (+ Drop); struct NativeHsaProgram { executable, step, kernarg, kernarg_size, grid_sync, free }; consts HSA_IMPLICIT_ARGUMENT_ALIGNMENT/BYTES, HSA_MULTIGRID_SYNC_POINTER_OFFSET = 88, HSA_GRID_SYNC_ALIGNMENT/BYTES/GROUPS_OFFSET
├── 19332 #[cfg(nvidia)] struct NativeCudaProgram { module, step, unload } + Drop; enum NativeBackend { Cpu, Amd, Nvidia, Remote }; struct NativeProgram { gpu, artifact, backend, forward, epoch, model_load, tile, shapes, schedule, shared_values, reduction_values, gradient_bytes }; Drop for NativeHsaProgram (kernarg)
├── 19381 enum NativeEntry { Forward, Epoch, ModelLoad }; fn native_symbol(name) (NUL-terminated); fn native_artifact_contract(artifact); Kernel::remote(shared, element, layout)
├── 19416 #[cfg(nvidia)] struct Cuda { _runtime, context, set/allocate/free/upload/download/clear/memory_info/synchronize/launch/load/unload/function/function_attribute/occupancy, cus, wave, workgroup, block_lds, sm_lds, registers, threads }; Kernel::cuda(object, shared, element, layout)
├── 19457 #[cfg(amd)] struct Hsa { _runtime, reader_create/destroy, executable_create/destroy/load/freeze, symbol, symbol_info, info, allocate/free/allow/copy/clear/store/wait/write, queue, signal, cpu_agent, vram_pool, kernarg_pool, agent, cus, wave, workgroup, lds, simd_per_cu, waves_per_simd, vgprs_per_simd }
├── 19492 const REMOTE_ALLOCATE/FREE/UPLOAD/DOWNLOAD/SYNCHRONIZE/LOAD/LAUNCH/MEMORY/CLEAR = 1..9; struct Wire<R, W> { input, output, role }: write_u8/u32/u64/bytes, flush, read_u8/u32/u64/into, read_status(action) (nonzero carries the worker's message), status(result)
├── 19568 type RemoteChannel; struct Remote { channel, wave, worker_threads }; enum Driver { Cpu, Hsa, Cuda, Remote }; struct Gpu { name, backend, native_target, driver, memory, shared_limit, dispatch: Mutex<()> } (Send + Sync); fn native_target_label(target)
├── 19601 #[cfg(amd)] repr(C) struct HsaQueue / HsaPacket; #[cfg(nvidia)] type NvQuery; struct Library(usize): open(name) via native_library, function<F>(name) via dlsym / GetProcAddress, Drop dlclose / FreeLibrary
├── 19667 fn load_native_cpu(artifact) / load_native_cpu_path(path, state_bytes, training, has_storage) (thread, forward, epoch symbol by state width f64/f32/f16/f8, model_load); fn driver_status(backend, status, action)
├── 19694 impl Gpu: status(), activate() (CUDA context set); native_program(graph, rows, precision, loss): resident waves, widest element, contraction shapes → limits + dominant, wave size, RECIPE_CONTRACTION_FRAGMENT_K, aligned plain attention check, matrix path (gfx11/gfx12 with fp16/bf16/int8/int4), chunk_k multiple of fragment, register_m/n
│   ├── 19776 shared budget per wave, state-byte ratio, per-contraction forward/gradient/previous tiles, shared values over contractions and attention tiles (RECIPE_ATTENTION_QUERY_TILE), register values, owned chunk partials, gradient scratch_base, NativeSchedule, compile_model + NativeProgram::load, shared-memory requirement check
│   ├── 19877 allocate(bytes) (traced with free-after) / allocate_bytes (CPU aligned alloc with a size header, CUDA, HSA vram pool with RECIPE_HOST_SPILL=1 fallback into system memory, remote opcode)
│   ├── 19925 free(pointer); upload(dst, src, bytes) (allocates when dst is 0); clear(pointer, bytes) (HSA word fill plus a byte tail copy)
│   └── 20010 download(dst, src, bytes); free_bytes() (host available bytes, CUDA memory_info, HSA pool info, remote memory opcode)
├── 20052 free_bytes tail (HSA agent info 0xA015, remote REMOTE_MEMORY); synchronize() (CUDA synchronize, HSA signal wait, remote opcode)
├── 20085 static DEVICES; fn cpu_worker_threads() (available parallelism capped by RECIPE_CPU_WORKER_THREADS); fn cpu_device(); fn host_available_bytes() (/proc/meminfo MemAvailable); fn shared_cpu_device()
├── 20105 pub fn device_names(selection) (dot-separated chain, `host:` prefixes, amd<n> / nv<n> / cpu only, no commas or numbered cpus); fn device_selection() (RECIPE_DEVICE with the local host prefix stripped)
├── 20141 fn devices() (RECIPE_FORCE_CPU, load_amd + load_nvidia, leaked &'static Gpu list, CPU appended last); fn device(name) (named lookup, or the single accelerator)
├── 20182 fn local_host() (unix gethostname(3) with a reserved terminator byte; windows GetComputerNameExW); static SELECTED; fn selected_gpus() (RECIPE_DEVICE chain, multi-device = false takes the first name, remote hosts through connect_remote, no duplicates); fn selected_gpu()
├── 20245 struct RemoteDirectory (ssh `rm -rf` on drop); fn command_output(command, action); fn remote_directory(host) (mktemp under ~/.cache/recipe/native with a strict path shape check)
├── 20278 fn connect_remote(host, device_name, canonical): REMOTES registry, RECIPE_BINARY scp + sha256sum comparison, `recipe --worker <device>` over ssh with piped stdio, handshake (status, backend byte, architecture, memory, shared limit, wave, worker threads) → leaked Gpu with Driver::Remote
├── 20353 #[cfg(amd)] type HsaInfo; struct HsaQuery + extern "C" collect_hsa; struct HsaGpuQuery + collect_discrete_hsa (device type 17 == GPU, 0xA114 bit 0 clear = discrete); types HsaSymbol / HsaSymbolInfo; unsafe fn hsa_kernel (attributes 22 object, 11 kernarg, 13 group, 14 private); fn kfd_property(text, name)
├── 20436 #[cfg(amd)] impl Hsa: native_dispatch (amd_kernel_vgprs, `.kd` symbol, AmdResidency, amd geometry); load_native (code-object reader, executable create/load/freeze, forward, single-position step at min(workgroup, 512)/wave waves when not training, epoch, model_load, kernarg + grid-sync allocation)
├── 20475 #[cfg(nvidia)] impl Cuda: native_dispatch (function attributes block/shared/registers, nvidia geometry, occupancy with the dynamic reduction buffer, groups × active); load_native (cubin load, forward, step with the forward's reduction buffer, epoch, model_load)
├── 20529 unsafe fn native_cpu_pointer / native_cpu_value / extern "C" native_cpu_barrier (std Barrier); macro launch_native_cpu_epoch! (12 pointers + 11 values); unsafe fn launch_native_cpu_entry(forward, epoch, model_load, entry, arguments) (argument counts checked against the layouts)
├── 20607 unsafe fn launch_native_cpu(cpu, entry, arguments, threads): Barrier context and one worker thread per lane
│   └── 20612 scoped worker threads: each runs thread(id, barrier, wait) then launch_native_cpu_entry; panics surface as errors
├── 20635 impl NativeProgram::load(gpu, artifact, graph, schedule, shapes, register_values, waves): artifact contract + epoch layout + backend match, per-driver load (CPU: worker-thread geometry with Kernel::remote; HSA / CUDA load_native; remote REMOTE_LOAD handshake reading forward/epoch/model-load geometries), load trace, reduction_values, gradient_bytes
│   ├── 20710 dispatch(entry); launch_forward(arguments, single) (single-position step dispatch on AMD/NVIDIA); launch_epoch; launch_model_load; launch; launch_dispatch (INTERRUPTED check, argument count vs layout, dynamic shared budget, device dispatch lock)
│   └── 20758 unsafe fn launch_backend(gpu, backend, dispatch, entry, arguments, threads, dynamic, shared): CPU → launch_native_cpu; AMD → kernarg packed by layout widths, optional 256-byte implicit block carrying the multigrid sync pointer, grid-sync words reset + group count, AQL packet, doorbell, wait loop reporting a 60 s stall
│       └── 20858 NVIDIA → cuLaunchKernel(threads/block, block, dynamic); Remote → REMOTE_LAUNCH with each argument's bytes; backend/driver mismatch error
├── 20891 fn load_amd(selection): hsa_init, hsa_iterate_agents for the CPU agent and discrete GPUs, amd<index> selection filter → load_amd_gpu
├── 20915 #[cfg(amd)] fn load_amd_gpu(runtime, info, cpu_agent, agent, index): VRAM and KERNARG pools, wave/workgroup/CU/KFD-node/cooperative-CU queries, gfx target from /sys/class/kfd properties, executable and symbol entry points, LDS/simd_per_cu/waves_per_simd/vgprs_per_simd, queue + completion signal, Hsa struct → Gpu "amd{index}"
├── 21001 fn load_nvidia(selection): cuInit, attribute ids, driver version → NVIDIA_DRIVER_VERSION, per device SM/warp/block/LDS/registers/threads/compute capability → sm_XY target, cuCtxCreate, Cuda struct with cuMem*/cuLaunchKernel/cuOccupancy fns → Gpu "nv{index}"; integrated devices skipped
├── 21106 type WorkerWire; struct WorkerProgram { backend, dispatches: [Option<Dispatch>; 3], shared_values, reduction_values, _temporary }; pub fn worker_serve(name): probes the local device, writes the handshake, command loop REMOTE_ALLOCATE / FREE / UPLOAD …
│   ├── 21172 REMOTE_DOWNLOAD / SYNCHRONIZE / MEMORY / CLEAR handlers
│   ├── 21197 REMOTE_LOAD: artifact bytes, waves, shared/register values, element, training, state width, storage flag; per-driver load (CPU through remote_native_artifact temp files, HSA, CUDA); replies each dispatch's shared/groups/block; stores a WorkerProgram
│   └── 21243 REMOTE_LAUNCH: entry byte, arguments read by layout widths into u64 slots, shared budget check, dispatch lock, launch_backend; unknown command error; flush per command
├── 21275 unix extern "C" dlopen/dlsym/dlclose/mmap/munmap; unsafe fn native_library (dlopen RTLD_NOW / LoadLibraryW); windows consts CTRL_C_EVENT, COMPUTER_NAME_DNS_HOSTNAME, MOVEFILE_*; kernel32 imports; signal(2) and write(2) imports
├── 21319 fn distance; fn nearest_bounds_distance(query, samples, bounds); fn nearest(query, state, features); fn graph_inputs(graph, samples, rows, gpu, precision) (forward, predictions at the source node); fn surrogate_model(hidden) = layer(hidden).tanh().layer(1); fn fit_surrogate(input, samples, targets, hidden, gpu, config) (frozen after training)
├── 21374 const NEAREST_IDENTICAL_LEAF; struct NearestIndex { features, permutation, nodes }: build(samples, features, rows), write(target), partition() (k-d tree split on the widest dimension, leaves of 16 rows, identical-row leaves)
│   └── 21421 nearest(index, samples, query, row, count, exclude, best): leaf scan with (distance, index) ordering, near/far recursion pruned by the far cell's bounds distance
├── 21447 struct PredictorProgram { code, locals, stack, table, nearest }; evaluate(row, query): stack machine over Feature / Row / Constant / Load / Store / Duplicate / Add / Subtract / Multiply / Divide / Greater / Choose
│   └── 21504 Nearest (mean target of the k nearest rows, self excluded when negative), Affine (means, scales, weights), Gaussian (per-class quadratic score, argmax label); depth check; fn finite_prediction
├── 21556 struct PredictorBuilder { code, locals, depth, stack, table, index }: new(), nearest(count, exclude, features, table), affine(table), gaussian(table), emit/push/feature/constant/binary/choose, finish() → PredictorProgram
├── 21609 struct Predictor { program, predict: Box<dyn Fn> }: new(program) (interpreted), fitted(program, teacher); enum TreeNode { Leaf(f64), Split { feature, threshold, left, right } }; fn tree_mean / tree_error
├── 21635 fn fit_tree(samples, targets, features, rows, depth, candidates, minimum) (best SSE split over candidate features, minimum leaf rows, midpoint thresholds); fn emit_tree(node, program) (constant/feature/Greater then both branches and Choose)
├── 21685 fn next_random(state) LCG; validators valid_estimator / positive_estimator / boosting_trees(count, configured) / cluster_estimator / neighbor_estimator; fn fit_svm(_, data, rows, config): feature means and capped inverse deviations, hinge-gradient descent over parallel row blocks
│   └── 21732 per-block epsilon-insensitive hinge direction, regularized gradient reduce, rate step; predictor = intercept + Affine(means, scales, weights)
├── 21777 fn fit_forest(trees, data, rows, config) (bootstrap rows, shuffled feature subset per tree, mean of trees; teacher predicts through the tree list)
├── 21801 fn solve_bayes(samples, targets, means, target_mean, config): ridge system in feature or row space (dual when rows < features), lower-triangle accumulation, noise variance and prior precision, Cholesky solve, dual weights recovered
├── 21865 fn fit_bayes(_, data, rows, config): regression → target mean + Affine(means, ones, weights); categorical → Gaussian table per class (means, −½/variance scales, log prior − ½ Σ log variance, labels)
├── 21917 fn tree_predict(tree, sample); fn boosted_predictor(base, trees, rate); fn xgboost_leaf(rows, gradients, regularization); fn fit_xgboost_tree(...) (parallel per-feature gain scan, minimum gain)
├── 21974 fn fit_xgboost(count, data, rows, config) (gradient = prediction − target, boost_rate steps; teacher folds the trees)
├── 21991 struct LightNode { candidate, value, split }; fn light_node; fn lightgbm_split(columns, residuals, residual_squares, rows, bins, minimum) (histogram bins per feature in parallel, best variance gain)
├── 22051 fn materialize_lightgbm(nodes, index); fn fit_lightgbm(count, data, rows, config) (columns transposed once, leaf-wise growth to lightgbm_leaves, boosted predictor)
├── 22089 struct CatboostBorders { thresholds, bins }; fn catboost_borders(samples, features, rows, count) (rank-spaced midpoints); fn ordered_split(borders, residuals, permutation, codes, level, prior, minimum) (ordered target statistics over the permutation, parallel per feature, first strict minimum)
├── 22156 fn oblivious_tree(splits, leaves, level, code); fn fit_catboost(count, data, rows, config) (random permutation per tree, depth-wise oblivious splits, prior-smoothed leaf means)
├── 22203 fn cluster(data, width, clusters, iterations, importance) (k-means, empty clusters reseeded from the farthest row); fn fit_kmeans(clusters, data, rows, config) → nearest(1) table with group labels
├── 22238 fn parallel_map<R>(count, work) (scoped threads over index spans, results in order); fn predict_rows(teacher, inputs, features); fn fit_knn(count, data, rows) (leave-one-out teacher, kept rows deduplicated to `count` copies); impl Estimator::fit
├── 22276 fn native_contraction_shapes(graph, rows) → per node (forward, gradient, previous, parameters) extents: Contraction (window = channels × kernel), Scan projection rows
│   └── 22292 Scan projection shapes; extents narrowed into Tile { m, n, k } per direction → NativeContractionShapes
├── 22316 fn native_attention_shared_values(extent, whole, inference) (forward, query-gradient, key/value-gradient and whole-sequence matrix phases; max over the phases a mode runs)
├── 22350 fn native_attention_tile(length, width, shared_values, query_tile, inference) (queries from RECIPE_ATTENTION_QUERY_TILE downward, keys from the remaining budget, matrix phase must fit when the tile spans the sequence); fn native_attention_tiles(graph, shared_values, query_tile, inference)
├── 22400 fn native_tiles(total, width, role); fn native_contraction_partial_per_chunk(m, n, register_m, register_n, block, ratio) (exchange lanes × register tile + bias lanes, in model elements)
├── 22420 struct NativeTrainingState { values, contexts, adjoints, input_adjoint, weights, frozen, moments, variances, gradient, metrics, contractions, tile, step, best_loss }; struct NativeEpochState { input_adjoint, weights, frozen, moments, variances, batch_normalizations, best_loss, step, objective }; fn write_contraction_schedule(contexts, layout, contractions) (nine i32 words per contraction slot)
├── 22472 fn fnv(text); fn native_device_identity(gpu, config, allocation) ("schedule-v3;device=…;driver=…;candidates=…" string); fn native_schedule_cache_path(artifact, identity) → schedule-{fnv}.tsv beside the artifact
├── 22513 fn load_schedule_cache(path, identity, schedule, shapes, ratio) (header `device <identity>`, `node <i> <9 extents>` lines, each tile must be among schedule_candidates, every contraction covered); fn replace_schedule_cache (rename / MoveFileExW)
├── 22570 fn store_schedule_cache(path, identity, contractions) (unique temp file, write, fsync, atomic replace, cleanup); fn dominant_tile(shapes, contractions) (gradient tile of the most gradient work)
├── 22605 fn schedule_candidates(limits, current, schedule, ratio, budget): current first; matrix path narrows 16-fragment M/N spans; vector path walks halving M/N lane ladders under block, shared-value and owned-chunk limits with K fixed
├── 22659 fn native_contraction_tile(limits, register_m, register_n, block, shared_values, fragment, ratio, matrix): matrix tile waves×16 by max(block/2, 32) with K from the shared budget; vector tile from √block lanes narrowing M then N until staging and partial-exchange K fit, K rounded to whole fragments
├── 22707 fn native_contraction_shared_values(extent, ...) (staging vs chunk partials); const NATIVE_SCRATCH_ROW_VALUES = 4; NATIVE_SPLIT_SPAN / NATIVE_MATRIX_SPLIT_SPAN / NATIVE_K_PARTITIONS from RECIPE_CONTRACTION_* via const fn parse_natural
├── 22742 fn native_gradient_bytes(graph, scratch_base, contractions) (split-K jobs × partitions, scratch rows per contraction); struct Resources { shared, max_block }; struct Geometry { groups, block } threads(); fn geometry(cus, wave, workgroup, lds, groups_per_cu, resources)
├── 22791 #[cfg(amd)] struct AmdResidency { vgprs, vgprs_per_simd, waves_per_simd, simd_per_cu }; fn amd(...) (waves halved until the VGPR budget of one CU holds the workgroup); fn amd_kernel_vgprs(bytes, name) (MessagePack `.vgpr_count` after the kernel's name); #[cfg(nvidia)] fn nvidia(...)
├── 22844 pub trait IntoDataSources { const AUTO; into_data_sources } with impls for Auto, strings and string lists
│   └── 22854 impls for &str, String, [T; N], Vec<T>, &[T]; impl Data: target(names), include(names) / exclude(names) (mutually exclusive), test(sources), set(source), norm(z_score), split(fraction)
├── 22912 type DataSchema = Vec<(kind, name)>; struct Prepared { samples, targets, target_width, rows, source_rows, features, schema, sequence, target_categorical, norm_mean, norm_scale, identities, fitted, bound }; Prepared::matrix(samples, targets, rows, target_width)
├── 22958 struct Table { name, headers, declared, rows, attention, path }; enum FeatureType { Numeric, Categorical(values), Text(width) }; fn prepare(data) (OnceLock'd prepare_data); fn column_match(name, table, header, column) (name, table.name, colN, dotted suffixes, `.row` headers); FeatureSelection::selects
├── 22997 fn load_tables(data, sources): collect_files over folders/archives, decode_tables per table file, directory_samples short-cut, resolve_references, merge_captures, merge_partitions, align_samples when several tables remain
├── 23030 fn resolve_references(grouped): a column whose first value is a plain relative path to a decoded table becomes a path column and every row must resolve
├── 23056 fn align_samples(tables): groups by headers; a lone table's columns that record exactly one sibling group's file paths give that group its sample order (ambiguous or partial records are errors)
│   └── 23126 sample count per source with a reading explanation, reference columns dropped from lone tables, unrecorded partitions joined row-wise, recorded groups flattened one row per file with `header.row` names
├── 23188 fn grouped_sample_table(name, targets, groups) (per-field shape consistency, chained attention shape, target values appended); fn manifest_sample(root, value, files) (`#page=` suffix, plain relative names only)
├── 23231 fn manifest_table(data, root, files, parsed): a table whose selected inputs all name files becomes one row of decoded values per manifest row (TIFF pages via tiff_pixels), targets kept as columns
├── 23290 fn directory_samples(data, sources, files, parsed): one source with declared targets; manifest_table first; flat sidecar samples (`<sample>.<target>`, `.label` / `.cls` fallbacks, `<sample>.<input>` views) → Table
│   └── 23369 name-labeled flat samples (`v1__v2__name[.input]`) → grouped_sample_table when every target varies; nested class samples keyed by relative parent path with qualified stems
│   ├── 23412 nested class layout: one relative parent component per target, qualified stems group input views → grouped_sample_table; otherwise one level of subdirectories is required
│   └── 23443 paired subdirectories (singularized directory names match targets or sample stems match; per-sample rows vs per-directory column groups, chained attention shape); class subdirectories (directory name = the one target) → SampleTableBuilder unless a declared table names the target
├── 23514 struct SampleTableBuilder { target, shape, headers, rows }: new(), push(path, bytes, target) (pixel.N / content headers, shape consistency), finish(name); fn sample_text; fn sample_values (png/jpeg pixels or trimmed UTF-8); fn is_image; fn is_document (md/html/htm)
├── 23564 fn jpeg_pixels(bytes): ZIGZAG table, marker walk (DQT 8-bit tables, DHT class/table, SOF0 baseline 8-bit 1 or 3 components at 1×1 sampling, DRI, SOS), other SOF types rejected
│   ├── 23638 struct Entropy bit reader (0xFF00 stuffing, marker check), receive(length), decode(table) canonical Huffman; fn extend; fn idct (libjpeg jpeg_idct_islow 13-bit fixed point, two passes, +128 clamp)
│   └── 23755 MCU loop with restart intervals and DC prediction reset, AC run/length decoding into zigzag positions, per-component planes, grayscale copy or libjpeg ycc_rgb_convert 16-bit fixed point → (width, height, components, pixels)
├── 23834 fn png_pixels(bytes): signature, IHDR (8-bit, color 0 or 2, no interlace), IDAT concatenation, zlib_inflate, scanline filters None/Sub/Up/Average/Paeth → pixels
├── 23898 fn tiff_pixels(bytes, page): II/MM order, directory chain walked to `page` with cycle detection, SHORT/LONG tags, width/height, compression 1 only, photometric, channels 1/3/4, 8-bit chunky, strips concatenated, WhiteIsZero inverted
├── 23972 fn name_headerless(data, tables) (include + target widths name a headerless table); fn prepare_data(data): load_tables, test tables appended (separate files, same headers), autoregressive branch, each target matches exactly one column or a numbered group, positional row alignment
│   └── 24035 feature columns via infer_feature in include order, sequence Shape from `.row` header groups, attention shape matching the feature count, target categories / categorical flag, per-row encode (channel-major reorder, rows missing any target skipped), imputed-percentage report, schema → finish_prepared
├── 24124 fn prepare_autoregression(data, tables) (CHAR_IDS one-hot prefixes over the longest string, next character as target); fn finish_prepared(...) (row identities from sample_identity mixed with occurrence count, shuffle, normalize_samples or impute_missing)
├── 24191 fn normalize_samples(samples, features, fit) (z-score fitted on the training split, missing → mean); fn impute_missing; fn sample_identity (bytewise FNV-1a); fn is_table (csv/tsv/txt/data/dat/all-data/jsonl/json/npz/sqlite/sqlite3/db/h5/hdf5/xml/gz/xlsx); fn is_archive (zip/tar); fn resolve_path (`~`)
├── 24238 fn collect_files(path, member, files) (directories recursed in sorted order, zip/tar members expanded under the archive path, JSON metadata objects skipped)
├── 24281 fn target_column(table, name); fn merge_captures(tables, targets) (per-directory captures where one single-row table holds each target → one row per capture with `table.header[.row]` names)
├── 24350 fn merge_partitions(tables, targets, features) (tables sharing every target column union their headers into one "data" table, other tables kept for alignment)
├── 24390 fn decode_tables(path, bytes): gz → gzip_inflate + recurse, xlsx_tables, jsonl / json → json_records_table, npz → npy_columns + array_table, sqlite/sqlite3/db → sqlite_tables, xml → xml_records, h5/hdf5 → hdf5_columns, else parse_table
├── 24441 fn sqlite_tables(bytes) (page size, sqlite_master walk, column names from CREATE TABLE); fn sqlite_rows (interior page 5 / leaf page 13 b-tree walk, no overflow pages); fn sqlite_varint; fn sqlite_record (serial types → text values)
│   └── 24532 serial types: NULL, 1/2/3/4/6/8-byte sign-extended integers, float, constants 0/1, odd ≥ 13 UTF-8 text; blobs rejected
├── 24565 fn inflate_consumed(bytes) (RFC 1951): LSB-first Bits reader, canonical Huffman { counts, symbols } build + decode, LENGTH_BASE/EXTRA and DISTANCE_BASE/EXTRA tables
│   └── 24631 block loop: stored (byte-aligned LEN), fixed tables, dynamic tables via the 19-code-length alphabet with repeats 16/17/18, literal/length/distance back-references, returns (output, bytes consumed)
├── 24707 fn inflate(bytes); fn zlib_inflate(bytes) (CMF/FLG check, no preset dictionary, Adler-32 verified)
├── 24723 fn hdf5_columns(bytes): version-0 superblock with 8-byte offsets and lengths, nested messages() following continuation (0x10) messages, object_messages for version-1 and OHDR version-2 headers, root symbol-table message (0x11)
│   ├── 24795 local HEAP data offset, group TREE / SNOD walk collecting dataset names and object headers; per dataset: dataspace (1) dims, datatype (3) little-endian int/float, layout (8) contiguous or chunked, filter (11) deflate only
│   └── 24867 raw buffer from the contiguous span or the chunk b-tree (zlib_inflate per chunk, coordinates scattered inside the dims), element decode by (float, signed, width), one column per trailing-dimension index named `dataset.N`
├── 24944 fn gzip_crc32; fn gzip_inflate(bytes) (concatenated members, FEXTRA/FNAME/FCOMMENT/FHCRC fields, inflate_consumed, CRC-32 and ISIZE verified, trailing zero / 'p' padding allowed after the last member)
├── 24998 fn tar_entries(bytes) (512-byte ustar headers, prefix joined, header checksum verified, regular files only); fn zip_entries(bytes) (end-of-central-directory search, central entries → local headers, stored or deflate members)
├── 25068 fn xml_entities(value) (amp/lt/gt/quot/apos and numeric &#…; / &#x…; entities)
├── 25097 fn xml_attribute(tag, name); fn xml_values(block, tag) (every `<tag>…</tag>` body unescaped); fn xml_tags(document, tag) (attribute strings of `<tag …>`); fn xlsx_column(reference) (A1 letters → index)
├── 25139 fn xlsx_tables(bytes): zip entries, workbook.xml.rels relationships, `<sheet>` list with r:id targets, sharedStrings `<si>` texts
│   └── 25183 per worksheet `<sheetData>` rows and `<c>` cells (inlineStr / shared / value) placed by column reference, first row is the header, rows padded to the header width → Table per sheet
├── 25237 fn npy_columns(name, bytes) (NPY v1/v2/v3 header dict, C order only, descr dtypes f4/f8/i1..i8/u1..u8 little-endian, trailing dims → `name.N` columns); fn array_table(name, columns) (row counts must agree)
├── 25299 fn json_array(text); enum JsonValue { Null, Bool, Number, Text, Array, Object } scalar(); fn json_value(text) recursive descent (literals, strings, arrays consumed, objects kept, numbers validated)
├── 25387 fn json_string(text) (escapes, \u with surrogate pairs)
├── 25430 fn xml_records(text) (optional declaration, root → one record per child, one text field per grandchild, nested elements rejected, entity unescape); fn json_records_table(name, records) (ordered key union, scalar fields)
├── 25522 fn parse_table(path, bytes) (delimiter tab / ; / , chosen by the widest rectangular parse, all-numeric first row → headerless colN + `target` names); fn records(bytes, delimiter) (double-quote escaping, CR trimming, blank records counted)
├── 25585 fn categories(table, column, rows); fn infer_feature(table, column, rows) (numeric, categorical under RECIPE_CATEGORICAL_RATIO, else text of the longest width); FeatureType::width; fn encode(value, kind, output) (empty → NaN fill, numeric, one-hot, byte codes)
├── 25631 fn shuffle(samples, targets, identities, features, source_rows, target_width) (RECIPE_RANDOM_SEED LCG Fisher-Yates over the source and test partitions separately)
├── 25653 pub enum RatPolicy { History, Rolling, Online, Learned, Full } with consts history / rolling / online / learned / full; validate(data) (rolling needs .split), capacity(data, rows)
├── 25697 struct RatCommand { path, policy }; evaluate(names, proposals): one child invocation, `name=value,…` records on stdin written from a scoped thread, one finite score per stdout line, any stderr or failure aborts
├── 25762 enum RatEvent { Line, Error, Closed, ErrorsClosed }; struct RatState { names, values, outputs, choices }; enum RatFrame { State, Score }; struct RatSession { child, input, events, path, errors_closed } + Drop (kill); fn rat_number; fn rat_names
├── 25806 impl RatSession: open(command) (RECIPE_RAT_PROTOCOL=1, stdout and stderr reader threads feeding an mpsc channel), send(line), frame() (records `score`, `state`, `actions`, `choice`, `ready`; INTERRUPTED-aware 100 ms polling)
├── 25906 finish() (`close`, drain both streams, exit status); RatState::select(proposal) (nearest valid choice under per-column range scaling)
├── 25956 struct RatEpisode { observations, score, action }; fn rat_episode(session, first, names, outputs, tape, proposal_node) (state → forward → nearest choice → `choose …` until the evaluator scores)
├── 25981 struct CommandRatComposition { graph, proposer, storage_model, evaluator, loss, proposal, offset }; fn command_rat_graph(model, prepared, rows, gpu, config) (proposer compiled without its downstream, evaluator over [features ; targets] emitting one reward, compose_scored_graph)
├── 26007 fn embed(graph, source, shape, channels) (frozen routing projection with ones); fn compose_scored_graph(proposer, scorer, config) (input and proposal embedded side by side and added, scorer appended and frozen, optimizer state resized) → (graph, scorer offset); fn extract_rat_proposer
├── 26056 struct RatReplay { width, capacity, rows, raw }: new(), observe(input, value) (FIFO at capacity), snapshot(); fn rat_fit_steps(tape, steps, rate, config)
├── 26091 struct RatFit { tape, graph, width, rows, gpu, loss, precision }: new(graph, gpu, loss, config), reserve(rows) (captures then rebuilds the tape for a new batch), fit(samples, targets, indices, steps, rate, config) → objective, predict(samples), weights(), fit_sequence(samples, target, rate, config)
├── 26163 fn rat_backward(tape, offset, teacher_weights, input, rate, config) (evaluator weights written into the composition, one epoch); structs LearnedSelector { graph, length }, LearnedSelectionScore { graph, fit, length }, LearnedActor { graph, tape, selector_node, selector_parameters, score_offset, length }, LearnedReplay { width, context_width, gpu, selector, selection_score, actor }; LearnedReplay::new
│   ├── 26212 compile_sequence(model, shape, gpu, config) (one row, no output projection); ensure_actor(length, config) (captures the previous actor, extracts the selector weights, recomposes selector + frozen scorer, new tape with target 1.0)
│   ├── 26248 ensure_selector (conv(surrogate_width, 1).tanh().conv(1, 1).sigmoid(), one value per observation, state carried across lengths); ensure_selection_score (conv.tanh().pool(length).layer(1) with its own RatFit)
│   └── 26282 fit_selected(target, samples, targets, steps, rate, selector_rate, config): context rows [observation ; score ; current prediction], selector forward, select ≥ 0.5, target fit on the selection, R² quality, scorer fit_sequence, rat_backward moves the selector toward the highest predicted quality
├── 26328 fn learned_channels(samples, width, length) (row-major → channel-major); fn learned_selection_input; fn copy_learned_state(old, new) (identical layout required, parameters/frozen/state copied); fn prepare_command_data(data) (every source row as features, declared targets become proposal names, no split)
├── 26383 pub struct Train { epochs, learning_rate, log_metrics, stop, resume, save, seed, rat, rat_target }; enum Compute { F(FloatFormat), Fp, Int(IntFormat), Bf, Tf } consts FP8/FP16/FP32/FP64/INT1/INT4/INT8/BF16/TF32; bytes(), pack(value), unpack(bits)
│   └── 26432 optimizer_epsilon(value) (smallest representable when it rounds to zero), below_one(value), saved(family, values) / saved_fields() bundle encoding, label() (fp16 / int8 / bf16 / tf32 / f(e,m))
├── 26478 impl Train: seed(), stop(value) (0 disables), optimizer(adamw), epochs(), lr(), log(metrics) (arms trace), save(path), resume(path), execute() (interrupt registration and INTERRUPTED_EXIT), run(model, data) (prints the evaluation when a split or tests exist), rat(policy, command), target(value)
├── 26551 print_rat(...) metric line with Score/Window/Choices; try_run_rat(model, data, prepared, command, gpu, config, started): policy capacity, no tests/resume, input names from the feature schema plus targets, single-row or full-set proposals, command_rat_graph composition, tape, initial proposal scored into a RatReplay
│   ├── 26616 epoch loop: full set rescored each epoch, evaluator fit (LearnedReplay selection or every observation), evaluator R² and loss, rat_backward through the frozen evaluator toward rat_target, new proposal scored and observed, print_rat with the schedule
│   └── 26682 final full-set scoring, predicted and measured reward, proposer weights and optimizer state extracted, trained_samples identities, save bundle (bn stats truncated to the proposer, norm stats, outputs, artifact key) → TrainingReport
├── 26756 try_run_stateful_rat(model, data, command, started): no data selectors or resume, RatSession `reset`, first decision state fixes the state and action schema
│   ├── 26772 Prepared::matrix from the first state values with a feature/target schema, seeded Config, command_rat_graph, tape on the first state, RatFit, unbounded RatReplay, learned selector, first rat_episode
│   ├── 26791 epoch loop: replay capacity per policy (Full clears), every episode observation labeled with the terminal score, evaluator fit (learned or all), rat_backward from a retained state row, `reset` and a fresh episode, print_rat
│   └── 26833 session.finish(), predicted reward at the last observation vs the episode score, proposer extracted, saved with an empty Data and truncated bn stats → TrainingReport
├── 26877 try_run(model, data, evaluation): RAT routing (stateful when autoregressive without sources, otherwise prepare_command_data → try_run_rat; estimators refused), prepare(data), training rows from the split, TargetScale for bce/focal losses, compile, logit output bias, stored_graph, bundle::restore on resume, trained_samples identities
│   ├── 26931 DeviceTape::new over the selected devices, initial forward with saved bn stats, print_devices, tune() (interrupt exits through finish_dispatch), initial predictions and loss, epoch loop: advance, live_epoch around tape.epoch, finish_dispatch checkpoints, print, INTERRUPTED_EXIT
│   └── 26984 final forward with refreshed bn stats, decoded predictions, autoregressive evaluation forwards one row at a time (streams characters with `blck`), held-out evaluation through tape.evaluate, R² over the right rows, final checkpoint → TrainingReport
├── 27054 finish_dispatch<T>(result, stored, schema, tape, request) (one checkpoint on interrupt via INTERRUPT_CHECKPOINTED); print(...) epoch metric line; print_evaluation(model, report) (Loss and R2 by default)
├── 27103 metric_line(loss, topology, metrics, epochs, schedule, measurement): ANSI-colored run / loss / r2 / time / epoch fields, block description once for blck-class metrics, tile schedule, score, checkpoint marker, choices and window counts
├── 27156 write_progress(line, replace, complete) (CR + erase-line + wrap control on stderr); live_epoch(model, run, epoch, epochs, config, schedule, action) (progress thread refreshing at RECIPE_PROGRESS_REFRESH_HZ while the epoch runs, interrupt-aware, erases on error)
├── 27226 struct Metrics { run, epoch, loss, r2, seconds, checkpoint, evaluation, score, window, choices }; pub struct TrainingReport { initial_loss, final_loss, initial_predictions, predictions, r2, evaluator_r2, validation_r2, predicted_reward, measured_reward, tile, schedule, run, epoch, seconds, epoch_seconds }
├── 27262 impl TrainingReport accessors: initial_loss(), final_loss(), initial_predictions(), predictions(), r2(), evaluator_r2(), validation_r2(), predicted_reward(), measured_reward(), tile() → [m, n, k], epoch_seconds()
└── 27301 struct TargetScale { minimum, span } fit(targets) / encode / decode (logistic) / logit; fn model_loss(predictions, targets, loss, threshold) (rmse takes the root); fn predicted_char(prediction); fn coefficient(targets, predictions) R²


build.rs (1384)
├── 1 imports (env, Error, fs, io, Path, PathBuf)
├── 7 struct FloatLayout { sign, exp, man }
│   ├── 13 impl: new(), bits()
│   └── 20 unpack(bits)→f64: inf/nan/zero/subnormal/normal by exponent field, sign from top bit
├── 35 struct FloatFormat { arithmetic, storage }
│   ├── 40 consts FP8(1,5,2) FP16 FP32 FP64 BF16(1,8,7) TF32(arith 1,8,10 / storage fp32); native(), bytes()
│   ├── 54 pack(f64)→u64: round through arithmetic layout, then store as 64/32/fp16/bf16(top 16 of f32)/8 bits
│   └── 66 unpack(u64)→f64 by storage width
├── 77 impl FloatLayout::pack_from(f64)→u64: nan/inf/zero, subnormal rounding ties-even, mantissa overflow bumps exponent, saturate to inf
├── 107 struct IntFormat { bits }: INT1/INT4/INT8, bytes(), pack() round+clamp to signed range, mask
├── 122 type BuildResult; const PARALLEL: workgroup.id decl + @global_id() = group*width+lane, @RECIPE_GRID_BARRIER@ placeholder
├── 129 const AMD_GRID_BARRIER: @grid_barrier → __ockl_grid_sync
├── 131 NVIDIA grid barrier comment: relaxed atomics + fences, one spinner lane per warp, bar.warp.sync reconverge (pre-sm_70 safe)
├── 143 const NVIDIA_GRID_BARRIER: @grid.count/@grid.phase globals, leader arrives, last arriver resets count + flips phase, spinner waits, acquire fence
├── 165 const AMD_WIDTH: @recipe.workgroup.size.x from dispatch.ptr+4 (i16)
├── 169 const AMD_WAVE_HELPERS (float state): wavefront.width, int8.dots=true, clock=__ockl_steadyctr_u64, wave.partner via ds.bpermute, wave.partner.f32
├── 176 const AMD_WAVE_HELPERS_DOUBLE: same with int8.dots=false, double partner via two bpermutes
├── 183 const IDENTITY_WAVE_HELPERS (CPU): width 1, no int8 dots, readcyclecounter clock, partner=identity, q4k/q6k/block32.slice stubs return zero
├── 200 nvidia_wave_helpers(state): shfl.sync.idx partner (lane = index>>2), width 32, globaltimer clock, slice stubs
├── 260 const IQ4_LEVELS [16 x i8]
├── 262 amd_block32_slice_helper(state, full): @recipe.block32.slice
│   ├── 265 signature; stub when !full; perm table words t0..t3 from IQ4_LEVELS
│   ├── 273 prologue IR: d (fp16), q8 d (f32), nibble shift, kind selects (iq4=1, q40=2, q41=3, q80=4), data offset, Q4_1 minimum m
│   ├── 275 loop w in 0..4: load word, shift/mask nibbles, amdgcn.perm level lookup with top-bit mask, sudot4 dots (iq4/q80/q4 select) and activation sum
│   └── 280 epilogue: Q4_0 subtract 8*sum, scale by d, add m*q8sum, multiply by q8 d, ret
├── 285 amd_q4_slice_helper(state, full): @recipe.q4k.slice
│   ├── 287 stub when !full; d/dmin fp16 loads
│   ├── 297 6-bit scale/minimum decode for group (low 4 groups direct, high 4 from packed high bits), group.d/group.dmin
│   ├── 313 q4 base offsets (pair*32+16, half*16), q8 block at group*36, q8 d
│   ├── 319 loop word 0..4: q4 word shift/mask, q8 word, sudot4 dot + sum chains
│   └── 333 epilogue: dot*group.d - q8sum*group.dmin, times q8 d, ret
├── 336 amd_q6_slice_helper(state, full): @recipe.q6k.slice
│   ├── 339 stub when !full; d at byte 208; chunk/group/local/half indices; ql/qh base and shifts; q8 block; int8 scale at 192+slice
│   ├── 355 loop word 0..4: ql nibbles | qh pair<<4 codes, sudot4 dot + sum chains
│   └── 369 epilogue: dot - 32*sum, times scale, times q8 d, times d, ret
├── 372 block_dot_helpers(): exact block dots in template placeholders (RECIPE_STATE), every backend
│   ├── 379 emit @recipe_block32_iq4_levels constant
│   ├── 382 closure activations: 4 quads of <4 x double> from quad-interleaved tile, decode each to RECIPE_STATE
│   ├── 429 closure words: 16 bytes from base as 4 words, byte extract to %<name>i.raw
│   ├── 439 closure sums: 4 independent madd/add chains per quad, join to %acc16 / %xsum16
│   ├── 452 @recipe.q4k.exact: 6-bit scale/min decode, nibble codes, dot*group.d - xsum*group.dmin
│   ├── 459 @recipe.q6k.exact: ql|qh<<4 - 32 codes, scale*acc*d
│   └── 467 @recipe.block32.exact(kind,...): iq4 level table / Q4_0 -8 / Q4_1 min / Q8_0 signed byte codes, d*acc + m*xsum
├── 477 parallel_ir(ir, width, grid_barrier): workitem.id→global_id, s.barrier→grid_barrier, insert PARALLEL after target triple, map recipe.local.id.x/group.id.x/local.barrier to amdgcn intrinsics
├── 483 word(text, from, to): identifier-bounded whole-word replace (alnum, _, . are identifier chars)
├── 496 numeric_region(ir): locate `; NUMERIC BEGIN` .. `; NUMERIC END` span
├── 501 const MATH_NAMES (recipe.math.exp/tanh/cos/sin/log); constant() hex f64 bits; math_literal() rounds through f32 unless double
├── 513 horner()/horner_typed(): fma chain innermost-first over coefficients, returns (ir, last value name)
├── 533 exponential_math(arith, name, bias, max, shift, high, low, terms): emits @name.pow2 (clamped exponent bits), @name.scale (two half powers), @name (nan/overflow/underflow, Cody-Waite reduce by log2e, Horner series, scale)
├── 551 reciprocal_factorials(first, step, last); alternating_factorials(first, last) sine/cosine signs
├── 571 shared_math(arith): deterministic transcendentals in double
│   ├── 581 declare f64 floor/fabs/fma when narrow; exp.wide (1023,2046,52, 13 terms); exp.narrow (127,254,23, 8 terms) when narrow
│   ├── 595 @recipe.math.expm1: series below 0.35 else exp.wide-1; @recipe.math.tanh.wide = u/(u+2), saturate at 20
│   ├── 601 @recipe.math.log.wide: nan/neg/zero/inf branches, subnormal boost by 2^54, mantissa split at sqrt2, atanh series 12 terms, two-part ln2
│   ├── 608 @recipe.math.trig(x, offset): three-term Cody-Waite by 2/pi, quadrant select sin/cos polys, negate upper quadrants; sin.wide/cos.wide wrappers
│   └── 616 per MATH_NAMES entry: double → .wide; narrow exp → exp.narrow; others fpext→wide→fptrunc
├── 632 native_codec(ty, rounded): optional @recipe.round (tf32 mantissa 13-bit round-nearest-even), @recipe.decode/@recipe.encode identity or rounded
├── 645 numeric_operations(prefix, value, arith, encoded, vector)
│   ├── 645 add/sub/mul/div (decode→op→encode when encoded); madd via llvm.fma
│   ├── 661 madd.vector: native <RECIPE_REGISTER_M x T> fma when half/float/double (declare once), else per-lane loop over @prefix.madd
│   ├── 680 neg; fcmp predicates oeq/oge/ogt/ole/olt/one/ord
│   ├── 692 from.u1/from.u32/from.s32; to.u32/to.s32
│   ├── 709 from.f32/from.f16; to.f16; to.f32
│   ├── 732 unary abs/floor/roundeven/sqrt + exp/tanh/cos/sin/log via MATH_NAMES
│   └── 749 sigmoid = 1/(1+exp(-x)) composed from prefix ops
├── 753 numeric_program(value, arith, codec): NUMERIC BEGIN header + declares, fma declare, shared_math, `recipe.*` encoded ops, default @recipe.set.format no-op, `recipe.state.*` ops, state.from.model / model.from.state, NUMERIC END
├── 775 widen_codec(codec, value): rename decode/encode to .narrow, add double decode (fpext) and encode (fptrunc through float)
├── 786 const ACC64 = "-acc64"; state_align(state) 8 or 4
├── 791 native_ir(ir, suffix, llvm, format, state): splice @RECIPE_NUMERIC@, word-replace double→llvm, contraction_tile suffix, align, RECIPE_MODEL_BYTES / STATE_ALIGN / STATE; re-literal 0.1, 0x3CB0.., 0x3FEF.. for half/bfloat/narrow
├── 816 custom_numeric(): runtime-parameterized `f(exp, man)` codec
│   ├── 819 @recipe_f_exp/@recipe_f_man LDS globals, declares, @recipe.set.format stores them, @recipe.f.power(exponent) inf/zero/normal/subnormal
│   ├── 831 @recipe.round(double): nan/inf/zero encode, source exponent (ctlz for subnormals), subnormal and normal mantissa rounding, saturate to max finite, pack via @recipe.f.decode
│   ├── 832 @recipe.f.decode(bits): special/subnormal/normal decode with sign
│   └── 835 decode/encode = round; numeric_program("double","double",block)
├── 838 custom_ir(ir, suffix): model bytes 8, state double, splice custom_numeric
├── 849 fp8_codec(): out-of-line (#3) e5m2 decode (subnormal 2^-16 scale) and encode (round-nearest-even, subnormal path, saturate to inf above 15)
├── 855 bf16_codec(): decode shl 16; encode nan-quiet, round-nearest-even on low 16
├── 859 encoded_ir(ir, suffix, bytes, codec, pack, state): llvm iN by bytes, optional widen, word-replace double→iN, align, literals -2/-1/0/0.1/0.5/1/2 and four hex constants through pack()
├── 881 half_ir(ir, suffix, state): half codec clamps ±65504 before fptrunc; same literal replacement with 0xH FP16 packs
├── 899 int_codec(format): decode sign-extend masked bits; encode roundeven, clamp to [-2^(b-1), 2^(b-1)-1], nan→0, mask to i8
├── 908 manifest readers: setting(key = ) line find; number() parse f64; text() strip quotes
├── 920 configured_entry(key, os) from `{ linux = "..." }` table; configured() resolves `$ENV/inside` roots
├── 931 precision_profiles(manifest): parse [precision] default-config and every [precision.<name>] into `name:k=v,...;...`, require default table exists
├── 959 native_configuration(manifest, os): FNV-1a over manifest + os + each `$VAR` reference (present/absent + value), emits rerun-if-env-changed
├── 989 platform(key, os) required; struct NvidiaToolkit { device_library, assembler, required }; nvidia_toolkit()
├── 1002 const CPU_REPLACEMENTS: contraction_tile → thread_local [CPU_SHARED_VALUES x double], drop addrspace(3), workitem.id→cpu.thread.id, local.id→0, group.id→thread.id, workgroup.size→1, barriers removed, grid_barrier→cpu.barrier, attributes #0 plain
├── 1020 const CPU_PARALLEL: thread_local @recipe.cpu.thread / barrier.context / barrier.wait; @recipe_model_thread(thread, context, wait) entry; cpu.thread.id; cpu.barrier calls wait(context)
├── 1026 struct Schedule { swizzle_m, partitions, split_span, matrix_split_span, local_chunks } (compile-time contraction shape; K split order is program property)
├── 1037 kv_codec(model, arith, kv, kv_bytes)
│   ├── 1040 closures widen/narrow (float↔state), to_float (half fpext / i16 bf16 shl16 / float), from_float (half clamp ±65504 / bf16 round-nearest-even / float)
│   ├── 1058 kv == model: pass-through encode/decode, kv.to.f32 / kv.from.f16 / kv.to.state via recipe.decode/encode
│   └── 1061 otherwise: kv.encode (decode→narrow→from_float), kv.decode (to_float→widen→encode), kv.to.f32, kv.from.f16, kv.to.state
├── 1070 precision_sources(ir, schedule): per base emit `<suffix>` with cache in model type + `<suffix>-kvf16/-kvbf16/-kvf32` where kv type differs (tf32 keeps -kvf32)
├── 1087 precision_bases(ir, schedule): substitute RECIPE_CONTRACTION_{SWIZZLE_M,K_PARTITIONS,MATRIX_SPLIT_SPAN,SPLIT_SPAN,LOCAL_CHUNKS}
│   ├── 1098 base "" = fp64 native double/double
│   ├── 1099 for state in [float, double(-acc64)]: -f32, -f16, -f8, -bf16, -tf32, -int8, -int4, -int1 with tile symbol `_<key>[_acc64]`
│   └── 1113 base "-f" = custom_ir (double/double)
├── 1114 template_state(base): double for "", "-f", *-acc64; else float
├── 1117 wmma_source(): strip `; RECIPE_WMMA ` catalog lines; wmma_method(source, key): find `key kind body` in ` || ` catalog, unescape \n
├── 1127 compose_wmma(ir, source, key): kind call → replace `@recipe.wmma(` call text; kind definition → replace the `declare ... @recipe.wmma(` line
├── 1138 compose_contraction(ir, matrix): map @contraction_{product_accumulate,a_index,b_index,output_m,output_n,store_lane} to vector_* or matrix_* variants
├── 1151 compile_amd(manifest, out, os, schedule)
│   ├── 1151 read amd-nv-cpu.ll, parallel_ir with AMD_WIDTH + AMD_GRID_BARRIER, splice block_dot_helpers at `; RECIPE_BLOCK_HELPERS`
│   ├── 1155 per precision source: wave helpers by state (double vs float + sudot4/perm declares + q4k/q6k/block32 slice helpers), write recipe-amd<suffix>.ll (vector contraction)
│   ├── 1165 for -f16/-bf16/-int8/-int4: gfx11 and gfx12 matrix variants via compose_wmma (gfx12-int4 uses gfx12-int8 method), write recipe-amd-<arch><suffix>.ll
│   └── 1175 emit RECIPE_AMD_IR (`;`-joined key=path) and RECIPE_HSA_* toolchain env vars
├── 1190 compile_nvidia(manifest, out, os, schedule)
│   ├── 1190 amd-nv-cpu.ll → nvptx64 triple, tid/ctaid/ntid sregs, barrier0, strip addrspace(5), NVIDIA_GRID_BARRIER
│   ├── 1201 per precision source: nvidia_wave_helpers(state), write recipe-nvidia<suffix>.ll
│   └── 1209 emit RECIPE_NV_IR, RECIPE_NV_COMPILER/CODEGEN/RUNTIME, toolkit DEVICE_LIBRARY/ASSEMBLER, RECIPE_NV_PTX_VERSION
├── 1219 compile_cpu(manifest, out, os, schedule)
│   ├── 1219 target triple from TARGET, IDENTITY_WAVE_HELPERS, block helpers, CPU_REPLACEMENTS, append CPU_PARALLEL with entry linkage; require absolute existing cpu-compiler/cpu-linker
│   ├── 1233 per precision source: strip addrspace(1)/(3)/(5), RECIPE_CONTRACTION_CPU_SHARED_VALUES, write recipe-cpu<suffix>.ll
│   └── 1246 emit RECIPE_CPU_IR, RECIPE_CPU_COMPILER/TARGET, RECIPE_NULL_DEVICE, RECIPE_CPU_LINKER/_DRIVER/_LIBRARY_FLAGS/_LINK_FLAGS/_ENTRY_LINKAGE/_MODULE_SUFFIX
└── 1262 main()
    ├── 1262 read Cargo.toml; Schedule from positive contraction-* keys
    ├── 1274 export every [package.metadata.train] and contraction/attention/delta/topology/placement/cpu-worker numeric key as RECIPE_* rustc-env
    ├── 1343 RECIPE_DEFAULT_CONFIG + RECIPE_PRECISION_PROFILES; RECIPE_MULTI_DEVICE (false/true/auto); OUT_DIR; check-cfg amd/nvidia
    ├── 1358 RECIPE_NATIVE_CONFIGURATION hash; installed() probe; compile_cpu always
    └── 1362 native (TARGET==HOST) gates: amd if hsa compiler+device library exist; nvidia if compiler + toolkit device library exist (required toolkit missing → error); rerun-if-changed Cargo.toml, amd-nv-cpu.ll


cli.rs (163)
├── 1 imports; USAGE string (`recipe [run] <source.rs> [--device ...] [--config ...] [export]` / `recipe --worker <device>`)
├── 5 invalid(): eprint message, exit 2
├── 10 mapped(): split `target=path;...` env string into `(target.suffix, path)` pairs
├── 14 export(source, selected)
│   ├── 14 validate source exists; device must be cpu / amd* / nv*
│   ├── 20 collect artifacts: RECIPE_HSA_CODE_OBJECTS→.hsaco, OUT_DIR/librecipe_cpu.a→cpu.a, RECIPE_HSA_ASSEMBLIES→.amd.s, RECIPE_NV_PTX→ptx
│   └── 24 retain by device kind; copy each to `recipe.<ext>` beside source; eprint `exported:`
├── 39 library_path(directory): pick newest of librecipe.rlib / deps/librecipe-*.rlib by mtime
├── 61 run(source, device, config)
│   ├── 61 locate exe dir, rlib, deps; per-pid output `recipe-script-<pid>`
│   ├── 68 rustc --edition=2024 --extern recipe=<rlib> -L dependency=deps -o output; exit on failure
│   └── 83 spawn output with RECIPE_BINARY / RECIPE_DEVICE / RECIPE_CONFIG env; remove file; exit with code (unix: 128+signal)
└── 101 main()
    ├── 101 arg loop: `--worker <name>` → recipe::worker_serve and return; `--device` once; `--config` once
    ├── 130 reject other `--` flags; `run` once; positional source then operation
    └── 150 require source; recipe::device_names(device); require .rs; dispatch None→run, "export"→export (one device), else usage


Cargo.toml (148)
├── 1 [package] recipe 0.1.0, edition 2024, build = build.rs
├── 7 [lib] path = recipe.rs; [[bin]] recipe → cli.rs; [dependencies] empty
├── 17 [package.metadata.recipe] build knobs
│   ├── 17 dependency-allowlist, multi-device=auto, contraction-* tile geometry (cpu-shared-values, register m/n, fragment-k, chunk-k, local-chunks, k-partitions, split spans, swizzle, waves)
│   ├── 33 attention-query-tile, delta-chunk, topology-probe-bytes, placement-launch-reserve-bytes (512 MB, kwin framebuffer note), cpu-worker-threads
│   ├── 46 per-OS toolchain: null-device, cpu-compiler/linker/linker-driver/library-flags/entry-linkage/link-flags/module-suffix
│   ├── 55 HSA: hsa-compiler, hsa-runtime, ocml/ockl/abi/finite/math bitcode libs, device-library-directory
│   └── 63 NVIDIA: nvidia-ptx=ptx71, compiler, codegen (llc), runtime, toolkit, libdevice, ptxas
├── 71 [package.metadata.train] defaults
│   ├── 71 epochs, learning-rate, initial-weight, adamw-*, kmeans/svm iterations+hyperparams, tree depth/min-rows, forest fraction
│   ├── 86 bayes-*, boost-*, catboost-*, xgboost-*, lightgbm-*, quantization-block-weights, surrogate-*
│   ├── 100 schedule-candidates/measurements/budget/warmups/minimum-improvement, random-seed, progress-refresh-hz, normalization-epsilon, categorical-ratio
│   └── 108 activation constants (leak/prelu/elu/selu/gelu), huber-threshold, output/gradient/backend tolerances
└── 124 [precision] default-config = recipe
    ├── 127 [precision.recipe] sum/embed/attn/rope/kv/atvn/norm/res = fp16, acc = fp32
    └── 139 [precision.llamacpp] sum = int8, kv = fp16, everything else fp32

```
