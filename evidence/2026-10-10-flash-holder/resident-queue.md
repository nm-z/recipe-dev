# Resident Flash queue and compact logit stores

The current allocation owner runs the complete 48-layer target and original Q8_0 MTP head on archy dies 4, 0, 1, 3, 5, and 7. The original model has 512 experts per layer and selects 10. Die 0 is capped at 6,186,598,400 bytes; physical dies 2 and 6 remain fenced. The existing placement includes 7,331,302,400 bytes of pinned RAM expert bundles. GPU allocation reservations precede upload completion.

## Module change

Terminal vocabulary projections have no graph consumers of inactive request positions. Their compact output length is five when MTP is selected. The worker writes complete owned rows contiguously. The active interval is `[begin, begin + active_columns)`; every active value is read from its original column-major matvec output, and inactive transient positions are zeroed. The bounds check requires the interval to fit the compact row. Nonterminal outputs and other layouts keep the existing scatter.

This extends the single-column store candidate to batched MTP verification. The 192-byte job descriptor, terminal marker, allocation sizes, peer addresses, reduction order, worker symbol, dynamic shared size, and synchronization sequence remain unchanged.

The replacement uses the public `recipe::expert_worker_ptx()` emitter. The release library and CLI build through the repository Makefile with one Cargo job and nice 10 on archy. A supplemental Makefile target assembles and links the emitted hand-written PTX with CUDA 11.4 `ptxas -O1 -c -arch=sm_52` and `nvlink --kernels-used expert_split_worker`.

CPU build, assembly, and link exit 0. The entry uses 22 registers, no stack frame, no spill stores, and no spill loads. These are static build observations. Resident reload, final-logit correctness, and full-model performance remain separate gates.

Source and module receipts: `/home/nate/codex/flash-logit-stores/{build.log,build.exit,module.exit,source-module.sha256}`. Worker cubin SHA256: `0c3419b499679a9b52e857f585159978d4d3a96754facd0a4f80fe6cfd3e83e0`.

## Queue

The live queue is `/home/nate/codex/flash-queue/{in,out}` on archy. Atomically rename UTF-8 newline-separated Recipe chat commands and prompts into `in/<UTC-sortable timestamp>-<worker>-<unique>.job`. Jobs execute in filename order. A trailing `/report` proves that commands reached the resident loop. One atomic `out/<job-id>.result` records the exact process identity, source revision, input hash, status, report path, and log path.

Supported controls are `/clear`, `/mtp N`, `/lookup N`, `/reply N`, `/artifacts`, `/reload_workers /absolute/cubin`, `/reload`, `/reload_mtp`, `/report /absolute/path`, `/dump_ids /absolute/path`, and `/dump_logits /absolute/path`. Main and MTP reloads use their resident artifact paths and require allocation-owner coordination. Every module must preserve buffers and ABI. No job sends `/exit`, opens another GPU context, acquires the allocation-owner lock, or writes the input FIFO directly.

The controller waits on busy locks. It never resubmits uncertain input. A claimed job requires review before controller recovery. Results distinguish completed consumption, observed command errors, rejection before input, and uncertain consumption. Reports retain the requests completed by each job rather than repeatedly storing the entire accumulated history.

When submitted jobs finish, the controller cycles plain, MTP2, and MTP4 benchmarks with the newest modules. It yields to submitted jobs at request boundaries. The current context is 128 tokens, so 256-token measurements aggregate shorter independent requests. Rates use total observed output divided by total generation time; request counts and actual emitted tokens must accompany those rates.

The existing CLI exposes final logits, token IDs, and per-operation clocks. Per-layer captures require a compatible diagnostic module and canonical restoration. A final-logit dump alone is not a per-layer dump.

## Allocation lifetime

The current executable owns allocations in its compute process. Module reload preserves those allocations, but a compute-process crash would still release them. Durable IPC allocation ownership and attach-only compute are a separate required gate. Do not report the current process as crash resilient. Never reload model weights from disk without CEO authorization.
