NVCC = /opt/cuda/bin/nvcc
PTXAS = /opt/cuda-11.4/bin/ptxas
RUSTC = rustc
LLAMA = /home/nate/llama.cpp-master
LIBS = /home/nate/llama-cuda118-master-bin
CUDART ?= $(LIBS)/libcudart.so.11.0
PACKED_EVIDENCE = evidence/2026-10-10-flash-packed
PEER_EVIDENCE = evidence/2026-10-10-m60-p2p
BUILD = /home/nate/codex/cx-fl-kern-build
UUIDS = GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28,GPU-c767d250-4454-d181-d648-8706b5f2fb64
all: barrier.cubin $(PEER_EVIDENCE)/bench packed.ptx $(BUILD)/packed.cubin $(BUILD)/bench $(PACKED_EVIDENCE)/gguf-header
barrier.cubin: barrier.ptx
	$(PTXAS) -arch=sm_52 -v -o $@ $<
target/debug/librecipe.rlib: recipe.rs barrier.ptx packed.ptx Cargo.toml build.rs amd-nv-cpu.ll
	cargo build --lib
$(PEER_EVIDENCE)/bench: $(PEER_EVIDENCE)/bench.rs target/debug/librecipe.rlib
	$(RUSTC) --edition=2024 -O -o $@ $< --extern recipe=target/debug/librecipe.rlib -L dependency=target/debug/deps -l cuda -C link-arg=$(CUDART) -C link-arg=-Wl,-rpath,$(dir $(CUDART))
run: all
	cd $(PEER_EVIDENCE) && CUDA_VISIBLE_DEVICES=$(UUIDS) RECIPE_DEVICE=nv0.nv1 ./bench
$(BUILD):
	mkdir -p $@
$(BUILD)/codebooks: $(PACKED_EVIDENCE)/codebooks.c $(LLAMA)/ggml/src/ggml-common.h | $(BUILD)
	$(CC) -O2 -Wall -Wextra -I$(LLAMA)/ggml/src $< -o $@
$(BUILD)/codebooks.inc: $(BUILD)/codebooks
	$< > $@.partial && mv $@.partial $@
$(BUILD)/probes.ptx: $(PACKED_EVIDENCE)/packed.cu $(BUILD)/codebooks.inc | $(BUILD)
	mkdir -p $(BUILD)/tmp
	TMPDIR=$(BUILD)/tmp $(NVCC) -Wno-deprecated-gpu-targets -arch=sm_52 -I$(LLAMA)/ggml/src -I$(BUILD) -ptx $< -o $@
	sed -i 's/^\.version .*/.version 7.4/' $@
$(BUILD)/selected.cu: $(PACKED_EVIDENCE)/packed.cu $(PACKED_EVIDENCE)/winners.tsv $(PACKED_EVIDENCE)/fixed-cta.tsv $(PACKED_EVIDENCE)/select-source.sh | $(BUILD)
	bash $(PACKED_EVIDENCE)/select-source.sh $(PACKED_EVIDENCE)/packed.cu $(PACKED_EVIDENCE)/winners.tsv $@ $(BUILD)/selected-configs.txt
$(BUILD)/selected.ptx: $(BUILD)/selected.cu $(BUILD)/codebooks.inc
	mkdir -p $(BUILD)/tmp
	TMPDIR=$(BUILD)/tmp $(NVCC) -Wno-deprecated-gpu-targets -arch=sm_52 -I$(LLAMA)/ggml/src -I$(BUILD) -ptx $< -o $@
	sed -i 's/^\.version .*/.version 7.4/' $@
packed.ptx: $(BUILD)/selected.ptx $(PACKED_EVIDENCE)/extract-helpers.awk $(PACKED_EVIDENCE)/prune-globals.awk
	awk -f $(PACKED_EVIDENCE)/prune-globals.awk $< $< > $(BUILD)/selected-pruned.ptx
	awk -f $(PACKED_EVIDENCE)/extract-helpers.awk $(BUILD)/selected-pruned.ptx > $@
$(BUILD)/packed.cubin: $(BUILD)/probes.ptx
	$(PTXAS) -arch=sm_52 -v $< -o $@
$(BUILD)/helpers.cubin: packed.ptx | $(BUILD)
	$(PTXAS) -arch=sm_52 -v $< -o $@
$(BUILD)/packed.sass: $(BUILD)/packed.cubin
	/opt/cuda/bin/cuobjdump -sass $< > $@
$(BUILD)/bench: $(PACKED_EVIDENCE)/bench.c | $(BUILD)
	$(CC) -O2 -Wall -Wextra -std=gnu11 $< -o $@ -I$(LLAMA)/ggml/include -I$(LLAMA)/ggml/src -I/opt/cuda/include -L$(LIBS) -lggml -lggml-base -lggml-cpu -lggml-cuda -lcuda -lm -Wl,-rpath,$(LIBS) -Wl,--disable-new-dtags
$(PACKED_EVIDENCE)/gguf-header: $(PACKED_EVIDENCE)/gguf-header.c
	$(CC) -O2 -Wall -Wextra $< -o $@
inventory: $(PACKED_EVIDENCE)/gguf-header
	bash $(PACKED_EVIDENCE)/inventory.sh
samples:
	bash $(PACKED_EVIDENCE)/fetch.sh
	bash $(PACKED_EVIDENCE)/fetch-experts.sh
benchmark: packed.ptx $(BUILD)/packed.cubin $(BUILD)/bench
	BENCH_BINARY=$(BUILD)/bench PACKED_CUBIN=$(BUILD)/packed.cubin ROW_LANES=8 WARPS_MIN=2 NITER=100 RUN_TAG=review bash $(PACKED_EVIDENCE)/run.sh
clean:
	rm -f barrier.cubin $(PEER_EVIDENCE)/bench
.PHONY: all run inventory samples benchmark clean
