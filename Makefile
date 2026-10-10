SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PTXAS := /opt/cuda-11.4/bin/ptxas
ARCH := sm_52
PTX := $(wildcard *.ptx)

.PHONY: all build kernels
all: build kernels

build:
	case "$$(hostname)" in archy|benji) ;; *) echo 'Build on archy or benji.' >&2; exit 1 ;; esac
	cargo build --release --lib --bin recipe

kernels: $(patsubst %.ptx,target/%.o,$(PTX))

target/%.o: %.ptx
	case "$$(hostname)" in archy|benji) ;; *) echo 'Assemble on archy or benji.' >&2; exit 1 ;; esac
	mkdir -p target
	$(PTXAS) -arch=$(ARCH) -O3 -c -v -o $@ $<

PACKED_EVIDENCE := evidence/2026-10-10-flash-packed
PACKED_BUILD := /home/nate/codex/cx-fl-kern-build/native-worker-provider
LLAMA_SOURCE := /home/nate/llama.cpp-master
NVCC := /opt/cuda/bin/nvcc

$(PACKED_BUILD)/codebooks: $(PACKED_EVIDENCE)/codebooks.c
	mkdir -p $(PACKED_BUILD)/tmp
	$(CC) -O2 -I$(LLAMA_SOURCE)/ggml/src $< -o $@
$(PACKED_BUILD)/codebooks.inc: $(PACKED_BUILD)/codebooks
	$< > $@
$(PACKED_BUILD)/selected.cu: $(PACKED_EVIDENCE)/packed.cu $(PACKED_EVIDENCE)/select-source.sh $(PACKED_BUILD)/codebooks.inc
	bash $(PACKED_EVIDENCE)/select-source.sh $< $(PACKED_EVIDENCE)/winners.tsv $@ $(PACKED_BUILD)/selected-configs.txt
$(PACKED_BUILD)/worker.ptx: $(PACKED_BUILD)/selected.cu $(PACKED_BUILD)/codebooks.inc
	TMPDIR=$(PACKED_BUILD)/tmp $(NVCC) -Wno-deprecated-gpu-targets -arch=sm_52 -DPACKED_WORKER -I$(LLAMA_SOURCE)/ggml/src -I$(PACKED_BUILD) -ptx $< -o $@
	sed -i 's/^\.version .*/.version 7.4/' $@
$(PACKED_BUILD)/worker-helpers.ptx: $(PACKED_BUILD)/worker.ptx $(PACKED_EVIDENCE)/implicit-shared.awk $(PACKED_EVIDENCE)/worker-helpers.awk
	awk -f $(PACKED_EVIDENCE)/implicit-shared.awk $< > $(PACKED_BUILD)/worker-implicit.ptx
	awk -f $(PACKED_EVIDENCE)/worker-helpers.awk $(PACKED_BUILD)/worker-implicit.ptx > $@
packed-worker-helpers: $(PACKED_BUILD)/worker-helpers.ptx
	awk '/^\/\/ BEGIN PRIVATE WORKER HELPERS/{exit}{print}' packed.ptx > $(PACKED_BUILD)/public.ptx
	cp $(PACKED_BUILD)/public.ptx packed.ptx
	printf '// BEGIN PRIVATE WORKER HELPERS\n' >> packed.ptx
	cat $< >> packed.ptx
.PHONY: packed-worker-helpers
