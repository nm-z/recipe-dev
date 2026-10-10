PTXAS := /opt/cuda-11.4/bin/ptxas
RUSTC := rustc
CUDART ?= /home/nate/llama-cuda118-master-bin/libcudart.so.11.0
EVIDENCE := evidence/2026-10-10-m60-p2p
UUIDS := GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28,GPU-c767d250-4454-d181-d648-8706b5f2fb64

all: barrier.cubin $(EVIDENCE)/bench mtp-ptx

# barrier.ptx is the owned source. No nvcc or CUDA C++ regeneration.
barrier.cubin: barrier.ptx
	$(PTXAS) -arch=sm_52 -v -o $@ $<

target/debug/librecipe.rlib: recipe.rs barrier.ptx mtp.ptx Cargo.toml build.rs amd-nv-cpu.ll
	cargo build --lib

$(EVIDENCE)/bench: $(EVIDENCE)/bench.rs target/debug/librecipe.rlib
	$(RUSTC) --edition=2024 -O -o $@ $< --extern recipe=target/debug/librecipe.rlib -L dependency=target/debug/deps -l cuda -C link-arg=$(CUDART) -C link-arg=-Wl,-rpath,$(dir $(CUDART))

run: all
	cd $(EVIDENCE) && CUDA_VISIBLE_DEVICES=$(UUIDS) RECIPE_DEVICE=nv0.nv1 ./bench

clean:
	rm -f barrier.cubin $(EVIDENCE)/bench

.PHONY: all run clean

JOBS ?= 4
.PHONY: recipe mtp-ptx
recipe:
	cargo build --release --lib --bin recipe -j $(JOBS)

mtp-ptx: target/mtp.o

target/mtp.o: mtp.ptx
	mkdir -p target
	$(PTXAS) -arch=sm_52 -c -v $< -o $@
