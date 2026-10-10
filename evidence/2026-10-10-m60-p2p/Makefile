NVCC := /opt/cuda/bin/nvcc
PTXAS := /opt/cuda-11.4/bin/ptxas
RUSTC := rustc
CUDART ?= /home/nate/llama-cuda118-master-bin/libcudart.so.11.0
UUIDS := GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28,GPU-c767d250-4454-d181-d648-8706b5f2fb64

all: barrier.cubin bench

barrier.ptx: barrier.cu packed.cuh
	$(NVCC) -Wno-deprecated-gpu-targets -arch=sm_52 -ptx -o $@ $<
	sed -i 's/^\.version .*/.version 7.4/' $@
	sed -i '$$ { /^$$/d; }' $@
barrier.cubin: barrier.ptx
	$(PTXAS) -arch=sm_52 -v -o $@ $<
bench: bench.rs
	$(RUSTC) --edition=2024 -O -o $@ $< -l cuda -C link-arg=$(CUDART) -C link-arg=-Wl,-rpath,$(dir $(CUDART))
run: all
	CUDA_VISIBLE_DEVICES=$(UUIDS) ./bench
clean:
	rm -f barrier.ptx barrier.cubin bench
.PHONY: all run clean
