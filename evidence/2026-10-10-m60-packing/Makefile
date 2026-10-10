NVCC   = /opt/cuda/bin/nvcc
PTXAS  = /opt/cuda-11.4/bin/ptxas
CUOBJ  = /opt/cuda/bin/cuobjdump
LIBS   = $(HOME)/llama-cuda118-master-bin
GGML   = $(HOME)/llama.cpp-master/ggml/include
NSET  ?= 1

all: kern.cubin kb

kern.ptx: kern.cu
	$(NVCC) -Wno-deprecated-gpu-targets -arch=sm_52 -DNSET=$(NSET) -ptx -o $@ $<
	sed -i 's/^\.version .*/.version 7.4/' $@
kern.cubin: kern.ptx
	$(PTXAS) -arch=sm_52 -v -o $@ $<
kern.sass: kern.cubin
	$(CUOBJ) -sass $< > $@
kb: kb.c
	gcc -O2 -std=gnu11 -o $@ $< -I$(GGML) -I/opt/cuda/include -L$(LIBS) -lggml -lggml-base -lggml-cuda -L/usr/lib -lcuda -lm -Wl,-rpath,$(LIBS) -Wl,--no-as-needed -Wl,--disable-new-dtags
clean:
	rm -f kern.ptx kern.cubin kern.sass kb
