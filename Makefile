PTXAS ?= /opt/cuda-11.4/bin/ptxas
ARCH ?= sm_52
JOBS ?= 4

.PHONY: all recipe ptx
all: recipe ptx

recipe:
	cargo build --release --lib --bin recipe -j $(JOBS)

ptx: target/mtp.o

target/mtp.o: mtp.ptx
	mkdir -p target
	$(PTXAS) -arch=$(ARCH) -c -v $< -o $@
