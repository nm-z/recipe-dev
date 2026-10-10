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
