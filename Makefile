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

# Compile an existing native PTX module without opening a device context.
.PHONY: native-module
native-module:
	case "$$(hostname)" in archy|benji) ;; *) echo 'Assemble on archy or benji.' >&2; exit 1 ;; esac
	test -n "$(NATIVE_PTX)" && test -n "$(NATIVE_OUTPUT)"
	mkdir -p "$$(dirname -- "$(NATIVE_OUTPUT)")"
	$(PTXAS) -arch=$(ARCH) -O1 -c --disable-optimizer-constants -v "$(NATIVE_PTX)" -o "$(NATIVE_OUTPUT).device.cubin"
	$(dir $(PTXAS))nvlink -arch=$(ARCH) --kernels-used "$$(awk '/\.entry / { for (i=1;i<=NF;i++) if ($$i==".entry") {name=$$(i+1);sub(/\(.*/,"",name); printf "%s%s",sep,name;sep=","}}' "$(NATIVE_PTX)")" -o "$(NATIVE_OUTPUT)" "$(NATIVE_OUTPUT).device.cubin"
