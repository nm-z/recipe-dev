SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PTXAS := /opt/cuda-11.4/bin/ptxas
ARCH := sm_52
NATIVE_ENTRIES ?= recipe_model_forward,recipe_model_step
PTX := $(wildcard *.ptx)

.PHONY: all build kernels native-module router-check
all: build kernels

build:
	case "$$(hostname)" in archy|benji) ;; *) echo 'Build on archy or benji.' >&2; exit 1 ;; esac
	cargo build --release --lib --bin recipe

kernels: $(patsubst %.ptx,target/%.o,$(PTX))

target/%.o: %.ptx
	case "$$(hostname)" in archy|benji) ;; *) echo 'Assemble on archy or benji.' >&2; exit 1 ;; esac
	mkdir -p target
	$(PTXAS) -arch=$(ARCH) -O3 -c -v -o $@ $<

native-module:
	case "$$(hostname)" in archy|benji) ;; *) echo 'Assemble on archy or benji.' >&2; exit 1 ;; esac
	[[ -n "$(NATIVE_PTX)" && -n "$(NATIVE_CUBIN)" ]]
	$(PTXAS) -arch=$(ARCH) -O1 -c --disable-optimizer-constants -v "$(NATIVE_PTX)" -o "$(NATIVE_CUBIN).device.cubin"
	$(dir $(PTXAS))nvlink -arch=$(ARCH) --kernels-used $(NATIVE_ENTRIES) "$(NATIVE_CUBIN).device.cubin" -o "$(NATIVE_CUBIN)"

router-check:
	case "$$(hostname)" in archy|benji) ;; *) echo 'Check on archy or benji.' >&2; exit 1 ;; esac
	[[ -n "$(ROUTER_CHECK)" ]]
	$(CC) -O2 -std=c11 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off evidence/2026-10-11-fl-topk/check-router.c -lm -o "$(ROUTER_CHECK)"
	"$(ROUTER_CHECK)"
