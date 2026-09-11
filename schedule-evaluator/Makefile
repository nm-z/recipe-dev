CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDLIBS ?= -lm

.PHONY: check

evaluate-c: evaluate.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

check: evaluate-c
	./evaluate-c --self-check
