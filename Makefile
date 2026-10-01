ifeq ($(origin CC),default)
    CC := gcc-16
endif

CFLAGS := -std=c11 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion -Wstrict-prototypes -Wmissing-prototypes -Iinclude -Iinclude/mlc -MMD -MP
TEST_CFLAGS	:= -Itests/support

CONFIG ?= debug

ifeq ($(CONFIG),debug)
    CFLAGS += -g -O0
else ifeq ($(CONFIG),release)
    CFLAGS += -g -O2
else ifeq ($(CONFIG),asan)
    CFLAGS += -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
else
    $(error Unknown CONFIG value: $(CONFIG))
endif

OS := $(shell uname -s)
RAW_ARCH := $(shell uname -m)

ifeq ($(RAW_ARCH),x86_64)
    ARCH := amd64
else ifeq ($(RAW_ARCH),arm64)
    ARCH := arm64
else ifeq ($(RAW_ARCH),aarch64)
    ARCH := arm64
else
    ARCH := $(RAW_ARCH)
endif

ifeq ($(ARCH),amd64)
    CFLAGS +=  -mavx2 -mfma
endif

CUDA ?= 0

SRCS_C := $(wildcard src/*/*.c)
SRCS_H := $(wildcard src/*/*.h)

TEST_SRCS := $(wildcard tests/*.c)
TEST_PROGS := $(patsubst tests/%.c,build/$(CONFIG)/tests/%,$(TEST_SRCS))
TEST_SUPPORT_C := $(wildcard tests/support/*.c)
TEST_SUPPORT_H := $(wildcard tests/support/*.h)

DEPS := $(TEST_PROGS:=.d)
-include $(DEPS)

.DEFAULT_GOAL := all
.PHONY: all info build test clean

all: info test

info:
	@echo "Operating System: $(OS)"
	@echo "Architecture: $(RAW_ARCH) ($(ARCH))"

build:
	@echo "Not implemented yet"

test: $(TEST_PROGS)
	@echo "Running tests..."
	@failures=0; \
	for test in $(TEST_PROGS); do \
	    if ! $$test; then \
	        failures=$$((failures + 1)); \
	    fi; \
	done; \
	if [ $$failures -ne 0 ]; then \
        exit 1; \
    fi;

build/$(CONFIG)/tests:
	@mkdir -p build/$(CONFIG)/tests

build/$(CONFIG)/tests/%: tests/%.c $(SRCS_C) $(SRCS_H) $(TEST_SUPPORT_C) $(TEST_SUPPORT_H) | build/$(CONFIG)/tests
	$(CC) $(CFLAGS) $(TEST_CFLAGS) $< $(TEST_SUPPORT_C) $(SRCS_C) -o $@

clean:
	@echo "Cleaning up..."
	@rm -rf build/*/tests/* || true
