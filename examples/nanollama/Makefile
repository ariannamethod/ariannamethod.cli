# examples/nanollama/Makefile
#
# Builds nanollama against the BAKED notorch (../../bake/notorch/) — no
# duplicate notorch.c/h in this directory. The example serves a second
# purpose besides being a usable trainer: it's a smoke check that the
# baked notorch headers and library are actually consumable from a
# user program, not just from notorch's own internal tests.

CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall

NOTORCH_DIR := ../../bake/notorch
INCLUDE     := -I$(NOTORCH_DIR)

UNAME := $(shell uname)
ifeq ($(USE_SIMD),1)
    # In-house AVX2+FMA cblas shim (notorch_simd.h), x86_64 Haswell+
    # (Intel 2013, AMD 2015). Zero external BLAS dependency — useful on
    # Linux hosts without OpenBLAS or for self-contained builds. The
    # shim re-#defines USE_BLAS internally so call sites in notorch.c
    # stay unchanged; mutually exclusive with -DUSE_BLAS.
    BLAS_CFLAGS := -DUSE_SIMD -mavx2 -mfma
    BLAS_LDLIBS := -lpthread
else ifeq ($(UNAME), Darwin)
    # ACCELERATE_NEW_LAPACK + ILP64: pick the modern Accelerate cblas
    # interface; silences cblas_sgemm/sgemv deprecation warnings on
    # macOS 13.3+. Mirrors notorch's own Makefile flags.
    BLAS_CFLAGS := -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
    BLAS_LDLIBS := -framework Accelerate
else
    # Linux glibc: M_PI (used in nanollama.c) and PATH_MAX live behind
    # _GNU_SOURCE; macOS exposes them unconditionally. Termux/Android
    # also tolerates -D_GNU_SOURCE harmlessly.
    BLAS_CFLAGS := -DUSE_BLAS -D_GNU_SOURCE
    BLAS_LDLIBS := -lopenblas
endif

BIN := nanollama

.PHONY: all clean

all: $(BIN)

# We compile notorch.c straight into the example rather than linking
# libnotorch.a; that keeps a single TU and matches what bake-notorch
# would produce. The shape is set entirely by CLI flags now (see usage),
# so there's nothing in this Makefile that hard-codes 89M.
$(BIN): nanollama.c $(NOTORCH_DIR)/notorch.c $(NOTORCH_DIR)/notorch.h
	$(CC) $(CFLAGS) $(BLAS_CFLAGS) $(INCLUDE) \
		nanollama.c $(NOTORCH_DIR)/notorch.c \
		$(BLAS_LDLIBS) -lm -o $(BIN)

clean:
	rm -f $(BIN) nanollama_ckpt.bin nanollama_final.bin
