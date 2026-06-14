# metaharmonix top-level Makefile
#
# Targets:
#   make              build ./mhx (REPL only — fast)
#   make bake         build vendored notorch (libnotorch.a) + AML (libaml.a, amlc)
#   make all          build mhx and the bake/ libraries
#   make test         build mhx and run unit + smoke tests
#   make test-bake    run vendored notorch + AML test suites
#   make clean        remove build artefacts
#
# Cross-platform notes:
#   - Compiles with any C11-capable cc (clang on macOS, gcc on Linux,
#     clang on Termux/Android aarch64).
#   - The REPL itself depends only on libc; bake/notorch and bake/aml
#     auto-detect BLAS (Accelerate on macOS, OpenBLAS on Linux/Termux
#     via pkg-config when available, scalar fallback otherwise).

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -std=c11
INCLUDE := -Iinclude

SRC     := src/mhx.c
BIN     := mhx

TEST_BIN := tests/test_mhx
TEST_SRC := tests/test_mhx.c src/mhx.c

UNAME := $(shell uname)

# Linux: PATH_MAX (used in src/mhx.c) requires _GNU_SOURCE in glibc.
# macOS and Termux/Android headers expose it without the feature flag.
ifeq ($(UNAME), Linux)
  CFLAGS += -D_GNU_SOURCE
endif

.PHONY: all bake bake-notorch bake-aml bake-dario bake-mesh-agent test test-bake clean

all: $(BIN) bake

$(BIN): $(SRC) include/mhx.h
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $(SRC)

# ── Vendored libraries (baked, not linked) ─────────────────────────────
# notorch / AML / mesh-agent are vendored under bake/ as full source. We
# delegate to their own Makefiles instead of duplicating their build logic
# — they already know how to detect BLAS / cross-compile / etc.

bake: bake-notorch bake-aml bake-dario bake-mesh-agent

bake-notorch:
	@echo "--- building bake/notorch ---"
	@$(MAKE) -C bake/notorch lib

bake-aml:
	@echo "--- building bake/aml ---"
	@$(MAKE) -C bake/aml libaml.a runner amlc

bake-dario:
	@echo "--- building vendor/dario (heart, no-web + sartre) ---"
	@$(MAKE) -C vendor/dario dario

bake-mesh-agent:
	@echo "--- building bake/mesh-agent (Go) ---"
	@$(MAKE) -C bake/mesh-agent native

# ── Tests ──────────────────────────────────────────────────────────────

$(TEST_BIN): $(TEST_SRC) include/mhx.h
	$(CC) $(CFLAGS) $(INCLUDE) -DMHX_NO_MAIN -o $@ $(TEST_SRC)

test: $(TEST_BIN) $(BIN) bake
	@echo "--- unit tests ---"
	@./$(TEST_BIN)
	@echo "--- smoke: mhx -c 'mhx version' ---"
	@./$(BIN) -c "mhx version"
	@echo "--- smoke: mhx -c 'mhx help' (head) ---"
	@./$(BIN) -c "mhx help" | head -3
	@echo "--- smoke: mhx --exec runs builtin argv directly ---"
	@./$(BIN) --exec mhx info | grep -q "CPU cores:" || { echo "FAIL: --exec builtin path"; exit 1; }
	@echo "--- smoke: mhx slots lists wired/runtime lanes ---"
	@./$(BIN) --exec mhx slots | grep -q "heart" || { echo "FAIL: slots missing heart"; exit 1; }
	@echo "--- smoke: mhx train nanollama --help passes through ---"
	@./$(BIN) --exec mhx train nanollama --help 2>&1 | grep -q "^mhx train nanollama <tokens.bin>" || { echo "FAIL: train help path"; exit 1; }
	@echo "--- smoke: host forward (echo hello) ---"
	@./$(BIN) -c "echo hello-from-host"
	@echo "--- smoke: line protocol mode is prompt-free ---"
	@printf "mhx version\nexit\n" | ./$(BIN) --line | grep -q "^metaharmonix " || { echo "FAIL: --line did not emit version"; exit 1; }
	@printf "mhx version\nexit\n" | ./$(BIN) --line | grep -q "^mhx>" && { echo "FAIL: --line printed prompt"; exit 1; } || true
	@echo "--- functional: mhx info reports CPU + RAM + BLAS ---"
	@./$(BIN) -c "mhx info" | grep -q "CPU cores:" || { echo "FAIL: no CPU line in info"; exit 1; }
	@./$(BIN) -c "mhx info" | grep -q "RAM total:" || { echo "FAIL: no RAM line in info"; exit 1; }
	@./$(BIN) -c "mhx info" | grep -q "BLAS:" || { echo "FAIL: no BLAS line in info"; exit 1; }
	@./$(BIN) -c "mhx info" | grep -q "Baked:" || { echo "FAIL: no Baked line in info"; exit 1; }
	@echo "  ok"
	@echo "--- functional: mhx aml version invokes baked runner ---"
	@./$(BIN) -c "mhx aml version" | grep -q "libaml linked" || { echo "FAIL: AML runner did not report libaml"; exit 1; }
	@echo "  ok"
	@echo "--- functional: mhx aml run executes a trivial program ---"
	@TMPROOT="$${TMPDIR:-/tmp}"; \
		SMOKE_FILE="$$TMPROOT/mhx_smoke.aml"; \
		printf "PROPHECY 0\nDESTINY 0\n" > "$$SMOKE_FILE"; \
		./$(BIN) -c "mhx aml run $$SMOKE_FILE" >/dev/null 2>&1 || { echo "FAIL: AML run on trivial program"; rm -f "$$SMOKE_FILE"; exit 1; }; \
		rm -f "$$SMOKE_FILE"
	@echo "  ok"
	@echo "--- functional: mhx notorch reports its BLAS branch ---"
	@./$(BIN) -c "mhx notorch" | grep -q "BLAS at build:" || { echo "FAIL: notorch did not report BLAS branch"; exit 1; }
	@echo "  ok"
	@echo "--- functional: mhx heart enters and exits cleanly ---"
	@printf "/quit\n" | ./$(BIN) -c "mhx heart" 2>&1 | grep -q "sartre" || { echo "FAIL: heart did not initialise SARTRE"; exit 1; }
	@printf "/quit\n" | ./$(BIN) -c "mhx heart" 2>&1 | grep -q "resonance unbroken" || { echo "FAIL: heart did not shut down cleanly"; exit 1; }
	@echo "  ok"
	@echo "--- functional: install/_detect.sh resolves a known manager ---"
	@MGR="$$(sh install/_detect.sh)"; \
		case "$$MGR" in brew|apt|pkg|pacman) echo "  detected: $$MGR" ;; \
		*) echo "FAIL: unknown manager '$$MGR' on this host"; exit 1 ;; esac
	@echo "OK"

test-bake: bake
	@echo "--- bake/notorch tests ---"
	@$(MAKE) -C bake/notorch test || true
	@echo "--- bake/aml tests ---"
	@$(MAKE) -C bake/aml test || true

clean:
	rm -f $(BIN) $(TEST_BIN)
	@$(MAKE) -C bake/notorch clean 2>/dev/null || true
	@$(MAKE) -C bake/aml clean 2>/dev/null || true
