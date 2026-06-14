# vendor/dario — build the metaharmonix heart.
#
# Builds dario.c + sartre_kernel.{c,h} into a single binary 'dario' with
# DARIO_NO_WEB (so 'mhx heart' is a pure REPL — no embedded HTTP server)
# and HAS_SARTRE / HAS_DARIO so the OS layer is wired in.
#
# The full upstream build (kk_kernel + sqlite3 + Janus inference) lives in
# ariannamethod/dario; the vendored copy here intentionally stops at the
# formula + the SARTRE chamber kernel.

CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lm

dario: dario.c sartre_kernel.c sartre_kernel.h
	$(CC) $(CFLAGS) -DDARIO_NO_WEB -DHAS_SARTRE -DHAS_DARIO \
		dario.c sartre_kernel.c $(LDLIBS) -o dario

clean:
	rm -f dario

.PHONY: clean
