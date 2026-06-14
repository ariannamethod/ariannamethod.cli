# metaharmonix — Linux build/test smoke image.
#
# Used as the cross-platform check on Railway: confirms that the REPL,
# the vendored notorch, and the vendored AML all build cleanly on
# Debian-stable amd64 with OpenBLAS, and that the wired smoke commands
# actually execute.
#
# Local: docker build -t metaharmonix-ci . && docker run --rm metaharmonix-ci

FROM debian:stable-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libopenblas-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /metaharmonix
COPY . /metaharmonix

# Build everything: the REPL, libnotorch.a, libaml.a, amlc, runner.
RUN make all

# Run the unit + smoke tests.
RUN make test

# Sanity-check the wired AML and notorch builtins on Linux+OpenBLAS:
# the version line proves the baked AML runner actually executed,
# and the notorch status confirms BLAS detection ran.
RUN ./mhx -c "mhx aml version"
RUN ./mhx -c "mhx notorch"

# Default: drop into the REPL so `railway run` gives a usable shell.
CMD ["./mhx"]
