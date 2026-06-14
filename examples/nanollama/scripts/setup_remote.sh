#!/bin/sh
# setup_remote.sh — prepare a fresh Linux GPU box for metaharmonix training.
#
# Idempotent: skips anything already installed. Designed for cloud GPU
# rental boxes (RunPod, Moebius, Lambda Labs, vast.ai, anything with
# Ubuntu/Debian + apt). Run AS the user that will train; the script
# elevates with sudo only for apt.
#
# Usage (run on the remote box, not locally):
#   bash setup_remote.sh

set -eu

echo "=== metaharmonix remote setup ==="

# 1) Detect package manager. We support apt only here — most cloud GPU
#    rentals ship Ubuntu or Debian. Add other branches if a host needs
#    them.
if ! command -v apt-get >/dev/null 2>&1; then
    echo "FATAL: this script expects apt-get (Debian/Ubuntu)." >&2
    echo "Install build-essential + pkg-config + libopenblas-dev + python3 + git manually." >&2
    exit 1
fi

# 2) Install build deps (no GPU drivers — RunPod/Lambda/Moebius ship
#    NVIDIA drivers + CUDA preinstalled). What metaharmonix actually
#    needs is just a C toolchain, OpenBLAS, and Python for the
#    tokenizer.
SUDO=""
[ "$(id -u)" = "0" ] || SUDO="sudo"

echo "+ apt-get update"
$SUDO apt-get update -qq

echo "+ apt-get install build deps"
$SUDO apt-get install -y --no-install-recommends \
    build-essential pkg-config libopenblas-dev \
    python3 python3-pip git ca-certificates curl

# 3) Sanity: print what we got. nvidia-smi if present (optional —
#    metaharmonix CPU path works fine, GPU path is for later when
#    notorch grows CUDA).
echo "=== toolchain ==="
cc --version | head -1
pkg-config --modversion openblas 2>/dev/null && echo "  openblas via pkg-config: ok"
python3 --version
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== gpu ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo "=== ready ==="
echo "Next: clone metaharmonix and run 'make all' from its root."
