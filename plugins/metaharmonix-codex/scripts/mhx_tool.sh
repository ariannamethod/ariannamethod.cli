#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PLUGIN_DIR/../.." && pwd)"
MHX_BIN="$REPO_ROOT/mhx"

if [[ ! -x "$MHX_BIN" ]]; then
  echo "metaharmonix-codex: mhx binary missing at $MHX_BIN" >&2
  echo "build it first with: make" >&2
  exit 1
fi

cmd="${1:-}"
shift || true

case "$cmd" in
  info)
    exec "$MHX_BIN" --exec mhx info
    ;;
  aml-version)
    exec "$MHX_BIN" --exec mhx aml version
    ;;
  aml-run)
    if [[ $# -lt 1 ]]; then
      echo "usage: $0 aml-run <file.aml>" >&2
      exit 1
    fi
    exec "$MHX_BIN" --exec mhx aml run "$1"
    ;;
  notorch-info)
    exec "$MHX_BIN" --exec mhx notorch info
    ;;
  notorch-test)
    exec "$MHX_BIN" --exec mhx notorch test
    ;;
  heart)
    exec "$MHX_BIN" --exec mhx heart "$@"
    ;;
  slots)
    exec "$MHX_BIN" --exec mhx slots "$@"
    ;;
  slot-show)
    if [[ $# -lt 1 ]]; then
      echo "usage: $0 slot-show <slot>" >&2
      exit 1
    fi
    exec "$MHX_BIN" --exec mhx slots show "$1"
    ;;
  slot-run)
    if [[ $# -lt 1 ]]; then
      echo "usage: $0 slot-run <slot> [args...]" >&2
      exit 1
    fi
    slot="$1"
    shift
    exec "$MHX_BIN" --exec mhx slots run "$slot" "$@"
    ;;
  nanollama-build)
    exec make -C "$REPO_ROOT/examples/nanollama"
    ;;
  nanollama-train)
    if [[ $# -lt 1 ]]; then
      echo "usage: $0 nanollama-train <tokens.bin> [extra nanollama args...]" >&2
      exit 1
    fi
    exec "$MHX_BIN" --exec mhx train nanollama "$@"
    ;;
  "")
    cat <<EOF
metaharmonix-codex helper

commands:
  info
  aml-version
  aml-run <file.aml>
  notorch-info
  notorch-test
  heart
  slots [list]
  slot-show <slot>
  slot-run <slot> [args...]
  nanollama-build
  nanollama-train <tokens.bin> [args...]
EOF
    ;;
  *)
    echo "unknown command: $cmd" >&2
    exit 1
    ;;
esac
