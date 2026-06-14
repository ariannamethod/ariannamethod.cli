#!/bin/sh
# run_remote.sh — launch a nanollama training run on a remote GPU box.
#
# Generic over SSH targets: works on RunPod, Moebius, Lambda Labs,
# vast.ai, or any Ubuntu/Debian host with passwordless ssh. The script
# does NOT depend on any cloud-provider CLI; the only contract is that
# `ssh <target>` works and the host has `apt`.
#
# Steps:
#   1) rsync this metaharmonix checkout to <target>:~/metaharmonix
#   2) run setup_remote.sh on the target (idempotent)
#   3) make all on the target — builds mhx + bake/notorch + bake/aml +
#      vendor/dario + examples/nanollama
#   4) optionally pre-tokenize a corpus through tokenize.py
#   5) launch training with the requested tier + steps
#
# Usage:
#   bash run_remote.sh <ssh_target> --tier nano --steps 5000 \
#                                   --corpus path/to/corpus.txt
#
# Required:
#   <ssh_target>      e.g. user@1.2.3.4 or runpod-style ssh alias
#
# Optional:
#   --tier <name>     micro|mini|nano|small (default: nano)
#   --steps N         training steps (default: 5000)
#   --ctx N           context length (default: 512)
#   --accum N         grad-accum micro-batches (default: 16)
#   --lr F            peak LR (default: 1.5e-4)
#   --corpus FILE     local corpus file to tokenize before training.
#                     Skipped if the remote already has tokens.bin.
#   --remote-dir D    target dir on the box (default: ~/metaharmonix)
#   --resume PATH     remote checkpoint to resume from
#   --no-rsync        skip the rsync step (assume the repo is already
#                     up to date on the target — useful for fast
#                     re-launches when only flags change)
#   --no-build        skip make all (assume already built)
#   --dry-run         print what would happen, do nothing

set -eu

if [ $# -lt 1 ]; then
    sed -n '2,/^set -eu/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

TARGET="$1"; shift

TIER="nano"
STEPS=5000
CTX=512
ACCUM=16
LR="1.5e-4"
CORPUS=""
REMOTE_DIR="metaharmonix"
RESUME=""
DO_RSYNC=1
DO_BUILD=1
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --tier)        TIER="$2"; shift 2 ;;
        --steps)       STEPS="$2"; shift 2 ;;
        --ctx)         CTX="$2"; shift 2 ;;
        --accum)       ACCUM="$2"; shift 2 ;;
        --lr)          LR="$2"; shift 2 ;;
        --corpus)      CORPUS="$2"; shift 2 ;;
        --remote-dir)  REMOTE_DIR="$2"; shift 2 ;;
        --resume)      RESUME="$2"; shift 2 ;;
        --no-rsync)    DO_RSYNC=0; shift ;;
        --no-build)    DO_BUILD=0; shift ;;
        --dry-run)     DRY=1; shift ;;
        -h|--help)
            sed -n '2,/^set -eu/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

run() {
    echo "+ $*"
    if [ "$DRY" -eq 0 ]; then
        "$@"
    fi
}

# Repo root: this script lives at examples/nanollama/scripts/run_remote.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== run_remote.sh ==="
echo "  target:      $TARGET"
echo "  remote dir:  $REMOTE_DIR"
echo "  tier:        $TIER"
echo "  steps:       $STEPS"
echo "  ctx/accum:   $CTX / $ACCUM"
echo "  lr:          $LR"
echo "  corpus:      ${CORPUS:-<remote tokens.bin reused>}"
[ -n "$RESUME" ] && echo "  resume:      $RESUME"

# 1) rsync checkout. Excludes build artefacts and weights so we ship
#    only sources to the remote.
if [ $DO_RSYNC -eq 1 ]; then
    run rsync -az --delete \
        --exclude '.git/' \
        --exclude 'mhx' \
        --exclude 'tests/test_mhx' \
        --exclude '*.o' --exclude '*.a' \
        --exclude 'bake/notorch/notorch_test' \
        --exclude 'bake/aml/runner/aml' \
        --exclude 'bake/aml/tools/amlc' \
        --exclude 'vendor/dario/dario' \
        --exclude 'examples/nanollama/nanollama' \
        --exclude 'examples/nanollama/nanollama_*.bin' \
        --exclude 'examples/nanollama/tokens.bin' \
        "$REPO_ROOT/" \
        "$TARGET:$REMOTE_DIR/"
fi

# 2) setup_remote.sh on the box (idempotent — re-runs are cheap).
run ssh "$TARGET" "cd $REMOTE_DIR && bash examples/nanollama/scripts/setup_remote.sh"

# 3) build everything: mhx + bake/* + vendor/dario + examples/nanollama.
if [ $DO_BUILD -eq 1 ]; then
    run ssh "$TARGET" "cd $REMOTE_DIR && make all && make -C examples/nanollama"
fi

# 4) tokenize corpus if one was passed. Otherwise assume tokens.bin
#    already lives on the remote (e.g. from a previous run).
if [ -n "$CORPUS" ]; then
    run rsync -az "$CORPUS" "$TARGET:$REMOTE_DIR/examples/nanollama/corpus.txt"
    run ssh "$TARGET" "cd $REMOTE_DIR/examples/nanollama && python3 tokenize.py corpus.txt"
fi

# 5) launch training. We invoke through 'mhx -c \"mhx train nanollama\"'
#    so the codepath the remote box runs is byte-for-byte the same as
#    'mhx train nanollama' from a local REPL. No script-vs-builtin
#    drift.
RESUME_FLAG=""
[ -n "$RESUME" ] && RESUME_FLAG="--resume $RESUME"

run ssh -t "$TARGET" \
    "cd $REMOTE_DIR && ./mhx -c \"mhx train nanollama examples/nanollama/tokens.bin --tier $TIER --steps $STEPS --ctx $CTX --accum $ACCUM --lr $LR $RESUME_FLAG\""

echo "=== done ==="
echo "Checkpoints saved to: $TARGET:$REMOTE_DIR/examples/nanollama/nanollama_{ckpt,final}.bin"
echo "Pull the final back with:"
echo "  rsync -az '$TARGET:$REMOTE_DIR/examples/nanollama/nanollama_final.bin' ./"
