#!/bin/bash
# train_lambda.sh — Janus 8.55M training on Lambda
# Pure Adam. No Chuck. No Python. No PyTorch.
#
# Usage:
#   scp -i ~/Downloads/Janus.pem -r ariannamethod.ai/ ubuntu@<IP>:~/
#   ssh -i ~/Downloads/Janus.pem ubuntu@<IP>
#   cd ~/ariannamethod.ai && bash janus/train_lambda.sh
#
set -euo pipefail

WORK_DIR=~/janus_train
DATA_FILE=$WORK_DIR/gutenberg_170m.txt
BINARY=$WORK_DIR/janus_train
LOG_FILE=$WORK_DIR/train_8m.log

# ── Model config (8.55M) ──
N_EMBD=384
N_HEADS=8
N_LAYERS=8
SEQ_LEN=256
LR=0.0003
STEPS=20000
SAVE_EVERY=1000
LOG_EVERY=10

echo "═══════════════════════════════════════════════════"
echo "  JANUS 8.55M — Lambda Training Setup"
echo "  Pure Adam. Byte-level. CPU."
echo "═══════════════════════════════════════════════════"

mkdir -p $WORK_DIR

# ── Step 1: Download ~170MB Gutenberg data ──
if [ ! -f "$DATA_FILE" ]; then
    echo "[1/3] Downloading Gutenberg data (~170MB)..."
    TEMP_DIR=$(mktemp -d)

    # Top English books from Project Gutenberg — diverse genres, clean UTF-8
    BOOKS=(
        1342  # Pride and Prejudice
        11    # Alice in Wonderland
        1661  # Sherlock Holmes
        84    # Frankenstein
        2701  # Moby Dick
        1952  # The Yellow Wallpaper
        98    # A Tale of Two Cities
        174   # The Picture of Dorian Gray
        1080  # A Modest Proposal
        2600  # War and Peace
        4300  # Ulysses
        1260  # Jane Eyre
        16    # Peter Pan
        5200  # Metamorphosis
        345   # Dracula
        1400  # Great Expectations
        76    # Adventures of Huckleberry Finn
        2554  # Crime and Punishment
        2542  # De Profundis
        219   # Heart of Darkness
        36    # The War of the Worlds
        43    # The Strange Case of Dr Jekyll and Mr Hyde
        55    # The Wonderful Wizard of Oz
        74    # Adventures of Tom Sawyer
        120   # Treasure Island
        158   # Emma
        161   # Sense and Sensibility
        768   # Wuthering Heights
        1184  # The Count of Monte Cristo
        730   # Oliver Twist
        244   # A Study in Scarlet
        3207  # Leviathan
        996   # Don Quixote
        1497  # Republic (Plato)
        2591  # Brothers Grimm Fairy Tales
        2680  # Meditations (Marcus Aurelius)
        4363  # Beyond Good and Evil
        28054 # Brothers Karamazov
        8800  # The Divine Comedy
        2814  # Dubliners
        5740  # Tractatus Logico-Philosophicus
        100   # Complete Works of Shakespeare
        1232  # The Prince (Machiavelli)
        3600  # Thus Spake Zarathustra
        2229  # The Idiot (Dostoevsky)
        600   # Notes from Underground
        135   # Les Misérables
        46    # A Christmas Carol
        514   # Little Women
        1998  # Thus Spake Zarathustra (alt)
    )

    echo "  Fetching ${#BOOKS[@]} books from Project Gutenberg..."
    for id in "${BOOKS[@]}"; do
        url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
        out="$TEMP_DIR/${id}.txt"
        curl -sL "$url" -o "$out" 2>/dev/null || true
        if [ -s "$out" ]; then
            printf "."
        else
            printf "x"
            rm -f "$out"
        fi
    done
    echo ""

    # Concatenate all into one file
    cat $TEMP_DIR/*.txt > "$DATA_FILE"
    rm -rf "$TEMP_DIR"

    SIZE=$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE" 2>/dev/null)
    SIZE_MB=$(echo "scale=1; $SIZE / 1048576" | bc)
    echo "  Data ready: $DATA_FILE ($SIZE_MB MB)"

    if [ "$SIZE" -lt 100000000 ]; then
        echo "  WARNING: only ${SIZE_MB}MB — may need more books"
        echo "  Target is ~170MB for byte-level tokenizer coverage"
    fi
else
    SIZE=$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE" 2>/dev/null)
    SIZE_MB=$(echo "scale=1; $SIZE / 1048576" | bc)
    echo "[1/3] Data exists: $DATA_FILE ($SIZE_MB MB)"
fi

# ── Step 2: Compile ──
echo "[2/3] Compiling janus_train..."
SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)

cc -std=c11 -O3 -march=native -I"$SRC_DIR" \
    -o "$BINARY" \
    "$SRC_DIR/janus/janus_train.c" \
    "$SRC_DIR/core/ariannamethod.c" \
    -lm -lpthread

echo "  Binary: $BINARY"
ls -lh "$BINARY"

# ── Step 3: Train ──
echo "[3/3] Starting training..."
echo "  Config: D=$N_EMBD, H=$N_HEADS, L=$N_LAYERS, seq=$SEQ_LEN"
echo "  LR=$LR, steps=$STEPS, save every $SAVE_EVERY"
echo "  Log: $LOG_FILE"
echo "═══════════════════════════════════════════════════"

cd $WORK_DIR

$BINARY "$DATA_FILE" \
    --n-embd $N_EMBD \
    --n-heads $N_HEADS \
    --n-layers $N_LAYERS \
    --seq-len $SEQ_LEN \
    --lr $LR \
    --steps $STEPS \
    --save-every $SAVE_EVERY \
    --log-every $LOG_EVERY \
    --log-file "$LOG_FILE" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Training complete. Checkpoints in $WORK_DIR/"
echo "═══════════════════════════════════════════════════"
ls -lh $WORK_DIR/janus_ckpt_*.bin 2>/dev/null || echo "  (no checkpoints found)"
