#!/usr/bin/env bash
# tests/test_amlc.sh — sanity test for the amlc transpiler.
#
# 1. `make amlc` builds tools/amlc from tools/amlc.c.
# 2. The binary parses a small synthetic .aml with two BLOOD COMPILE
#    blocks, one BLOOD MAIN, one BLOOD LINK, and one ECHO directive,
#    then compiles the generated C and runs it.
# 3. Asserts the emitted program prints the expected marker.
#
# This guards the happy-path syntax tree of amlc — brace tracker,
# directive recogniser, ECHO injection, accel auto-link.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== test_amlc =="

make amlc >/tmp/amlc_build.log 2>&1
if [ ! -x tools/amlc ]; then
    echo "  FAIL: make amlc did not produce tools/amlc"
    cat /tmp/amlc_build.log
    exit 1
fi
echo "  PASS [build]: tools/amlc"

# ── Synthetic .aml + helper header ──────────────────────────────────────
WORK=$(mktemp -d)
trap "rm -rf '$WORK'" EXIT

cat > "$WORK/test.aml" <<'EOF'
# tiny smoke .aml — exercises BLOOD COMPILE / BLOOD MAIN / BLOOD LINK / ECHO

ECHO "test_echo.h"

BLOOD COMPILE smoke_helpers {
#include <stdio.h>
static void say(const char *s) { printf("%s\n", s); }
}

BLOOD COMPILE smoke_logic {
static int compute(int x) {
    if (x > 0) { return x * 2; } else { return -1; }
}
}

BLOOD LINK -lm

BLOOD MAIN {
int main(void) {
    say("AMLC_OK");
    int v = compute(21);
    if (v != 42) { say("FAIL_COMPUTE"); return 1; }
    say(ECHO_MARKER);
    return 0;
}
}
EOF

cat > "$WORK/test_echo.h" <<'EOF'
#define ECHO_MARKER "ECHO_OK"
EOF

# ── Compile + run ───────────────────────────────────────────────────────
cd "$WORK"
"$OLDPWD/tools/amlc" test.aml -o test_smoke --no-accel >/tmp/amlc_compile.log 2>&1
if [ ! -x test_smoke ]; then
    echo "  FAIL: amlc did not emit test_smoke binary"
    cat /tmp/amlc_compile.log
    exit 1
fi
echo "  PASS [compile]: test.aml → test_smoke"

OUT=$(./test_smoke)
EXPECTED=$'AMLC_OK\nECHO_OK'
if [ "$OUT" = "$EXPECTED" ]; then
    echo "  PASS [run]: emitted output matches"
else
    echo "  FAIL: output mismatch"
    echo "  expected: $EXPECTED"
    echo "  got:      $OUT"
    exit 1
fi

echo
echo "amlc OK"
