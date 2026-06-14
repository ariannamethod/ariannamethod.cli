#!/bin/sh
# Detect the host package manager. Prints one of:
#   brew | apt | pkg | pacman | unknown
# Termux's `pkg` is checked before `apt` because Termux ships both.

set -eu

if [ -n "${PREFIX-}" ] && [ -d "$PREFIX/etc/apt" ] && command -v pkg >/dev/null 2>&1; then
    echo "pkg"
elif command -v brew >/dev/null 2>&1; then
    echo "brew"
elif command -v apt >/dev/null 2>&1 || command -v apt-get >/dev/null 2>&1; then
    echo "apt"
elif command -v pacman >/dev/null 2>&1; then
    echo "pacman"
else
    echo "unknown"
fi
