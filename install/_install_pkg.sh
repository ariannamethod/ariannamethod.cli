#!/bin/sh
# Shared install backend used by every install/<lang>.sh wrapper.
#
# Usage:
#   install/_install_pkg.sh <brew_pkg> <apt_pkg> <termux_pkg> <pacman_pkg>
#
# Empty string for an argument means "this language is not packaged on
# that manager" — we exit 2 with a clear note in that case so the
# caller (mhx install <lang>) reports a real failure instead of
# pretending it worked.

set -eu

brew_pkg="${1-}"
apt_pkg="${2-}"
termux_pkg="${3-}"
pacman_pkg="${4-}"

here="$(cd "$(dirname "$0")" && pwd)"
mgr="$(sh "$here/_detect.sh")"

run() {
    echo "+ $*"
    "$@"
}

case "$mgr" in
    brew)
        [ -n "$brew_pkg" ] || { echo "no brew package mapping"; exit 2; }
        run brew install "$brew_pkg"
        ;;
    apt)
        [ -n "$apt_pkg" ] || { echo "no apt package mapping"; exit 2; }
        if [ "$(id -u)" = 0 ]; then
            run apt-get update
            run apt-get install -y "$apt_pkg"
        else
            run sudo apt-get update
            run sudo apt-get install -y "$apt_pkg"
        fi
        ;;
    pkg)
        [ -n "$termux_pkg" ] || { echo "no termux/pkg mapping"; exit 2; }
        run pkg install -y "$termux_pkg"
        ;;
    pacman)
        [ -n "$pacman_pkg" ] || { echo "no pacman mapping"; exit 2; }
        run sudo pacman -S --noconfirm "$pacman_pkg"
        ;;
    *)
        echo "metaharmonix: no supported package manager detected." >&2
        echo "PATH probed for: brew, apt/apt-get, pkg (Termux), pacman." >&2
        exit 3
        ;;
esac
