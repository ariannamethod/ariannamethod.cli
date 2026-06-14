#!/bin/sh
# Install clang on the host. macOS ships clang via Xcode CLT, but we
# install the brew formula too so 'brew upgrade' keeps it fresh.
exec sh "$(dirname "$0")/_install_pkg.sh" \
    llvm clang clang clang
