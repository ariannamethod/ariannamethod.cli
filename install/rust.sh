#!/bin/sh
# Termux/pkg ships 'rust' as a metapackage; everywhere else we go via
# the host's Rust package, NOT rustup — keeps install repeatable and
# avoids touching the user's home with a curl|sh installer.
exec sh "$(dirname "$0")/_install_pkg.sh" \
    rust rustc rust rust
