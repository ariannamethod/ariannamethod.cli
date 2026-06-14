# install/

Host-aware wrappers for `mhx install <lang>`. Each wrapper is a small
shell script that:

1. Asks `_detect.sh` what package manager the host runs;
2. Maps `<lang>` to the right package name on that manager;
3. Calls the manager with sane defaults (`-y`, `--noconfirm`, etc.).

Supported package managers:

| Host                | Detected as | Installer used                    |
|---------------------|-------------|-----------------------------------|
| macOS + Homebrew    | `brew`      | `brew install <pkg>`              |
| Debian / Ubuntu     | `apt`       | `sudo apt install -y <pkg>`       |
| Termux / Android    | `pkg`       | `pkg install -y <pkg>`            |
| Arch / pacman       | `pacman`    | `sudo pacman -S --noconfirm <pkg>`|

Supported languages (in this pass): `clang`, `python`, `go`, `rust`,
`lua`. Adding more is a matter of appending one row to the package map
inside the relevant wrapper.

Run `bash install/_detect.sh` standalone to see what `mhx` thinks the
host is. Run `bash install/<lang>.sh` directly without `mhx` if you
want to bypass the REPL.
