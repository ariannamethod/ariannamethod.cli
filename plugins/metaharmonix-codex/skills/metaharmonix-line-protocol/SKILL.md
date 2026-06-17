---
name: metaharmonix-line-protocol
description: Use metaharmonix through `mhx -c` or its line protocol to inspect AML, notorch, heart, and baked-runtime status from the local repo checkout.
---

# Metaharmonix Line Protocol

Use this skill when the task is about the local `metaharmonix` checkout and the
goal is to inspect or drive it through its terminal surface.

## Local assumptions

- Repo root: `<metaharmonix-repo-root>`
- Binary: `./mhx`
- Preferred non-interactive form: `./mhx -c "<command>"`

## Good commands

```sh
./mhx -c "mhx version"
./mhx -c "mhx info"
./mhx -c "mhx aml version"
./mhx -c "mhx notorch info"
./mhx -c "mhx heart"
```

Repo-local helper wrapper:

```sh
bash plugins/metaharmonix-codex/scripts/mhx_tool.sh info
bash plugins/metaharmonix-codex/scripts/mhx_tool.sh aml-version
bash plugins/metaharmonix-codex/scripts/mhx_tool.sh notorch-info
bash plugins/metaharmonix-codex/scripts/mhx_tool.sh nanollama-build
```

## Notes

- Prefer `-c` for one-shot plugin/tool calls.
- `mhx --line` is the prompt-free multi-command mode.
- Baked tools resolve from the repo root discovered by `mhx`, or by
  `MHX_ROOT` / `MHX_BAKE_PATH` when explicitly set.
