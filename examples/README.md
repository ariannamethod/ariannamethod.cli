# examples/

Concrete programs that exercise the metaharmonix stack end-to-end.
Each example is buildable on its own (`make` inside the directory),
links against the **baked** notorch / AML in `../../bake/`, and is
small enough to read top-to-bottom.

## nanollama

Llama 3-style trainer in C, on notorch, no PyTorch. Runtime-configurable
shape via `--tier <name>` plus per-flag overrides (`--dim`, `--layers`,
`--heads`, `--ffn`, `--vocab`). Exposes the same training knobs you'd
expect (`--ctx`, `--steps`, `--accum`, `--lr`, `--log`, `--save`,
`--resume`, `--seed`).

```sh
cd examples/nanollama
make                           # builds against ../../bake/notorch/
./nanollama --help             # see all tiers + flags
python3 tokenize.py            # produce tokens.bin from your corpus
./nanollama tokens.bin --tier micro --steps 5000   # first try
./nanollama tokens.bin --tier nano                  # default 89M
./nanollama tokens.bin --tier small                 # 336M, slower
./nanollama tokens.bin --tier mini --layers 8       # custom shape
```

Origin: ported from
[`ariannamethod/nanollama-notorch`](https://github.com/ariannamethod/nanollama-notorch).
The metaharmonix copy:

- Drops the duplicated `notorch.c/h` — links against the baked
  `../../bake/notorch/` source tree, which doubles as a smoke check
  that user programs can consume the baked headers.
- Replaces hard-coded architectural `#define`s with runtime statics
  driven by CLI flags, so the same binary trains 28M, 50M, 89M, or
  336M models without recompilation.
- Adds named tier presets (`micro`, `mini`, `nano`, `small`) mirroring
  `config/train_*.py` from the upstream nanollama.
