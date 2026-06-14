# examples/nanollama/scripts/

Tiny shell glue for running nanollama training on a rented GPU box.
Generic over SSH targets — works on **RunPod**, **Moebius**,
**Lambda Labs**, **vast.ai**, or any Ubuntu/Debian host with
passwordless ssh. No cloud-provider CLI dependency; the only contract
is `ssh <target>` works and the host has `apt`.

## Two scripts

### `setup_remote.sh`

Run **on** the GPU box. Idempotent. Installs build-essential +
pkg-config + libopenblas-dev + python3 + git, prints the resulting
toolchain, and shows `nvidia-smi` if a GPU is attached. No NVIDIA
driver install — every GPU rental we use ships drivers preinstalled.

### `run_remote.sh`

Run **locally**, points at a remote SSH target. End-to-end:

1. `rsync` the metaharmonix checkout to `<target>:~/metaharmonix`
   (excludes build artefacts and weights — only source ships).
2. Runs `setup_remote.sh` on the box (idempotent).
3. `make all` on the remote — builds `mhx` + `bake/notorch` + `bake/aml`
   + `vendor/dario` + `examples/nanollama`.
4. Optionally tokenises a local corpus through `tokenize.py`.
5. Launches training **through `mhx -c "mhx train nanollama …"`** —
   so the codepath is byte-identical to a local `mhx train nanollama`,
   no script-vs-builtin drift.

## Quickstart

Spin up an H100 / H200 / A100 box on RunPod (or wherever), copy the
SSH command they give you, then locally:

```sh
# First run: ship the repo, install deps, build, train.
bash examples/nanollama/scripts/run_remote.sh \
    root@<runpod-ip> \
    --tier nano \
    --steps 5000 \
    --corpus ./my_corpus.txt

# Re-launch with different flags (skip rsync + skip rebuild):
bash examples/nanollama/scripts/run_remote.sh \
    root@<runpod-ip> \
    --no-rsync --no-build \
    --tier small --steps 20000 \
    --resume examples/nanollama/nanollama_ckpt.bin

# Pull the final weights back home:
rsync -az 'root@<runpod-ip>:metaharmonix/examples/nanollama/nanollama_final.bin' ./
```

`--dry-run` prints what would happen without actually touching the
remote — useful for double-checking flags before paying for compute.

## Tier sizing for cloud rentals

| tier  | params | minimum useful GPU                |
|-------|--------|-----------------------------------|
| micro | ~28M   | T4 / A10 — 8 GB enough            |
| mini  | ~50M   | A10 / A100-40 — comfortable       |
| nano  | ~89M   | A100-40 / A100-80                 |
| small | ~336M  | A100-80 / H100 — needs the room   |

H100 / H200 boxes on RunPod / Moebius oversize all four tiers; pick
the one that matches your training budget.

## Why no cloud CLI

`runpodctl`, `lambdalabs` and `moebius` clients all exist, all do
slightly different things, and all break differently. A plain `ssh`
+ `rsync` script depends only on tools that have been stable since
the 90s and works on every host that opens port 22.
