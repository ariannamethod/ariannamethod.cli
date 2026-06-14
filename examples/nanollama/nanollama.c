/*
 * nanollama.c — Llama 3-style trainer on notorch
 * No PyTorch. No Python. Just C.
 *
 * Architecture is now fully runtime-configurable; the file no longer
 * hard-codes the 89M nano shape. CLI flags drive vocab / dim / layers /
 * heads / ffn alongside the existing training knobs (steps, accum,
 * context, lr).
 *
 * Build:
 *   make
 * Run (defaults reproduce the original 89M nano):
 *   ./nanollama tokens.bin
 * Run a different shape (e.g. ~28M micro):
 *   ./nanollama tokens.bin --layers 8 --dim 384 --heads 6 --ffn 1024
 *
 * Architectural flags (any subset; missing ones keep nano defaults):
 *   --vocab N      vocabulary size       (default 32000)
 *   --dim N        hidden dim            (default 576)
 *   --layers N     transformer layers    (default 13)
 *   --heads N      attention heads       (default 9, must divide --dim)
 *   --ffn N        FFN hidden dim        (default 1536)
 *
 * Training flags:
 *   --resume PATH  resume from a checkpoint
 *   --ctx N        context length        (default 512)
 *   --steps N      effective steps       (default 15000)
 *   --accum N      grad-accum micro-batches per step (default 16)
 *   --lr F         peak learning rate    (default 1.5e-4)
 *   --log N        log every N effective steps (default 10)
 *   --save N       save every N effective steps (default 1000)
 *   --seed N       RNG seed              (default 42)
 *
 * Co-authored: Oleg Ataeff & Claude
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ═══ Runtime model config (no longer #define'd) ════════════════════════
 * Defaults reproduce the original 89M nano. CLI flags overwrite. */
static int VOCAB    = 32000;
static int DIM      = 576;
static int NLAYER   = 13;
static int NHEAD    = 9;
static int FFN      = 1536;
static int HDIM     = 64;        /* derived = DIM / NHEAD, recomputed below */
static int NWEIGHTS = 0;         /* derived = 1 + NLAYER * 9 + 2 */

/* ═══ Weight layout ═════════════════════════════════════════════════════
 * wte                          [VOCAB, DIM]
 * per layer (×NLAYER):
 *   rms_attn                   [DIM]
 *   wq, wk, wv, wo             [DIM, DIM]
 *   rms_ffn                    [DIM]
 *   w_gate, w_up               [FFN, DIM]
 *   w_down                     [DIM, FFN]
 * rms_final                    [DIM]
 * w_out                        [VOCAB, DIM]
 *
 * Total tensors: 1 + NLAYER × 9 + 2
 */
static nt_tensor** W = NULL;     /* allocated once NWEIGHTS is known */

#define W_WTE       0
#define W_LAYER(l,i) (1 + (l) * 9 + (i))
/* layer offsets: 0=rms_attn 1=wq 2=wk 3=wv 4=wo 5=rms_ffn 6=wgate 7=wup 8=wdown */
#define W_RMS_FINAL (1 + NLAYER * 9)
#define W_OUT       (1 + NLAYER * 9 + 1)

/* ═══ Init weights ══════════════════════════════════════════════════════
 * Llama/GPT-2 residual scaling: wo and w_down get Xavier * rs,
 * rs = 0.02 / sqrt(2 * NLAYER), so stacked residuals stay bounded. */
static void init_model(void) {
    float rs = 0.02f / sqrtf(2.0f * (float)NLAYER);

    W[W_WTE] = nt_tensor_new2d(VOCAB, DIM);
    nt_tensor_xavier(W[W_WTE], VOCAB, DIM);

    for (int l = 0; l < NLAYER; l++) {
        W[W_LAYER(l,0)] = nt_tensor_new(DIM);
        nt_tensor_fill(W[W_LAYER(l,0)], 1.0f);

        for (int i = 1; i <= 3; i++) {
            W[W_LAYER(l,i)] = nt_tensor_new2d(DIM, DIM);
            nt_tensor_xavier(W[W_LAYER(l,i)], DIM, DIM);
        }

        W[W_LAYER(l,4)] = nt_tensor_new2d(DIM, DIM);
        nt_tensor_xavier(W[W_LAYER(l,4)], DIM, DIM);
        for (int i = 0; i < W[W_LAYER(l,4)]->len; i++)
            W[W_LAYER(l,4)]->data[i] *= rs / 0.1f;

        W[W_LAYER(l,5)] = nt_tensor_new(DIM);
        nt_tensor_fill(W[W_LAYER(l,5)], 1.0f);

        W[W_LAYER(l,6)] = nt_tensor_new2d(FFN, DIM);
        nt_tensor_xavier(W[W_LAYER(l,6)], DIM, FFN);
        W[W_LAYER(l,7)] = nt_tensor_new2d(FFN, DIM);
        nt_tensor_xavier(W[W_LAYER(l,7)], DIM, FFN);

        W[W_LAYER(l,8)] = nt_tensor_new2d(DIM, FFN);
        nt_tensor_xavier(W[W_LAYER(l,8)], FFN, DIM);
        for (int i = 0; i < W[W_LAYER(l,8)]->len; i++)
            W[W_LAYER(l,8)]->data[i] *= rs / 0.1f;
    }

    W[W_RMS_FINAL] = nt_tensor_new(DIM);
    nt_tensor_fill(W[W_RMS_FINAL], 1.0f);

    W[W_OUT] = nt_tensor_new2d(VOCAB, DIM);
    nt_tensor_xavier(W[W_OUT], DIM, VOCAB);
}

/* ═══ Count parameters ══════════════════════════════════════════════════ */
static long count_params(void) {
    long n = 0;
    for (int i = 0; i < NWEIGHTS; i++)
        n += W[i]->len;
    return n;
}

/* ═══ Forward pass on tape — returns loss tape index ════════════════════ */
static int forward(int* tokens, int* targets, int ctx, float* loss_out) {
    nt_tape_start();

    /* Register params */
    int* ti = (int*)malloc(NWEIGHTS * sizeof(int));
    for (int i = 0; i < NWEIGHTS; i++)
        ti[i] = nt_tape_param(W[i]);
    nt_tape_no_decay(ti[W_WTE]);  /* no weight decay on embedding */

    /* Input tensors on tape */
    nt_tensor* tok_t = nt_tensor_new(ctx);
    nt_tensor* tgt_t = nt_tensor_new(ctx);
    for (int i = 0; i < ctx; i++) {
        tok_t->data[i] = (float)tokens[i];
        tgt_t->data[i] = (float)targets[i];
    }
    int tok_idx = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_idx = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);

    /* Embedding (no positional — RoPE handles positions) */
    int h = nt_seq_embedding(ti[W_WTE], -1, tok_idx, ctx, DIM);

    /* Transformer blocks */
    for (int l = 0; l < NLAYER; l++) {
        int ra  = ti[W_LAYER(l,0)];
        int wq_ = ti[W_LAYER(l,1)];
        int wk_ = ti[W_LAYER(l,2)];
        int wv_ = ti[W_LAYER(l,3)];
        int wo_ = ti[W_LAYER(l,4)];
        int rf  = ti[W_LAYER(l,5)];
        int wg  = ti[W_LAYER(l,6)];
        int wu  = ti[W_LAYER(l,7)];
        int wd  = ti[W_LAYER(l,8)];

        /* Attention: norm → Q/K/V → RoPE → MHA → proj → residual */
        int xn   = nt_seq_rmsnorm(h, ra, ctx, DIM);
        int q    = nt_rope(nt_seq_linear(wq_, xn, ctx), ctx, HDIM);
        int k    = nt_rope(nt_seq_linear(wk_, xn, ctx), ctx, HDIM);
        int v    = nt_seq_linear(wv_, xn, ctx);
        int attn = nt_mh_causal_attention(q, k, v, ctx, HDIM);
        int proj = nt_seq_linear(wo_, attn, ctx);
        h = nt_add(h, proj);

        /* FFN: norm → SwiGLU(gate, up) → down → residual */
        xn       = nt_seq_rmsnorm(h, rf, ctx, DIM);
        int gate = nt_seq_linear(wg, xn, ctx);
        int up   = nt_seq_linear(wu, xn, ctx);
        int sg   = nt_swiglu(gate, up);
        int down = nt_seq_linear(wd, sg, ctx);
        h = nt_add(h, down);
    }

    /* Head: norm → linear → cross-entropy */
    int hf     = nt_seq_rmsnorm(h, ti[W_RMS_FINAL], ctx, DIM);
    int logits = nt_seq_linear(ti[W_OUT], hf, ctx);
    int loss   = nt_seq_cross_entropy(logits, tgt_idx, ctx, VOCAB);

    /* Read loss value from tape */
    nt_tape_entry* entries = (nt_tape_entry*)nt_tape_get();
    *loss_out = entries[loss].output->data[0];

    free(ti);
    return loss;
}

/* ═══ Load pre-tokenized binary ═════════════════════════════════════════ */
static int* load_tokens(const char* path, int* n_tokens) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fread(n_tokens, sizeof(int), 1, f);
    int* tokens = (int*)malloc(*n_tokens * sizeof(int));
    fread(tokens, sizeof(int), *n_tokens, f);
    fclose(f);
    return tokens;
}

/* ═══ Save / Load weights ═══════════════════════════════════════════════ */
static void save_weights(const char* path) {
    nt_save(path, (void**)W, NWEIGHTS);
    printf("  saved %s\n", path);
}

static int load_weights(const char* path) {
    int n = 0;
    nt_tensor** loaded = (nt_tensor**)nt_load(path, &n);
    if (!loaded) return 0;
    if (n != NWEIGHTS) {
        fprintf(stderr,
            "  warning: checkpoint has %d tensors, current arch needs %d. "
            "shape mismatch will skip extras.\n", n, NWEIGHTS);
    }
    int copy_n = n < NWEIGHTS ? n : NWEIGHTS;
    for (int i = 0; i < copy_n; i++) {
        if (loaded[i]->len != W[i]->len) {
            fprintf(stderr,
                "  warning: tensor %d size mismatch (ckpt=%d, model=%d), "
                "skipping.\n", i, loaded[i]->len, W[i]->len);
            nt_tensor_free(loaded[i]);
            continue;
        }
        memcpy(W[i]->data, loaded[i]->data, W[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    printf("  resumed from %s (%d tensors)\n", path, copy_n);
    return 1;
}

/* ═══ Cosine LR with warmup ═════════════════════════════════════════════ */
static float get_lr(int step, int total, int warmup, float peak, float min_lr) {
    if (step < warmup)
        return peak * (float)step / (float)warmup;
    float progress = (float)(step - warmup) / (float)(total - warmup);
    return min_lr + (peak - min_lr) * 0.5f * (1.0f + cosf(M_PI * progress));
}

/* ═══ Tier presets ══════════════════════════════════════════════════════
 * Named shapes that mirror config/train_*.py from
 * github.com/ariannamethod/nanollama. Only the shapes that fit on a
 * laptop without GPU are exposed here; large/big/medium need machines
 * we don't deploy from this Makefile yet. Tier sets all five
 * architectural sizes; individual --dim / --layers / etc. flags
 * passed AFTER --tier still override the preset, so '--tier mini
 * --layers 8' is a legitimate one-off shape. */
typedef struct {
    const char* name;
    int dim, layers, heads, ffn;
    const char* note;
} tier_preset;

static const tier_preset TIERS[] = {
    /* head_dim is dim/heads; ffn ≈ 8/3 * dim rounded to a multiple of 256
     * so SwiGLU costs round to a sensible matmul. */
    {"micro",  512,  12,  8,  1408,
        "~28M params; first thing to try on a laptop"},
    {"mini",   640,  12,  8,  1792,
        "~50M params; bigger micro, still CPU-friendly"},
    {"nano",   576,  13,  9,  1536,
        "~89M params; the default — matches the upstream nano shape"},
    {"small", 1024,  24, 16,  2816,
        "~336M params; CPU-trainable but slow"},
    {NULL, 0, 0, 0, 0, NULL},
};

static int apply_tier(const char* name) {
    for (const tier_preset* t = TIERS; t->name; t++) {
        if (!strcmp(t->name, name)) {
            DIM    = t->dim;
            NLAYER = t->layers;
            NHEAD  = t->heads;
            FFN    = t->ffn;
            return 1;
        }
    }
    return 0;
}

/* ═══ Usage ═════════════════════════════════════════════════════════════ */
static void usage(const char* arg0) {
    fprintf(stderr,
        "usage: %s tokens.bin [arch and training flags]\n"
        "\n"
        "  --tier <name>  named preset (overrideable by flags below)\n",
        arg0);
    for (const tier_preset* t = TIERS; t->name; t++) {
        fprintf(stderr,
            "      %-7s dim=%-5d layers=%-3d heads=%-3d ffn=%-5d  %s\n",
            t->name, t->dim, t->layers, t->heads, t->ffn, t->note);
    }
    fprintf(stderr,
        "\n"
        "  architecture (each override is applied after --tier):\n"
        "    --vocab N      default 32000\n"
        "    --dim N        default 576 (nano)\n"
        "    --layers N     default 13\n"
        "    --heads N      default 9 (must divide --dim)\n"
        "    --ffn N        default 1536\n"
        "  training:\n"
        "    --ctx N        default 512\n"
        "    --steps N      default 15000\n"
        "    --accum N      default 16\n"
        "    --lr F         default 1.5e-4\n"
        "    --log N        default 10\n"
        "    --save N       default 1000\n"
        "    --resume PATH  load weights from a checkpoint first\n"
        "    --seed N       default 42\n");
}

/* ═══ Main ══════════════════════════════════════════════════════════════ */
int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    /* --help / -h at argv[1] must be caught before the positional
     * data_path swallows it. Without this guard, `nanollama --help`
     * tries to open "--help" as tokens.bin and exits with an error,
     * which breaks `mhx slots run nanollama --help` smoke test. */
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage(argv[0]);
        return 0;
    }

    const char* data_path   = argv[1];
    const char* resume_path = NULL;
    int ctx        = 512;
    int steps      = 15000;
    int accum      = 16;
    int log_every  = 10;
    int save_every = 1000;
    float lr_peak  = 1.5e-4f;
    int seed       = 42;

    /* Parse args — architecture flags update the static globals so
     * init_model / forward / save / load all see the same shape. */
    for (int i = 2; i < argc; i++) {
        const char* a = argv[i];
        if      (!strcmp(a, "--resume") && i+1 < argc) resume_path = argv[++i];
        else if (!strcmp(a, "--ctx")    && i+1 < argc) ctx        = atoi(argv[++i]);
        else if (!strcmp(a, "--steps")  && i+1 < argc) steps      = atoi(argv[++i]);
        else if (!strcmp(a, "--lr")     && i+1 < argc) lr_peak    = (float)atof(argv[++i]);
        else if (!strcmp(a, "--accum")  && i+1 < argc) accum      = atoi(argv[++i]);
        else if (!strcmp(a, "--log")    && i+1 < argc) log_every  = atoi(argv[++i]);
        else if (!strcmp(a, "--save")   && i+1 < argc) save_every = atoi(argv[++i]);
        else if (!strcmp(a, "--seed")   && i+1 < argc) seed       = atoi(argv[++i]);
        else if (!strcmp(a, "--tier")   && i+1 < argc) {
            const char* name = argv[++i];
            if (!apply_tier(name)) {
                fprintf(stderr, "unknown tier: %s\n", name);
                usage(argv[0]);
                return 1;
            }
        }
        else if (!strcmp(a, "--vocab")  && i+1 < argc) VOCAB      = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")    && i+1 < argc) DIM        = atoi(argv[++i]);
        else if (!strcmp(a, "--layers") && i+1 < argc) NLAYER     = atoi(argv[++i]);
        else if (!strcmp(a, "--heads")  && i+1 < argc) NHEAD      = atoi(argv[++i]);
        else if (!strcmp(a, "--ffn")    && i+1 < argc) FFN        = atoi(argv[++i]);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "unknown flag: %s\n", a);
            usage(argv[0]); return 1;
        }
    }

    /* Validate architecture, derive HDIM and NWEIGHTS. */
    if (VOCAB <= 0 || DIM <= 0 || NLAYER <= 0 || NHEAD <= 0 || FFN <= 0) {
        fprintf(stderr, "all architectural sizes must be > 0\n");
        return 1;
    }
    if (DIM % NHEAD != 0) {
        fprintf(stderr,
            "--dim (%d) must be divisible by --heads (%d) for MHA\n",
            DIM, NHEAD);
        return 1;
    }
    HDIM     = DIM / NHEAD;
    NWEIGHTS = 1 + NLAYER * 9 + 2;
    W        = (nt_tensor**)calloc(NWEIGHTS, sizeof(nt_tensor*));
    if (!W) { fprintf(stderr, "out of memory\n"); return 1; }

    float lr_min = lr_peak * 0.1f;
    int warmup   = steps < 1000 ? steps / 20 : 200;
    int raw_total = steps * accum;

    printf("════════════════════════════════════════════\n");
    printf("  nanollama on notorch — Llama 3-style trainer\n");
    printf("  arch: vocab=%d dim=%d layers=%d heads=%d hdim=%d ffn=%d\n",
           VOCAB, DIM, NLAYER, NHEAD, HDIM, FFN);
    printf("  Chuck optimizer. No PyTorch.\n");
    printf("════════════════════════════════════════════\n");

    /* Load data */
    int n_tokens;
    int* data = load_tokens(data_path, &n_tokens);
    printf("data: %d tokens from %s\n", n_tokens, data_path);
    if (n_tokens < ctx + 1) {
        fprintf(stderr, "need at least %d tokens, got %d\n", ctx + 1, n_tokens);
        return 1;
    }

    /* Init model */
    nt_seed(seed);
    init_model();
    long np = count_params();
    printf("model: %ld params (%.1fM)\n", np, np / 1e6);
    printf("config: ctx=%d accum=%d eff_batch=%d lr=%.2e→%.2e warmup=%d steps=%d seed=%d\n",
           ctx, accum, ctx * accum, lr_peak, lr_min, warmup, steps, seed);

    /* Resume */
    if (resume_path) load_weights(resume_path);

    /* Train */
    srand(seed);
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    float loss_accum = 0.0f;
    float best_loss  = 1e9f;
    int   eff_step   = 0;

    int raw;
    for (raw = 1; raw <= raw_total; raw++) {
        /* Random chunk */
        int off = rand() % (n_tokens - ctx - 1);
        int* tokens  = data + off;
        int* targets = data + off + 1;

        /* Forward + backward + accumulate */
        float loss_val;
        int loss_idx = forward(tokens, targets, ctx, &loss_val);
        loss_accum += loss_val;

        nt_tape_backward(loss_idx);
        nt_tape_accum_grads();
        nt_tape_clear();

        /* Every accum steps: optimizer update */
        if (raw % accum == 0) {
            eff_step++;
            float avg_loss = loss_accum / (float)accum;
            loss_accum = 0.0f;

            float lr = get_lr(eff_step, steps, warmup, lr_peak, lr_min);

            nt_tape_start();
            for (int i = 0; i < NWEIGHTS; i++)
                nt_tape_param(W[i]);
            nt_tape_no_decay(0);  /* embedding = param 0 */
            nt_tape_apply_accum(accum);
            nt_tape_clip_grads(1.0f);
            nt_tape_chuck_step(lr, avg_loss);
            nt_tape_clear();

            if (avg_loss < best_loss) best_loss = avg_loss;

            if (eff_step % log_every == 0 || eff_step == 1) {
                double elapsed = (double)clock() / CLOCKS_PER_SEC - t0;
                long tokens_seen = (long)raw * ctx;
                printf("step %5d/%d | train %.4f | best %.4f | lr %.2e | %.1fM tok | %.0fs\n",
                       eff_step, steps, avg_loss, best_loss, lr, tokens_seen / 1e6, elapsed);
                fflush(stdout);
            }

            if (eff_step % save_every == 0) {
                save_weights("nanollama_ckpt.bin");
            }
        }
    }

    /* Final save */
    save_weights("nanollama_final.bin");
    double elapsed = (double)clock() / CLOCKS_PER_SEC - t0;
    printf("\ndone. %d steps, %.0fs, final train=%.4f best=%.4f\n",
           steps, elapsed, loss_accum, best_loss);
    printf("tokens seen: %.1fM\n", (long)raw_total * ctx / 1e6);

    free(data);
    free(W);
    return 0;
}
