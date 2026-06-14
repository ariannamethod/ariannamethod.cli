/*
 * infer_doe.c — Sample from a doe checkpoint trained by train_doe.c
 *
 * Architecture: V=256, E=224, H=8, FFN=896, CTX=128, L=12 (char-level)
 * Loads:        doe_ckpt.bin / doe_best.bin / doe_10m.bin
 * Sampling:     temperature + top-k, prompts can be piped via -p or stdin
 *
 * Build: make infer_doe
 * Run:   ./infer_doe doe_ckpt.bin -p "<|user|>Who are you?<|assistant|>" -n 200 -t 0.85 -k 40
 *        echo "<|user|>What is the parliament?<|assistant|>" | ./infer_doe doe_best.bin -n 240
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V     256
#define EMB   224
#define HEADS 8
#define HD    (EMB / HEADS)
#define FFN_D 896
#define CTX   128
#define NLAYERS 12

typedef struct {
    nt_tensor *wte, *wpe;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo;
        nt_tensor *rms2, *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static Model* model_new(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(V, EMB);
    m->wpe = nt_tensor_new2d(CTX, EMB);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(EMB);
        m->L[l].wq = nt_tensor_new2d(EMB, EMB);
        m->L[l].wk = nt_tensor_new2d(EMB, EMB);
        m->L[l].wv = nt_tensor_new2d(EMB, EMB);
        m->L[l].wo = nt_tensor_new2d(EMB, EMB);
        m->L[l].rms2 = nt_tensor_new(EMB);
        m->L[l].w_gate = nt_tensor_new2d(FFN_D, EMB);
        m->L[l].w_up = nt_tensor_new2d(FFN_D, EMB);
        m->L[l].w_down = nt_tensor_new2d(EMB, FFN_D);
    }
    m->rms_f = nt_tensor_new(EMB);
    m->head = nt_tensor_new2d(V, EMB);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte); nt_tensor_free(m->wpe);
    for (int l = 0; l < NLAYERS; l++) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq); nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv); nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up);
        nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head); free(m);
}

static int model_n_tensors(void) { return 2 + NLAYERS * 9 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte; p[i++] = m->wpe;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].rms1; p[i++]=m->L[l].wq; p[i++]=m->L[l].wk;
        p[i++]=m->L[l].wv; p[i++]=m->L[l].wo; p[i++]=m->L[l].rms2;
        p[i++]=m->L[l].w_gate; p[i++]=m->L[l].w_up; p[i++]=m->L[l].w_down;
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static int load_weights(Model* m, const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) return -1;
    int expected = model_n_tensors();
    if (n_loaded != expected) {
        fprintf(stderr, "weights tensor count mismatch: got %d, want %d\n", n_loaded, expected);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded);
        return -1;
    }
    nt_tensor** mp = model_param_array(m);
    for (int i = 0; i < expected; i++) {
        memcpy(mp[i]->data, loaded[i]->data, mp[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    free(loaded); free(mp);
    return 0;
}

static int forward(Model* m, int* tokens, int* targets) {
    int wte_i = nt_tape_param(m->wte);
    int wpe_i = nt_tape_param(m->wpe);
    int li[NLAYERS][9];
    for (int l = 0; l < NLAYERS; l++) {
        li[l][0] = nt_tape_param(m->L[l].rms1);
        li[l][1] = nt_tape_param(m->L[l].wq);
        li[l][2] = nt_tape_param(m->L[l].wk);
        li[l][3] = nt_tape_param(m->L[l].wv);
        li[l][4] = nt_tape_param(m->L[l].wo);
        li[l][5] = nt_tape_param(m->L[l].rms2);
        li[l][6] = nt_tape_param(m->L[l].w_gate);
        li[l][7] = nt_tape_param(m->L[l].w_up);
        li[l][8] = nt_tape_param(m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) { tok_t->data[i] = (float)tokens[i]; tgt_t->data[i] = (float)targets[i]; }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, CTX, EMB);
    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, EMB);
        int q = nt_seq_linear(li[l][1], xn, CTX);
        int k = nt_seq_linear(li[l][2], xn, CTX);
        int v = nt_seq_linear(li[l][3], xn, CTX);
        int attn = nt_mh_causal_attention(q, k, v, CTX, HD);
        int proj = nt_seq_linear(li[l][4], attn, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l][5], CTX, EMB);
        int gate = nt_silu(nt_seq_linear(li[l][6], xn, CTX));
        int up = nt_seq_linear(li[l][7], xn, CTX);
        int down = nt_seq_linear(li[l][8], nt_mul(gate, up), CTX);
        h = nt_add(h, down);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, EMB);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, V);
}

/* top-k filter: zero out everything outside top-k by logit */
static void filter_top_k(float* logits, int k) {
    if (k <= 0 || k >= V) return;
    /* simple O(V*k) selection */
    int idx[V]; for (int i = 0; i < V; i++) idx[i] = i;
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < V; j++) if (logits[idx[j]] > logits[idx[best]]) best = j;
        int t = idx[i]; idx[i] = idx[best]; idx[best] = t;
    }
    char keep[V] = {0};
    for (int i = 0; i < k; i++) keep[idx[i]] = 1;
    for (int i = 0; i < V; i++) if (!keep[i]) logits[i] = -1e9f;
}

static int sample_token(Model* m, int* ctx, int gen_len, float temp, int top_k) {
    int tokens[CTX], targets[CTX];
    for (int i = 0; i < gen_len; i++) tokens[i] = ctx[i];
    for (int i = gen_len; i < CTX; i++) tokens[i] = 0;
    memset(targets, 0, sizeof(targets));
    nt_tape_start();
    int loss_idx = forward(m, tokens, targets);
    nt_tape* tape = nt_tape_get();
    int logits_idx = tape->entries[loss_idx].parent1;
    float* last = tape->entries[logits_idx].output->data + (gen_len-1) * V;

    float buf[V];
    for (int i = 0; i < V; i++) buf[i] = last[i] / (temp > 1e-6f ? temp : 1.0f);
    filter_top_k(buf, top_k);
    float mx = buf[0]; for (int i=1;i<V;i++) if(buf[i]>mx) mx=buf[i];
    float sm = 0; for (int i=0;i<V;i++) { buf[i]=expf(buf[i]-mx); sm+=buf[i]; }
    for (int i=0;i<V;i++) buf[i]/=sm;
    float r = (float)rand()/(float)RAND_MAX, cum=0;
    int next = 0;
    for (int i = 0; i < V; i++) { cum += buf[i]; if (cum >= r) { next = i; break; } }
    nt_tape_clear();
    return next;
}

int main(int argc, char** argv) {
    const char* weights_path = NULL;
    const char* prompt = NULL;
    int max_new = 240;
    float temp = 0.85f;
    int top_k = 40;
    unsigned int seed = 1729;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i+1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned int)atoi(argv[++i]);
        else if (argv[i][0] != '-' && !weights_path) weights_path = argv[i];
    }
    if (!weights_path) {
        fprintf(stderr, "usage: %s <weights.bin> [-p PROMPT] [-n N] [-t T] [-k K] [--seed S]\n", argv[0]);
        return 1;
    }

    /* read prompt from stdin if not provided */
    static char stdin_buf[4096];
    if (!prompt) {
        size_t got = fread(stdin_buf, 1, sizeof(stdin_buf)-1, stdin);
        stdin_buf[got] = 0;
        /* trim trailing newline */
        while (got > 0 && (stdin_buf[got-1] == '\n' || stdin_buf[got-1] == '\r')) stdin_buf[--got] = 0;
        prompt = stdin_buf;
    }
    if (!prompt || !prompt[0]) {
        fprintf(stderr, "no prompt (use -p or pipe to stdin)\n");
        return 1;
    }

    fprintf(stderr, "[doe] weights: %s\n", weights_path);
    fprintf(stderr, "[doe] prompt:  %s\n", prompt);
    fprintf(stderr, "[doe] n=%d temp=%.2f top_k=%d seed=%u\n", max_new, temp, top_k, seed);

    nt_seed(seed);
    srand(seed);
    Model* model = model_new();
    if (load_weights(model, weights_path) != 0) {
        fprintf(stderr, "[doe] failed to load weights\n");
        model_free(model);
        return 1;
    }
    nt_train_mode(0);

    int ctx[CTX]; int gen_len = 0;
    for (int i = 0; prompt[i] && gen_len < CTX/2; i++) ctx[gen_len++] = (unsigned char)prompt[i];
    fputs(prompt, stdout); fflush(stdout);

    for (int s = 0; s < max_new; s++) {
        int next = sample_token(model, ctx, gen_len, temp, top_k);
        unsigned char c = (unsigned char)next;
        if (c >= 32 && c < 127) fputc(c, stdout);
        else if (c == '\n')     fputc('\n', stdout);
        else                     fputc('.', stdout);
        fflush(stdout);
        ctx[gen_len++] = next;
        if (gen_len >= CTX - 1) {
            /* shift left to keep room — char-level: drop oldest 32 chars */
            int shift = 32;
            memmove(ctx, ctx + shift, sizeof(int) * (gen_len - shift));
            gen_len -= shift;
        }
    }
    fputc('\n', stdout);

    model_free(model);
    return 0;
}
