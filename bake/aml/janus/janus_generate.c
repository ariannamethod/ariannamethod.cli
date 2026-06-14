// janus_generate.c — Interactive text generation with Janus
// v4: LEFT-ALIGNED prompt, dynamic seq_len, NO chat template
//
// Usage: ./janus_generate <checkpoint> [--temp F] [--max-tokens N] [--top-k N]
//
// Build:
//   cc -O3 -I. -o janus_generate janus_generate.c core/ariannamethod.c -lm -lpthread

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "core/ariannamethod.h"
#include "janus/janus_tokenizer.h"

typedef struct {
    int   vocab_size, n_embd, hidden_dim, n_heads, n_layers, seq_len;
    float temperature;
    int   max_tokens, top_k;
    char  ckpt_path[512];
} GenConfig;

static int load_checkpoint(GenConfig* cfg, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    int hdr[8];
    if (fread(hdr, sizeof(int), 8, f) != 8 || hdr[0] != 0x4A414E55) {
        fprintf(stderr, "ERROR: bad checkpoint\n"); fclose(f); return -1;
    }
    cfg->vocab_size = hdr[1]; cfg->n_embd = hdr[2]; cfg->n_heads = hdr[3];
    cfg->n_layers = hdr[4]; cfg->seq_len = hdr[5];
    int step = hdr[6];
    cfg->hidden_dim = hdr[7];
    if (cfg->hidden_dim <= 0) cfg->hidden_dim = cfg->n_embd;
    
    printf("Checkpoint: step %d, V=%d D=%d hidden=%d H=%d L=%d seq=%d\n",
           step, cfg->vocab_size, cfg->n_embd, cfg->hidden_dim,
           cfg->n_heads, cfg->n_layers, cfg->seq_len);

    struct { const char* name; int rows; int cols; } g[] = {
        {"wte", cfg->vocab_size, cfg->n_embd},
        {"wpe", cfg->seq_len, cfg->n_embd},
        {"lm_head", cfg->vocab_size, cfg->n_embd},
    };
    for (int i = 0; i < 3; i++) {
        int len; fread(&len, sizeof(int), 1, f);
        float* buf = (float*)malloc(len * sizeof(float));
        fread(buf, sizeof(float), len, f);
        am_set_var_matrix(g[i].name, buf, g[i].rows, g[i].cols);
        free(buf);
    }
    for (int l = 0; l < cfg->n_layers; l++) {
        const char* attn_sfx[] = {"wq", "wk", "wv", "wo", NULL};
        for (int s = 0; attn_sfx[s]; s++) {
            char name[32]; snprintf(name, 32, "%s%d", attn_sfx[s], l);
            int len; fread(&len, sizeof(int), 1, f);
            float* buf = (float*)malloc(len * sizeof(float));
            fread(buf, sizeof(float), len, f);
            am_set_var_matrix(name, buf, cfg->n_embd, cfg->n_embd);
            free(buf);
        }
        {
            char name[32]; int len; float* buf;
            snprintf(name, 32, "w1_%d", l); fread(&len, sizeof(int), 1, f);
            buf = (float*)malloc(len * sizeof(float)); fread(buf, sizeof(float), len, f);
            am_set_var_matrix(name, buf, cfg->hidden_dim, cfg->n_embd); free(buf);
            snprintf(name, 32, "w3_%d", l); fread(&len, sizeof(int), 1, f);
            buf = (float*)malloc(len * sizeof(float)); fread(buf, sizeof(float), len, f);
            am_set_var_matrix(name, buf, cfg->hidden_dim, cfg->n_embd); free(buf);
            snprintf(name, 32, "w2_%d", l); fread(&len, sizeof(int), 1, f);
            buf = (float*)malloc(len * sizeof(float)); fread(buf, sizeof(float), len, f);
            am_set_var_matrix(name, buf, cfg->n_embd, cfg->hidden_dim); free(buf);
        }
    }
    fclose(f);
    return step;
}

static char* gen_forward(GenConfig* cfg) {
    int bs = cfg->n_layers * 512 + 512;
    char* s = (char*)malloc(bs);
    int n = 0;
    n += snprintf(s+n, bs-n, "h = seq_embed(wte, wpe, tokens, seq_len)\n");
    for (int l = 0; l < cfg->n_layers; l++) {
        n += snprintf(s+n, bs-n,
            "h_norm = seq_rmsnorm(h, seq_len, n_embd)\n"
            "q = seq_matvec(wq%d, h_norm, seq_len)\n"
            "k = seq_matvec(wk%d, h_norm, seq_len)\n"
            "v = seq_matvec(wv%d, h_norm, seq_len)\n"
            "attn_out = multi_head_attention(q, k, v, seq_len, n_embd, n_heads)\n"
            "attn_proj = seq_matvec(wo%d, attn_out, seq_len)\n"
            "h = add(h, attn_proj)\n"
            "h_norm = seq_rmsnorm(h, seq_len, n_embd)\n"
            "gate_pre = seq_matvec(w1_%d, h_norm, seq_len)\n"
            "gate = silu(gate_pre)\n"
            "up = seq_matvec(w3_%d, h_norm, seq_len)\n"
            "mlp_out = mul(gate, up)\n"
            "mlp_proj = seq_matvec(w2_%d, mlp_out, seq_len)\n"
            "h = add(h, mlp_proj)\n", l,l,l,l,l,l,l);
    }
    n += snprintf(s+n, bs-n,
        "h_norm = seq_rmsnorm(h, seq_len, n_embd)\n"
        "logits = seq_matvec(lm_head, h_norm, seq_len)\n");
    return s;
}

static int sample_topk(const float* logits, int V, float temp, int K) {
    typedef struct { float v; int i; } P;
    P* p = (P*)malloc(V * sizeof(P));
    for (int i = 0; i < V; i++) { p[i].v = logits[i]; p[i].i = i; }
    if (K > V) K = V;
    for (int i = 0; i < K; i++) {
        int b = i;
        for (int j = i+1; j < V; j++) if (p[j].v > p[b].v) b = j;
        if (b != i) { P t = p[i]; p[i] = p[b]; p[b] = t; }
    }
    float mx = p[0].v, sum = 0;
    for (int i = 0; i < K; i++) { p[i].v = expf((p[i].v - mx) / temp); sum += p[i].v; }
    float r = (float)rand() / (float)RAND_MAX * sum, c = 0;
    int tok = p[0].i;
    for (int i = 0; i < K; i++) { c += p[i].v; if (r <= c) { tok = p[i].i; break; } }
    free(p);
    return tok;
}

static void generate(GenConfig* cfg, EvolvingTokenizer* tok, const char* fwd,
                     const char* prompt_text, float temp, int max_tok, int topk) {
    int V = cfg->vocab_size;
    int S = cfg->seq_len;
    float* tokens = (float*)calloc(S, sizeof(float));

    int prompt_ids[8192];
    int prompt_len = tok_encode_raw(tok, (const unsigned char*)prompt_text,
                                     strlen(prompt_text), prompt_ids, 8192);

    // LEFT-ALIGN: prompt starts at position 0
    int ctx = prompt_len < S ? prompt_len : S;
    for (int i = 0; i < ctx; i++)
        tokens[i] = (float)prompt_ids[prompt_len - ctx + i];
    int cur_len = ctx;

    for (int t = 0; t < max_tok; t++) {
        // Dynamic seq_len = only real tokens
        char cmd[64]; snprintf(cmd, 64, "seq_len = %d", cur_len);
        am_exec(cmd);

        am_set_var_array("tokens", tokens, cur_len);
        if (am_exec(fwd) != 0) { fprintf(stderr, "\n[AML error: %s]\n", am_get_error()); break; }

        int llen = 0;
        const float* all = am_get_var_array("logits", &llen);
        if (!all || llen < V) break;

        // Sample from LAST real position
        int next = sample_topk(all + (cur_len-1)*V, V, temp, topk);

        unsigned char out[64];
        int out_len = tok_decode_token(tok, next, out, 64);
        for (int i = 0; i < out_len; i++) putchar(out[i]);
        fflush(stdout);

        // Append or slide
        if (cur_len < S) {
            tokens[cur_len] = (float)next;
            cur_len++;
        } else {
            for (int i = 0; i < S-1; i++) tokens[i] = tokens[i+1];
            tokens[S-1] = (float)next;
        }
    }
    printf("\n");
    free(tokens);
}

int main(int argc, char** argv) {
    GenConfig cfg = {0};
    cfg.temperature = 0.8f; cfg.max_tokens = 512; cfg.top_k = 40;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') snprintf(cfg.ckpt_path, 512, "%s", argv[i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc) cfg.temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-tokens") && i+1 < argc) cfg.max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i+1 < argc) cfg.top_k = atoi(argv[++i]);
    }
    if (!cfg.ckpt_path[0]) { fprintf(stderr, "Usage: janus_generate <checkpoint> [options]\n"); return 1; }

    srand(time(NULL));
    am_init();
    am_persistent_mode(1);

    int step = load_checkpoint(&cfg, cfg.ckpt_path);
    if (step < 0) return 1;

    { char h[256]; snprintf(h, 256, "n_embd=%d\nn_heads=%d\nseq_len=%d\nvocab_size=%d\n",
          cfg.n_embd, cfg.n_heads, cfg.seq_len, cfg.vocab_size); am_exec(h); }

    EvolvingTokenizer tok;
    tok_init(&tok, 0, 0, 0);
    {
        char tpath[512];
        snprintf(tpath, 512, "%s", cfg.ckpt_path);
        char* p = strstr(tpath, "ckpt_step");
        if (p) { memmove(p + 3, p + 4, strlen(p + 4) + 1); memcpy(p, "tok", 3); }
        if (tok_load(&tok, tpath) == 0) {
            printf("Tokenizer: vocab=%d, merges=%d, bpe=%s\n",
                   tok.vocab_size, tok.n_merges, tok.bpe_enabled ? "ON" : "OFF");
        } else {
            printf("Tokenizer: byte-level (no tokenizer file)\n");
        }
    }

    char* fwd = gen_forward(&cfg);

    long total = (long)cfg.vocab_size*cfg.n_embd*2 + (long)cfg.seq_len*cfg.n_embd +
                 (long)cfg.n_layers*4*cfg.n_embd*cfg.n_embd +
                 (long)cfg.n_layers*(2*cfg.hidden_dim*cfg.n_embd + cfg.n_embd*cfg.hidden_dim);
    printf("═══════════════════════════════════════════════════\n");
    printf("  JANUS — %.2fM params, vocab=%d (%s)\n",
           (double)total / 1e6,
           cfg.vocab_size, tok.bpe_enabled ? "BPE" : "byte-level");
    printf("  D=%d hidden=%d (%.1fx) heads=%d layers=%d\n",
           cfg.n_embd, cfg.hidden_dim, (float)cfg.hidden_dim/cfg.n_embd,
           cfg.n_heads, cfg.n_layers);
    printf("  temp=%.2f, top_k=%d, max_tokens=%d\n", cfg.temperature, cfg.top_k, cfg.max_tokens);
    printf("  Type text to continue. Empty line = quit.\n");
    printf("═══════════════════════════════════════════════════\n\n");

    char input[4096];
    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = strlen(input);
        while (len > 0 && (input[len-1]=='\n' || input[len-1]=='\r')) input[--len] = 0;
        if (len == 0) break;

        // Raw text completion — NO chat template
        generate(&cfg, &tok, fwd, input, cfg.temperature, cfg.max_tokens, cfg.top_k);
    }

    free(fwd);
    printf("[bye]\n");
    return 0;
}
