// test_janus_c.c — Janus C API test
// Build: cc test_janus_c.c -L. -ljanus -o test_janus_c
// Run: DYLD_LIBRARY_PATH=. ./test_janus_c [gguf_path] [delta_path]

#include <stdio.h>
#include <string.h>
#include "libjanus.h"

// Test logit callback — just prints that it was called
static int callback_count = 0;
void test_logit_callback(float* logits __attribute__((unused)), int n) {
    callback_count++;
    if (callback_count == 1) {
        printf("[callback] first call — %d logits\n", n);
    }
}

int main(int argc, char** argv) {
    printf("=== JANUS — First Transformer in AML ===\n");
    printf("\"Janus will grow like mycelium\"\n\n");

    // Init
    janus_init();

    // === Basic tests (no model) ===
    printf("--- Basic API tests ---\n");
    int loaded = janus_model_loaded();
    printf("model_loaded = %d (expect 0) %s\n", loaded, loaded == 0 ? "OK" : "FAIL");

    printf("vocab_size = %d (expect 0) %s\n",
        janus_get_vocab_size(), janus_get_vocab_size() == 0 ? "OK" : "FAIL");

    // Set auto-detect
    janus_set_alpha(-1.0f);

    // === Model loading test ===
    if (argc < 2) {
        printf("\n--- Skipping model test (no GGUF path) ---\n");
        printf("Usage: %s <gguf_path> [delta_path]\n", argv[0]);
        janus_shutdown();
        return 0;
    }

    printf("\n--- Loading model: %s ---\n", argv[1]);
    int rc = janus_load_model(argv[1]);
    if (rc != 0) {
        printf("FAIL: model load returned %d\n", rc);
        janus_shutdown();
        return 1;
    }

    loaded = janus_model_loaded();
    printf("model_loaded = %d (expect 1) %s\n", loaded, loaded == 1 ? "OK" : "FAIL");
    printf("vocab_size   = %d\n", janus_get_vocab_size());
    printf("embed_dim    = %d\n", janus_get_embed_dim());
    printf("num_layers   = %d\n", janus_get_num_layers());

    // Load delta if provided
    if (argc >= 3) {
        printf("\n--- Loading delta: %s ---\n", argv[2]);
        rc = janus_load_delta(argv[2]);
        printf("delta load = %d %s\n", rc, rc == 0 ? "OK" : "FAIL");
    }

    // Load gamma if provided
    if (argc >= 4) {
        printf("\n--- Loading gamma: %s ---\n", argv[3]);
        rc = janus_load_gamma(argv[3]);
        printf("gamma load = %d %s\n", rc, rc == 0 ? "OK" : "FAIL");
    }

    // Register logit callback
    janus_set_logit_callback(test_logit_callback);

    // === Generation test: English ===
    printf("\n--- Generate (English) ---\n");
    janus_set_alpha(0.0f);
    char* response = janus_generate("Who are you?", 50, 0.8f, 0.9f);
    if (response) {
        printf("Response: %s\n", response);
        janus_free_string(response);
    }

    // === Generation test: Russian (auto-detect) ===
    if (argc >= 3) {
        printf("\n--- Generate (Russian, auto-detect) ---\n");
        janus_set_alpha(-1.0f);  // auto
        response = janus_generate("\xd0\x9a\xd1\x82\xd0\xbe \xd1\x82\xd1\x8b?", 50, 0.8f, 0.9f);
        if (response) {
            printf("Response: %s\n", response);
            janus_free_string(response);
        }
    }

    printf("\n--- Logit callback was called %d times ---\n", callback_count);

    // Cleanup
    janus_unload_model();
    janus_shutdown();

    printf("\n=== JANUS TESTS COMPLETE ===\n");
    return 0;
}
