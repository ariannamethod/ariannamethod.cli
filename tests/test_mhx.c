/* Unit tests for the metaharmonix REPL core.
 *
 * Compiled with -DMHX_NO_MAIN so we link the implementation without the
 * regular `main()` and provide our own here. */

#include "mhx.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        failures++;                                                 \
    }                                                               \
} while (0)

static void test_version_constant(void)
{
    CHECK(MHX_VERSION != NULL, "MHX_VERSION defined");
    CHECK(strlen(MHX_VERSION) > 0, "MHX_VERSION non-empty");
}

static void test_dispatch_empty_is_ok(void)
{
    CHECK(mhx_dispatch("") == MHX_OK, "empty line dispatches to MHX_OK");
    CHECK(mhx_dispatch("   ") == MHX_OK, "whitespace-only dispatches to MHX_OK");
    CHECK(mhx_dispatch(NULL) == MHX_OK, "NULL dispatches to MHX_OK");
}

static void test_exit_builtin(void)
{
    CHECK(mhx_dispatch("exit") == MHX_EXIT, "'exit' returns MHX_EXIT");
    CHECK(mhx_dispatch("quit") == MHX_EXIT, "'quit' returns MHX_EXIT");
}

static void test_install_requires_arg(void)
{
    /* `mhx install` with no language is a usage error. */
    mhx_status st = mhx_dispatch("mhx install");
    CHECK(st == MHX_BAD_ARGS, "'mhx install' without arg → MHX_BAD_ARGS");
}

static void test_install_unknown_lang(void)
{
    /* `mhx install zzzlang` is a usage error — caught by the whitelist
     * before we ever try to spawn a script. */
    CHECK(mhx_dispatch("mhx install zzzlang") == MHX_BAD_ARGS,
          "'mhx install <unknown-lang>' → MHX_BAD_ARGS");
}

static void test_train_requires_target(void)
{
    CHECK(mhx_dispatch("mhx train") == MHX_BAD_ARGS,
          "'mhx train' without target → MHX_BAD_ARGS");
}

static void test_train_unknown_target(void)
{
    CHECK(mhx_dispatch("mhx train nonsense") == MHX_BAD_ARGS,
          "'mhx train <unknown>' → MHX_BAD_ARGS");
}

static void test_slots_status_ok(void)
{
    CHECK(mhx_dispatch("mhx slots") == MHX_OK,
          "'mhx slots' → MHX_OK");
}

static void test_slots_unknown_name(void)
{
    CHECK(mhx_dispatch("mhx slots show no-such-slot") == MHX_BAD_ARGS,
          "'mhx slots show <unknown>' → MHX_BAD_ARGS");
}

static void test_slots_run_help_ok(void)
{
    CHECK(mhx_dispatch("mhx slots run nanollama --help") == MHX_OK,
          "'mhx slots run nanollama --help' → MHX_OK");
}

static void test_dispatch_argv_builtin_ok(void)
{
    char *argv[] = { (char *)"mhx", (char *)"info", NULL };
    CHECK(mhx_dispatch_argv(2, argv) == MHX_OK,
          "mhx_dispatch_argv runs builtin directly");
}

static void test_dispatch_argv_no_host_forward(void)
{
    char *argv[] = { (char *)"echo", (char *)"hello", NULL };
    CHECK(mhx_dispatch_argv(2, argv) == MHX_BAD_ARGS,
          "mhx_dispatch_argv rejects host forwarding");
}

static void test_info_returns_ok(void)
{
    /* 'mhx info' always succeeds — even on a host where some baked
     * binaries are missing, info is supposed to report the gap rather
     * than fail. */
    CHECK(mhx_dispatch("mhx info") == MHX_OK,
          "'mhx info' → MHX_OK");
}

static void test_mhx_version_runs(void)
{
    /* Just ensure dispatching mhx version returns MHX_OK; output goes
     * to stdout and is verified by the smoke test in the Makefile. */
    CHECK(mhx_dispatch("mhx version") == MHX_OK, "'mhx version' → MHX_OK");
}

static void test_unknown_verb(void)
{
    /* `mhx <unknown>` is a metaharmonix usage error, NOT forwarded to
     * the host shell — otherwise typos would silently run as host
     * commands. */
    CHECK(mhx_dispatch("mhx zzz-not-a-verb") == MHX_BAD_ARGS,
          "unknown mhx verb → MHX_BAD_ARGS (not forwarded to host)");
}

static void test_non_mhx_forwarded(void)
{
    /* A line that doesn't start with "mhx"/"exit"/"quit" is forwarded
     * to the host shell. `true` is portable and exits 0 → MHX_OK. */
    CHECK(mhx_dispatch("true") == MHX_OK,
          "non-builtin command forwarded to host");
}

static void test_nonzero_host_fails(void)
{
    CHECK(mhx_dispatch("false") == MHX_HOST_FAIL,
          "non-zero host exit propagates as MHX_HOST_FAIL");
}

static void test_aml_status_no_subverb(void)
{
    /* 'mhx aml' with no subverb prints status and returns MHX_OK
     * regardless of whether bake/aml/ has been built — the status line
     * itself reflects the build state. */
    CHECK(mhx_dispatch("mhx aml") == MHX_OK,
          "'mhx aml' (status mode) → MHX_OK");
}

static void test_notorch_status_no_subverb(void)
{
    CHECK(mhx_dispatch("mhx notorch") == MHX_OK,
          "'mhx notorch' (status mode) → MHX_OK");
}

static void test_aml_unknown_subverb(void)
{
    /* 'mhx aml <bad>' is a usage error, NOT forwarded. */
    CHECK(mhx_dispatch("mhx aml frobnicate") == MHX_BAD_ARGS,
          "'mhx aml <unknown>' → MHX_BAD_ARGS");
}

static void test_notorch_unknown_subverb(void)
{
    CHECK(mhx_dispatch("mhx notorch frobnicate") == MHX_BAD_ARGS,
          "'mhx notorch <unknown>' → MHX_BAD_ARGS");
}

static void test_resolve_baked(void)
{
    /* Resolution of a known baked binary — works only when the test is
     * run from the repo root with bake/ already built. We do a soft
     * check: if the binary exists, resolution must succeed; if it
     * doesn't, this test is a no-op so we don't break CI before bake
     * is built. */
    char path[1024];
    int rc = mhx_resolve_baked("aml/runner/aml", path, sizeof(path));
    if (rc == 0) {
        CHECK(strstr(path, "aml/runner/aml") != NULL,
              "resolved path contains the relative");
    }

    /* A path that definitely does not exist must return -1. */
    CHECK(mhx_resolve_baked("nope/nope/nope", path, sizeof(path)) == -1,
          "non-existent baked path returns -1");
}

int main(void)
{
    test_version_constant();
    test_dispatch_empty_is_ok();
    test_exit_builtin();
    test_install_requires_arg();
    test_install_unknown_lang();
    test_train_requires_target();
    test_train_unknown_target();
    test_slots_status_ok();
    test_slots_unknown_name();
    test_slots_run_help_ok();
    test_dispatch_argv_builtin_ok();
    test_dispatch_argv_no_host_forward();
    test_mhx_version_runs();
    test_unknown_verb();
    test_non_mhx_forwarded();
    test_nonzero_host_fails();
    test_aml_status_no_subverb();
    test_notorch_status_no_subverb();
    test_aml_unknown_subverb();
    test_notorch_unknown_subverb();
    test_info_returns_ok();
    test_resolve_baked();

    if (failures) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    printf("all unit tests passed\n");
    return 0;
}
