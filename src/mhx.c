/* metaharmonix terminal core.
 *
 * Minimal REPL host: reads a line, splits into argv, runs builtins
 * directly or forwards to the host shell. Heart (dario), AML and notorch
 * builtins are stubbed out for now and answer with their slot status —
 * they will be wired in subsequent passes. */

#include "mhx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>      /* fork, execv, access, sysconf */
#include <sys/types.h>
#include <sys/wait.h>    /* waitpid */
#include <sys/utsname.h> /* uname */
#include <sys/statvfs.h> /* statvfs */
#include <errno.h>
#include <limits.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>  /* sysctlbyname for hw.memsize */
#endif

static char g_mhx_root[PATH_MAX];

typedef struct {
    char name[64];
    char label[128];
    char kind[32];
    char state[32];
    char target[PATH_MAX];
    char notes[256];
} mhx_slot;

static void mhx_init_root(const char *argv0)
{
    char resolved[PATH_MAX];
    char cwd[PATH_MAX];
    const char *from_env = getenv("MHX_ROOT");
    char *slash;

    g_mhx_root[0] = '\0';
    if (from_env && *from_env) {
        snprintf(g_mhx_root, sizeof(g_mhx_root), "%s", from_env);
        return;
    }
    if (argv0 && strchr(argv0, '/')) {
        if (realpath(argv0, resolved)) {
            slash = strrchr(resolved, '/');
            if (slash) {
                *slash = '\0';
                snprintf(g_mhx_root, sizeof(g_mhx_root), "%s", resolved);
                return;
            }
        }
    }
    if (getcwd(cwd, sizeof(cwd))) {
        snprintf(g_mhx_root, sizeof(g_mhx_root), "%s", cwd);
    }
}

static int mhx_path_join(char *out, size_t size, const char *root, const char *rel)
{
    int n;
    if (!out || size == 0 || !root || !rel) return -1;
    n = snprintf(out, size, "%s/%s", root, rel);
    if (n <= 0 || (size_t)n >= size) return -1;
    return 0;
}

static void mhx_chomp(char *s)
{
    size_t n;
    if (!s) return;
    n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) {
        s[--n] = '\0';
    }
}

static const char *mhx_skip_ws(const char *s)
{
    if (!s) return "";
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

static int mhx_slots_manifest_path(char *out, size_t size)
{
    return mhx_path_join(out, size,
                         g_mhx_root[0] ? g_mhx_root : ".",
                         "runtime/slots.tsv");
}

static int mhx_slot_resolve_target(const mhx_slot *slot, char *out, size_t size)
{
    if (!slot || !out || size == 0) return -1;
    if (strcmp(slot->kind, "baked") == 0) {
        return mhx_resolve_baked(slot->target, out, size);
    }
    if (strcmp(slot->kind, "repo") == 0) {
        if (mhx_path_join(out, size, g_mhx_root[0] ? g_mhx_root : ".", slot->target) != 0)
            return -1;
        return access(out, X_OK) == 0 ? 0 : -1;
    }
    return -1;
}

static int mhx_load_slots(mhx_slot *slots, int max_slots)
{
    char manifest[PATH_MAX];
    FILE *fp;
    char line[2048];
    int count = 0;
    if (!slots || max_slots <= 0) return -1;
    if (mhx_slots_manifest_path(manifest, sizeof(manifest)) != 0) return -1;
    fp = fopen(manifest, "r");
    if (!fp) return -1;
    while (fgets(line, sizeof(line), fp) && count < max_slots) {
        char *fields[6];
        char *p = line;
        int i = 0;
        mhx_chomp(line);
        p = (char *)mhx_skip_ws(p);
        if (!*p || *p == '#') continue;
        while (i < 6) {
            fields[i++] = p;
            p = strchr(p, '\t');
            if (!p) break;
            *p++ = '\0';
        }
        if (i < 6) continue;
        if (strcmp(fields[0], "slot") == 0) continue;
        snprintf(slots[count].name, sizeof(slots[count].name), "%s", fields[0]);
        snprintf(slots[count].label, sizeof(slots[count].label), "%s", fields[1]);
        snprintf(slots[count].kind, sizeof(slots[count].kind), "%s", fields[2]);
        snprintf(slots[count].state, sizeof(slots[count].state), "%s", fields[3]);
        snprintf(slots[count].target, sizeof(slots[count].target), "%s", fields[4]);
        snprintf(slots[count].notes, sizeof(slots[count].notes), "%s", fields[5]);
        count++;
    }
    fclose(fp);
    return count;
}

static mhx_slot *mhx_find_slot(mhx_slot *slots, int nslots, const char *name)
{
    for (int i = 0; i < nslots; i++) {
        if (strcmp(slots[i].name, name) == 0) return &slots[i];
    }
    return NULL;
}

/* ---- baked binary resolver + spawn ------------------------------------ */

int mhx_resolve_baked(const char *rel, char *out, size_t size)
{
    if (!rel || !out || size == 0) return -1;

    const char *prefix = getenv("MHX_BAKE_PATH");
    int n;
    if (prefix && *prefix) {
        n = snprintf(out, size, "%s/%s", prefix, rel);
        if (n > 0 && (size_t)n < size && access(out, X_OK) == 0) return 0;
    }
    /* Development layout: bake/ holds the runtimes (AML, notorch),
     * vendor/ holds organisms (the dario heart). Both are resolved by
     * relative path from cwd. */
    n = snprintf(out, size, "%s/bake/%s", g_mhx_root[0] ? g_mhx_root : ".", rel);
    if (n > 0 && (size_t)n < size && access(out, X_OK) == 0) return 0;
    n = snprintf(out, size, "%s/vendor/%s", g_mhx_root[0] ? g_mhx_root : ".", rel);
    if (n > 0 && (size_t)n < size && access(out, X_OK) == 0) return 0;

    return -1;
}

int mhx_spawn_baked(const char *rel, char *const argv[])
{
    char path[1024];
    if (mhx_resolve_baked(rel, path, sizeof(path)) != 0) {
        fprintf(stderr,
                "mhx: baked binary '%s' not found — "
                "run 'make bake' from the metaharmonix repo, or set "
                "MHX_BAKE_PATH to its install prefix.\n", rel);
        return -1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "mhx: fork failed: %s\n", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        /* Child. argv[0] is conventionally the program name; we override
         * with the resolved path so execv finds the actual file. */
        execv(path, argv);
        fprintf(stderr, "mhx: execv(%s) failed: %s\n", path, strerror(errno));
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "mhx: waitpid failed: %s\n", strerror(errno));
        return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

static int mhx_spawn_path(const char *path, char *const argv[])
{
    pid_t pid;
    int status = 0;
    if (!path || !argv) return -1;
    pid = fork();
    if (pid < 0) {
        fprintf(stderr, "mhx: fork failed: %s\n", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        execv(path, argv);
        fprintf(stderr, "mhx: execv(%s) failed: %s\n", path, strerror(errno));
        _exit(127);
    }
    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "mhx: waitpid failed: %s\n", strerror(errno));
        return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

/* ---- input ------------------------------------------------------------- */

int mhx_readline(char *buf, size_t size)
{
    if (!buf || size == 0) return -1;
    if (!fgets(buf, (int)size, stdin)) return -1;
    size_t n = strlen(buf);
    if (n > 0 && buf[n - 1] == '\n') { buf[n - 1] = '\0'; n--; }
    return (int)n;
}

/* ---- argv split (whitespace, no quoting yet) --------------------------- */

#define MHX_MAX_ARGV 64

static int split_argv(char *line, char **argv)
{
    int argc = 0;
    char *p = line;
    while (*p && argc < MHX_MAX_ARGV - 1) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        argv[argc++] = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        if (*p) { *p = '\0'; p++; }
    }
    argv[argc] = NULL;
    return argc;
}

/* ---- builtins ---------------------------------------------------------- */

/* Forward declaration: spawn_with_passthrough is defined further down,
 * but builtin_heart calls it for argv pass-through. */
static mhx_status spawn_with_passthrough(const char *rel, const char *prog_name,
                                         int argc, char **argv,
                                         int forward_from);

static mhx_status builtin_help(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf(
        "metaharmonix " MHX_VERSION " — Arianna Method terminal\n"
        "\n"
        "builtins:\n"
        "  mhx help                       show this message\n"
        "  mhx version                    print version\n"
        "  mhx info                       host introspection: CPU, RAM,\n"
        "                                 disk, BLAS, baked binaries\n"
        "  mhx heart                      enter the dario heart REPL\n"
        "                                 (formula + SARTRE chambers)\n"
        "  mhx aml                        baked AML status\n"
        "  mhx aml run <file.aml>         execute an AML program\n"
        "  mhx aml compile <file.aml>     compile via amlc\n"
        "  mhx aml version                baked AML runner version\n"
        "  mhx notorch                    baked notorch status + BLAS\n"
        "  mhx notorch test               run vendored notorch tests\n"
        "  mhx install <lang>             install clang|python|go|rust|lua\n"
        "                                 via host pkg manager\n"
        "  mhx train nanollama <tokens>   run examples/nanollama trainer\n"
        "                                 with any extra nanollama flags\n"
        "  mhx slots                      list orchestrated slots/models\n"
        "  mhx slots show <slot>          inspect one slot target/state\n"
        "  mhx slots run <slot> [args...] run a wired slot target\n"
        "  mhx mesh                       cross-node messaging status\n"
        "  mhx mesh peers                 list discovered peers\n"
        "  mhx mesh status [PEER]         /presence of local agent or PEER\n"
        "  mhx mesh send <peer|@all|@TAG> TEXT  unicast/broadcast/multicast TEXT\n"
        "  mhx mesh listen [flags]        tail ~/.mesh/inbox chronologically\n"
        "                                 flags: --since DURATION, --watch\n"
        "  mhx mesh ask <peer|@all|@TAG> TEXT  send, then collect replies\n"
        "  mhx mesh tags <list|show|add|rm|del>  manage tag selectors\n"
        "  mhx mesh inbox <count|show|dump|clear|archive>  inbox admin\n"
        "  mhx mesh selfcheck                local readiness check\n"
        "                                 flag: --timeout DURATION (default 60s)\n"
        "  exit | quit                    leave the REPL\n"
        "\n"
        "anything else is forwarded to the host shell.\n");
    return MHX_OK;
}

static mhx_status builtin_version(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("metaharmonix %s\n", MHX_VERSION);
    return MHX_OK;
}

static mhx_status builtin_heart(int argc, char **argv)
{
    /* 'mhx heart' (no subverb) reports status. With any extra args, we
     * forward them to the dario binary — for example 'mhx heart --help'
     * or future 'mhx heart --kernel'. */
    char path[1024];
    int built = mhx_resolve_baked("dario/dario", path, sizeof(path)) == 0;
    if (argc < 3) {
        if (!built) {
            printf("heart: vendor/dario/  (not built — run "
                   "'make bake-dario' to compile dario.c + sartre_kernel "
                   "into the heart binary)\n");
            return MHX_OK;
        }
        /* Built — drop the user into the dario REPL. dario inherits our
         * stdin/stdout, prints its own banner, and exits when the user
         * types '/quit' or sends EOF. */
        char *cargv[] = { (char *)"dario", NULL };
        int rc = mhx_spawn_baked("dario/dario", cargv);
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }
    /* Pass-through args (e.g. mhx heart --version when dario gains one). */
    return spawn_with_passthrough("dario/dario", "dario", argc, argv, 2);
}

/* Helper: dispatch to a baked binary, passing through the trailing argv.
 * argv comes from "mhx <verb> <subverb> [args...]" — we forward
 * argv[2..argc-1] as argv[1..] of the baked binary. */
static mhx_status spawn_with_passthrough(const char *rel, const char *prog_name,
                                         int argc, char **argv,
                                         int forward_from)
{
    char *child_argv[MHX_MAX_ARGV + 1];
    int j = 0;
    child_argv[j++] = (char *)prog_name;
    for (int i = forward_from; i < argc && j < MHX_MAX_ARGV; i++)
        child_argv[j++] = argv[i];
    child_argv[j] = NULL;

    int rc = mhx_spawn_baked(rel, child_argv);
    if (rc < 0) return MHX_HOST_FAIL;
    if (rc != 0) return MHX_BAD_ARGS;
    return MHX_OK;
}

static mhx_status builtin_aml(int argc, char **argv)
{
    if (argc < 3) {
        /* Status mode: "mhx aml" with no subverb. */
        char path[1024];
        int has_runner = mhx_resolve_baked("aml/runner/aml", path, sizeof(path)) == 0;
        int has_amlc   = mhx_resolve_baked("aml/tools/amlc", path, sizeof(path)) == 0;
        printf("aml: vendored at bake/aml/  (runner: %s, amlc: %s)\n"
               "  mhx aml run <file.aml>      execute via baked runner\n"
               "  mhx aml compile <file.aml>  compile via baked amlc\n"
               "  mhx aml version             baked runner version\n",
               has_runner ? "built" : "not built — run 'make bake-aml'",
               has_amlc   ? "built" : "not built");
        return MHX_OK;
    }
    const char *sub = argv[2];
    if (strcmp(sub, "run") == 0)
        return spawn_with_passthrough("aml/runner/aml", "aml", argc, argv, 3);
    if (strcmp(sub, "compile") == 0)
        return spawn_with_passthrough("aml/tools/amlc", "amlc", argc, argv, 3);
    if (strcmp(sub, "version") == 0) {
        char *cargv[] = { (char *)"aml", (char *)"--version", NULL };
        int rc = mhx_spawn_baked("aml/runner/aml", cargv);
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }
    fprintf(stderr, "mhx aml: unknown subverb '%s' (run|compile|version)\n", sub);
    return MHX_BAD_ARGS;
}

static mhx_status builtin_notorch(int argc, char **argv)
{
    if (argc < 3) {
        /* Status mode: report what's baked and which BLAS would be used. */
        char lib_path[PATH_MAX];
        char src_path[PATH_MAX];
        int has_lib = 0;
        if (mhx_path_join(lib_path, sizeof(lib_path), g_mhx_root[0] ? g_mhx_root : ".", "bake/notorch/libnotorch.a") == 0 &&
            access(lib_path, R_OK) == 0) has_lib = 1;
        if (!has_lib &&
            mhx_path_join(src_path, sizeof(src_path), g_mhx_root[0] ? g_mhx_root : ".", "bake/notorch/notorch.c") == 0 &&
            access(src_path, R_OK) == 0) has_lib = 1;
        const char *blas;
#if defined(__APPLE__)
        blas = "Accelerate (compile-time on macOS)";
#elif defined(__linux__)
        blas = "OpenBLAS via pkg-config (compile-time on Linux/Termux)";
#else
        blas = "scalar fallback";
#endif
        printf("notorch: vendored at bake/notorch/  (source: %s)\n"
               "  BLAS at build:  %s\n"
               "  mhx notorch test    run vendored test_notorch suite\n"
               "  mhx notorch info    print this status\n",
               has_lib ? "present" : "missing — re-run vendoring",
               blas);
        return MHX_OK;
    }
    const char *sub = argv[2];
    if (strcmp(sub, "info") == 0) {
        char *cargv[] = { (char *)"mhx", (char *)"notorch", NULL };
        return builtin_notorch(2, cargv);
    }
    if (strcmp(sub, "test") == 0) {
        /* notorch test is built by 'make -C bake/notorch' (default
         * target = notorch_test), then we exec it. */
        char *cargv[] = { (char *)"notorch_test", NULL };
        int rc = mhx_spawn_baked("notorch/notorch_test", cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx notorch test: build it first with "
                "'make -C bake/notorch'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_BAD_ARGS;
    }
    fprintf(stderr, "mhx notorch: unknown subverb '%s' (info|test)\n", sub);
    return MHX_BAD_ARGS;
}

/* ---- host introspection (mhx info) ---------------------------------- */

/* Total physical RAM in bytes. Returns 0 if it can't be determined. */
static unsigned long long mhx_total_ram_bytes(void)
{
#if defined(__APPLE__)
    int64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0 && mem > 0)
        return (unsigned long long)mem;
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return (unsigned long long)pages * (unsigned long long)page_size;
    return 0;
#endif
}

/* Available RAM in bytes — best-effort. macOS lacks _SC_AVPHYS_PAGES, so
 * we report 0 there and let the user infer from total. */
static unsigned long long mhx_free_ram_bytes(void)
{
#if defined(_SC_AVPHYS_PAGES)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return (unsigned long long)pages * (unsigned long long)page_size;
#endif
    return 0;
}

/* Free disk bytes on '/'. Returns 0 if statvfs fails. */
static unsigned long long mhx_disk_free_bytes(void)
{
    struct statvfs s;
    if (statvfs("/", &s) != 0) return 0;
    return (unsigned long long)s.f_bavail * (unsigned long long)s.f_frsize;
}

static void mhx_print_human(unsigned long long bytes, char *out, size_t size)
{
    if (bytes == 0) { snprintf(out, size, "n/a"); return; }
    static const char *units[] = { "B", "KB", "MB", "GB", "TB" };
    int u = 0;
    double v = (double)bytes;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; u++; }
    snprintf(out, size, "%.1f %s", v, units[u]);
}

static mhx_status builtin_info(int argc, char **argv)
{
    (void)argc; (void)argv;

    /* Compile-time BLAS branch — same logic as builtin_notorch's
     * status mode but reported as one line for scripting. */
    const char *blas;
#if defined(__APPLE__)
    blas = "Accelerate";
#elif defined(__linux__)
    blas = "OpenBLAS (pkg-config at build time)";
#else
    blas = "scalar fallback";
#endif

    /* Baked binary presence — answers the operational question 'do I
     * need to run make bake before using AML / notorch / heart?'. */
    char path[1024];
    char lib_path[PATH_MAX];
    int has_aml_runner = mhx_resolve_baked("aml/runner/aml", path, sizeof(path)) == 0;
    int has_aml_amlc   = mhx_resolve_baked("aml/tools/amlc", path, sizeof(path)) == 0;
    int has_notorch    = mhx_path_join(lib_path, sizeof(lib_path), g_mhx_root[0] ? g_mhx_root : ".", "bake/notorch/libnotorch.a") == 0
                      && access(lib_path, R_OK) == 0;
    int has_heart      = mhx_resolve_baked("dario/dario", path, sizeof(path)) == 0;

    struct utsname u;
    if (uname(&u) != 0) memset(&u, 0, sizeof(u));

    long cores = sysconf(_SC_NPROCESSORS_ONLN);

    char ram_total[32], ram_free[32], disk_free[32];
    mhx_print_human(mhx_total_ram_bytes(), ram_total, sizeof(ram_total));
    mhx_print_human(mhx_free_ram_bytes(),  ram_free,  sizeof(ram_free));
    mhx_print_human(mhx_disk_free_bytes(), disk_free, sizeof(disk_free));

    printf("metaharmonix %s on %s %s (%s)\n",
           MHX_VERSION,
           u.sysname[0] ? u.sysname : "?",
           u.release[0] ? u.release : "?",
           u.machine[0] ? u.machine : "?");
    printf("  CPU cores:    %ld\n", cores > 0 ? cores : 0L);
    printf("  RAM total:    %s\n", ram_total);
    printf("  RAM free:     %s\n", ram_free);
    printf("  Disk free /:  %s\n", disk_free);
    printf("  BLAS:         %s\n", blas);
    printf("  Baked:        aml-runner=%s  amlc=%s  notorch=%s  heart=%s\n",
           has_aml_runner ? "yes" : "no",
           has_aml_amlc   ? "yes" : "no",
           has_notorch    ? "yes" : "no",
           has_heart      ? "yes" : "no");
    return MHX_OK;
}

static mhx_status builtin_install(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: mhx install <clang|python|go|rust|lua>\n");
        return MHX_BAD_ARGS;
    }
    const char *lang = argv[2];
    /* Whitelist: every wrapper that exists in install/. New languages
     * get appended here AND get an install/<lang>.sh — the C side and
     * the script set must move together so 'mhx install <typo>' fails
     * loudly instead of silently exec'ing a missing script. */
    static const char *known[] = {
        "clang", "python", "go", "rust", "lua", NULL
    };
    int ok = 0;
    for (int i = 0; known[i]; i++) {
        if (strcmp(lang, known[i]) == 0) { ok = 1; break; }
    }
    if (!ok) {
        fprintf(stderr, "mhx install: unknown lang '%s' "
                "(known: clang, python, go, rust, lua)\n", lang);
        return MHX_BAD_ARGS;
    }
    char script[256];
    int n = snprintf(script, sizeof(script), "%s/install/%s.sh", g_mhx_root[0] ? g_mhx_root : ".", lang);
    if (n <= 0 || (size_t)n >= sizeof(script) || access(script, X_OK) != 0) {
        fprintf(stderr, "mhx install: %s not executable — "
                "run 'chmod +x install/*.sh' or re-clone\n", script);
        return MHX_HOST_FAIL;
    }
    char *cargv[] = { (char *)"sh", script, NULL };
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "mhx install: fork failed: %s\n", strerror(errno));
        return MHX_HOST_FAIL;
    }
    if (pid == 0) {
        execvp("sh", cargv);
        fprintf(stderr, "mhx install: execvp failed: %s\n", strerror(errno));
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return MHX_OK;
    return MHX_BAD_ARGS;
}

static mhx_status builtin_train(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: mhx train <target> [args...]\n");
        return MHX_BAD_ARGS;
    }
    if (strcmp(argv[2], "nanollama") == 0) {
        char makefile_path[PATH_MAX];
        char bin_path[PATH_MAX];
        char cwd[PATH_MAX];
        char *child_argv[MHX_MAX_ARGV + 1];
        int j = 0;
        int rc;
        if (argc < 4) {
            fprintf(stderr, "usage: mhx train nanollama <tokens.bin> [extra args...]\n");
            return MHX_BAD_ARGS;
        }
        if (strcmp(argv[3], "--help") == 0 || strcmp(argv[3], "-h") == 0) {
            puts("mhx train nanollama <tokens.bin> [extra args...]");
            puts("  builds examples/nanollama on demand, then runs the vendored");
            puts("  trainer with the remaining args.");
            puts("");
            puts("examples:");
            puts("  mhx train nanollama ./tokens.bin --steps 200");
            puts("  mhx train nanollama ./tokens.bin --resume ckpt.bin --steps 1000");
            return MHX_OK;
        }
        if (mhx_path_join(makefile_path, sizeof(makefile_path),
                          g_mhx_root[0] ? g_mhx_root : ".",
                          "examples/nanollama/Makefile") != 0 ||
            mhx_path_join(bin_path, sizeof(bin_path),
                          g_mhx_root[0] ? g_mhx_root : ".",
                          "examples/nanollama/nanollama") != 0) {
            return MHX_HOST_FAIL;
        }
        if (access(bin_path, X_OK) != 0) {
            if (access(makefile_path, R_OK) != 0) {
                fprintf(stderr, "mhx train nanollama: missing example build files\n");
                return MHX_HOST_FAIL;
            }
            if (!getcwd(cwd, sizeof(cwd))) return MHX_HOST_FAIL;
            if (chdir(g_mhx_root[0] ? g_mhx_root : ".") != 0) {
                fprintf(stderr, "mhx train nanollama: chdir failed: %s\n", strerror(errno));
                return MHX_HOST_FAIL;
            }
            rc = system("make -C examples/nanollama");
            (void)chdir(cwd);
            if (!(WIFEXITED(rc) && WEXITSTATUS(rc) == 0)) {
                fprintf(stderr, "mhx train nanollama: build failed\n");
                return MHX_HOST_FAIL;
            }
        }
        child_argv[j++] = (char *)"nanollama";
        for (int i = 3; i < argc && j < MHX_MAX_ARGV; i++) {
            child_argv[j++] = argv[i];
        }
        child_argv[j] = NULL;
        rc = mhx_spawn_path(bin_path, child_argv);
        if (rc < 0) return MHX_HOST_FAIL;
        if (rc != 0) return MHX_BAD_ARGS;
        return MHX_OK;
    }
    fprintf(stderr, "mhx train: unknown target '%s' (known: nanollama)\n", argv[2]);
    return MHX_BAD_ARGS;
}

static mhx_status builtin_slots(int argc, char **argv)
{
    mhx_slot slots[32];
    char resolved[PATH_MAX];
    int nslots = mhx_load_slots(slots, 32);
    if (nslots < 0) {
        fprintf(stderr, "mhx slots: could not load runtime/slots.tsv\n");
        return MHX_HOST_FAIL;
    }
    if (argc < 3 || strcmp(argv[2], "list") == 0) {
        printf("metaharmonix slots (%d)\n", nslots);
        for (int i = 0; i < nslots; i++) {
            const char *resolved_state =
                mhx_slot_resolve_target(&slots[i], resolved, sizeof(resolved)) == 0
                ? "wired" : "missing";
            printf("  %-12s %-18s kind=%-6s state=%-9s target=%s\n",
                   slots[i].name, slots[i].label, slots[i].kind,
                   resolved_state, slots[i].target);
        }
        return MHX_OK;
    }
    if (strcmp(argv[2], "show") == 0) {
        mhx_slot *slot;
        if (argc < 4) {
            fprintf(stderr, "usage: mhx slots show <slot>\n");
            return MHX_BAD_ARGS;
        }
        slot = mhx_find_slot(slots, nslots, argv[3]);
        if (!slot) {
            fprintf(stderr, "mhx slots: unknown slot '%s'\n", argv[3]);
            return MHX_BAD_ARGS;
        }
        printf("slot:   %s\n", slot->name);
        printf("label:  %s\n", slot->label);
        printf("kind:   %s\n", slot->kind);
        printf("state:  %s\n", slot->state);
        printf("target: %s\n", slot->target);
        printf("notes:  %s\n", slot->notes);
        if (strcmp(slot->name, "nanollama") == 0) {
            printf("hint:   mhx train nanollama <tokens.bin> [args...]\n");
        }
        if (mhx_slot_resolve_target(slot, resolved, sizeof(resolved)) == 0) {
            printf("wired:  yes\n");
            printf("path:   %s\n", resolved);
        } else {
            printf("wired:  no\n");
        }
        return MHX_OK;
    }
    if (strcmp(argv[2], "run") == 0) {
        mhx_slot *slot;
        char *child_argv[MHX_MAX_ARGV + 1];
        int j = 0;
        int rc;
        if (argc < 4) {
            fprintf(stderr, "usage: mhx slots run <slot> [args...]\n");
            return MHX_BAD_ARGS;
        }
        slot = mhx_find_slot(slots, nslots, argv[3]);
        if (!slot) {
            fprintf(stderr, "mhx slots: unknown slot '%s'\n", argv[3]);
            return MHX_BAD_ARGS;
        }
        /* `--help` is a pass-through to the slot's wrapper help — it must
         * succeed regardless of whether the slot's target binary is built
         * yet (a fresh clone has slots.tsv but not examples/.../<bin>).
         * Handle help before target resolution. Without this order, the
         * unit suite fails on first checkout because resolve returns
         * HOST_FAIL before the bypass below ever runs. */
        if (strcmp(slot->name, "nanollama") == 0 && argc >= 5 &&
            (strcmp(argv[4], "--help") == 0 || strcmp(argv[4], "-h") == 0)) {
            char *train_help_argv[] = {
                (char *)"mhx", (char *)"train", (char *)"nanollama", (char *)"--help", NULL
            };
            return builtin_train(4, train_help_argv);
        }
        if (mhx_slot_resolve_target(slot, resolved, sizeof(resolved)) != 0) {
            fprintf(stderr, "mhx slots run: slot '%s' target is not wired yet\n", slot->name);
            return MHX_HOST_FAIL;
        }
        child_argv[j++] = slot->name;
        for (int i = 4; i < argc && j < MHX_MAX_ARGV; i++) {
            child_argv[j++] = argv[i];
        }
        child_argv[j] = NULL;
        rc = mhx_spawn_path(resolved, child_argv);
        if (rc < 0) return MHX_HOST_FAIL;
        if (rc != 0) return MHX_BAD_ARGS;
        return MHX_OK;
    }
    fprintf(stderr, "mhx slots: unknown subverb '%s' (list|show|run)\n", argv[2]);
    return MHX_BAD_ARGS;
}

static mhx_status builtin_mesh(int argc, char **argv)
{
    if (argc < 3) {
        char path[1024];
        int has_bin = mhx_resolve_baked("mesh-agent/mesh-agent",
                                        path, sizeof(path)) == 0;
        printf("mesh: vendored at bake/mesh-agent/  (binary: %s)\n"
               "  mhx mesh peers                       list discovered peers + slot counts\n"
               "  mhx mesh status [PEER]               /presence of local agent (or PEER)\n"
               "  mhx mesh send <peer> TEXT...         POST /msg to one peer\n"
               "  mhx mesh send @all TEXT...           broadcast to every peer except self\n"
               "  mhx mesh send @TAG TEXT...           broadcast to peers in tag (see tags)\n"
               "  mhx mesh listen [--since D] [--watch] tail ~/.mesh/inbox chronologically\n"
               "  mhx mesh ask <peer|@all|@TAG> TEXT... [--timeout D]  send + collect replies\n"
               "  mhx mesh tags <list|show|add|rm|del> ...  manage ~/.mesh/tags.json\n"
               "  mhx mesh inbox <count|show|dump|clear|archive> ...  inbox admin\n"
               "  mhx mesh selfcheck                   local readiness check (PASS/WARN/FAIL rows)\n",
               has_bin ? "built" : "not built — run 'make -C bake/mesh-agent'");
        return MHX_OK;
    }
    const char *sub = argv[2];

    if (strcmp(sub, "peers") == 0) {
        char *cargv[] = { (char *)"mesh-agent", (char *)"peers", NULL };
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx mesh: build it first with 'make -C bake/mesh-agent'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    if (strcmp(sub, "listen") == 0) {
        /* mhx mesh listen [--since DURATION] [--watch] — pass-through */
        int extra = argc - 3;
        int n = 2 + extra + 1;
        char **cargv = (char **)calloc((size_t)n, sizeof(char *));
        if (!cargv) return MHX_HOST_FAIL;
        int j = 0;
        cargv[j++] = (char *)"mesh-agent";
        cargv[j++] = (char *)"listen";
        for (int i = 3; i < argc; i++) cargv[j++] = argv[i];
        cargv[j] = NULL;
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        free(cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx mesh: build it first with 'make -C bake/mesh-agent'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    if (strcmp(sub, "status") == 0) {
        if (argc >= 4) {
            char *cargv[] = { (char *)"mesh-agent", (char *)"status",
                              argv[3], NULL };
            int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
            if (rc < 0) return MHX_HOST_FAIL;
            return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
        }
        char *cargv[] = { (char *)"mesh-agent", (char *)"status", NULL };
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        if (rc < 0) return MHX_HOST_FAIL;
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    if (strcmp(sub, "send") == 0) {
        if (argc < 5) {
            fprintf(stderr,
                "mhx mesh send: usage 'mhx mesh send <peer|@all|@TAG> TEXT...'\n");
            return MHX_BAD_ARGS;
        }
        const char *target = argv[3];
        int broadcast_all = (strcmp(target, "@all") == 0);
        int is_tag = (target[0] == '@' && !broadcast_all);
        /* @all  → mesh-agent broadcast TEXT...
         * @TAG  → mesh-agent broadcast @TAG TEXT...
         * PEER  → mesh-agent send PEER TEXT... */
        int extra = argc - 4;
        int n;
        if (broadcast_all)  n = 2 + extra + 1;
        else if (is_tag)    n = 3 + extra + 1;
        else                n = 3 + extra + 1;
        char **cargv = (char **)calloc((size_t)n, sizeof(char *));
        if (!cargv) return MHX_HOST_FAIL;
        int j = 0;
        cargv[j++] = (char *)"mesh-agent";
        if (broadcast_all) {
            cargv[j++] = (char *)"broadcast";
        } else if (is_tag) {
            cargv[j++] = (char *)"broadcast";
            cargv[j++] = (char *)target;
        } else {
            cargv[j++] = (char *)"send";
            cargv[j++] = (char *)target;
        }
        for (int i = 4; i < argc; i++) cargv[j++] = argv[i];
        cargv[j] = NULL;
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        free(cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx mesh: build it first with 'make -C bake/mesh-agent'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    if (strcmp(sub, "tags") == 0) {
        /* mhx mesh tags <list|show|add|rm|del> ... — pass-through */
        int extra = argc - 3;
        int n = 2 + extra + 1;
        char **cargv = (char **)calloc((size_t)n, sizeof(char *));
        if (!cargv) return MHX_HOST_FAIL;
        int j = 0;
        cargv[j++] = (char *)"mesh-agent";
        cargv[j++] = (char *)"tags";
        for (int i = 3; i < argc; i++) cargv[j++] = argv[i];
        cargv[j] = NULL;
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        free(cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx mesh: build it first with 'make -C bake/mesh-agent'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    if (strcmp(sub, "inbox") == 0) {
        /* mhx mesh inbox <count|show|dump|clear|archive> ... — pass-through */
        int extra = argc - 3;
        int n = 2 + extra + 1;
        char **cargv = (char **)calloc((size_t)n, sizeof(char *));
        if (!cargv) return MHX_HOST_FAIL;
        int j = 0;
        cargv[j++] = (char *)"mesh-agent";
        cargv[j++] = (char *)"inbox";
        for (int i = 3; i < argc; i++) cargv[j++] = argv[i];
        cargv[j] = NULL;
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        free(cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx mesh: build it first with 'make -C bake/mesh-agent'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    if (strcmp(sub, "selfcheck") == 0) {
        /* mhx mesh selfcheck — local node readiness check, pass-through */
        char *cargv[] = { (char *)"mesh-agent", (char *)"selfcheck", NULL };
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx mesh: build it first with 'make -C bake/mesh-agent'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    if (strcmp(sub, "ask") == 0) {
        /* mhx mesh ask <peer|@all> TEXT... [--timeout D] — pass-through */
        int extra = argc - 3;
        int n = 2 + extra + 1;
        char **cargv = (char **)calloc((size_t)n, sizeof(char *));
        if (!cargv) return MHX_HOST_FAIL;
        int j = 0;
        cargv[j++] = (char *)"mesh-agent";
        cargv[j++] = (char *)"ask";
        for (int i = 3; i < argc; i++) cargv[j++] = argv[i];
        cargv[j] = NULL;
        int rc = mhx_spawn_baked("mesh-agent/mesh-agent", cargv);
        free(cargv);
        if (rc < 0) {
            fprintf(stderr,
                "mhx mesh: build it first with 'make -C bake/mesh-agent'\n");
            return MHX_HOST_FAIL;
        }
        return rc == 0 ? MHX_OK : MHX_HOST_FAIL;
    }

    fprintf(stderr,
        "mhx mesh: unknown subverb '%s' "
        "(peers|status|send|listen|ask|tags|inbox|selfcheck)\n", sub);
    return MHX_BAD_ARGS;
}

mhx_status mhx_run_builtin(int argc, char **argv)
{
    if (argc == 0) return MHX_OK;

    /* "exit" is a top-level builtin, not "mhx exit". */
    if (strcmp(argv[0], "exit") == 0 || strcmp(argv[0], "quit") == 0)
        return MHX_EXIT;

    if (strcmp(argv[0], "mhx") != 0) return MHX_UNKNOWN;
    if (argc < 2) return builtin_help(argc, argv);

    const char *verb = argv[1];
    if (strcmp(verb, "help") == 0)    return builtin_help(argc, argv);
    if (strcmp(verb, "version") == 0) return builtin_version(argc, argv);
    if (strcmp(verb, "info") == 0)    return builtin_info(argc, argv);
    if (strcmp(verb, "heart") == 0)   return builtin_heart(argc, argv);
    if (strcmp(verb, "aml") == 0)     return builtin_aml(argc, argv);
    if (strcmp(verb, "notorch") == 0) return builtin_notorch(argc, argv);
    if (strcmp(verb, "install") == 0) return builtin_install(argc, argv);
    if (strcmp(verb, "train") == 0)   return builtin_train(argc, argv);
    if (strcmp(verb, "slots") == 0)   return builtin_slots(argc, argv);
    if (strcmp(verb, "mesh") == 0)    return builtin_mesh(argc, argv);

    /* `mhx <something>` is committed to the builtin namespace, so an
     * unknown verb is a usage error — we do not silently forward
     * "mhx zzz" to the host shell. */
    fprintf(stderr, "mhx: unknown verb '%s' (try 'mhx help')\n", verb);
    return MHX_BAD_ARGS;
}

/* ---- host forwarding --------------------------------------------------- */

mhx_status mhx_run_host(const char *line)
{
    if (!line || !*line) return MHX_OK;
    int rc = system(line);
    if (rc == -1) return MHX_HOST_FAIL;
    if (WIFEXITED(rc) && WEXITSTATUS(rc) == 0) return MHX_OK;
    return MHX_HOST_FAIL;
}

/* ---- dispatch ---------------------------------------------------------- */

mhx_status mhx_dispatch(const char *line)
{
    if (!g_mhx_root[0]) mhx_init_root(NULL);
    if (!line) return MHX_OK;
    /* skip leading whitespace */
    while (*line && isspace((unsigned char)*line)) line++;
    if (!*line) return MHX_OK;

    char buf[4096];
    size_t n = strlen(line);
    if (n >= sizeof(buf)) return MHX_BAD_ARGS;
    memcpy(buf, line, n + 1);

    char *argv[MHX_MAX_ARGV];
    /* Need a working copy because split_argv writes NUL terminators. */
    char work[4096];
    memcpy(work, buf, n + 1);
    int argc = split_argv(work, argv);

    mhx_status st = mhx_run_builtin(argc, argv);
    if (st != MHX_UNKNOWN) return st;
    return mhx_run_host(buf);
}

mhx_status mhx_dispatch_argv(int argc, char **argv)
{
    if (!g_mhx_root[0]) mhx_init_root(NULL);
    if (argc <= 0 || !argv) return MHX_OK;
    mhx_status st = mhx_run_builtin(argc, argv);
    if (st != MHX_UNKNOWN) return st;
    fprintf(stderr, "mhx --exec: host forwarding is not supported; use -c for shell commands\n");
    return MHX_BAD_ARGS;
}

/* ---- repl -------------------------------------------------------------- */

static int mhx_repl_inner(int prompt)
{
    char line[4096];
    for (;;) {
        if (prompt) {
            fputs("mhx> ", stdout);
            fflush(stdout);
        }
        int n = mhx_readline(line, sizeof(line));
        if (n < 0) { fputc('\n', stdout); break; }
        mhx_status st = mhx_dispatch(line);
        if (st == MHX_EXIT) break;
    }
    return 0;
}

int mhx_repl(void)
{
    return mhx_repl_inner(1);
}

int mhx_repl_no_prompt(void)
{
    return mhx_repl_inner(0);
}

#ifndef MHX_NO_MAIN
int main(int argc, char **argv)
{
    mhx_init_root(argv[0]);
    if (argc >= 3 && strcmp(argv[1], "--exec") == 0) {
        return (int)mhx_dispatch_argv(argc - 2, argv + 2);
    }
    /* Single-shot mode: `mhx -c "<cmd>"` runs one command and exits. */
    if (argc >= 3 && strcmp(argv[1], "-c") == 0) {
        return (int)mhx_dispatch(argv[2]);
    }
    if (argc >= 2 && (strcmp(argv[1], "--line") == 0 || strcmp(argv[1], "--quiet") == 0)) {
        return mhx_repl_no_prompt();
    }
    return mhx_repl();
}
#endif
