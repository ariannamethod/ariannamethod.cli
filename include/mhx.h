#ifndef MHX_H
#define MHX_H

/* metaharmonix terminal — public interface for the core REPL.
 *
 * The terminal is built as a thin host above the user's shell:
 *   - mhx_repl()  drives the read/dispatch/print loop;
 *   - mhx_run_builtin() handles "mhx <verb>" commands directly;
 *   - anything else is forwarded to the host shell via mhx_run_host().
 *
 * The heart (dario.c) and the bake/ runtimes (aml, notorch) plug into
 * the same dispatcher through builtins, not through magic. */

#include <stddef.h>

#define MHX_VERSION "0.4.0-info"

typedef enum {
    MHX_OK = 0,
    MHX_EXIT = 1,        /* user asked to leave the REPL */
    MHX_UNKNOWN = 2,     /* unknown builtin verb */
    MHX_HOST_FAIL = 3,   /* host shell forwarder failed */
    MHX_BAD_ARGS = 4
} mhx_status;

/* Read a single line from stdin into buf (size includes NUL).
 * Returns number of bytes read excluding NUL, or -1 on EOF/error. */
int mhx_readline(char *buf, size_t size);

/* Dispatch a single command line. Empty input is a no-op (MHX_OK). */
mhx_status mhx_dispatch(const char *line);
mhx_status mhx_dispatch_argv(int argc, char **argv);

/* Built-in verb table entrypoint. Returns MHX_UNKNOWN if not a builtin. */
mhx_status mhx_run_builtin(int argc, char **argv);

/* Forward a command to the host shell. Blocking. */
mhx_status mhx_run_host(const char *line);

/* Drive the REPL until EOF or the "exit" builtin. */
int mhx_repl(void);
int mhx_repl_no_prompt(void);

/* Resolve the on-disk path of a baked-in or vendored binary
 * (e.g. "aml/runner/aml", "dario/dario"). Resolution order:
 *   1. $MHX_BAKE_PATH/<rel>   if MHX_BAKE_PATH is set;
 *   2. ./bake/<rel>           (baked runtimes — AML, notorch);
 *   3. ./vendor/<rel>         (vendored organisms — dario heart).
 * Writes the resolved path into out (size bytes). Returns 0 on success,
 * -1 if no candidate exists or the buffer is too small. */
int mhx_resolve_baked(const char *rel, char *out, size_t size);

/* fork+execv a baked-in binary; argv must be NULL-terminated.
 * Returns the child's exit status (>= 0) on success, or -1 if the
 * binary could not be resolved or spawn failed. */
int mhx_spawn_baked(const char *rel, char *const argv[]);

#endif
