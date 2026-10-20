"""Generate Claude launchers from the effective environment and borrowed confinement.

Instructions stay on an anonymous descriptor. The conversation names its required CLI/config/runtime paths; the physical owner applies those bindings and builds the confined launch. No physical resource implementation is imported here.
"""

from __future__ import annotations

import ctypes
import os
import shlex
from collections.abc import Mapping
from pathlib import Path

from aibuildai_conversation.environment import prepare_claude_runtime
from aibuildai_conversation._impl.spec import Confinement
from aibuildai_conversation.assets import claude_path
from aibuildai_exec.launch import environment_preamble

_PROMPT_FD = 9
_LIBC = ctypes.CDLL(None, use_errno=True)


def _create_memfd(name: str) -> int:
    fd = _LIBC.memfd_create(os.fsencode(name), 1)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def build_agent_launcher(
    conf: "Confinement | None",
    *,
    prompt_source: str,
    env: Mapping[str, str],
) -> tuple[str, int | None]:
    """Build a launcher from the same environment the SDK receives; keep prompt bytes on an anonymous descriptor."""
    environment_keys = (*env, "CLAUDE_AGENT_SDK_VERSION", "PWD", "SHLVL", "_", "TRACEPARENT", "TRACESTATE", "BAGGAGE")
    real_claude = str(claude_path())
    # The SDK probes this executable with -v. Prompt-file options make that
    # metadata query initialize the CLI; only model calls need the descriptor.
    command = shlex.quote(real_claude)
    dispatch = (
        'if [ "$#" -eq 1 ] && { [ "$1" = -v ] || [ "$1" = --version ]; }; then\n'
        f'  exec {command} "$@"\nfi\n'
        f'exec {command} --append-system-prompt-file /proc/self/fd/{_PROMPT_FD} "$@"'
    )
    inner_argv = ["/bin/bash", "-c", dispatch, "aibuildai-claude"]
    reopen_fds = {_PROMPT_FD: prompt_source}
    token_file = shlex.quote(str(Path.home() / ".aibuildai" / "claude_token"))
    prelude = (
        f"if [ -r {token_file} ]; then\n"
        f'  CLAUDE_CODE_OAUTH_TOKEN="$(cat {token_file})"\n'
        "  export CLAUDE_CODE_OAUTH_TOKEN\n"
        "fi\n"
    )
    if conf is None:
        script = (
            "#!/usr/bin/env bash\nset -u\n"
            + environment_preamble(environment_keys)
            + f"exec {_PROMPT_FD}<{shlex.quote(prompt_source)}\n"
            + prelude
            + "exec "
            + " ".join(shlex.quote(arg) for arg in inner_argv)
            + ' "$@"\n'
        )
        launcher_fd = _create_memfd("aibuildai-claude-launcher")
        try:
            with os.fdopen(os.dup(launcher_fd), "w") as stream:
                stream.write(script)
            os.fchmod(launcher_fd, 0o500)
        except OSError:
            os.close(launcher_fd)
            raise
        return f"/proc/{os.getpid()}/fd/{launcher_fd}", launcher_fd
    if not conf.apply_bwrap:
        return conf.as_script(inner_argv, reopen_fds=reopen_fds, environment_keys=environment_keys, prelude=prelude), None

    own_read, own_write = prepare_claude_runtime(env)
    return conf.as_script(
        inner_argv,
        setenv={"PATH": os.pathsep.join(part for part in (own_read[0], env.get("PATH", "")) if part)},
        reopen_fds=reopen_fds,
        environment_keys=environment_keys, read_paths=own_read, write_paths=own_write, prelude=prelude,
    ), None
