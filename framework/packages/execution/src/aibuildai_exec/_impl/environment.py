"""Compile explicit process environments for shell launchers."""

import shlex
import json
import os
import sys
from typing import NoReturn
from collections.abc import Collection, Sequence


def environment_preamble(keys: Collection[str]) -> str:
    """Keep only the named exported variables before a generated shell launch."""
    allowed = "|".join(shlex.quote(key) for key in sorted(set(keys)))
    if not allowed:
        raise ValueError("a shell launch requires an explicit environment")
    return (
        'for _aibuildai_env_name in $(compgen -e); do\n'
        'case "$_aibuildai_env_name" in\n'
        f'{allowed}) ;;\n'
        '*) unset "$_aibuildai_env_name" || exit 125 ;;\n'
        'esac\ndone\n'
    )


def make_killable_preamble() -> str:
    """The one line that keeps a memory limit from hanging the machine (issue #1298).

    Every payload the product launches can inherit ``oom_score_adj = -1000`` from the ssh or VS Code Remote session that started the orchestrator. aibuildai never sets it; it is inherited. At -1000 the kernel's memcg OOM killer, on finding a cgroup over ``memory.max``, has NO eligible victim -- so instead of killing, it LIVELOCKS: it re-reports the OOM forever, pins single-threaded journald at 100% CPU, and takes the shared host down with it.

    ``echo 0 > /proc/self/oom_score_adj`` raises the inherited value back to 0. Raising toward 0 needs no capability, writes ``/proc/self`` so it needs no cgroup delegation, and is inherited across fork and exec -- so one line at the top of the launcher makes the payload killable on every launch path there is.

    Reproduced on this host while designing the tree: without it, a 150 MB process in a 100 MB cgroup hangs for 60 seconds and neither dies nor progresses. With it, the kernel kills it immediately. ``memory.oom.group`` does not save you -- killing a group still requires one killable member.

    An implementation that copies the tree and drops this line hangs the machine.

    The write is ASSERTED, never best-effort. It used to end in ``|| true``, which is the one thing this line must not do: a host where ``/proc/self/oom_score_adj`` is not writable is exactly the host where the payload stays OOM-immune, the memcg killer finds no victim, and the machine livelocks -- and it would have done so with no error anywhere. A launch that cannot make its payload killable must not start.

    What the assertion must test is the VALUE, not the write. The kernel refuses the write whenever 0 is below ``signal->oom_score_adj_min``, a floor a privileged writer set -- and a systemd user manager started with a POSITIVE ``OOMScoreAdjust`` propagates a positive floor to every service beneath it, this launcher included. A payload already at >= 0 is eligible for the OOM killer, which is the entire property asserted here: the ceiling has a victim and nothing can livelock. Failing the launch there rejects a host that is behaving correctly. So a refused write falls through to reading the value back, and only a negative (or unreadable) one aborts -- fail-closed on the case that actually hangs the machine.

    Emits no single quotes, so it embeds inside an outer ``bash -c '...'``.
    """
    return (
        'echo 0 > /proc/self/oom_score_adj 2>/dev/null || {\n'
        '  oom_cur=$(cat /proc/self/oom_score_adj 2>/dev/null)\n'
        '  case ${oom_cur:-unreadable} in\n'
        '    [0-9]*) ;;\n'
        '    *) echo "aibuildai: cannot reset oom_score_adj (it is '
        '${oom_cur:-unreadable}); this payload would be OOM-immune and would '
        'livelock the host instead of being killed" >&2\n'
        '       exit 1 ;;\n'
        '  esac\n'
        '}'
    )


def restore_environment(descriptor: int, argv: Sequence[str]) -> NoReturn:
    """Restore the inherited descriptor's environment and execute argv.

    Frozen hosts dispatch __exec_environment to this entry before their normal application startup.
    """
    with os.fdopen(descriptor) as stream:
        environment = json.load(stream)
    os.execvpe(argv[0], list(argv), environment)


if __name__ == "__main__":
    restore_environment(int(sys.argv[1]), sys.argv[2:])
