"""Process-entry scrub of inherited Claude-Code feature env (#304).

The maintenance loop / testers launch the product from inside Claude-Code sessions, whose harness exports feature flags (CLAUDE_CODE_FORK_SUBAGENT, CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS, CLAUDE_CODE_SSE_PORT, CLAUDE_EFFORT, ...) to child processes. The Claude Agent SDK transport merges the product's full os.environ into every bundled-CLI subprocess (options.env is only an overlay), so any inherited flag silently mutates child-CLI behavior — a leaked CLAUDE_CODE_FORK_SUBAGENT=1 turned every sub-agent dispatch into a detached background fork (#273). #298 pins three known flags per agent spec; this module is the bootstrap-level chokepoint that removes the whole CLAUDE* class from os.environ at process entry, so every CLI child is clean by construction rather than by per-callsite pinning. The keep list below is the part that is NOT a flag, and a credential belongs on it: a scrubbed login leaves the child CLI unable to authenticate at all.
"""
import os
from collections.abc import Mapping, MutableMapping

from aibuildai_conversation._impl.claude import config_paths

# The product's own legitimate Claude config surface — never scrubbed.
# CLAUDE_CONFIG_DIR: native Claude CLI config dir for the subprocess.
# CLAUDE_CODE_OAUTH_TOKEN: the host's headless Claude Code login. It is a
#   CREDENTIAL, not a feature flag, and the bundled CLI authenticates with it
#   exactly as it does with a .credentials.json sign-in. Scrubbing it took a
#   host whose `claude auth status` reported `loggedIn: true` and made every
#   child CLI answer "Not logged in · Please run /login".
# CLAUDE_AGENT_SDK_*: read by the parent-side SDK client inside THIS
#   process (e.g. CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK), not a child-CLI
#   behavior flag.
_KEEP_EXACT = frozenset({"CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN"})
_KEEP_PREFIX = "CLAUDE_AGENT_SDK_"


def scrub_claude_feature_env(env: MutableMapping[str, str] | None = None) -> list[str]:
    """Delete inherited Claude feature flags from the supplied environment, or from process env at entry.

    Returns the sorted list of removed names (empty when already clean). The bare "CLAUDE" prefix covers CLAUDECODE, CLAUDE_CODE_*, CLAUDE_EFFORT and CLAUDE_AUTO_* alike — future Claude-Code flags are covered by construction.
    """
    target = os.environ if env is None else env
    removed = sorted(
        k for k in target
        if k.startswith("CLAUDE")
        and k not in _KEEP_EXACT
        and not k.startswith(_KEEP_PREFIX)
    )
    for k in removed:
        del target[k]
    return removed



def prepare_claude_runtime(env: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Prepare native state and return exact read/write paths for the effective environment.

    The caller grants these paths before launch. This does not grant HOME or
    change a Scope; config and state locations use the existing native rules.
    """
    # CLI discovery loads the provider SDK; entry-point environment scrubbing does not.
    from aibuildai_conversation.assets import claude_path

    reads = [str(claude_path().resolve().parent)]
    config = config_paths.config_dir(env)
    config.mkdir(parents=True, exist_ok=True)
    writes = [str(config)]
    state = config_paths.state_file(env)
    if state.parent != config:
        try:
            with state.open("x") as stream:
                stream.write("{}")
        except FileExistsError:
            pass
        writes.append(str(state))
    return tuple(reads), tuple(writes)


__all__ = ('scrub_claude_feature_env', 'prepare_claude_runtime')
