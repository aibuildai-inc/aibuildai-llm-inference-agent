#!/usr/bin/env bash
set -e
# claude_agent_sdk spawns a Claude Code subprocess; unset to allow nested invocation
unset CLAUDECODE
cd "$(dirname "$(dirname "$(realpath "$0")")")"

echo "=== rebuilding all built-in pattern memories $(date) ==="
python -m memory.curate
echo "=== done $(date) ==="
