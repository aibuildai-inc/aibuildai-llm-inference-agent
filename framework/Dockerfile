FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Run as an arbitrary host uid so
# files written to bind-mounted host dirs are owned by the invoking user, not
# root. Following the OpenShift "arbitrary uid in group 0" convention: HOME lives
# off /root at a group-0-writable path and no USER is pinned, so any uid in the
# root group can write HOME (e.g. the `aibuildai login` session) while the binary
# still resolves on PATH. The chgrp/chmod at the end bakes these perms in so the
# aibuildai-home named volume inherits them on first mount.
ENV HOME=/opt/aibuildai
RUN mkdir -p /opt/aibuildai /workspace

# System essentials
RUN apt-get update && apt-get install -y \
    curl wget git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda init bash

# Install Node.js (for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install the aibuildai binary from the public rolling-latest release. The
# v2.5-latest pointer always serves the newest 2.5.x patch, so this never goes
# stale. install.sh installs into $HOME/.local (= /opt/aibuildai/.local here).
RUN curl -fsSL -o /tmp/aibuildai.tar.gz \
      https://github.com/aibuildai/AI-Build-AI/releases/download/v2.5-latest/aibuildai-linux-x86_64-v2.5-latest.tar.gz \
    && mkdir -p /tmp/aibuildai-extract \
    && tar -xzf /tmp/aibuildai.tar.gz -C /tmp/aibuildai-extract --strip-components=1 \
    && bash /tmp/aibuildai-extract/install.sh \
    && rm -rf /tmp/aibuildai.tar.gz /tmp/aibuildai-extract
ENV PATH="/opt/aibuildai/.local/bin:$PATH"

# Make HOME + WORKDIR writable by any uid in the root group (gid 0). New files
# created at runtime get primary group 0 (from `--user <uid>:0`), so the process
# can read/execute the binary and persist credentials/run state under either dir.
RUN chgrp -R 0 /opt/aibuildai /workspace \
    && chmod -R g=u /opt/aibuildai /workspace

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
