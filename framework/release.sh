#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# APP_VERSION is the bare derived version ("2.5.5", or "2026.8.28" on a named
# line) and APP_LINE the codename of a named line ("science", empty on a direct
# line). The release tag, tarball name, and GitHub release all use the tag that
# aibuildai_version.py derives from the pair: "v2.5.5" or "science-2026-08-28". Both are
# injected into _version.py below. AIBUILDAI_BUILD_VERSION and
# AIBUILDAI_BUILD_LINE let the caller pass the host-derived pair in — required
# inside the Docker build, where .git is excluded from the context so the
# in-container `git describe` cannot resolve them. A local source build leaves
# them unset and derives from git as usual.
BARE_VERSION="${AIBUILDAI_BUILD_VERSION:-$(python -c 'from aibuildai_version import APP_VERSION; print(APP_VERSION)')}"
BUILD_LINE="${AIBUILDAI_BUILD_LINE-$(python -c 'from aibuildai_version import APP_LINE; print(APP_LINE or "")')}"
VERSION="$(python -c 'import sys; from aibuildai_version import release_tag; print(release_tag(sys.argv[1], sys.argv[2] or None))' "${BARE_VERSION}" "${BUILD_LINE}")"
ARCH="linux-x86_64"
NAME="aibuildai-${ARCH}-${VERSION}"
TARBALL="${NAME}.tar.gz"
# The tarball carries the tag: the auto-updater resolves a release asset by the
# name `aibuildai-<arch>-<tag>.tar.gz`. The directory a CUSTOMER unpacks carries
# the bare codename on a named line (`science`), so every release on the line
# unpacks to the same place; a direct line unpacks to its tag (`v2.5.5`).
STAGING="aibuildai-${ARCH}-$(python -c 'import sys; from aibuildai_version import release_dir_label; print(release_dir_label(sys.argv[1], sys.argv[2] or None))' "${BARE_VERSION}" "${BUILD_LINE}")"
PUBLISH=1
DRAFT=0
NOTES_FILE=""
REPO="ruz048/ML-agent"
USE_GH=0
GITHUB_TOKEN_RESOLVED=""
SOURCE_ROOT="$(pwd)"
BUILD_ROOT=""
CHECK_RUNTIME=""
PG_ROOT=""
PG_DATA=""
CHECK_USER=()
STAGE_ROOT=""

usage() {
    cat <<EOF
Usage: $0 [--no-publish] [--draft] [--notes-file PATH]

Options:
  --no-publish       Build the tarball locally but skip GitHub release publish
  --draft            Create the GitHub release as a draft
  --notes-file PATH  Markdown file used as the release notes source
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-publish)
            PUBLISH=0
            shift
            ;;
        --draft)
            DRAFT=1
            shift
            ;;
        --notes-file)
            [[ $# -ge 2 ]] || { echo "--notes-file requires a path" >&2; exit 2; }
            NOTES_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

resolve_github_token() {
    if [[ -n "${GITHUB_TOKEN_RESOLVED}" ]]; then
        printf '%s\n' "${GITHUB_TOKEN_RESOLVED}"
        return 0
    fi

    if [[ -n "${GH_TOKEN:-}" ]]; then
        GITHUB_TOKEN_RESOLVED="${GH_TOKEN}"
        printf '%s\n' "${GITHUB_TOKEN_RESOLVED}"
        return 0
    fi

    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        GITHUB_TOKEN_RESOLVED="${GITHUB_TOKEN}"
        printf '%s\n' "${GITHUB_TOKEN_RESOLVED}"
        return 0
    fi

    local cred password
    cred="$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill 2>/dev/null || true)"
    password="$(printf '%s\n' "${cred}" | awk -F= '/^password=/{print $2}')"
    if [[ -n "${password}" ]]; then
        GITHUB_TOKEN_RESOLVED="${password}"
        printf '%s\n' "${GITHUB_TOKEN_RESOLVED}"
        return 0
    fi

    return 1
}

cleanup_temp_build() {
    if [[ -n "${PG_DATA}" && -x "${PG_ROOT}/bin/pg_ctl" ]]; then
        "${CHECK_USER[@]}" env -i LD_LIBRARY_PATH="${PG_ROOT}/lib" \
            "${PG_ROOT}/bin/pg_ctl" stop -D "${PG_DATA}" -m fast -w -t 30 >/dev/null 2>&1 || true
    fi
    if [[ -n "${CHECK_RUNTIME}" && -d "${CHECK_RUNTIME}" ]]; then
        rm -rf "${CHECK_RUNTIME}"
    fi
    if [[ -n "${BUILD_ROOT}" && -d "${BUILD_ROOT}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}

prepare_build_tree() {
    BUILD_ROOT="$(mktemp -d "${SOURCE_ROOT}/tmp.release.stage.XXXXXX")"
    # The staging tree pyinstaller freezes starts as the copied source tree;
    # build_runtime_tree then Cython-compiles every product module in it to
    # native .so before packaging (see scripts/cython_compile_tree.py).
    STAGE_ROOT="${BUILD_ROOT}"

    python - "${SOURCE_ROOT}" "${STAGE_ROOT}" <<'PY'
from pathlib import Path
import shutil
import sys

src_root = Path(sys.argv[1])
stage_root = Path(sys.argv[2])

sys.path.insert(0, str(src_root))
from aibuildai_version import RUNTIME_TOP_LEVEL_MODULES, RUNTIME_PACKAGES, RUNTIME_ROOT_DATA_FILES

# Always ship aibuildai.spec alongside the staging tree so pyinstaller can
# consume it inside STAGE_ROOT regardless of where it ends up. RUNTIME_ROOT_DATA_FILES
# are repo-root data files bundled into the frozen build (now empty; example.yaml is
# generated from the models at runtime, no longer staged).
top_level_files = list(RUNTIME_TOP_LEVEL_MODULES) + [
    "aibuildai.spec",
    "pyproject.toml",
    "README.md",
] + list(RUNTIME_ROOT_DATA_FILES)
for relative in top_level_files:
    shutil.copy2(src_root / relative, stage_root / relative)

for package in RUNTIME_PACKAGES:
    shutil.copytree(src_root / package, stage_root / package, dirs_exist_ok=True)
PY
}

build_runtime_tree() {
    prepare_build_tree

    # Programs run from this exact wheel inside the run's own Program environment.
    # Build it from the untouched release source before the staging tree is
    # Cython-compiled for the manager binary.
    APP_LINE_LITERAL="$(python -c 'import sys; print(repr(sys.argv[1] or None))' "${BUILD_LINE}")"
    sed -i "s/^APP_VERSION, APP_LINE = _derive_version()$/APP_VERSION = \"${BARE_VERSION}\"\\nAPP_LINE = ${APP_LINE_LITERAL}/" "${STAGE_ROOT}/aibuildai_version.py"
    grep -qxF "APP_VERSION = \"${BARE_VERSION}\"" "${STAGE_ROOT}/aibuildai_version.py"
    grep -qxF "APP_LINE = ${APP_LINE_LITERAL}" "${STAGE_ROOT}/aibuildai_version.py"
    mkdir -p "${STAGE_ROOT}/program-runtime"
    AIBUILDAI_BUILD_VERSION="${BARE_VERSION}" python -m pip wheel --no-deps --wheel-dir "${STAGE_ROOT}/program-runtime" "${SOURCE_ROOT}/packages/conversation" "${SOURCE_ROOT}/packages/execution" "${STAGE_ROOT}"
    cp "${SOURCE_ROOT}/aibuildai_version.py" "${STAGE_ROOT}/aibuildai_version.py"

    # Generate the plugin bundle: embed all plugin content as Python string
    # constants in the staging tree's startup/_plugin_bundle/ package. The whole
    # bundle as ONE module expanded to a ~500 MB .c that gcc -O2 could not build,
    # so it is split into bounded _chunk_NN.py modules (~4 MiB content each) plus
    # an __init__.py aggregator that merges them into PLUGIN_BUNDLE. Every chunk
    # IS Cython-compiled to .so below, like the rest of the tree.
    python "${SOURCE_ROOT}/scripts/build_plugin_bundle.py" \
        "${SOURCE_ROOT}/plugins" \
        "${STAGE_ROOT}/startup/_plugin_bundle"

    # Embed product prompt and memory resources before Cython compilation.
    # PyInstaller collects the installed Conversation catalog as package data.
    python "${SOURCE_ROOT}/scripts/build_resource_bundle.py" \
        "${STAGE_ROOT}"

    # Cython-compile every product module in the staging tree to native .so and
    # drop the plaintext .py (renamed to .py.bak so aibuildai.spec's AST scan
    # still reads it). Runs LAST in staging, once all .py plus the generated
    # bundles (the _plugin_bundle/ chunk package + the resource bundles) are
    # present, so every bundle chunk is compiled too. The small skip list in
    # that script keeps files as Python only when the runtime needs Python
    # function behavior. PyInstaller still puts those files inside the binary.
    python "${SOURCE_ROOT}/scripts/cython_compile_tree.py" "${STAGE_ROOT}"
}

validate_notes_file() {
    [[ -n "${NOTES_FILE}" ]] || {
        echo "Publishing requires --notes-file PATH." >&2
        echo "Write the note by hand from the aibuildai-releasing skill's notes template." >&2
        exit 1
    }
    [[ -f "${NOTES_FILE}" ]] || {
        echo "Release notes file not found: ${NOTES_FILE}" >&2
        exit 1
    }
    if grep -nP "TODO:|Describe This Release.*template|Replace this section" "${NOTES_FILE}" >/dev/null 2>&1; then
        echo "Release notes file still contains template placeholders: ${NOTES_FILE}" >&2
        exit 1
    fi
}

render_release_notes() {
    local asset_id="$1"
    local output_file="$2"
    python - "$NOTES_FILE" "$output_file" "$VERSION" "$NAME" "$asset_id" <<'PY'
from pathlib import Path
import sys

src, dst, version, name, asset_id = sys.argv[1:]
text = Path(src).read_text()
text = text.replace("{{VERSION}}", version)
text = text.replace("{{NAME}}", name)
text = text.replace("{{ASSET_ID}}", asset_id)
Path(dst).write_text(text)
PY
}

json_get() {
    local expr="$1"
    python -c '
import json
import sys

expr = sys.argv[1]
data = json.load(sys.stdin)
value = data
for part in expr.split("."):
    if not part:
        continue
    value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
' "$expr"
}

github_api_json() {
    local method="$1"
    local url="$2"
    local data="${3:-}"
    local token
    token="$(resolve_github_token)"
    if [[ -n "${data}" ]]; then
        curl -fsSL \
            -X "${method}" \
            -H "Authorization: Bearer ${token}" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            -H "Content-Type: application/json" \
            "${url}" \
            -d "${data}"
    else
        curl -fsSL \
            -X "${method}" \
            -H "Authorization: Bearer ${token}" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "${url}"
    fi
}

github_release_exists() {
    local releases_json
    releases_json="$(github_api_json GET "https://api.github.com/repos/${REPO}/releases")"
    printf '%s' "${releases_json}" | python -c '
import json
import sys

version = sys.argv[1]
releases = json.load(sys.stdin)
sys.exit(0 if any(r.get("tag_name") == version for r in releases) else 1)
' "${VERSION}"
}

publish_with_gh() {
    local draft_flag initial_notes rendered_notes asset_id
    draft_flag=()
    if [[ "${DRAFT}" -eq 1 ]]; then
        draft_flag+=(--draft)
    fi

    initial_notes="$(mktemp)"
    rendered_notes="$(mktemp)"
    printf 'Preparing release artifacts for %s.\n' "${VERSION}" > "${initial_notes}"

    gh release create "${VERSION}" \
        --repo "${REPO}" \
        --title "aibuildai ${VERSION}" \
        --notes-file "${initial_notes}" \
        "${draft_flag[@]}"

    gh release upload "${VERSION}" "${TARBALL}" SHA256SUMS \
        --repo "${REPO}" \
        --clobber

    asset_id="$(gh api "repos/${REPO}/releases/tags/${VERSION}" --jq '.assets[] | select(.name == "'"${TARBALL}"'") | .id')"
    render_release_notes "${asset_id}" "${rendered_notes}"
    gh release edit "${VERSION}" \
        --repo "${REPO}" \
        --notes-file "${rendered_notes}"

    rm -f "${initial_notes}" "${rendered_notes}"
}

publish_with_api() {
    local initial_notes rendered_notes create_payload create_response
    local release_id upload_url upload_response asset_id edit_payload html_url target_commitish

    initial_notes="$(mktemp)"
    rendered_notes="$(mktemp)"
    printf 'Preparing release artifacts for %s.\n' "${VERSION}" > "${initial_notes}"
    target_commitish="$(git rev-parse HEAD)"

    create_payload="$(python - "${VERSION}" "${target_commitish}" "${DRAFT}" "${initial_notes}" <<'PY'
from pathlib import Path
import json
import sys

version, target_commitish, draft_flag, notes_path = sys.argv[1:]
payload = {
    "tag_name": version,
    "target_commitish": target_commitish,
    "name": f"aibuildai {version}",
    "body": Path(notes_path).read_text(),
    "draft": draft_flag == "1",
    "prerelease": False,
}
print(json.dumps(payload))
PY
)"

    create_response="$(github_api_json POST "https://api.github.com/repos/${REPO}/releases" "${create_payload}")"
    release_id="$(printf '%s' "${create_response}" | json_get "id")"
    upload_url="$(printf '%s' "${create_response}" | json_get "upload_url")"
    upload_url="${upload_url%\{*}"

    upload_response="$(curl -fsSL \
        -X POST \
        -H "Authorization: Bearer $(resolve_github_token)" \
        -H "Content-Type: application/gzip" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        --data-binary @"${TARBALL}" \
        "${upload_url}?name=${TARBALL}")"
    asset_id="$(printf '%s' "${upload_response}" | json_get "id")"

    curl -fsSL \
        -X POST \
        -H "Authorization: Bearer $(resolve_github_token)" \
        -H "Content-Type: text/plain" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        --data-binary @SHA256SUMS \
        "${upload_url}?name=SHA256SUMS" >/dev/null

    render_release_notes "${asset_id}" "${rendered_notes}"
    edit_payload="$(python - "${rendered_notes}" <<'PY'
from pathlib import Path
import json
import sys

print(json.dumps({"body": Path(sys.argv[1]).read_text()}))
PY
)"
    github_api_json PATCH "https://api.github.com/repos/${REPO}/releases/${release_id}" "${edit_payload}" >/dev/null
    html_url="$(printf '%s' "${create_response}" | json_get "html_url")"

    rm -f "${initial_notes}" "${rendered_notes}"
    printf 'Draft URL: %s\n' "${html_url}"
}

trap 'cleanup_temp_build' EXIT

if [[ "${PUBLISH}" -eq 1 ]]; then
    validate_notes_file
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        USE_GH=1
    elif ! resolve_github_token >/dev/null 2>&1; then
        echo "Publishing requires either authenticated gh CLI or a GitHub token in GH_TOKEN/GITHUB_TOKEN/git credential store." >&2
        exit 1
    fi
fi

# Build tools: PyInstaller freezes the tree; Cython compiles every product
# module to native .so (scripts/cython_compile_tree.py, invoked from
# build_runtime_tree). Both are build-only deps, installed here like the rest of
# the minimal builder image's tooling.
pip install pyinstaller cython --quiet
require_cmd pyinstaller

# Pre-build machine gate — no unchecked binary leaves this script. check.sh runs
# pyright, ruff, semgrep, the Agent public-method check, and the lock-file check.
# These are questions a machine can settle. Everything a machine cannot settle
# is a written-down rule, not a gate here.
#
# The minimal Ubuntu-20.04 builder image deliberately does not carry that tooling
# (semgrep cannot even be installed there: it hard-pins mcp==1.23.3, which
# conflicts with the product's mcp). So the Docker release build sets
# AIBUILDAI_SKIP_PRECHECK=1 and check.sh is run on the HOST before `docker build`
# instead (see README.md "Releasing"). A local source build leaves the flag unset
# and gates here as usual. The skip is explicit, never silent.
if [ -n "${AIBUILDAI_SKIP_PRECHECK:-}" ]; then
    echo "===== Pre-build machine gate SKIPPED (AIBUILDAI_SKIP_PRECHECK set) ====="
    echo "Run './check.sh' on the host before this build — see README.md Releasing."
else
    echo "===== Pre-build machine gate ====="
    ./check.sh
    echo "===== Machine gate passed ====="
fi

build_runtime_tree
echo "Staging tree prepared at ${STAGE_ROOT}"

# Build-time version injection (the PyInstaller equivalent of setuptools-scm).
# Freeze the BARE git-tag-derived version and the line into a plain
# `_version.py` that aibuildai_version.py reads first at runtime (APP_VERSION is stored
# without a leading "v"; APP_LINE is None on a direct line). The frozen binary
# therefore never shells out to git and never depends on package metadata.
# `_version` is declared as a hidden import in aibuildai.spec so PyInstaller
# bundles it.
if [[ -n "${BUILD_LINE}" ]]; then
    printf 'APP_VERSION = "%s"\nAPP_LINE = "%s"\n' "${BARE_VERSION}" "${BUILD_LINE}" > "${STAGE_ROOT}/_version.py"
else
    printf 'APP_VERSION = "%s"\nAPP_LINE = None\n' "${BARE_VERSION}" > "${STAGE_ROOT}/_version.py"
fi
echo "Injected version ${BARE_VERSION} (line: ${BUILD_LINE:-none}) into ${STAGE_ROOT}/_version.py"

(
    cd "${STAGE_ROOT}"
    AIBUILDAI_AST_ROOT="${SOURCE_ROOT}" pyinstaller aibuildai.spec --noconfirm
)
echo "Build complete: ${STAGE_ROOT}/dist/aibuildai/"

# Clean previous staging
rm -rf "${STAGING}" "${TARBALL}"

# Stage the binary and _internal/
mkdir -p "${STAGING}"
cp -r "${STAGE_ROOT}/dist/aibuildai/"* "${STAGING}/"
cp -r "${STAGE_ROOT}/program-runtime" "${STAGING}/program-runtime"
cp release/guide.md "${STAGING}/README.md"

# The machine-readable build identity, shipped beside the binary. `--version`
# prints the customer-facing release label, not the version triple the install
# tree is keyed by, so install.sh and the auto-update self-check read WHICH build
# this tree is from here instead.
printf '%s\n' "${BARE_VERSION}" > "${STAGING}/VERSION"

# Stage the one owned tools directory next to the executable. The checkout's
# own pins/locks/submodule commit are the authority: stage_tools() runs the
# same setup_owned_tools() path a source run uses and copies the resolved
# bytes, so no second list of executables exists anywhere.
python - "${SOURCE_ROOT}" "${STAGING}" <<'PY'
from pathlib import Path
import sys

src_root = Path(sys.argv[1])
staging = Path(sys.argv[2])

sys.path.insert(0, str(src_root))
from infra.owned_tools import stage_tools

stage_tools(staging / "tools")
print("Staged owned tools directory: tools")
PY

# Stage raw runtime resource directories (plugins, etc.) next to the binary.
# These are NOT obfuscated or bundled through PyInstaller — they ship as-is
# so the binary can discover them via sys.executable at runtime.
python - "${SOURCE_ROOT}" "${STAGING}" <<'PY'
from pathlib import Path
import shutil
import sys

src_root = Path(sys.argv[1])
staging = Path(sys.argv[2])

sys.path.insert(0, str(src_root))
from aibuildai_version import RUNTIME_RESOURCE_DIRS

for name in RUNTIME_RESOURCE_DIRS:
    src = src_root / name
    if not src.is_dir():
        print(f"WARNING: resource dir missing, skipping: {name}", file=sys.stderr)
        continue
    dst = staging / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
    )
    print(f"Staged resource dir: {name}")
PY

# What install.sh is allowed to copy, derived from what this build actually
# staged. Deriving it here is what keeps it true: a second hand-written list of
# package contents would drift the moment staging above adds or drops an entry.
# The two files the installer must never copy into the install tree are named
# here rather than left to statement order: install.sh and SHA256SUMS are
# written into the staging tree below, and a list that excluded them only by
# being computed first would start naming them the day that order changes.
PACKAGE_ENTRY_LINES="$(find "${STAGING}" -mindepth 1 -maxdepth 1 \
    ! -name install.sh ! -name SHA256SUMS -printf '%f\n' | sort)"
# install.sh expands this list unquoted in `for ENTRY in ...`, and the list
# reaches the script through a sed replacement, so only a plain name survives
# both: whitespace would split one entry into two and the installer would then
# blame the user's download for a packaging fault, a glob character would be
# expanded against the install directory, and `|`, `&`, or `\` would corrupt
# the replacement. Every name staged today is plain, but the list comes from
# `find`, which carries any name. Refuse the build here, naming the entry.
while IFS= read -r ENTRY; do
    if [[ ! "${ENTRY}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "release.sh failed: staged entry '${ENTRY}' is not a plain name; install.sh expands its entry list unquoted and cannot carry it." >&2
        exit 1
    fi
done <<< "${PACKAGE_ENTRY_LINES}"
PACKAGE_ENTRIES="$(tr '\n' ' ' <<< "${PACKAGE_ENTRY_LINES}")"
echo "Package entries install.sh will copy: ${PACKAGE_ENTRIES}"

# Copy the one installer used by both manual install and automatic update.
cp release/install.sh "${STAGING}/install.sh"
POINTER_TARBALL="aibuildai-${ARCH}-$(python -c 'import sys; from aibuildai_version import pointer_tag; print(pointer_tag(sys.argv[1], sys.argv[2] or None))' "${BARE_VERSION}" "${BUILD_LINE}").tar.gz"
sed -i -e "s|@RELEASE_TARBALL@|${TARBALL}|" -e "s|@POINTER_TARBALL@|${POINTER_TARBALL}|" -e "s|@PACKAGE_ENTRIES@|${PACKAGE_ENTRIES}|" "${STAGING}/install.sh"
chmod +x "${STAGING}/install.sh"

# Generate checksums
cd "${STAGING}"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
cd ..

# Fail the build if any bundled ELF requires a higher glibc than this host provides.
# Inside Dockerfile.build (Ubuntu 20.04, glibc 2.31) this is the customer floor.
HOST_GLIBC="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
echo "glibc guard: host provides GLIBC ${HOST_GLIBC}"
HOST_GLIBC_MAJ="${HOST_GLIBC%%.*}"
HOST_GLIBC_MIN="${HOST_GLIBC#*.}"
objdump --version >/dev/null
GLIBC_VIOLATIONS=""
GLIBC_SCANNED=0
GLIBC_WITH_SYMBOLS=0
while IFS= read -r elf; do
    highest="$(objdump -T "$elf" 2>/dev/null \
        | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
        | sed 's/GLIBC_//' \
        | sort -t. -k1,1n -k2,2n -u \
        | tail -1 \
        || true)"
    GLIBC_SCANNED=$((GLIBC_SCANNED + 1))
    if [[ -z "$highest" ]]; then
        continue
    fi
    GLIBC_WITH_SYMBOLS=$((GLIBC_WITH_SYMBOLS + 1))
    file_maj="${highest%%.*}"
    file_min="${highest#*.}"
    if [[ "$file_maj" -gt "$HOST_GLIBC_MAJ" ]] \
       || { [[ "$file_maj" -eq "$HOST_GLIBC_MAJ" ]] && [[ "$file_min" -gt "$HOST_GLIBC_MIN" ]]; }; then
        rel="${elf#"${STAGING}/"}"
        GLIBC_VIOLATIONS="${GLIBC_VIOLATIONS}  GLIBC_${highest}  ${rel}\n"
    fi
done < <(find "${STAGING}" -type f | while read -r f; do
    head -c4 "$f" 2>/dev/null | grep -q $'\x7fELF' && printf '%s\n' "$f"
done)

if [[ "${GLIBC_SCANNED}" -eq 0 ]]; then
    echo "glibc guard FAILED: scanned 0 ELF files (expected hundreds)" >&2
    exit 1
fi
if [[ "${GLIBC_WITH_SYMBOLS}" -eq 0 ]]; then
    echo "glibc guard FAILED: ${GLIBC_SCANNED} ELFs scanned but none had GLIBC symbols (objdump broken?)" >&2
    exit 1
fi
if [[ -n "${GLIBC_VIOLATIONS}" ]]; then
    echo "glibc guard FAILED (scanned ${GLIBC_SCANNED} ELFs): these files require a higher glibc than ${HOST_GLIBC}:" >&2
    printf '%b' "${GLIBC_VIOLATIONS}" >&2
    exit 1
fi
echo "glibc guard passed: ${GLIBC_SCANNED} ELFs scanned, none exceeds GLIBC_${HOST_GLIBC}"

# Create tarball
tar -czf "${TARBALL}" "${STAGING}/"

echo "Created ${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"

CHECK_ROOT="${BUILD_ROOT}/release-check"
CHECK_RUNTIME="$(mktemp -d /tmp/aibuildai-release-runtime.XXXXXX)"
mkdir -p "${CHECK_ROOT}/home" "${CHECK_ROOT}/journal" "${CHECK_RUNTIME}/aibuildai/postgresql/18"
tar -xzf "${TARBALL}" -C "${CHECK_ROOT}"
if [[ "$(id -u)" -eq 0 ]]; then
    chown -R aibuildai-build:aibuildai-build "${CHECK_ROOT}"
    CHECK_USER=(runuser -u aibuildai-build --)
fi
PG_ROOT="${CHECK_ROOT}/${STAGING}/tools/postgresql"
PG_DATA="${CHECK_ROOT}/home/.aibuildai/postgresql/18/data"
PG_SOCKET="${CHECK_RUNTIME}/aibuildai/postgresql/18"
mkdir -p "$(dirname "${PG_DATA}")"
if [[ "$(id -u)" -eq 0 ]]; then
    chown -R aibuildai-build:aibuildai-build "${CHECK_ROOT}/home" "${CHECK_RUNTIME}"
fi
"${CHECK_USER[@]}" env -i HOME="${CHECK_ROOT}/home" XDG_RUNTIME_DIR="${CHECK_RUNTIME}" PATH="/usr/bin:/bin" LD_LIBRARY_PATH="${PG_ROOT}/lib" \
    "${PG_ROOT}/bin/pg_ctl" initdb -D "${PG_DATA}" -o "--auth-local=peer --auth-host=reject --encoding=UTF8 --no-locale" >/dev/null
"${CHECK_USER[@]}" env -i HOME="${CHECK_ROOT}/home" XDG_RUNTIME_DIR="${CHECK_RUNTIME}" PATH="/usr/bin:/bin" LD_LIBRARY_PATH="${PG_ROOT}/lib" \
    "${PG_ROOT}/bin/pg_ctl" start -D "${PG_DATA}" -l "${CHECK_ROOT}/postgresql.log" -w -t 30 \
    -p "${PG_ROOT}/bin/postgres" -o "-h '' -k '${PG_SOCKET}' -c unix_socket_permissions=0700"
(
    cd "${CHECK_ROOT}"
    "${CHECK_USER[@]}" env -i HOME="${CHECK_ROOT}/home" XDG_RUNTIME_DIR="${CHECK_RUNTIME}" PATH="/usr/bin:/bin" \
        "./${STAGING}/aibuildai" __release_check "${CHECK_ROOT}/journal"
)
"${CHECK_USER[@]}" env -i LD_LIBRARY_PATH="${PG_ROOT}/lib" \
    "${PG_ROOT}/bin/pg_ctl" stop -D "${PG_DATA}" -m fast -w -t 30
PG_DATA=""
echo "Built binary wrote and restored its PostgreSQL journal."

if [[ "${PUBLISH}" -eq 0 ]]; then
    echo "Skipping GitHub release publish (--no-publish)."
    rm -rf "${STAGING}"
    exit 0
fi

# Write the checksum the auto-updater checks a download against — a PUBLISH
# concern only. A --no-publish build (Dockerfile.build, local/dev) exits above and
# produces just the tarball. The public release writes the same file host-side
# (see README.md "Releasing").
(cd "$(dirname "${TARBALL}")" && sha256sum "$(basename "${TARBALL}")" > SHA256SUMS)

# Publish GitHub release
if [[ "${USE_GH}" -eq 1 ]]; then
    if gh release view "${VERSION}" --repo "${REPO}" >/dev/null 2>&1; then
        echo "Release ${VERSION} already exists on ${REPO}." >&2
        exit 1
    fi
elif github_release_exists; then
    echo "Release ${VERSION} already exists on ${REPO}." >&2
    exit 1
fi

if [[ "${USE_GH}" -eq 1 ]]; then
    publish_with_gh
else
    publish_with_api
fi

echo "Release ${VERSION} published."

# Cleanup staging
rm -rf "${STAGING}"
