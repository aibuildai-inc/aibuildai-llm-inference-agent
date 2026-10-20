#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OS="$(uname -s)"
MACHINE="$(uname -m)"
if [[ "${OS}" != "Linux" || "${MACHINE}" != "x86_64" ]]; then
    echo "aibuildai install failed: this release supports Linux x86_64 only; detected ${OS} ${MACHINE}." >&2
    exit 1
fi

LIB="${HOME}/.local/lib/aibuildai"
VER="$(cat "${SCRIPT_DIR}/VERSION")"
if [[ ! "${VER}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "aibuildai install failed: invalid build version '${VER}'." >&2
    exit 1
fi
for REQUIRED_COMMAND in flock timeout; do
    command -v "${REQUIRED_COMMAND}" >/dev/null 2>&1 || {
        echo "aibuildai install failed: the required '${REQUIRED_COMMAND}' command is unavailable." >&2
        exit 1
    }
done

for ENTRY in @PACKAGE_ENTRIES@; do
    if [[ ! -e "${SCRIPT_DIR}/${ENTRY}" ]]; then
        echo "aibuildai install failed: this package has no ${ENTRY}; the download is incomplete." >&2
        exit 1
    fi
done

VERSIONS="${LIB}/versions"
BIN="${HOME}/.local/bin/aibuildai"
mkdir -p "${VERSIONS}" "$(dirname "${BIN}")"
exec 9>"${LIB}/install.lock"
flock 9

STAGED="$(mktemp -d "${LIB}/.installing-${VER}.XXXXXX")"
DEST="${VERSIONS}/${VER}"
BACKUP="${LIB}/.rollback-${VER}.$$"
OLD_CURRENT="$(readlink "${LIB}/current" 2>/dev/null || true)"
OLD_BIN="$(readlink "${BIN}" 2>/dev/null || true)"
COMMITTED=0
INSTALLED_NEW=0

rollback_install() {
    if [[ "${COMMITTED}" -eq 0 ]]; then
        if [[ "${INSTALLED_NEW}" -eq 1 ]]; then
            rm -rf "${DEST}"
        fi
        if [[ -d "${BACKUP}" ]]; then
            mv "${BACKUP}" "${DEST}"
        fi
        if [[ -n "${OLD_CURRENT}" ]]; then
            rm -f "${LIB}/current.restore"
            ln -sfn "${OLD_CURRENT}" "${LIB}/current.restore"
            mv -Tf "${LIB}/current.restore" "${LIB}/current"
        else
            rm -f "${LIB}/current"
        fi
        if [[ -n "${OLD_BIN}" ]]; then
            ln -sf "${OLD_BIN}" "${BIN}"
        else
            rm -f "${BIN}"
        fi
    fi
    [[ -z "${STAGED}" || ! -d "${STAGED}" ]] || rm -rf "${STAGED}"
}
trap rollback_install EXIT

for ENTRY in @PACKAGE_ENTRIES@; do
    cp -r "${SCRIPT_DIR}/${ENTRY}" "${STAGED}/"
done

for CANDIDATE in "@RELEASE_TARBALL@" "@POINTER_TARBALL@"; do
    TARBALL="$(dirname "${SCRIPT_DIR}")/${CANDIDATE}"
    if [[ -f "${TARBALL}" ]]; then
        sha256sum "${TARBALL}" | awk '{print $1}' > "${STAGED}/tarball.sha256"
        break
    fi
done

timeout 60s env AIBUILDAI_AUTO_UPDATE=0 AIBUILDAI_IN_SERVICE=1 "${STAGED}/aibuildai" setup >/dev/null

if [[ -d "${DEST}" ]]; then
    rm -rf "${BACKUP}"
    mv "${DEST}" "${BACKUP}"
fi
mv "${STAGED}" "${DEST}"
STAGED=""
INSTALLED_NEW=1
ln -sfn "versions/${VER}" "${LIB}/current.next"
mv -Tf "${LIB}/current.next" "${LIB}/current"
ln -sf "${LIB}/current/aibuildai" "${BIN}"
timeout 60s env AIBUILDAI_AUTO_UPDATE=0 AIBUILDAI_IN_SERVICE=1 "${BIN}" setup >/dev/null

COMMITTED=1
find "${VERSIONS}" -mindepth 1 -maxdepth 1 ! -name "${VER}" -exec rm -rf -- {} +
rm -rf "${LIB}/aibuildai" "${LIB}/_internal" "${LIB}/README.md" "${BACKUP}"
find "${LIB}" -mindepth 1 -maxdepth 1 \( -name '.installing-*' -o -name '.rollback-*' -o -name 'current.*' \) -exec rm -rf -- {} +
trap - EXIT

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
       echo "Added ~/.local/bin to PATH in ~/.bashrc (restart shell or run: source ~/.bashrc)" ;;
esac

echo "Installed $("${DEST}/aibuildai" --version): ${BIN}"
