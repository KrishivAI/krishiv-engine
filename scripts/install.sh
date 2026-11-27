#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Installs the `krishiv` command line binary:
#
#   curl -fsSL https://krishiv.ai/install.sh | sh
#
# Picks the build for this machine (Linux x86-64 and aarch64 on glibc, macOS arm64 and x86-64), downloads it and
# SHA256SUMS from the GitHub Release, checks the checksum and only then installs it. On x86-64 Linux
# it takes the `x86-64-v3` build when the CPU has the features it needs and the plain `krishiv-compat`
# build otherwise.
#
# Environment:
# Flags: --version X.Y.Z (same as KRISHIV_VERSION), --dry-run (say what would happen, change nothing),
# --uninstall (remove the installed binary), --help.
#
#   KRISHIV_VERSION      release to install, e.g. 0.3.1 or v0.3.1 (default: the latest release)
#   KRISHIV_INSTALL_DIR  where to put `krishiv` (default: $HOME/.local/bin)
#   KRISHIV_COMPAT=1     install the compat build even when the CPU could run the fast one
#   KRISHIV_BASE_URL     where release files are served (default: the GitHub Release; for mirrors and tests)
#   KRISHIV_NO_ATTEST=1  skip the build attestation check that runs when `gh` is installed
#   KRISHIV_CPU_FLAGS    override the CPU flag list read from /proc/cpuinfo (tests)
set -eu

REPO="KrishivAI/krishiv-engine"

say() { printf '%s\n' "krishiv-install: $*" >&2; }
die() { say "error: $*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# The flags the x86-64-v3 build needs (the same list as the CPU check at the binary's start).
V3_FLAGS="avx avx2 bmi1 bmi2 fma f16c movbe abm"

fetch() { # url dest
  if have curl; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif have wget; then
    wget -q -O "$2" "$1"
  else
    die "need curl or wget"
  fi
}

sha256_of() {
  if have sha256sum; then
    sha256sum "$1" | cut -d ' ' -f 1
  elif have shasum; then
    shasum -a 256 "$1" | cut -d ' ' -f 1
  else
    die "need sha256sum or shasum to verify the download"
  fi
}

cpu_flags() {
  if [ -n "${KRISHIV_CPU_FLAGS+set}" ]; then
    printf '%s' "$KRISHIV_CPU_FLAGS"
  elif [ -r /proc/cpuinfo ]; then
    grep -m1 '^flags' /proc/cpuinfo | cut -d ':' -f 2
  fi
}

has_v3() {
  flags=" $(cpu_flags) "
  for f in $V3_FLAGS; do
    case "$flags" in *" $f "*) ;; *) return 1 ;; esac
  done
  return 0
}

# True on a musl libc system (Alpine): the Linux release binaries are glibc builds.
is_musl() {
  [ "${KRISHIV_LIBC:-}" = musl ] && return 0
  [ -n "${KRISHIV_LIBC:-}" ] && return 1
  for f in /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do
    [ -e "$f" ] && return 0
  done
  return 1
}

pick_artifact() {
  os=$(uname -s)
  arch=$(uname -m)
  case "$os" in
    MINGW* | MSYS* | CYGWIN*)
      die "Windows is not installed by this script: use  pip install krishiv  or  winget install Krishiv.Krishiv  (or scripts/install.ps1)" ;;
    Linux)
      if is_musl; then
        die "no musl (Alpine) build of the krishiv binary yet: use  pip install krishiv  (the musllinux wheel), the Docker image, or a glibc distribution"
      fi ;;
  esac
  case "$os/$arch" in
    Linux/x86_64 | Linux/amd64)
      if [ "${KRISHIV_COMPAT:-}" = 1 ] || ! has_v3; then
        echo krishiv-compat-linux-x86_64
      else
        echo krishiv-linux-x86_64
      fi ;;
    Linux/aarch64 | Linux/arm64) echo krishiv-linux-aarch64 ;;
    Darwin/arm64) echo krishiv-macos-arm64 ;;
    Darwin/x86_64) echo krishiv-macos-x86_64 ;;
    *) die "no build for $os $arch" ;;
  esac
}

usage() {
  cat >&2 <<'USAGE'
usage: install.sh [--version X.Y.Z] [--dry-run] [--uninstall]
  --version X.Y.Z  install that release (default: the latest)
  --dry-run        print what would be downloaded and installed; change nothing
  --uninstall      remove \$KRISHIV_INSTALL_DIR/krishiv (default \$HOME/.local/bin/krishiv)
Environment: KRISHIV_VERSION, KRISHIV_INSTALL_DIR, KRISHIV_COMPAT=1, KRISHIV_NO_ATTEST=1, KRISHIV_BASE_URL
USAGE
}

main() {
  dry=0
  uninstall=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --version) [ $# -ge 2 ] || die "--version needs a value"; KRISHIV_VERSION=$2; shift 2 ;;
      --version=*) KRISHIV_VERSION=${1#--version=}; shift ;;
      --dry-run) dry=1; shift ;;
      --uninstall) uninstall=1; shift ;;
      -h | --help) usage; exit 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  dir="${KRISHIV_INSTALL_DIR:-$HOME/.local/bin}"
  if [ "$uninstall" = 1 ]; then
    [ -e "$dir/krishiv" ] || die "$dir/krishiv is not installed"
    if [ "$dry" = 1 ]; then say "dry run: would remove $dir/krishiv"; return 0; fi
    rm -f "$dir/krishiv" || die "could not remove $dir/krishiv"
    say "removed $dir/krishiv"
    return 0
  fi
  artifact=$(pick_artifact)

  if [ -n "${KRISHIV_BASE_URL:-}" ]; then
    base="$KRISHIV_BASE_URL"
  elif [ -n "${KRISHIV_VERSION:-}" ]; then
    base="https://github.com/$REPO/releases/download/v${KRISHIV_VERSION#v}"
  else
    base="https://github.com/$REPO/releases/latest/download"
  fi

  if [ "$dry" = 1 ]; then
    say "dry run: would download $base/$artifact and $base/SHA256SUMS, verify the checksum, and install $dir/krishiv"
    return 0
  fi
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT INT TERM

  say "downloading $artifact"
  fetch "$base/$artifact" "$tmp/$artifact" || die "could not download $base/$artifact (no such release or no build of $artifact in it; pip install krishiv is the fallback)"
  fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" || die "could not download $base/SHA256SUMS"

  want=$(awk -v f="$artifact" '$2 == f || $2 == "*" f { print $1; exit }' "$tmp/SHA256SUMS")
  [ -n "$want" ] || die "SHA256SUMS has no entry for $artifact"
  got=$(sha256_of "$tmp/$artifact")
  [ "$want" = "$got" ] || die "checksum mismatch for $artifact (expected $want, got $got); nothing installed"
  say "checksum ok"

  if have gh && [ "${KRISHIV_NO_ATTEST:-}" != 1 ]; then
    gh attestation verify "$tmp/$artifact" --repo "$REPO" >/dev/null 2>&1 \
      || die "build attestation check failed for $artifact; nothing installed (set KRISHIV_NO_ATTEST=1 to skip it)"
    say "attestation ok"
  fi

  mkdir -p "$dir"
  chmod +x "$tmp/$artifact"
  # Move within the target directory first so a running `krishiv` is replaced atomically.
  cp "$tmp/$artifact" "$dir/.krishiv.new"
  mv -f "$dir/.krishiv.new" "$dir/krishiv"
  say "installed $dir/krishiv ($("$dir/krishiv" --version 2>/dev/null || echo 'version check failed'))"

  case ":$PATH:" in
    *":$dir:"*) ;;
    *) say "add it to your PATH:  export PATH=\"$dir:\$PATH\"" ;;
  esac
}

main "$@"
