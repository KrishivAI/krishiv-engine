#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Tests scripts/install.sh against a fake release directory (file:// URLs): artifact choice per CPU,
# checksum verification, refusal on a mismatch, version selection. Runs on Linux and macOS.
# shellcheck disable=SC2015,SC2016 # "A && echo ok || fail" is the test idiom; the sh -c bodies are single-quoted on purpose
set -eu
here=$(cd "$(dirname "$0")" && pwd)
script="$here/../../scripts/install.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fails=0
# The fake binaries run anywhere; pretend glibc so an Alpine host still exercises the other paths.
export KRISHIV_LIBC=glibc

sum() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }

rel="$work/rel"
mkdir -p "$rel"
for a in krishiv-linux-x86_64 krishiv-compat-linux-x86_64 krishiv-linux-aarch64 krishiv-macos-arm64; do
  printf '#!/bin/sh\necho %s\n' "$a" > "$rel/$a"
done
(cd "$rel" && for f in krishiv-*; do sum "$f"; done > SHA256SUMS)

check() { # name, expected-exit(0|1), expected output of installed binary (or "-")
  name=$1; want_rc=$2; want_out=$3; shift 3
  dest="$work/bin-$name"
  rc=0
  env KRISHIV_NO_ATTEST=1 KRISHIV_BASE_URL="file://$rel" KRISHIV_INSTALL_DIR="$dest" "$@" sh "$script" >"$work/log-$name" 2>&1 || rc=$?
  if [ "$rc" != "$want_rc" ]; then echo "FAIL $name: exit $rc, wanted $want_rc"; cat "$work/log-$name"; fails=$((fails + 1)); return; fi
  if [ "$want_out" = "-" ]; then
    [ ! -e "$dest/krishiv" ] || { echo "FAIL $name: installed despite refusal"; fails=$((fails + 1)); return; }
  else
    got=$("$dest/krishiv")
    [ "$got" = "$want_out" ] || { echo "FAIL $name: installed $got, wanted $want_out"; fails=$((fails + 1)); return; }
  fi
  echo "ok $name"
}

os=$(uname -s); arch=$(uname -m)
v3="avx avx2 bmi1 bmi2 fma f16c movbe abm sse4_2"
if [ "$os" = Linux ] && { [ "$arch" = x86_64 ] || [ "$arch" = amd64 ]; }; then
  check v3-cpu 0 krishiv-linux-x86_64 KRISHIV_CPU_FLAGS="$v3"
  check old-cpu 0 krishiv-compat-linux-x86_64 KRISHIV_CPU_FLAGS="sse4_2 avx"
  check no-flags 0 krishiv-compat-linux-x86_64 KRISHIV_CPU_FLAGS=""
  check forced-compat 0 krishiv-compat-linux-x86_64 KRISHIV_CPU_FLAGS="$v3" KRISHIV_COMPAT=1
  want=krishiv-linux-x86_64
elif [ "$os" = Linux ]; then
  want=krishiv-linux-aarch64
elif [ "$os/$arch" = Darwin/arm64 ]; then
  want=krishiv-macos-arm64
else
  echo "skip: no build for $os $arch"; exit 0
fi
check host 0 "$want" KRISHIV_CPU_FLAGS="$v3"

# A tampered binary is refused and nothing is installed.
cp -R "$rel" "$work/rel-bad"
printf 'tampered\n' >> "$work/rel-bad/$want"
rel="$work/rel-bad"
check mismatch 1 - KRISHIV_CPU_FLAGS="$v3"
grep -q 'checksum mismatch' "$work/log-mismatch" || { echo "FAIL mismatch: no message"; fails=$((fails + 1)); }

# A SHA256SUMS without the artifact is refused.
rel="$work/rel-missing"; mkdir -p "$rel"; cp "$work/rel/$want" "$rel/"; : > "$rel/SHA256SUMS"
check no-entry 1 - KRISHIV_CPU_FLAGS="$v3"

# A version is turned into the release download URL (no network: only the URL is checked).
out=$(env KRISHIV_VERSION=v0.3.1 KRISHIV_INSTALL_DIR="$work/bin-ver" KRISHIV_CPU_FLAGS="$v3" PATH="$work/fakebin:$PATH" sh -c '
  mkdir -p "$0/fakebin"; printf "#!/bin/sh\necho \"\$@\" >> \"$0/curl.log\"; exit 22\n" > "$0/fakebin/curl"; chmod +x "$0/fakebin/curl"
  PATH="$0/fakebin:$PATH" sh "$1" 2>&1 || true; cat "$0/curl.log"' "$work" "$script")
case "$out" in *"releases/download/v0.3.1/"*) echo "ok version-url" ;; *) echo "FAIL version-url: $out"; fails=$((fails + 1)) ;; esac

# --dry-run changes nothing; --version selects the release; unknown flags are refused.
rel="$work/rel"
rc=0; sh "$script" --dry-run --version 0.3.1 KRISHIV_X >/dev/null 2>&1 || rc=$?
[ "$rc" != 0 ] && echo "ok unknown-arg" || { echo "FAIL unknown-arg"; fails=$((fails + 1)); }
out=$(KRISHIV_CPU_FLAGS="$v3" KRISHIV_INSTALL_DIR="$work/bin-dry" sh "$script" --dry-run --version 0.3.1 2>&1)
case "$out" in *"releases/download/v0.3.1/"*) [ ! -e "$work/bin-dry" ] && echo "ok dry-run" || { echo "FAIL dry-run: created dir"; fails=$((fails + 1)); } ;; *) echo "FAIL dry-run: $out"; fails=$((fails + 1)) ;; esac

# With gh present the attestation is checked: a failing check installs nothing, KRISHIV_NO_ATTEST=1 skips it.
mkdir -p "$work/ghbin"; printf '#!/bin/sh\nexit 1\n' > "$work/ghbin/gh"; chmod +x "$work/ghbin/gh"
rc=0; env PATH="$work/ghbin:$PATH" KRISHIV_NO_ATTEST= KRISHIV_BASE_URL="file://$rel" KRISHIV_INSTALL_DIR="$work/bin-att" KRISHIV_CPU_FLAGS="$v3" sh "$script" >"$work/log-att" 2>&1 || rc=$?
if [ "$rc" != 0 ] && [ ! -e "$work/bin-att/krishiv" ] && grep -q 'attestation' "$work/log-att"; then echo "ok attestation-refused"; else echo "FAIL attestation-refused"; fails=$((fails + 1)); fi
rc=0; env PATH="$work/ghbin:$PATH" KRISHIV_NO_ATTEST=1 KRISHIV_BASE_URL="file://$rel" KRISHIV_INSTALL_DIR="$work/bin-att2" KRISHIV_CPU_FLAGS="$v3" sh "$script" >/dev/null 2>&1 || rc=$?
[ "$rc" = 0 ] && [ -e "$work/bin-att2/krishiv" ] && echo "ok attestation-skipped" || { echo "FAIL attestation-skipped"; fails=$((fails + 1)); }

# --uninstall removes the binary; a second one fails loudly; --dry-run keeps it.
d="$work/bin-un"; mkdir -p "$d"; printf x > "$d/krishiv"
KRISHIV_INSTALL_DIR="$d" sh "$script" --uninstall --dry-run >/dev/null 2>&1 && [ -e "$d/krishiv" ] && echo "ok uninstall-dry-run" || { echo "FAIL uninstall-dry-run"; fails=$((fails + 1)); }
KRISHIV_INSTALL_DIR="$d" sh "$script" --uninstall >/dev/null 2>&1 && [ ! -e "$d/krishiv" ] && echo "ok uninstall" || { echo "FAIL uninstall"; fails=$((fails + 1)); }
rc=0; KRISHIV_INSTALL_DIR="$d" sh "$script" --uninstall >/dev/null 2>&1 || rc=$?
[ "$rc" != 0 ] && echo "ok uninstall-missing" || { echo "FAIL uninstall-missing"; fails=$((fails + 1)); }

# musl (Alpine) is refused with the pip/Docker advice on Linux.
if [ "$os" = Linux ]; then
  rc=0; env KRISHIV_LIBC=musl KRISHIV_BASE_URL="file://$rel" KRISHIV_INSTALL_DIR="$work/bin-musl" sh "$script" >"$work/log-musl" 2>&1 || rc=$?
  if [ "$rc" != 0 ] && grep -q 'musl' "$work/log-musl" && [ ! -e "$work/bin-musl/krishiv" ]; then echo "ok musl-refused"; else echo "FAIL musl-refused"; fails=$((fails + 1)); fi
fi

# On a real musl host (the Alpine container) the script itself must detect it, with no override.
for f in /lib/ld-musl-*.so.1; do
  if [ -e "$f" ]; then
    rc=0; env -u KRISHIV_LIBC KRISHIV_BASE_URL="file://$rel" KRISHIV_INSTALL_DIR="$work/bin-realmusl" sh "$script" >"$work/log-realmusl" 2>&1 || rc=$?
    if [ "$rc" != 0 ] && grep -q 'musl' "$work/log-realmusl"; then echo "ok musl-detected"; else echo "FAIL musl-detected"; fails=$((fails + 1)); fi
  fi
  break
done

[ "$fails" = 0 ] || { echo "$fails failure(s)"; exit 1; }
echo "all install tests passed"
