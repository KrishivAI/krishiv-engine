#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds the embedded engine's Node-API addon (crates/krishiv-node) with the `dist` profile and copies it to
# OUT/krishiv.node (publish-js.yml), with OUT/THIRD-PARTY-NOTICES.txt beside it (the pack script requires it). On Linux it runs inside the PyPA manylinux_2_28 / musllinux_1_2 image, so the
# glibc floor is the Python wheels' (2.28); it installs the pinned toolchain (rust-toolchain.toml) when the image has
# none. On macOS it runs on the runner.
#
#   node-addon-build.sh OUT [OWNER_UID:GID]   (OWNER: chown OUT to it, for a build run as root in a container)
#
# The addon is built for the baseline CPU of each target (no target-cpu): it has no import-time CPU check as the
# wheel does, so a CPU-level build would die with SIGILL on an older CPU instead of failing with a message.
set -euo pipefail
out=$1
owner=${2:-}
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"

if ! command -v cargo >/dev/null 2>&1; then
  export CARGO_HOME=${CARGO_HOME:-/tmp/cargo} RUSTUP_HOME=${RUSTUP_HOME:-/tmp/rustup}
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
  export PATH="$CARGO_HOME/bin:$PATH"
fi
# The toolchain rust-toolchain.toml names, installed explicitly (rustup's install on first use is not on in every
# rustup release).
channel=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
rustup toolchain install "$channel" --profile minimal
rustc --version

case "$(uname -s)" in
  Darwin)
    export MACOSX_DEPLOYMENT_TARGET=11.0
    lib=libkrishiv_node.dylib
    ;;
  *)
    lib=libkrishiv_node.so
    # A cdylib needs the dynamic C runtime; the musl target links it statically by default. The host
    # triple decides `crt-static` (musl's `ldd --version` exits 1, which `pipefail` would turn into a miss).
    # `sed` reads rustc's whole output, so no early exit can end the pipe with EPIPE.
    case "$(rustc -vV | sed -n 's/^host: //p')" in
      *-musl) export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=-crt-static" ;;
    esac
    ;;
esac

cargo build -p krishiv-node --profile dist --locked
mkdir -p "$out"
target=${CARGO_TARGET_DIR:-target}
cp "$target/dist/$lib" "$out/krishiv.node"
ls -l "$out/krishiv.node"
# A dlopen'ed library marked STATIC_TLS (initial-exec TLS somewhere in it) loads only while its whole TLS block fits
# glibc's small static-TLS surplus, and fails with "cannot allocate memory in static TLS block" once it grows.
# Checked on glibc (the manylinux images carry readelf; musllinux has none, and macOS has no ELF to check).
host=$(rustc -vV | sed -n 's/^host: //p')
if [[ $host == *-linux-gnu ]]; then
  # Read first, so a missing readelf fails the build (`set -e`) instead of passing the check.
  dynamic=$(readelf -dW "$out/krishiv.node")
  if grep -q STATIC_TLS <<<"$dynamic"; then
    echo "krishiv.node is marked STATIC_TLS: a dependency uses initial-exec TLS (mimalloc needs local_dynamic_tls)" >&2
    exit 1
  fi
fi
# The crates linked into the addon, with their license texts. Written here, where cargo is known to be: on Linux
# the runner outside the PyPA image may have none.
python3 -I .github/scripts/third_party_notices.py --package krishiv-node --target "$host" \
  --title "the krishiv Node-API addon ($host)" --out "$out/THIRD-PARTY-NOTICES.txt"
if [ -n "$owner" ]; then
  chown -R "$owner" "$out" "$target"
fi
