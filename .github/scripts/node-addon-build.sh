#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds the embedded engine's Node-API addon (crates/krishiv-node) with the `dist` profile and copies it to
# OUT/krishiv.node (publish-js.yml). On Linux it runs inside the PyPA manylinux_2_28 / musllinux_1_2 image, so the
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
    # A cdylib needs the dynamic C runtime; the musl target links it statically by default.
    if ldd --version 2>&1 | grep -qi musl; then
      export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=-crt-static"
    fi
    ;;
esac

cargo build -p krishiv-node --profile dist --locked
mkdir -p "$out"
target=${CARGO_TARGET_DIR:-target}
cp "$target/dist/$lib" "$out/krishiv.node"
ls -l "$out/krishiv.node"
if [ -n "$owner" ]; then
  chown -R "$owner" "$out" "$target"
fi
