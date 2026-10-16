#!/usr/bin/env bash
# Build the UI and copy dist/ into crates/krishiv-ui-assets/assets/ (committed,
# so cargo build never needs Node; ADR 0024).
set -euo pipefail
cd "$(dirname "$0")/.."
pnpm install --frozen-lockfile
pnpm build
dest=../crates/krishiv-ui-assets/assets
rm -rf "$dest"
mkdir -p "$dest"
cp -R dist/. "$dest"/
echo "synced $(find "$dest" -type f | wc -l) files to $dest"
