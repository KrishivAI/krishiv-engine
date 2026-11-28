#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails when the extension module of a Linux wheel in DIST is marked STATIC_TLS.
#
#   wheel-tls-check.sh DIST
#
# Python loads the module with dlopen. A library marked STATIC_TLS (initial-exec TLS somewhere in it, mimalloc's
# default) loads on glibc only while its whole TLS block fits glibc's small static-TLS surplus, and fails with
# "cannot allocate memory in static TLS block" once it grows past it (or when other modules took the surplus first);
# musl crashed at import. The module must use the dynamic TLS models (mimalloc's `local_dynamic_tls`).
set -euo pipefail
shopt -s nullglob
dist=$1
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
checked=0
for wheel in "$dist"/*.whl; do
  rm -rf "$work/w"
  mkdir "$work/w"
  unzip -q "$wheel" 'krishiv/*.so' -d "$work/w"
  for so in "$work"/w/krishiv/*.so; do
    # Read first, so a missing readelf fails the check (`set -e`) instead of passing it.
    dynamic=$(readelf -dW "$so")
    if grep -q STATIC_TLS <<<"$dynamic"; then
      echo "$(basename "$wheel"): $(basename "$so") is marked STATIC_TLS: a dependency uses initial-exec TLS" \
        "(mimalloc needs local_dynamic_tls)" >&2
      exit 1
    fi
    checked=$((checked + 1))
  done
done
if [ "$checked" = 0 ]; then
  echo "no extension module found in $dist/*.whl" >&2
  exit 1
fi
echo "$checked extension module(s) without STATIC_TLS"
