#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Shell test for .github/scripts/mutants-memory-guard.sh. Run: bash .github/scripts/test-mutants-memory-guard.sh
set -euo pipefail

guard="$(cd "$(dirname "$0")" && pwd)/mutants-memory-guard.sh"
tmp=$(mktemp -d)
trap 'pkill -P $$ 2>/dev/null || true; kill $(jobs -p) 2>/dev/null || true; rm -rf "$tmp"' EXIT
mkdir -p "$tmp/target/debug/deps"
# A test binary: argv[0] is a path under target/*/deps (`exec -a` sets it; a copied `sleep` would not
# work, multi-call coreutils pick the applet from argv[0]).
bash -c 'exec -a "$0" sleep 300' "$tmp/target/debug/deps/big_test" >/dev/null 2>&1 & big=$!
# A rustc-like process: the path is only in its arguments (`$0` of the shell), never argv[0].
bash -c 'sleep 300; :' "$tmp/target/debug/deps/rustc_like" >/dev/null 2>&1 & rustc_like=$!
sleep 300 >/dev/null 2>&1 & unrelated=$!
sleep 0.2

mem() { printf 'MemTotal: 16000000 kB\nMemAvailable: %s kB\n' "$1" > "$tmp/meminfo"; }

# Plenty of memory: nothing is killed.
mem 8000000
MEMINFO="$tmp/meminfo" ONCE=1 bash "$guard" "$tmp" 3072 1 > "$tmp/out"
kill -0 "$big" || { echo "FAIL: killed a process with memory to spare"; exit 1; }

# Below the threshold: the test binary of this tree dies, an unrelated process lives, and it is named.
mem 1000000
MEMINFO="$tmp/meminfo" ONCE=1 bash "$guard" "$tmp" 3072 1 > "$tmp/out"
sleep 0.2
if kill -0 "$big" 2>/dev/null; then echo "FAIL: the test binary survived"; exit 1; fi
kill -0 "$unrelated" || { echo "FAIL: killed an unrelated process"; exit 1; }
kill -0 "$rustc_like" || { echo "FAIL: killed a process that only mentions deps/ in its arguments"; exit 1; }
grep -q "killing pid $big" "$tmp/out" || { echo "FAIL: the kill was not logged: $(cat "$tmp/out")"; exit 1; }

# Nothing to kill: a warning, no failure.
MEMINFO="$tmp/meminfo" ONCE=1 bash "$guard" "$tmp" 3072 1 > "$tmp/out"
grep -q "no test binary to kill" "$tmp/out" || { echo "FAIL: no warning: $(cat "$tmp/out")"; exit 1; }
echo "ok"
