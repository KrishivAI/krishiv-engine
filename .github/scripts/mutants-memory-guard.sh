#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Memory guard for the nightly mutation job. A mutant can turn a loop or a capacity into an unbounded
# allocation; within seconds the hosted runner (16 GB) freezes and the job ends "The operation was
# canceled" or "lost communication", with no mutant named and the whole slice lost. This loop watches
# MemAvailable and, below the threshold, kills the test binary (a process running from
# <root>/target/*/deps/) with the largest resident set and says which, so the mutant counts as caught
# (its test failed) and the slice goes on.
#
# Usage: mutants-memory-guard.sh ROOT [THRESHOLD_MB [INTERVAL_SECONDS]]
# Environment: MEMINFO (default /proc/meminfo) and ONCE=1 (one check, for the test).
set -u
root=${1:?usage: mutants-memory-guard.sh ROOT [THRESHOLD_MB [INTERVAL_SECONDS]]}
threshold_kb=$(( ${2:-3072} * 1024 ))
interval=${3:-1}
meminfo=${MEMINFO:-/proc/meminfo}

check() {
  local avail_kb pid rss cmd
  avail_kb=$(awk '/^MemAvailable:/ {print $2}' "$meminfo")
  [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$threshold_kb" ] || return 0
  # Largest resident set among the test binaries of this tree.
  # Only argv[0] counts: rustc and the linker mention target/*/deps in their arguments (--out-dir,
  # -L dependency=, -o) and are often the largest processes during a build; killing one would
  # silently drop the mutant (unviable) or end the baseline build.
  read -r pid rss cmd < <(ps -eo pid=,rss=,args= --sort=-rss | awk -v pat="$root/target/" \
    'index($3, pat) == 1 && $3 ~ /\/deps\/[^\/]+$/ {print; exit}')
  if [ -z "${pid:-}" ]; then
    echo "::warning::memory guard: $((avail_kb / 1024)) MB available and no test binary to kill"
    return 0
  fi
  echo "::warning::memory guard: $((avail_kb / 1024)) MB available; killing pid $pid ($((rss / 1024)) MB): ${cmd:0:200}"
  kill -KILL "$pid" 2>/dev/null || true
}

if [ "${ONCE:-}" = 1 ]; then
  check
else
  while sleep "$interval"; do check; done
fi
