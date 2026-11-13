#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# ClickBench for Krishiv. Prints, as ClickBench requires:
#   Load time: <seconds>
#   Data size: <bytes>
#   43 lines of the form [first, second, third],     (null for a failed run)
#
# Inside a ClickBench checkout (../lib/benchmark-common.sh exists) this hands over to ClickBench's own driver, which
# also measures startup and concurrency. Standalone (this repository) it runs the loop below: each query three times,
# the first run cold (page cache dropped before it; Krishiv is embedded, so every run is also a fresh process).
#
#   HITS_PARQUET=/path/to/hits.parquet   use this file instead of downloading ~14 GB
#   KRISHIV_VERSION=0.2.0 | KRISHIV_WHEEL=<path or URL>   what `install` installs
#   BENCH_TRIES=3                        runs per query
set -uo pipefail
cd "$(dirname "$0")"

if [ -f ../lib/benchmark-common.sh ]; then
    export BENCH_DOWNLOAD_SCRIPT=download-hits-parquet-single BENCH_RESTARTABLE=no
    export BENCH_CONCURRENT_DURATION="${BENCH_CONCURRENT_DURATION:-0}"
    exec ../lib/benchmark-common.sh
fi

TRIES="${BENCH_TRIES:-3}"
./install || exit 1

if [ ! -e hits.parquet ]; then
    if [ -n "${HITS_PARQUET:-}" ]; then ln -s "$(readlink -f "$HITS_PARQUET")" hits.parquet
    else wget --continue --progress=dot:giga https://datasets.clickhouse.com/hits_compatible/hits.parquet || exit 1
    fi
fi

./start
LOAD_START=$(date +%s.%N)
./load || exit 1
LOAD_END=$(date +%s.%N)
awk -v a="$LOAD_START" -v b="$LOAD_END" 'BEGIN { printf "Load time: %.3f\n", b - a }'
echo "Data size: $(./data-size)"

COLD=yes
if ! { sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }; then
    echo "warning: cannot drop the page cache (needs sudo); the first runs are not cold, tag the result no-cold" >&2
    COLD=no
fi

while IFS= read -r query; do
    [ -z "$query" ] && continue
    if [ "$COLD" = yes ]; then sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; fi
    times=()
    for i in $(seq 1 "$TRIES"); do
        err=$(printf '%s\n' "$query" | ./query 2>&1 >/dev/null)
        status=$?
        t=$(printf '%s\n' "$err" | tr '\r' '\n' | grep -E '^[0-9]+(\.[0-9]+)?$' | tail -n1)
        if [ "$status" -eq 0 ] && [ -n "$t" ]; then times+=("$t"); else times+=("null"); printf '%s\n' "$err" | tail -n 5 >&2; fi
    done
    line="["
    for i in "${!times[@]}"; do [ "$i" -gt 0 ] && line+=", "; line+="${times[$i]}"; done
    echo "$line],"
done < queries.sql
./stop
