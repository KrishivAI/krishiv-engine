#!/usr/bin/env bash
# Thin wrapper over `krishiv-datasets` (implementation: src/krishiv_bench/datasets.py).
#   datasets/gen.sh tpch --sf 1       datasets/gen.sh tpcds --sf 1
#   datasets/gen.sh clickbench --yes  (14 GB download; never run automatically)
#   datasets/gen.sh nexmark --events 10000000
#   datasets/gen.sh zipf --rows 100000000 --keys 1000000 --alpha 1.2
# Output: ${KRISHIV_DATA:-$HOME/krishiv-data}/<dataset>/..., sha256 manifests in .../manifests/
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run krishiv-datasets "$@"
