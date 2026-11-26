#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# A dry run of db-benchmark (https://github.com/duckdblabs/db-benchmark) with its own launcher, for the duckdb and
# krishiv solutions, in a container. One data set at a time: the benchmark's R generator writes the CSVs, the
# benchmark's run.sh runs both solutions on them, the CSVs are deleted. Results add up in <work>/db-benchmark/
# time.csv and logs.csv; compare the answers afterwards with
#   uv run python -m krishiv_bench.dbbench_report --ours <work>/db-benchmark/time.csv --check duckdb
#
#   bench/dbbench/dryrun.sh <work dir> <rows: 1e7|1e8|1e9> [groupby|join]...
#
# Environment:
#   KRISHIV_WHEEL         a wheel file to install instead of the released package (copied into the work dir)
#   KRISHIV_MEMORY_LIMIT  krishiv.memory.limit of the solution (default: the engine's own default)
#   DBB_RUN_MEMORY        memory of the container that runs the solutions (default 16g); the answer a solution
#                         holds is outside the engine's limit, so keep the limit well below this
#                         (measured with 0.3.0: the 1e8 join with an 8g limit reaches 16g and one set is
#                         killed there; give the 1e8 join 24g)
#   DBB_GEN_MEMORY        memory of the container that generates data (default 24g; 1e9 needs far more: the R
#                         generator holds the whole table)
#   DBB_COMMIT            the benchmark commit to check out (default: the one below)
#   DBB_DATA_DIR          a directory that already holds the CSVs (the benchmark's file names): nothing is
#                         generated or deleted, and a data set whose CSV is missing is skipped
#   DBB_DATA              only data sets whose name matches this extended regular expression
# Disk: a 1e8 group-by CSV is 5 GB, a 1e9 one about 50 GB; each solution keeps its own copy while it runs.
set -euo pipefail

work=${1:?usage: dryrun.sh <work dir> <rows> [groupby|join]...}
rows=${2:?usage: dryrun.sh <work dir> <rows> [groupby|join]...}
shift 2
tasks=("$@")
[[ ${#tasks[@]} -gt 0 ]] || tasks=(groupby join)
case "$rows" in 1e7 | 1e8 | 1e9) ;; *) echo "rows must be 1e7, 1e8 or 1e9" >&2; exit 2 ;; esac

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image=${DBB_IMAGE:-krishiv-dbbench:1}
commit=${DBB_COMMIT:-2f05a6f}
mkdir -p "$work"
work=$(cd "$work" && pwd)
board=$work/db-benchmark

docker image inspect "$image" >/dev/null 2>&1 || docker build -q -t "$image" "$here"

wheel=
if [[ -n ${KRISHIV_WHEEL:-} ]]; then
  cp -f "$KRISHIV_WHEEL" "$work/"
  wheel=/dbb/$(basename "$KRISHIV_WHEEL")
fi

data_mount=()
if [[ -n ${DBB_DATA_DIR:-} ]]; then data_mount=(-v "$(cd "$DBB_DATA_DIR" && pwd):/work/data:ro"); fi

# box <memory> <command...>: run a command in the container, in the benchmark checkout, as this user.
box() {
  local mem=$1
  shift
  docker run --rm --user "$(id -u):$(id -g)" --memory "$mem" --memory-swap "$mem" \
    -v "$board:/work" -v "$work:/dbb:ro" "${data_mount[@]}" -w /work \
    -e HOME=/work/.home -e IGNORE_SWAP=true -e TEST_RUN=true -e TEST_MOUNT_DIR=/work/mount \
    -e MACHINE_TYPE=c6id.metal -e KRISHIV_WHEEL="$wheel" -e KRISHIV_ON_DISK -e KRISHIV_MEMORY_LIMIT \
    "$image" "$@"
}

if [[ ! -d $board ]]; then
  git clone -q https://github.com/duckdblabs/db-benchmark "$board"
  git -C "$board" checkout -q "$commit"
  (cd "$here/.." && uv run -q python -m krishiv_bench.dbbench_solution --checkout "$board")
  sed -i -e 's/^export RUN_SOLUTIONS=.*/export RUN_SOLUTIONS="duckdb krishiv"/' \
    -e 's/^export DO_REPORT=.*/export DO_REPORT=false/' -e 's/^export DO_PUBLISH=.*/export DO_PUBLISH=false/' \
    -e 's/^export FORCE_RUN=.*/export FORCE_RUN=true/' "$board/run.conf"
  mkdir -p "$board/.home" "$board/mount" "$board/data"
  # The checkout carries the published results; the dry run's files start with the header alone.
  for f in time logs; do
    mv "$board/$f.csv" "$board/$f.published.csv"
    head -1 "$board/$f.published.csv" >"$board/$f.csv"
  done
  box 4g ./duckdb/setup-duckdb.sh
  box 4g ./krishiv/setup-krishiv.sh
fi

# one <task> <generator> <data set> <nrow,k,na,sort> <generator arguments...>
one() {
  local task=$1 gen=$2 name=$3 fields=$4
  shift 4
  if [[ -n ${DBB_DATA:-} && ! $name =~ $DBB_DATA ]]; then return 0; fi
  echo "=== $name $(date +%H:%M:%S), free $(df -BG --output=avail "$work" | tail -1 | tr -d ' ')"
  if [[ -n ${DBB_DATA_DIR:-} ]]; then
    # A join reads three more tables, named as the benchmark's helpers name them (rows / 1e6, / 1e3 and rows).
    local need=("$name") e=${rows#1e} f
    if [[ $task == join ]]; then need+=("${name/NA/1e$((e - 6))}" "${name/NA/1e$((e - 3))}" "${name/NA/$rows}"); fi
    for f in "${need[@]}"; do
      if [[ ! -f $DBB_DATA_DIR/$f.csv ]]; then
        echo "SKIPPED: no $f.csv in $DBB_DATA_DIR"
        return 0
      fi
    done
  elif ! box "${DBB_GEN_MEMORY:-24g}" bash -c "cd data && Rscript ../_data/$gen $*"; then
    echo "GENERATOR FAILED: $name"
    rm -f "$board"/data/*_"${rows}"_*.csv
    failed=1
    return 0
  fi
  sed -i "s/^export RUN_TASKS=.*/export RUN_TASKS=\"$task\"/" "$board/run.conf"
  printf 'task,data,nrow,k,na,sort,active\n%s,%s,%s,1\n' "$task" "$name" "$fields" >"$board/_control/data.csv"
  # The launcher removes a solution's .ret only when it starts that solution, and the checkout outlives this run:
  # a .ret of an earlier run must not stand in for a solution that never started.
  rm -f "$board/run.lock" "$board"/out/run_*_"${task}"_"${name}".ret
  box "${DBB_RUN_MEMORY:-16g}" ./run.sh 2>&1 | grep -E '^(start|finish|# Bench)|error|Error|timed out|Killed' || true
  for s in duckdb krishiv; do
    if [[ $(cat "$board/out/run_${s}_${task}_${name}.ret" 2>/dev/null) != 0 ]]; then
      echo "FAILED: $s $task $name (see $board/out/run_${s}_${task}_${name}.err)"
      failed=1
    fi
  done
  [[ -n ${DBB_DATA_DIR:-} ]] || rm -f "$board"/data/*_"${rows}"_*.csv
  rm -rf "$board"/mount/*
}

failed=0
for task in "${tasks[@]}"; do
  case "$task" in
    groupby)
      one groupby groupby-datagen.R "G1_${rows}_1e2_0_0" "$rows,1e2,0,0" "$rows" 1e2 0 0
      one groupby groupby-datagen.R "G1_${rows}_1e1_0_0" "$rows,1e1,0,0" "$rows" 1e1 0 0
      one groupby groupby-datagen.R "G1_${rows}_2e0_0_0" "$rows,2e0,0,0" "$rows" 2e0 0 0
      one groupby groupby-datagen.R "G1_${rows}_1e2_0_1" "$rows,1e2,0,1" "$rows" 1e2 0 1
      one groupby groupby-datagen.R "G1_${rows}_1e2_5_0" "$rows,1e2,5,0" "$rows" 1e2 5 0
      ;;
    join)
      one join join-datagen.R "J1_${rows}_NA_0_0" "$rows,NA,0,0" "$rows" 0 0 0
      one join join-datagen.R "J1_${rows}_NA_5_0" "$rows,NA,5,0" "$rows" 0 5 0
      one join join-datagen.R "J1_${rows}_NA_0_1" "$rows,NA,0,1" "$rows" 0 0 1
      ;;
    *) echo "unknown task $task" >&2; exit 2 ;;
  esac
done
echo "=== done $(date +%H:%M:%S), failed=$failed"
exit "$failed"
