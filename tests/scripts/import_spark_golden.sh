#!/usr/bin/env bash
# Vendor Spark's SQLQueryTestSuite golden files (P0-T22).
# Usage: tests/scripts/import_spark_golden.sh <tag>      e.g. v4.2.0
# Sparse-fetches sql/core/src/test/resources/sql-tests/{inputs,results} from apache/spark at <tag>
# into tests/spark-golden/<tag>/{inputs,results}, plus MANIFEST and NOTICE.
# Only v4.2.0 is committed; other tags are gitignored (see tests/spark-golden/.gitignore).
set -euo pipefail

TAG="${1:?usage: $0 <tag> (e.g. v4.2.0)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/tests/spark-golden/$TAG"
REPO="${SPARK_GIT_URL:-https://github.com/apache/spark.git}"
SUBDIR="sql/core/src/test/resources/sql-tests"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none --sparse "$REPO" "$WORK/spark"
git -C "$WORK/spark" sparse-checkout set --no-cone "/$SUBDIR/inputs/" "/$SUBDIR/results/"
SHA="$(git -C "$WORK/spark" rev-parse HEAD)"

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$WORK/spark/$SUBDIR/inputs" "$DEST/inputs"
cp -R "$WORK/spark/$SUBDIR/results" "$DEST/results"

FILES="$(find "$DEST/inputs" "$DEST/results" -type f | wc -l | tr -d ' ')"
SQL="$(find "$DEST/inputs" -type f -name '*.sql' | wc -l | tr -d ' ')"
OUT="$(find "$DEST/results" -type f -name '*.sql.out' | wc -l | tr -d ' ')"
cat > "$DEST/MANIFEST" <<EOM
tag: $TAG
commit: $SHA
source: $REPO ($SUBDIR/{inputs,results})
file_count: $FILES
sql_files: $SQL
out_files: $OUT
imported: $(date -u +%Y-%m-%d)
EOM
cat > "$DEST/NOTICE" <<'EON'
These files are copied unmodified from the Apache Spark project
(https://github.com/apache/spark, sql/core/src/test/resources/sql-tests), which is
licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
Copyright The Apache Software Foundation. See MANIFEST for the exact tag and commit.
They are used only as conformance test data for Krishiv; Apache Spark, Spark and
the Spark logo are trademarks of the Apache Software Foundation.
EON
echo "imported $FILES files ($SQL .sql, $OUT .sql.out) at $TAG ($SHA) -> $DEST"
