// SPDX-License-Identifier: Apache-2.0
// Reads tests/fixtures/parquet/variant/krishiv-written.parquet (written by Krishiv, see
// crates/krishiv-session/tests/variant_parquet.rs) with Spark 4.2.0 and writes what Spark reads
// next to it, as variant_parquet_fixture.scala does for Spark's own files. Run (repo root):
//   docker run --rm -v "$PWD/tests/fixtures/parquet/variant:/out" -v "$PWD/tests/scripts:/s" apache/spark:4.2.0 \
//     /opt/spark/bin/spark-shell --conf spark.ui.enabled=false -i /s/variant_parquet_read.scala
import java.nio.file.{Files, Paths}

spark.conf.set("spark.sql.session.timeZone", "UTC")
val back = spark.read.parquet("/out/krishiv-written.parquet").selectExpr("id",
  "to_json(v) AS json", "try_variant_get(v, '$.a', 'bigint') AS a",
  "try_variant_get(v, '$.b', 'string') AS b", "CAST(v AS STRING) AS s",
  "schema_of_variant(v) AS schema").orderBy("id").collect()
def q(x: Any): String = if (x == null) "null" else "\"" + x.toString.flatMap {
  case '"' => "\\\""; case '\\' => "\\\\"; case '\n' => "\\n"; case c => c.toString } + "\""
val lines = back.map { r =>
  s"""  {"id": ${r.getLong(0)}, "json": ${q(r.get(1))}, "a": ${if (r.isNullAt(2)) "null" else r.getLong(2)}, "b": ${q(r.get(3))}, "s": ${q(r.get(4))}, "schema": ${q(r.get(5))}}"""
}
Files.write(Paths.get("/out/krishiv-written.json"), ("[\n" + lines.mkString(",\n") + "\n]\n").getBytes("UTF-8"))
Files.write(Paths.get("/out/krishiv-written.schema"), spark.read.parquet("/out/krishiv-written.parquet").schema.simpleString.getBytes("UTF-8"))
System.exit(0)
