// SPDX-License-Identifier: Apache-2.0
// Writes the VARIANT Parquet fixtures (P2-T001c) with Spark 4.2.0's own writer and the values
// Spark reads back from each file, into tests/fixtures/parquet/variant/. Run (from the repo root):
//   docker run --rm -v "$PWD/tests/fixtures/parquet/variant:/out" -v "$PWD/tests/scripts:/s" apache/spark:4.2.0 \
//     /opt/spark/bin/spark-shell --conf spark.ui.enabled=false -i /s/variant_parquet_fixture.scala
// Deterministic content.
import java.nio.file.{Files, Paths, StandardCopyOption}

spark.conf.set("spark.sql.session.timeZone", "UTC")
val rowsSql = """
SELECT id, CASE id % 10
  WHEN 0 THEN parse_json(format_string('{"a":%d,"b":"s%d"}', id, id))
  WHEN 1 THEN parse_json(format_string('{"a":"x%d","b":"s%d","c":[1,2,{"d":%d}]}', id, id, id))
  WHEN 2 THEN parse_json(format_string('{"b":"s%d"}', id))
  WHEN 3 THEN NULL
  WHEN 4 THEN parse_json('null')
  WHEN 5 THEN parse_json(format_string('[%d, 1.50, "q", null]', id))
  WHEN 6 THEN to_variant_object(named_struct('a', id, 'b', CAST(id AS STRING),
      'd', date_add(DATE'2024-01-01', CAST(id AS INT)),
      't', TIMESTAMP'2024-01-01 10:00:00.5' + make_interval(0, 0, 0, 0, 0, 0, id),
      'n', CAST(TIMESTAMP'2024-01-01 10:00:00' AS TIMESTAMP_NTZ), 'bin', X'0102ff',
      'dec', CAST(id AS DECIMAL(20, 3)), 'f', CAST(id AS FLOAT) / 4, 'flag', id % 4 = 2))
  WHEN 7 THEN CAST(CAST(id AS DECIMAL(12, 2)) / 7 AS VARIANT)
  WHEN 8 THEN parse_json(format_string('{"a":%d,"b":null,"e":{"f":1e10,"g":[true]}}', id))
  ELSE parse_json(format_string('{"a":%d,"b":"%s","é":%d}', id * 1000000000000, repeat('z', 70), id))
END AS v FROM range(0, 300)
"""
def write(name: String, confs: Seq[(String, String)]): Unit = {
  confs.foreach { case (k, v) => spark.conf.set(k, v) }
  val dir = s"/tmp/variant-$name"
  spark.sql(rowsSql).coalesce(1).write.mode("overwrite").option("parquet.block.size", 16384).parquet(dir)
  confs.foreach { case (k, _) => spark.conf.unset(k) }
  val part = new java.io.File(dir).listFiles().filter(_.getName.endsWith(".parquet")).head
  Files.copy(part.toPath, Paths.get(s"/out/$name.parquet"), StandardCopyOption.REPLACE_EXISTING)
  // What Spark reads back from the file it wrote.
  val back = spark.read.parquet(s"/out/$name.parquet").selectExpr("id",
    "to_json(v) AS json", "try_variant_get(v, '$.a', 'bigint') AS a",
    "try_variant_get(v, '$.b', 'string') AS b", "CAST(v AS STRING) AS s",
    "schema_of_variant(v) AS schema").orderBy("id").collect()
  def q(x: Any): String = if (x == null) "null" else "\"" + x.toString.flatMap {
    case '"' => "\\\""; case '\\' => "\\\\"; case '\n' => "\\n"; case c => c.toString } + "\""
  val lines = back.map { r =>
    s"""  {"id": ${r.getLong(0)}, "json": ${q(r.get(1))}, "a": ${if (r.isNullAt(2)) "null" else r.getLong(2)}, "b": ${q(r.get(3))}, "s": ${q(r.get(4))}, "schema": ${q(r.get(5))}}"""
  }
  Files.write(Paths.get(s"/out/$name.json"), ("[\n" + lines.mkString(",\n") + "\n]\n").getBytes("UTF-8"))
}
write("unshredded", Seq())
write("unshredded-value-first", Seq("spark.sql.variant.writeShredding.enabled" -> "false"))
write("unannotated", Seq("spark.sql.parquet.variant.annotateLogicalType.enabled" -> "false",
  "spark.sql.variant.writeShredding.enabled" -> "false"))
write("shredded", Seq("spark.sql.variant.forceShreddingSchemaForTest" -> "a bigint, b string"))
write("shredded-nested", Seq("spark.sql.variant.forceShreddingSchemaForTest" ->
  "a bigint, b string, e struct<f double, g array<boolean>>, c array<variant>"))
// The row-6 objects shred these types (dates, timestamps, decimals, floats, binary, ints).
write("shredded-types", Seq("spark.sql.variant.forceShreddingSchemaForTest" ->
  "a int, b string, d date, t timestamp, n timestamp_ntz, dec decimal(20, 3), f float, bin binary, flag boolean"))
// Variants inside a struct, an array and a map value: found through the footer's Spark schema.
def nestedColumns(): Unit = {
  val dir = "/tmp/variant-nested-columns"
  spark.sql(rowsSql).selectExpr("id", "named_struct('x', v, 'n', id) AS s", "array(v, NULL) AS a",
    "map('k', v) AS m").coalesce(1).write.mode("overwrite").parquet(dir)
  val part = new java.io.File(dir).listFiles().filter(_.getName.endsWith(".parquet")).head
  Files.copy(part.toPath, Paths.get("/out/nested-columns.parquet"), StandardCopyOption.REPLACE_EXISTING)
  val back = spark.read.parquet("/out/nested-columns.parquet").selectExpr("id",
    "to_json(s.x) AS sx", "to_json(a[0]) AS a0", "to_json(m['k']) AS mk").orderBy("id").collect()
  def q(x: Any): String = if (x == null) "null" else "\"" + x.toString.flatMap {
    case '"' => "\\\""; case '\\' => "\\\\"; case c => c.toString } + "\""
  val lines = back.map(r => s"""  {"id": ${r.getLong(0)}, "sx": ${q(r.get(1))}, "a0": ${q(r.get(2))}, "mk": ${q(r.get(3))}}""")
  Files.write(Paths.get("/out/nested-columns.json"), ("[\n" + lines.mkString(",\n") + "\n]\n").getBytes("UTF-8"))
}
nestedColumns()
System.exit(0)
