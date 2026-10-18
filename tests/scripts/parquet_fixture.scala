// SPDX-License-Identifier: Apache-2.0
// Writes tests/fixtures/parquet/spark.parquet with Spark 4.2's own writer (defaults: INT96
// timestamps, snappy). Run by gen_parquet_fixtures.py's usage; deterministic content.
import java.math.BigDecimal
import java.sql.{Date, Timestamp}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

val schema = StructType(Seq(
  StructField("id", IntegerType, false),
  StructField("small", DecimalType(9, 2)),
  StructField("medium", DecimalType(18, 4)),
  StructField("large", DecimalType(38, 10)),
  StructField("ts", TimestampType),
  StructField("ntz", TimestampNTZType),
  StructField("d", DateType),
  StructField("s", StringType),
  StructField("st", StructType(Seq(StructField("x", IntegerType), StructField("y", StringType)))),
  StructField("arr", ArrayType(LongType)),
  StructField("m", MapType(StringType, IntegerType))
))
val rows = (0 until 500).map { i =>
  def n[T](v: T): Any = if (i % 9 == 4) null else v
  Row(
    i,
    n(new BigDecimal(i * 1234 - 300000).movePointLeft(2)),
    n(new BigDecimal(i.toLong * 98765431L - 7).movePointLeft(4)),
    n(new BigDecimal(BigInt(i).pow(9).bigInteger).add(BigDecimal.ONE).movePointLeft(10)),
    n(new Timestamp(i * 86400000L * 37 - 2000000000000L)),
    n(java.time.LocalDateTime.of(1990 + i % 30, 1 + i % 12, 1 + i % 28, i % 24, i % 60, i % 60)),
    n(Date.valueOf(java.time.LocalDate.of(1900 + i % 150, 1 + i % 12, 1 + i % 28))),
    n(s"row-$i-${"é" * (i % 4)}"),
    n(Row(i * 2, s"y$i")),
    n((0 until i % 4).map(k => (i * 10 + k).toLong)),
    n((0 until i % 3).map(k => s"k$k" -> (i + k)).toMap)
  )
}
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.createDataFrame(java.util.Arrays.asList(rows: _*), schema).coalesce(1)
  .write.mode("overwrite").option("parquet.block.size", 16384).parquet("/tmp/spark-fixture")
val part = new java.io.File("/tmp/spark-fixture").listFiles.filter(_.getName.endsWith(".parquet")).head
java.nio.file.Files.copy(part.toPath, java.nio.file.Paths.get("/out/spark.parquet"),
  java.nio.file.StandardCopyOption.REPLACE_EXISTING)
System.exit(0)
