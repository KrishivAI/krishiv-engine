// SPDX-License-Identifier: Apache-2.0
// Reads every Krishiv-written file in /in with Spark 4.2 and writes, for each, the Spark types it
// read (/out/<variant>.schema) and a copy written by Spark's own writer (/out/<variant>.parquet,
// microsecond timestamps), so check_parquet_written.py can hash exactly what Spark read.
import java.io.File
import java.nio.file.{Files, Paths, StandardCopyOption}

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
for (f <- new File("/in").listFiles.filter(_.getName.endsWith(".parquet")).sortBy(_.getName)) {
  val variant = f.getName.stripSuffix(".parquet")
  val df = spark.read.parquet(f.getPath)
  Files.write(Paths.get(s"/out/$variant.schema"), df.schema.simpleString.getBytes("UTF-8"))
  val tmp = s"/tmp/written-$variant"
  df.coalesce(1).write.mode("overwrite").parquet(tmp)
  val part = new File(tmp).listFiles.filter(_.getName.endsWith(".parquet")).head
  Files.copy(part.toPath, Paths.get(s"/out/$variant.parquet"), StandardCopyOption.REPLACE_EXISTING)
}
System.exit(0)
