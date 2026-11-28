// SPDX-License-Identifier: Apache-2.0
// The Spark 4.3.0 files of tests/fixtures/parquet/nanos (see gen_parquet_nanos_fixtures.py):
// nanosecond timestamps written with spark.sql.timestampNanosTypes.enabled.
import java.nio.file.{Files, Paths, StandardCopyOption}
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.timestampNanosTypes.enabled", "true")
def write(name: String, q: String): Unit = {
  val tmp = Files.createTempDirectory(s"nanos-$name")
  val dir = tmp.resolve("out").toString
  spark.sql(q).coalesce(1).write.parquet(dir)
  val part = new java.io.File(dir).listFiles().filter(_.getName.endsWith(".parquet")).head
  Files.copy(part.toPath, Paths.get(s"/out/$name.parquet"), StandardCopyOption.REPLACE_EXISTING)
  org.apache.commons.io.FileUtils.deleteDirectory(tmp.toFile)
}
write("spark-flat", "SELECT CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9)) AS n9, " +
  "CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9)) AS l9, " +
  "CAST('2020-01-01 00:00:00.1234567' AS TIMESTAMP_NTZ(7)) AS n7")
write("spark-nested", "SELECT array(CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9))) AS a, " +
  "named_struct('x', CAST('2020-01-01 00:00:00.1234567' AS TIMESTAMP_LTZ(7))) AS s")
System.exit(0)
