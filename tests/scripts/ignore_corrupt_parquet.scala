// SPDX-License-Identifier: Apache-2.0
// The Spark side of crates/krishiv-session/tests/ignore_corrupt_parquet.rs (RS-26): good and truncated Parquet
// files in both name orders and a bad file alone, under spark.sql.files.ignoreCorruptFiles and
// spark.sql.parquet.mergeSchema, and the per-read ignoreCorruptFiles option against the session conf. Each run
// writes its layouts under a fresh directory of its own in /w. Run (Spark 4.2.0), with <dir> writable by the container:
//   docker run --rm -i -e HOME=/tmp -v <dir>:/w apache/spark:4.2.0 /opt/spark/bin/spark-shell --master 'local[1]' \
//     < ignore_corrupt_parquet.scala 2>&1 | grep -o '@@.*'
import java.nio.file._
val root = Files.createTempDirectory(Paths.get("/w"), "rs26-")
val w = root.toString
spark.range(0, 1000).selectExpr("id", "cast(id as string) as s").coalesce(1).write.mode("overwrite").parquet(s"$w/src")
val good = Files.list(Paths.get(s"$w/src")).filter(p => p.toString.endsWith(".parquet")).findFirst.get
val bytes = Files.readAllBytes(good)
def mk(dir: String, files: Seq[(String, Array[Byte])]) = {
  val d = root.resolve(dir); Files.createDirectories(d)
  files.foreach { case (n, b) => Files.write(d.resolve(n), b) }
}
val half = java.util.Arrays.copyOf(bytes, bytes.length / 2)
val tailcut = java.util.Arrays.copyOf(bytes, bytes.length - 3)
mk("good_bad", Seq("a.parquet" -> bytes, "b.parquet" -> half))
mk("bad_good", Seq("a.parquet" -> half, "b.parquet" -> bytes))
mk("bad_only", Seq("a.parquet" -> half))
mk("good_tailcut", Seq("a.parquet" -> bytes, "b.parquet" -> tailcut))
mk("tailcut_good", Seq("a.parquet" -> tailcut, "b.parquet" -> bytes))
def cls(t: Throwable): String = t match {
  case e: org.apache.spark.SparkThrowable if e.getCondition != null => e.getCondition + " | " + Option(t.getMessage).map(_.take(160).replace("\n"," ")).getOrElse("")
  case _ => if (t.getCause != null) cls(t.getCause) else t.getClass.getName + ": " + Option(t.getMessage).map(_.take(160)).getOrElse("")
}
// SQL over parquet.`dir`; key off, spark.read.parquet gives the inner class (CANNOT_READ_FILE_FOOTER) where the
// SQL form wraps it in FAILED_TO_CREATE_PLAN_FOR_DIRECT_QUERY.
for (dir <- Seq("good_bad","bad_good","bad_only","good_tailcut","tailcut_good"); ign <- Seq("false","true"); merge <- Seq("false","true")) {
  spark.conf.set("spark.sql.files.ignoreCorruptFiles", ign)
  spark.conf.set("spark.sql.parquet.mergeSchema", merge)
  val r = try {
    val df = spark.sql(s"select count(*) c, sum(id) s from parquet.`$w/$dir`")
    val schema = spark.read.parquet(s"$w/$dir").schema.simpleString
    "OK " + df.collect().mkString(",") + " schema=" + schema
  } catch { case t: Throwable => "ERR " + cls(t) }
  println(s"@@ dir=$dir ignore=$ign merge=$merge -> $r")
}
// The per-read option against the session conf (DataFrame reader, both orders of a good and a cut file).
for (dir <- Seq("good_bad","bad_good"); conf <- Seq("false","true"); opt <- Seq("false","true"); merge <- Seq("false","true")) {
  spark.conf.set("spark.sql.files.ignoreCorruptFiles", conf)
  spark.conf.set("spark.sql.parquet.mergeSchema", merge)
  val r = try {
    val df = spark.read.option("ignoreCorruptFiles", opt).parquet(s"$w/$dir")
    "OK " + df.selectExpr("count(*)", "sum(id)").collect().mkString(",")
  } catch { case t: Throwable => "ERR " + cls(t) }
  println(s"@@ option dir=$dir conf=$conf option=$opt merge=$merge -> $r")
}
