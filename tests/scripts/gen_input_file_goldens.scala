// SPDX-License-Identifier: Apache-2.0
// Fed to spark-shell on stdin by gen_input_file_goldens.py. Writes the data under /w/data with fixed file names,
// runs every case of /w/cases.jsonl (fields: n, conf, pre, sql) and prints `@@`-prefixed JSON lines.
:paste
object H {
  val root = "/w/data"
  def esc(s: String): String = {
    val b = new StringBuilder("\"")
    s.foreach {
      case '"' => b.append("\\\"")
      case '\\' => b.append("\\\\")
      case '\n' => b.append("\\n")
      case '\r' => b.append("\\r")
      case '\t' => b.append("\\t")
      case c if c < ' ' => b.append("\\u%04x".format(c.toInt))
      case c => b.append(c)
    }
    b.append("\"").toString
  }
  def rel(s: String): String = s.replace(root, "<root>")
  def js(v: Any): String = if (v == null) "null" else esc(rel(String.valueOf(v)))
  // Writes `df` as exactly one file `<dir>/<name>` (Spark's random part-file name is renamed).
  def one(df: org.apache.spark.sql.DataFrame, fmt: String, dir: String, name: String, opts: Map[String, String] = Map()): Unit = {
    val tmp = new java.io.File(s"$root/_tmp_$name")
    df.coalesce(1).write.options(opts).mode("overwrite").format(fmt).save(tmp.getPath)
    val part = tmp.listFiles().filter(f => f.getName.startsWith("part-") && !f.getName.endsWith(".crc")).head
    new java.io.File(s"$root/$dir").mkdirs()
    java.nio.file.Files.move(part.toPath, new java.io.File(s"$root/$dir/$name").toPath)
    tmp.listFiles().foreach(_.delete()); tmp.delete()
  }
  def setup(): Unit = {
    import spark.implicits._
    one(Seq(1, 2).toDF("id"), "parquet", "a", "a1.parquet")
    one(Seq(3).toDF("id"), "parquet", "a", "a2.parquet")
    one(Seq(1, 10).toDF("id"), "parquet", "b", "b1.parquet")
    one(Seq(1, 7).toDF("id"), "csv", "c", "c1.csv", Map("header" -> "true"))
    one(Seq(2, 8).toDF("id"), "json", "d", "d1.json")
    spark.sparkContext.hadoopConfiguration.setInt("parquet.block.size", 65536)
    val big = spark.range(0, 200000).selectExpr("id", "sha2(cast(id as string), 256) as h")
    one(big, "parquet", "big", "big.parquet")
    spark.read.parquet(s"$root/a").createOrReplaceTempView("a")
    spark.read.parquet(s"$root/b").createOrReplaceTempView("b")
    spark.read.option("header", "true").option("inferSchema", "true").csv(s"$root/c").createOrReplaceTempView("c")
    spark.read.json(s"$root/d").createOrReplaceTempView("d")
    spark.read.parquet(s"$root/big").createOrReplaceTempView("big")
    spark.sql(s"CREATE TABLE t USING parquet LOCATION '$root/a'")
    val files = new java.io.File(root).listFiles().filterNot(_.getName.startsWith("_")).flatMap { d =>
      d.listFiles().filter(f => f.isFile && !f.getName.startsWith(".") && !f.getName.startsWith("_"))
    }
    files.sortBy(_.getPath).foreach(f => println("@@FILE " + js(f.getPath.replace(root, "<root>") ) + " " + f.length))
  }
  def runDf(sql: String): Seq[Seq[Any]] = {
    import org.apache.spark.sql.functions._
    val df = sql match {
      case "DF:select" => spark.read.parquet(s"$root/a").select(input_file_name(), col("id"))
      case "DF:filter" => spark.read.parquet(s"$root/a").filter(input_file_name().contains("a1")).select(input_file_name(), col("id"))
      case "DF:blocks" => spark.read.parquet(s"$root/a").select(input_file_name(), input_file_block_start(), input_file_block_length(), col("id"))
    }
    df.collect().map(_.toSeq).toSeq
  }
  def main(): Unit = {
    setup()
    println("@@VERSION " + spark.version + " " + spark.sparkContext.master)
    val cases = spark.read.json("/w/cases.jsonl").orderBy("n").collect()
    cases.foreach { r =>
      val n = r.getAs[Long]("n"); val conf = r.getAs[String]("conf"); val pre = r.getAs[String]("pre"); val sql = r.getAs[String]("sql")
      val keys = if (conf == null || conf.isEmpty) Array[String]() else conf.split("\\|")
      keys.foreach { kv => val i = kv.indexOf('='); spark.conf.set(kv.substring(0, i), kv.substring(i + 1)) }
      val out = try {
        if (pre != null && pre.nonEmpty) pre.split(";;").foreach(p => spark.sql(p).collect())
        val rows: Seq[Seq[Any]] = if (sql.startsWith("DF:")) runDf(sql) else spark.sql(sql).collect().map(_.toSeq).toSeq
        val enc = rows.map(rw => "[" + rw.map(js).mkString(",") + "]").sorted
        "\"rows\": [" + enc.mkString(",") + "]"
      } catch {
        case e: org.apache.spark.SparkThrowable =>
          "\"error\": {\"class\": " + js(e.getCondition) + ", \"sqlstate\": " + js(e.getSqlState) + ", \"message\": " + js(e.asInstanceOf[Throwable].getMessage) + "}"
        case e: Throwable => "\"error\": {\"class\": " + js("JVM:" + e.getClass.getName) + ", \"sqlstate\": null, \"message\": " + js(e.getMessage) + "}"
      }
      println("@@CASE {\"n\": " + n + ", " + out + "}")
      keys.foreach { kv => spark.conf.unset(kv.substring(0, kv.indexOf('='))) }
      spark.catalog.clearCache()
      spark.sql("DROP VIEW IF EXISTS ca"); spark.sql("DROP VIEW IF EXISTS av2")
    }
    println("@@END")
  }
}
H.main()
System.exit(0)
