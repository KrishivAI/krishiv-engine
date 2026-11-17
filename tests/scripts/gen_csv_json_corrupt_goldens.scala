// SPDX-License-Identifier: Apache-2.0
// Spark 4.2.0 probes for PY-076 (corrupt-record-only queries), PY-067 (row order of a split CSV file) and SQL-30
// (UTF-16 CSV read as UTF-8). Run by tests/scripts/gen_csv_json_corrupt_goldens.py through spark-shell; inputs are
// mounted read-only at /w. Every result line is "@@<label>\t<OK|ERR>\t<fields...>"; strings are \u-escaped.
import org.apache.spark.SparkThrowable
import org.apache.spark.sql.{DataFrame, Row}

def esc(s: String): String = if (s == null) "<null>" else s.flatMap { c =>
  if (c == '\\') "\\\\" else if (c < 0x20 || c > 0x7e) "\\u%04x".format(c.toInt) else c.toString
}
def rows(df: DataFrame): String = df.collect().map(r => esc(r.toString)).mkString(" | ")
def probe(label: String)(f: => String): Unit = {
  try { println("@@" + label + "\tOK\t" + f) }
  catch { case t: Throwable =>
    val chain = Iterator.iterate[Throwable](t)(_.getCause).takeWhile(_ != null).toList
    val st = chain.collectFirst { case s: SparkThrowable if s.getCondition != null => s }
    val cls = t.getClass.getName
    val (cond, sqlstate, params) = st match {
      case Some(s) => (s.getCondition, s.getSqlState, s.getMessageParameters.toString)
      case None => ("<none>", "<none>", "<none>")
    }
    println("@@" + label + "\tERR\t" + cls + "\t" + cond + "\t" + sqlstate + "\t" + esc(params) + "\t" + esc(t.getMessage))
  }
}
spark.conf.set("spark.sql.ansi.enabled", "true")

// ---- PY-076 ----
val pq = "/tmp/pq-dir"
spark.range(0, 3).selectExpr("id as a", "cast(id as string) as b").coalesce(1).write.mode("overwrite").parquet(pq)
probe("pq_as_json.schema")(spark.read.json(pq).schema.simpleString)
probe("pq_as_json.collect_count")(spark.read.json(pq).collect().length.toString)
probe("pq_as_json.count")(spark.read.json(pq).count().toString)
probe("pq_as_json.select_star")(spark.read.json(pq).select("*").collect().length.toString)
probe("pq_as_json.select_corrupt")(spark.read.json(pq).select("_corrupt_record").collect().length.toString)
probe("pq_as_json.cached_collect")({ val d = spark.read.json(pq).cache(); d.count(); d.collect().length.toString })

val sch = "a INT, _corrupt_record STRING"
val sch2 = "a INT, _bad STRING"
for ((fmt, path) <- Seq(("json", "/w/bad.json"), ("csv", "/w/bad.csv"))) {
  def rd(s: String = sch, opts: Map[String, String] = Map.empty) =
    spark.read.format(fmt).schema(s).options(opts).load(path)
  val p = fmt + "."
  probe(p + "select_corrupt")(rows(rd().select("_corrupt_record")))
  probe(p + "select_corrupt_analysis_only")(rd().select("_corrupt_record").schema.simpleString + " (no action)")
  probe(p + "count")(rd().count().toString)
  probe(p + "select_star")(rows(rd()))
  probe(p + "where_corrupt_notnull")(rows(rd().where("_corrupt_record is not null")))
  probe(p + "where_corrupt_notnull_count")(rd().where("_corrupt_record is not null").count().toString)
  probe(p + "where_corrupt_notnull_select_a")(rows(rd().where("_corrupt_record is not null").select("a")))
  probe(p + "select_corrupt_a")(rows(rd().select("_corrupt_record", "a")))
  probe(p + "select_a")(rows(rd().select("a")))
  probe(p + "cached_select_corrupt")({ val d = rd().cache(); d.count(); rows(d.select("_corrupt_record")) })
  probe(p + "cached_where_corrupt")({ val d = rd().cache(); d.count(); rows(d.where("_corrupt_record is not null")) })
  probe(p + "cache_then_select_corrupt_no_prior_action")(rows(rd().cache().select("_corrupt_record")))
  probe(p + "renamed.select_bad")(rows(rd(sch2, Map("columnNameOfCorruptRecord" -> "_bad")).select("_bad")))
  probe(p + "renamed.select_bad_a")(rows(rd(sch2, Map("columnNameOfCorruptRecord" -> "_bad")).select("_bad", "a")))
  probe(p + "renamed.select_default_name")(rows(rd(sch2, Map("columnNameOfCorruptRecord" -> "_bad")).select("_corrupt_record")))
  probe(p + "renamed.default_name_in_schema.select_corrupt")(rows(rd(sch, Map("columnNameOfCorruptRecord" -> "_bad")).select("_corrupt_record")))
  probe(p + "renamed.select_bad_dropmalformed")(rows(rd(sch2, Map("columnNameOfCorruptRecord" -> "_bad", "mode" -> "DROPMALFORMED")).select("_bad")))
  probe(p + "failfast.select_corrupt")(rows(rd(sch, Map("mode" -> "FAILFAST")).select("_corrupt_record")))
  probe(p + "sql_session_conf_name")({
    spark.conf.set("spark.sql.columnNameOfCorruptRecord", "_c2")
    try rows(spark.read.format(fmt).schema("a INT, _c2 STRING").load(path).select("_c2")) finally spark.conf.unset("spark.sql.columnNameOfCorruptRecord")
  })
  probe(p + "sql_table_function")(rows(spark.sql(s"SELECT _corrupt_record FROM $fmt.`$path`")))
  probe(p + "sql_table_function_star")(rows(spark.sql(s"SELECT * FROM $fmt.`$path`")))
}
probe("json.inferred.schema")(spark.read.json("/w/bad.json").schema.simpleString)
probe("json.inferred.select_corrupt")(rows(spark.read.json("/w/bad.json").select("_corrupt_record")))
probe("json.inferred.select_star")(rows(spark.read.json("/w/bad.json")))
probe("json.inferred.count")(spark.read.json("/w/bad.json").count().toString)
probe("json.inferred.cached_select_corrupt")({ val d = spark.read.json("/w/bad.json").cache(); d.count(); rows(d.select("_corrupt_record")) })
probe("json.inferred.where_corrupt")(rows(spark.read.json("/w/bad.json").where("_corrupt_record is not null")))
probe("json.multiline.select_corrupt")(rows(spark.read.schema(sch).option("multiLine", "true").json("/w/bad.json").select("_corrupt_record")))
// ansi off: same check is analysis-time, independent of ansi
spark.conf.set("spark.sql.ansi.enabled", "false")
probe("ansi_off.json.select_corrupt")(rows(spark.read.schema(sch).json("/w/bad.json").select("_corrupt_record")))
spark.conf.set("spark.sql.ansi.enabled", "true")

// ---- PY-067 ----
spark.conf.set("spark.sql.files.maxPartitionBytes", (1024 * 1024).toString)
{
  val big = "/w/big.csv"
  def rd = spark.read.option("header", "false").csv(big)
  probe("py067.partitions")(rd.rdd.getNumPartitions.toString)
  for (run <- 1 to 3) {
    probe(s"py067.first.run$run")(esc(rd.first().toString))
    probe(s"py067.head3.run$run")(rd.head(3).map(r => esc(r.toString)).mkString(" | "))
    probe(s"py067.limit3_collect.run$run")(rd.limit(3).collect().map(r => esc(r.toString)).mkString(" | "))
    probe(s"py067.take3.run$run")(rd.take(3).map(r => esc(r.toString)).mkString(" | "))
  }
  probe("py067.show_head")(rd.take(2).map(r => esc(r.toString)).mkString(" | "))
  probe("py067.input_file_order_of_partitions")(rd.rdd.mapPartitionsWithIndex((i, it) => if (it.hasNext) Iterator((i, it.next().getString(0))) else Iterator.empty).collect().map { case (i, s) => s"$i:$s" }.mkString(" "))
  probe("py067.limit1000_first_last")({ val a = rd.limit(1000).collect(); esc(a.head.toString) + " .. " + esc(a.last.toString) })
  // two files: the scan sorts splits by length descending
  probe("py067.two_files_first")(esc(spark.read.csv("/w/big2").first().toString))
  probe("py067.two_files_partition_firsts")(spark.read.csv("/w/big2").rdd.mapPartitionsWithIndex((i, it) => if (it.hasNext) Iterator((i, it.next().getString(0))) else Iterator.empty).collect().map { case (i, s) => s"$i:$s" }.mkString(" "))
}

// ---- SQL-30 ----
for (name <- Seq("utf16le", "utf16be")) {
  val path = s"/w/$name.csv"
  for (hdr <- Seq("true", "false"); enc <- Seq(None, Some("UTF-16"))) {
    val tag = s"sql030.$name.header=$hdr.enc=${enc.getOrElse("default")}"
    def rd = { val r = spark.read.option("header", hdr); enc.fold(r)(e => r.option("encoding", e)).csv(path) }
    probe(tag + ".columns")(rd.columns.map(esc).mkString(","))
    probe(tag + ".rows")(rows(rd))
    probe(tag + ".count")(rd.count().toString)
  }
}
System.exit(0)
