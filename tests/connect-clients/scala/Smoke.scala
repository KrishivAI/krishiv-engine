// SPDX-License-Identifier: Apache-2.0

// Smoke run of the Apache Spark Connect Scala client (spark-connect-client-jvm) against a
// Spark Connect server (Krishiv, or real Spark for a baseline). Report-only: it always writes
// its results file. Run with scala-cli: see run.sh.
//> using scala 2.13
//> using dep org.apache.spark:spark-connect-client-jvm_2.13:4.0.0
//> using dep com.lihaoyi::ujson:4.0.2
//> using javaOpt --add-opens=java.base/java.nio=ALL-UNNAMED

import java.nio.file.{Files, Paths}
import org.apache.spark.sql.SparkSession
import scala.util.{Failure, Success, Try}

object Smoke {
  final case class Res(name: String, status: String, detail: String, ms: Long)

  // The shared text form of a value: NULL for null, String.valueOf otherwise (the Go client's %v
  // agrees for the types the queries return).
  def render(v: Any): String = if (v == null) "NULL" else String.valueOf(v)

  def firstLine(s: String): String = {
    val l = Option(s).getOrElse("").linesIterator.nextOption().getOrElse("")
    if (l.length > 300) l.take(300) else l
  }

  def timed(name: String)(body: => (String, String)): Res = {
    val start = System.nanoTime()
    val (status, detail) = Try(body) match {
      case Success(r) => r
      case Failure(e) => ("error", firstLine(e.toString))
    }
    Res(name, status, detail, (System.nanoTime() - start) / 1000000)
  }

  def runQuery(spark: SparkSession, q: ujson.Value): (String, String) = {
    val sqlText = q("sql").str
    val outcome = Try {
      val rows = spark.sql(sqlText).collect()
      val cols = rows.headOption.map(_.schema.fieldNames.toList)
      (cols, rows.map(r => (0 until r.length).map(i => render(r.get(i))).toList).toList)
    }
    q.obj.get("error") match {
      case Some(code) =>
        outcome match {
          case Success(_) => ("fail", s"expected error ${code.str}, the query succeeded")
          case Failure(e) if String.valueOf(e).contains(code.str) => ("pass", "")
          case Failure(e) => ("fail", s"expected error ${code.str}, got: ${firstLine(e.toString)}")
        }
      case None =>
        outcome match {
          case Failure(e) => ("error", firstLine(e.toString))
          case Success((cols, rows)) =>
            val want = q("rows").arr.map(_.arr.map(_.str).toList).toList
            val wantCols = q.obj.get("columns").map(_.arr.map(_.str).toList)
            if (rows != want) ("fail", s"rows: want $want, got $rows")
            else if (wantCols.isDefined && rows.nonEmpty && cols != wantCols)
              ("fail", s"columns: want ${wantCols.get}, got ${cols.get}")
            else ("pass", "")
        }
    }
  }

  def apiChecks(spark: SparkSession): Seq[Res] = Seq(
    timed("api_df_count") {
      val n = spark.sql("SELECT id FROM range(25)").count()
      if (n == 25L) ("pass", "") else ("fail", s"count: want 25, got $n")
    },
    timed("api_df_schema_names") {
      val names = spark.sql("SELECT 1 AS a, 'x' AS b").schema.fieldNames.toList
      if (names == List("a", "b")) ("pass", "") else ("fail", s"field names: $names")
    },
    timed("api_config_round_trip") {
      spark.conf.set("spark.sql.shuffle.partitions", "7")
      val got = spark.conf.get("spark.sql.shuffle.partitions")
      if (got == "7") ("pass", "") else ("fail", s"config value: $got")
    },
    timed("api_range_collect") {
      val rows = spark.range(3).collect()
      if (rows.length == 3) ("pass", "") else ("fail", s"rows: ${rows.length}")
    }
  )

  def main(args: Array[String]): Unit = {
    val Seq(remote, queriesFile, outFile) = args.toSeq
    val queries = ujson.read(Files.readString(Paths.get(queriesFile))).arr
    val results = Try(SparkSession.builder().remote(remote).getOrCreate()) match {
      case Failure(e) => Seq(Res("connect", "error", firstLine(e.toString), 0))
      case Success(spark) =>
        try queries.map(q => timed(q("name").str)(runQuery(spark, q))).toSeq ++ apiChecks(spark)
        finally Try(spark.close())
    }
    val out = ujson.Obj(
      "client" -> "scala",
      "version" -> "spark-connect-client-jvm 4.0.0",
      "remote" -> remote,
      "results" -> ujson.Arr.from(results.map(r =>
        ujson.Obj("name" -> r.name, "status" -> r.status, "detail" -> r.detail, "ms" -> r.ms)))
    )
    Files.writeString(Paths.get(outFile), ujson.write(out, indent = 2))
    println(s"scala client: ${results.count(_.status == "pass")}/${results.size} pass")
    sys.exit(0)
  }
}
