// Plans file scans with Spark's own FileSourceScanExec and prints the file partitions it produced, as JSON
// lines prefixed with "@@S ". Each case's files are sparse files of the given sizes (nothing is read: the
// text source plans splits from listing sizes only). Also runs the schema-mapping error probes ("@@E ").
// Run by tests/scripts/gen_file_splits.py inside the Spark container via `spark-shell -I`; never edit the
// output by hand (P2-T117).
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ObjectNode
import org.apache.spark.SparkThrowable
import org.apache.spark.sql.execution.FileSourceScanExec
import org.apache.spark.sql.execution.datasources.FilePartition
import java.io.{File, RandomAccessFile}
import scala.jdk.CollectionConverters._

val mapper = new ObjectMapper()
val base = new File("/tmp/krishiv_splits")
def rmrf(f: File): Unit = { if (f.isDirectory) f.listFiles.foreach(rmrf); f.delete() }
rmrf(base)
base.mkdirs()
println("@@VERSION " + spark.version)
println("@@PARALLELISM " + spark.sparkContext.defaultParallelism)
val confKeys = Seq(
  "spark.sql.files.maxPartitionBytes",
  "spark.sql.files.openCostInBytes",
  "spark.sql.files.minPartitionNum",
  "spark.sql.files.maxPartitionNum",
  "spark.sql.leafNodeDefaultParallelism")

val in = scala.io.Source.fromFile("/tmp/krishiv_cases.jsonl", "UTF-8")
in.getLines().foreach { line =>
  val c = mapper.readTree(line)
  val id = c.get("id").asText()
  val dir = new File(base, id)
  dir.mkdirs()
  c.get("files").elements().asScala.foreach { f =>
    val raf = new RandomAccessFile(new File(dir, f.get("name").asText()), "rw")
    raf.setLength(f.get("size").asLong())
    raf.close()
  }
  confKeys.foreach(k => spark.conf.unset(k))
  c.get("conf").fields().asScala.foreach { e => spark.conf.set(e.getKey, e.getValue.asText()) }
  val m = mapper.createObjectNode()
  m.put("id", id)
  try {
    val df = spark.read.format("text").load(dir.getPath)
    val scan = df.queryExecution.sparkPlan.collectFirst { case s: FileSourceScanExec => s }.get
    val listing = m.putArray("listing")
    scan.relation.location.listFiles(Nil, Nil).flatMap(_.files).foreach { f =>
      val a = listing.addArray(); a.add(f.getPath.getName); a.add(f.getLen)
    }
    val parts = m.putArray("partitions")
    scan.inputRDD.partitions.foreach { p =>
      val arr = parts.addArray()
      p.asInstanceOf[FilePartition].files.foreach { f =>
        val s = arr.addArray(); s.add(f.toPath.getName); s.add(f.start); s.add(f.length)
      }
    }
  } catch { case e: Throwable => m.put("error", e.toString) }
  println("@@S " + mapper.writeValueAsString(m))
}
in.close()
confKeys.foreach(k => spark.conf.unset(k))

def conditions(e: Throwable): Seq[(String, String, java.util.Map[String, String])] = {
  val out = scala.collection.mutable.ArrayBuffer[(String, String, java.util.Map[String, String])]()
  var t = e
  while (t != null) {
    t match {
      case s: SparkThrowable => out += ((Option(s.getCondition).getOrElse("UNCLASSIFIED"), Option(s.getSqlState).getOrElse(""), s.getMessageParameters))
      case other => out += (("UNCLASSIFIED", other.getClass.getName, java.util.Collections.emptyMap[String, String]()))
    }
    t = t.getCause
  }
  out.toSeq
}
def probe(name: String)(body: => Seq[org.apache.spark.sql.Row]): Unit = {
  val m = mapper.createObjectNode()
  m.put("probe", name)
  try {
    val rows = m.putArray("rows")
    body.foreach(r => rows.add(r.toString))
  } catch {
    case e: Throwable =>
      m.remove("rows")
      val arr = m.putArray("errors")
      conditions(e).foreach { case (cond, state, params) =>
        val o = arr.addObject(); o.put("condition", cond); o.put("sql_state", state)
        val p = o.putObject("parameters")
        params.asScala.toSeq.sortBy(_._1).filter(_._1 != "path").foreach { case (k, v) => p.put(k, v) }
      }
  }
  println("@@E " + mapper.writeValueAsString(m))
}
val pdir = new File(base, "probes")
pdir.mkdirs()
spark.conf.set("spark.sql.caseSensitive", "true")
spark.range(3).selectExpr("cast(id as int) as a", "cast(id * 10 as int) as A").coalesce(1).write.parquet(new File(pdir, "dup").getPath)
spark.range(3).selectExpr("cast(id as int) as i", "cast(id as string) as s", "named_struct('x', cast(id as int)) as st").coalesce(1).write.parquet(new File(pdir, "base").getPath)
spark.conf.set("spark.sql.caseSensitive", "false")
probe("case_insensitive_duplicate")(spark.read.schema("a int").parquet(new File(pdir, "dup").getPath).collect().toSeq)
probe("case_insensitive_match")(spark.read.schema("I int").parquet(new File(pdir, "base").getPath).orderBy("I").collect().toSeq)
probe("widen_int_to_long")(spark.read.schema("i bigint").parquet(new File(pdir, "base").getPath).orderBy("i").collect().toSeq)
probe("widen_int_to_double")(spark.read.schema("i double").parquet(new File(pdir, "base").getPath).orderBy("i").collect().toSeq)
probe("mismatch_string_to_long")(spark.read.schema("s bigint").parquet(new File(pdir, "base").getPath).collect().toSeq)
probe("narrow_int_to_short")(spark.read.schema("i smallint").parquet(new File(pdir, "base").getPath).collect().toSeq)
probe("missing_top_level")(spark.read.schema("i int, z string").parquet(new File(pdir, "base").getPath).orderBy("i").collect().toSeq)
probe("missing_nested")(spark.read.schema("i int, st struct<x:int,y:string>").parquet(new File(pdir, "base").getPath).orderBy("i").collect().toSeq)
spark.conf.set("spark.sql.caseSensitive", "true")
probe("case_sensitive_no_match")(spark.read.schema("I int").parquet(new File(pdir, "base").getPath).collect().toSeq)
spark.conf.set("spark.sql.caseSensitive", "false")
rmrf(base)
println("@@END")
