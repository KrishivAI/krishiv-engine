// Classifies generated function-matrix cases as deterministic or not, using Spark's own analyzer
// (P2-T106a). Reads JSON lines {"i": n, "sql": "...", "conf": {...}} from /tmp/krishiv_cases.jsonl and
// prints "@@ND <i>" for every case whose analyzed plan contains a non-deterministic expression
// (`Expression.deterministic == false`: rand, uuid, shuffle, reflect, ...) or a CURRENT_LIKE expression
// (current_date, now, current_user, ...: deterministic per query but not across runs). Cases that fail
// analysis are skipped (they have no values). Run by gen_function_matrix.py via `spark-shell -I`.
import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.spark.sql.catalyst.trees.TreePattern.CURRENT_LIKE
import scala.jdk.CollectionConverters._

val mapper = new ObjectMapper()
val src = scala.io.Source.fromFile("/tmp/krishiv_cases.jsonl", "UTF-8")
var set = Set.empty[String]
for (line <- src.getLines() if line.trim.nonEmpty) {
  val node = mapper.readTree(line)
  val conf = node.get("conf").fields().asScala.map(e => e.getKey -> e.getValue.asText()).toMap
  (set -- conf.keySet).foreach(k => spark.conf.unset(k))
  conf.foreach { case (k, v) => spark.conf.set(k, v) }
  set = conf.keySet
  try {
    val plan = spark.sql(node.get("sql").asText()).queryExecution.analyzed
    val unstable = plan.exists(_.expressions.exists(e => !e.deterministic || e.containsPattern(CURRENT_LIKE)))
    if (unstable) println("@@ND " + node.get("i").asInt())
  } catch { case _: Throwable => () }
}
println("@@END")
System.exit(0)
