// Reports Spark's own view of each built-in function (P2-T153): for every function-matrix case
// {"f": name, "sql": "...", "conf": {...}} in /tmp/krishiv_cases.jsonl that Spark analyzes, prints
// "@@T <name> <deterministic> <kind>" where deterministic is `Expression.deterministic` over the
// analyzed plan (the cases' other inputs are literals and VALUES rows, so only the function can make
// it false) and kind is "aggregate" when the plan holds an AggregateExpression, "window" when it holds
// a WindowFunction (`rank`, `lag`, ...; an aggregate over a window stays "aggregate"), "generator" when
// it holds a Generate node, else "scalar". Run by gen_function_registry.py in `spark-shell` (`:load`).
import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.spark.sql.catalyst.expressions.WindowFunction
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.plans.logical.Generate
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
    val det = !plan.exists(_.expressions.exists(e => !e.deterministic))
    val agg = plan.exists(_.expressions.exists(_.exists(_.isInstanceOf[AggregateExpression])))
    val win = plan.exists(_.expressions.exists(_.exists(_.isInstanceOf[WindowFunction])))
    val gen = plan.exists(_.isInstanceOf[Generate])
    val kind = if (agg) "aggregate" else if (win) "window" else if (gen) "generator" else "scalar"
    println("@@T " + node.get("f").asText() + " " + det + " " + kind)
  } catch { case _: Throwable => () }
}
println("@@END")
System.exit(0)
