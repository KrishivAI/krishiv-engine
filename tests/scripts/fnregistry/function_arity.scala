// Probes each built-in function's argument counts (P2-T153): reads one name per line from
// /tmp/krishiv_cases.jsonl and analyzes `SELECT name(NULL, ...)` with 0 to 6 arguments. Prints
// "@@A <name> <n> ok" when n arguments pass the arity check (the call may still fail later, on types)
// and "@@A <name> <n> <expectedNum>" with Spark's own `expectedNum` text when it fails with
// WRONG_NUM_ARGS. Run by gen_function_registry.py in `spark-shell` (`:load`).
import org.apache.spark.sql.AnalysisException

val src = scala.io.Source.fromFile("/tmp/krishiv_cases.jsonl", "UTF-8")
for (name <- src.getLines().map(_.trim) if name.nonEmpty) {
  for (n <- 0 to 6) {
    val args = Seq.fill(n)("NULL").mkString(", ")
    val verdict = try {
      spark.sql(s"SELECT `$name`($args)").queryExecution.analyzed
      "ok"
    } catch {
      case e: AnalysisException if Option(e.getCondition).exists(_.startsWith("WRONG_NUM_ARGS")) =>
        e.getMessageParameters.get("expectedNum").replace(' ', '_')
      case _: Throwable => "ok"
    }
    println("@@A " + name + " " + n + " " + verdict)
  }
}
println("@@END")
System.exit(0)
