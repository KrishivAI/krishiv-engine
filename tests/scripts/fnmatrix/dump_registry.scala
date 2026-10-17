// Dumps Spark's built-in function registry (scalar/aggregate/window/generator functions, table-valued
// functions and parser-level operators) as JSON lines prefixed with "@@FN ". Run by
// tests/scripts/gen_function_matrix.py inside the Spark container via `spark-shell -I`; never edit the
// output by hand (P2-T106a, review I12).
import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.spark.sql.catalyst.analysis.FunctionRegistry
import org.apache.spark.sql.catalyst.expressions.ExpressionInfo

val mapper = new ObjectMapper()
def emit(registry: String, name: String, info: ExpressionInfo): Unit = {
  val m = new java.util.LinkedHashMap[String, String]()
  m.put("name", name)
  m.put("registry", registry)
  m.put("group", Option(info.getGroup).getOrElse(""))
  m.put("class", Option(info.getClassName).getOrElse(""))
  m.put("since", Option(info.getSince).getOrElse(""))
  m.put("deprecated", Option(info.getDeprecated).getOrElse(""))
  m.put("source", Option(info.getSource).getOrElse(""))
  m.put("usage", Option(info.getUsage).getOrElse(""))
  m.put("examples", Option(info.getExamples).getOrElse(""))
  println("@@FN " + mapper.writeValueAsString(m))
}
val reg = spark.sessionState.functionRegistry
reg.listFunction().foreach(id => reg.lookupFunction(id).foreach(i => emit("function", id.funcName, i)))
val treg = spark.sessionState.tableFunctionRegistry
treg.listFunction().foreach(id => treg.lookupFunction(id).foreach(i => emit("table_function", id.funcName, i)))
FunctionRegistry.builtinOperators.foreach { case (n, i) => emit("operator", n, i) }
println("@@VERSION " + spark.version)
println("@@END")
System.exit(0)
