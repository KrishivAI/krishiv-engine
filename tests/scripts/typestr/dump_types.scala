// Parses Spark DDL type strings with Spark's own parser and prints the resulting DataType's
// renderings (simpleString, catalogString, sql, typeName, json) or the error Spark raised, as JSON
// lines prefixed with "@@T ". Also builds DecimalType(p, s) directly ("@@D ") and prints the
// error-condition entries (message template + SQLSTATE) for every error class seen ("@@EC ").
// Run by tests/scripts/gen_type_strings.py inside the Spark container via `spark-shell -I`; never
// edit the output by hand (P2-T001a).
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ObjectNode
import org.apache.spark.SparkThrowable
import org.apache.spark.sql.catalyst.parser.CatalystSqlParser
import org.apache.spark.sql.types._
import scala.jdk.CollectionConverters._

val mapper = new ObjectMapper()
// ASCII-only output: the reader splits lines on Unicode line separators too.
mapper.getFactory.configure(com.fasterxml.jackson.core.JsonGenerator.Feature.ESCAPE_NON_ASCII, true)
val seen = scala.collection.mutable.LinkedHashSet[String]()

def putError(m: ObjectNode, e: Throwable): Unit = e match {
  case t: SparkThrowable =>
    val cond = Option(t.getCondition).getOrElse("UNCLASSIFIED")
    seen += cond
    m.put("error_class", cond)
    m.put("sql_state", Option(t.getSqlState).getOrElse(""))
    val p = m.putObject("message_parameters")
    t.getMessageParameters.asScala.toSeq.sortBy(_._1).foreach { case (k, v) => p.put(k, v) }
  case other =>
    m.put("error_class", "UNCLASSIFIED")
    m.put("exception", other.getClass.getName)
}

def putType(m: ObjectNode, dt: DataType): Unit = {
  m.put("simple_string", dt.simpleString)
  m.put("catalog_string", dt.catalogString)
  m.put("sql", dt.sql)
  m.put("type_name", dt.typeName)
  m.put("json", dt.json)
  m.put("simple_string_reparses", CatalystSqlParser.parseDataType(dt.simpleString) == dt)
}

val in = scala.io.Source.fromFile("/tmp/krishiv_cases.jsonl", "UTF-8")
in.getLines().foreach { line =>
  val c = mapper.readTree(line)
  val m = mapper.createObjectNode()
  if (c.has("ddl")) {
    val ddl = c.get("ddl").asText()
    m.put("ddl", ddl)
    try putType(m, CatalystSqlParser.parseDataType(ddl)) catch { case e: Throwable => putError(m, e) }
    println("@@T " + mapper.writeValueAsString(m))
  } else {
    val p = c.get("precision").asInt()
    val s = c.get("scale").asInt()
    m.put("precision", p)
    m.put("scale", s)
    try putType(m, DecimalType(p, s)) catch { case e: Throwable => putError(m, e) }
    println("@@D " + mapper.writeValueAsString(m))
  }
}
in.close()

val conditions = mapper.readTree(
  getClass.getClassLoader.getResource("error/error-conditions.json"))
seen.foreach { cond =>
  val top = cond.split('.').head
  val node = conditions.get(top)
  if (node != null) {
    val m = mapper.createObjectNode()
    m.put("name", top)
    m.set[ObjectNode]("entry", node)
    println("@@EC " + mapper.writeValueAsString(m))
  }
}
println("@@VERSION " + spark.version)
println("@@END")
System.exit(0)
