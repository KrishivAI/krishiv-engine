// Parses SQL statements with Spark's session parser (`sqlParser.parsePlan`, the parser of
// `spark.sql`) under per-case configs and prints, as JSON lines prefixed with "@@P ", whether each
// parsed or the error Spark raised (condition, SQLSTATE, message parameters, query-context
// indexes, and the line and position a `ParseException` reports). A case's `mode` picks another
// entry point of the same grammar: `datatype` (`sqlParser.parseDataType`), `tableschema`
// (`sqlParser.parseTableSchema`) or `fromddl` (`DataType.fromDDL`, the schema strings of
// `from_json` and friends). Run by tests/scripts/gen_sql_parse_probe.py inside the Spark container via
// `spark-shell -I`; never edit the output by hand (P2-T070).
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ObjectNode
import org.apache.spark.SparkThrowable
import org.apache.spark.sql.catalyst.parser.ParseException
import org.apache.spark.sql.types.DataType
import scala.jdk.CollectionConverters._

val mapper = new ObjectMapper()
mapper.getFactory.configure(com.fasterxml.jackson.core.JsonGenerator.Feature.ESCAPE_NON_ASCII, true)
val in = scala.io.Source.fromFile("/tmp/krishiv_cases.jsonl", "UTF-8")
in.getLines().foreach { line =>
  val c = mapper.readTree(line)
  val sql = c.get("sql").asText()
  val mode = Option(c.get("mode")).map(_.asText()).getOrElse("plan")
  val conf = c.get("conf").fields().asScala.map(e => e.getKey -> e.getValue.asText()).toSeq
  conf.foreach { case (k, v) => spark.conf.set(k, v) }
  val m = mapper.createObjectNode()
  m.put("sql", sql)
  try {
    mode match {
      case "plan" => spark.sessionState.sqlParser.parsePlan(sql)
      case "datatype" => spark.sessionState.sqlParser.parseDataType(sql)
      case "tableschema" => spark.sessionState.sqlParser.parseTableSchema(sql)
      case "fromddl" => DataType.fromDDL(sql)
    }
    m.put("parses", true)
  } catch {
    case t: Throwable with SparkThrowable =>
      m.put("parses", false)
      m.put("error_class", Option(t.getCondition).getOrElse("UNCLASSIFIED"))
      m.put("sql_state", Option(t.getSqlState).getOrElse(""))
      val p = m.putObject("message_parameters")
      t.getMessageParameters.asScala.toSeq.sortBy(_._1).foreach { case (k, v) => p.put(k, v) }
      t.getQueryContext.headOption.foreach { q => m.put("start", q.startIndex); m.put("stop", q.stopIndex) }
      t match {
        case pe: ParseException =>
          pe.start.line.foreach(l => m.put("line", l))
          pe.start.startPosition.foreach(p => m.put("pos", p))
        case _ =>
      }
    case other: Throwable =>
      m.put("parses", false)
      m.put("error_class", "UNCLASSIFIED")
  }
  conf.foreach { case (k, _) => spark.conf.unset(k) }
  println("@@P " + mapper.writeValueAsString(m))
}
in.close()
println("@@VERSION " + spark.version)
println("@@END")
System.exit(0)
