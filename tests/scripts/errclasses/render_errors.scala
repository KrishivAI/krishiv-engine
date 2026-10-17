// SPDX-License-Identifier: Apache-2.0
// Raises real SparkExceptions for sampled (errorClass, parameters) pairs and prints what Spark
// itself reports. Input: /tmp/krishiv_cases.jsonl, one {"c": class, "p": {param: value}} per line.
// Run by tests/scripts/gen_error_classes.py inside the Spark 4.2.0 container via `spark-shell -I`.
import org.apache.spark.SparkException
import org.json4s._
import org.json4s.jackson.JsonMethods

implicit val formats: Formats = DefaultFormats
println("@@VERSION " + spark.version)
scala.io.Source.fromFile("/tmp/krishiv_cases.jsonl", "UTF-8").getLines().filter(_.nonEmpty).foreach { line =>
  val j = JsonMethods.parse(line)
  val cls = (j \ "c").extract[String]
  val params = (j \ "p").extract[Map[String, String]]
  val e = new SparkException(cls, params, null)
  val out = JObject(
    "c" -> JString(cls),
    "state" -> Option(e.getSqlState).map(JString(_)).getOrElse(JNull),
    "full" -> JString(e.getMessage),
    "params" -> JObject(params.toList.sortBy(_._1).map { case (k, v) => k -> JString(v) }))
  println("@@CASE " + JsonMethods.compact(JsonMethods.render(out)))
}
println("@@END")
