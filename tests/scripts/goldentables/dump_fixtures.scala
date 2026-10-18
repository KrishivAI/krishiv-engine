// Dumps what the golden runner needs from a real Spark 4.2 session, as JSON lines prefixed "@@":
//   @@TABLE  the six tables SQLQueryTestSuite.createTestTables builds (built the same way, from the
//            test-data files copied into the container) and the TPC-DS tables TPCDSSchema declares
//            (created empty, as the suite does for pipe-operators.sql);
//   @@ROW    one row of the table before it, encoded by type (see encode());
//   @@PROBE  a query run through HiveResult.hiveResultString, with its result values and the exact
//            text Spark printed;
//   @@SORTED the analysed plan of a query and what SQLQueryTestHelper.isSemanticallySorted says.
// Run by tests/scripts/gen_golden_fixtures.py inside the Spark container via `spark-shell -I`; never
// edit the output by hand (P2-T152).
import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import com.fasterxml.jackson.databind.node.{ArrayNode, JsonNodeFactory, ObjectNode}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.planning.PhysicalOperation
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.util.{ArrayData, CharVarcharUtils, MapData}
import org.apache.spark.sql.execution.HiveResult
import org.apache.spark.sql.execution.command.{DescribeColumnCommand, DescribeCommandBase}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.CalendarInterval
import scala.io.Source

val mapper = new ObjectMapper()
// ASCII-only output: the reader splits lines on Unicode line separators too.
mapper.getFactory.configure(com.fasterxml.jackson.core.JsonGenerator.Feature.ESCAPE_NON_ASCII, true)
val nf = JsonNodeFactory.instance
val dataDir = "/tmp/krishiv_golden_data"

def emit(tag: String, node: JsonNode): Unit = println("@@" + tag + " " + mapper.writeValueAsString(node))

/** Encodes an internal Spark value by its type: integers as numbers; float and double as Java's
 *  toString text; decimals as the unscaled integer text; strings as text; binary as hex; dates as
 *  days, timestamps as microseconds, intervals as months or microseconds; arrays as lists, maps as
 *  lists of [key, value] in storage order, structs as lists in field order. */
def encode(v: Any, dt: DataType): JsonNode = {
  if (v == null) return nf.nullNode()
  dt match {
    case NullType => nf.nullNode()
    case BooleanType => nf.booleanNode(v.asInstanceOf[Boolean])
    case ByteType => nf.numberNode(v.asInstanceOf[Byte].toLong)
    case ShortType => nf.numberNode(v.asInstanceOf[Short].toLong)
    case IntegerType | DateType | _: YearMonthIntervalType => nf.numberNode(v.asInstanceOf[Int].toLong)
    case LongType | TimestampType | TimestampNTZType | _: DayTimeIntervalType =>
      nf.numberNode(v.asInstanceOf[Long])
    case FloatType => nf.textNode(java.lang.Float.toString(v.asInstanceOf[Float]))
    case DoubleType => nf.textNode(java.lang.Double.toString(v.asInstanceOf[Double]))
    case _: DecimalType => nf.textNode(v.asInstanceOf[org.apache.spark.sql.types.Decimal].toJavaBigDecimal.unscaledValue.toString)
    case _: StringType => nf.textNode(v.toString)
    case BinaryType => nf.textNode(v.asInstanceOf[Array[Byte]].map(b => f"${b & 0xff}%02x").mkString)
    case CalendarIntervalType =>
      val c = v.asInstanceOf[CalendarInterval]
      val o = nf.objectNode()
      o.put("months", c.months); o.put("days", c.days); o.put("microseconds", c.microseconds)
      o
    case ArrayType(et, _) =>
      val a = v.asInstanceOf[ArrayData]
      val out = nf.arrayNode()
      (0 until a.numElements()).foreach(i => out.add(if (a.isNullAt(i)) nf.nullNode() else encode(a.get(i, et), et)))
      out
    case MapType(kt, vt, _) =>
      val m = v.asInstanceOf[MapData]
      val out = nf.arrayNode()
      (0 until m.numElements()).foreach { i =>
        val pair = nf.arrayNode()
        pair.add(encode(m.keyArray().get(i, kt), kt))
        pair.add(if (m.valueArray().isNullAt(i)) nf.nullNode() else encode(m.valueArray().get(i, vt), vt))
        out.add(pair)
      }
      out
    case StructType(fs) =>
      val r = v.asInstanceOf[InternalRow]
      val out = nf.arrayNode()
      fs.zipWithIndex.foreach { case (f, i) => out.add(if (r.isNullAt(i)) nf.nullNode() else encode(r.get(i, f.dataType), f.dataType)) }
      out
    case other => throw new IllegalArgumentException("unsupported type " + other.catalogString)
  }
}

def encodeRow(r: InternalRow, types: Seq[DataType]): ArrayNode = {
  val out = nf.arrayNode()
  types.zipWithIndex.foreach { case (t, i) => out.add(if (r.isNullAt(i)) nf.nullNode() else encode(r.get(i, t), t)) }
  out
}

def tableHeader(name: String, schema: StructType): Unit = {
  val h = nf.objectNode()
  h.put("name", name)
  val cols = h.putArray("columns")
  schema.fields.foreach { f =>
    val c = cols.addObject()
    c.put("name", f.name); c.put("type", f.dataType.catalogString); c.put("nullable", f.nullable)
  }
  emit("TABLE", h)
}

def dumpTable(name: String): Unit = {
  val df = spark.table(name)
  tableHeader(name, df.schema)
  val types = df.schema.fields.map(_.dataType).toSeq
  df.queryExecution.executedPlan.executeCollect().foreach { r =>
    val row = nf.objectNode(); row.put("table", name); row.set[JsonNode]("row", encodeRow(r, types)); emit("ROW", row)
  }
}

// ---- the tables, built as SQLQueryTestSuite.createTestTables builds them ----
import spark.implicits._
val tabSep = Map("delimiter" -> "\t", "header" -> "false")
val onekSchema = ("unique1 int, unique2 int, two int, four int, ten int, twenty int, hundred int, thousand int, " +
  "twothousand int, fivethous int, tenthous int, odd int, even int, stringu1 string, stringu2 string, string4 string")
(1 to 100).map(i => (i, i.toString)).toDF("key", "value").repartition(1).write.format("parquet").saveAsTable("testdata")
((Seq(1, 2, 3), Seq(Seq(1, 2, 3))) :: (Seq(2, 3, 4), Seq(Seq(2, 3, 4))) :: Nil)
  .toDF("arraycol", "nestedarraycol").write.format("parquet").saveAsTable("arraydata")
(Tuple1(Map(1 -> "a1", 2 -> "b1", 3 -> "c1", 4 -> "d1", 5 -> "e1")) ::
  Tuple1(Map(1 -> "a2", 2 -> "b2", 3 -> "c2", 4 -> "d2")) ::
  Tuple1(Map(1 -> "a3", 2 -> "b3", 3 -> "c3")) ::
  Tuple1(Map(1 -> "a4", 2 -> "b4")) ::
  Tuple1(Map(1 -> "a5")) :: Nil).toDF("mapcol").write.format("parquet").saveAsTable("mapdata")
spark.read.format("csv").options(tabSep).schema("a int, b float").load(dataDir + "/agg.data")
  .write.format("parquet").saveAsTable("aggtest")
spark.read.format("csv").options(tabSep).schema(onekSchema).load(dataDir + "/onek.data")
  .write.format("parquet").saveAsTable("onek")
spark.read.format("csv").options(tabSep).schema(onekSchema).load(dataDir + "/tenk.data")
  .write.format("parquet").saveAsTable("tenk1")
Seq("testdata", "arraydata", "mapdata", "aggtest", "onek", "tenk1").foreach(dumpTable)

// ---- TPC-DS table schemas (TPCDSSchema; created empty, `USING parquet`, as the suite does) ----
val tpcds = Source.fromFile(dataDir + "/tpcds_tables.tsv").getLines().filter(_.nonEmpty).map { l =>
  val Array(name, cols) = l.split("\t", 2); (name, cols)
}.toSeq
tpcds.foreach { case (name, cols) =>
  spark.sql(s"CREATE TABLE `$name` ($cols) USING parquet")
  val meta = spark.sessionState.catalog.getTableMetadata(spark.sessionState.sqlParser.parseTableIdentifier(name))
  tableHeader(name, CharVarcharUtils.getRawSchema(meta.schema))
}

// ---- HiveResult probes ----
// SQLQueryTestSuite runs with the Java 8 time API (java.time values in the collected rows), which is
// what HiveResult formats there; without it dates and timestamps go through java.sql and the hybrid calendar.
spark.conf.set("spark.sql.datetime.java8API.enabled", "true")
def probe(sql: String, tz: String, style: String): Unit = {
  val o = nf.objectNode()
  o.put("sql", sql); o.put("time_zone", tz); o.put("binary_style", style)
  try {
    spark.conf.set("spark.sql.session.timeZone", tz)
    spark.conf.set("spark.sql.binaryOutputStyle", style)
    val plan = spark.sql(sql).queryExecution.executedPlan
    val types = plan.output.map(_.dataType)
    val ts = o.putArray("types"); types.foreach(t => ts.add(t.catalogString))
    val rows = o.putArray("rows"); plan.executeCollect().foreach(r => rows.add(encodeRow(r, types)))
    val hive = o.putArray("hive"); HiveResult.hiveResultString(plan).foreach(s => hive.add(s))
  } catch { case e: Throwable => o.put("error", e.getClass.getName + ": " + String.valueOf(e.getMessage).take(200)) }
  emit("PROBE", o)
}

// ---- SQLQueryTestHelper.isSemanticallySorted, verbatim, over the analysed plan ----
def isSemanticallySorted(plan: LogicalPlan): Boolean = plan match {
  case _: Join | _: Aggregate | _: Generate | _: Sample | _: Distinct => false
  case _: DescribeCommandBase | _: DescribeColumnCommand | _: DescribeRelation | _: DescribeTablePartition |
       _: DescribeColumn => true
  case PhysicalOperation(_, _, Sort(_, true, _, _)) => true
  case _ => plan.children.iterator.exists(isSemanticallySorted)
}

spark.sql("CREATE TEMPORARY VIEW t1 AS SELECT * FROM VALUES (1, 'x'), (2, 'y') AS t1(a, b)")
spark.sql("CREATE TEMPORARY VIEW t2 AS SELECT * FROM VALUES (1, 'x'), (3, 'z') AS t2(a, b)")

Source.fromFile(dataDir + "/cases.jsonl").getLines().filter(_.nonEmpty).foreach { line =>
  val c = mapper.readTree(line)
  c.get("kind").asText match {
    case "probe" => probe(c.get("sql").asText, c.get("time_zone").asText, c.get("binary_style").asText)
    case "sorted" =>
      val sql = c.get("sql").asText
      val o = nf.objectNode()
      o.put("sql", sql)
      try {
        val analyzed = spark.sql(sql).queryExecution.analyzed
        o.put("sorted", isSemanticallySorted(analyzed))
        o.put("plan", analyzed.treeString.replaceAll("#\\d+L?", "#").trim)
      } catch { case e: Throwable => o.put("error", e.getClass.getName + ": " + String.valueOf(e.getMessage).take(200)) }
      emit("SORTED", o)
  }
}
println("@@VERSION " + spark.version)
println("@@END")
