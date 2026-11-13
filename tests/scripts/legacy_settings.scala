// SPDX-License-Identifier: Apache-2.0
// Run by gen_legacy_settings.py inside a Spark release's spark-shell: prints every registered
// `spark.sql.legacy.*` SQL config with its default. `SQLConf.getConfigEntries` is private[spark] (it is public
// bytecode), so it is called by reflection; the entries are read the same way (Scala 2.12 and 2.13).
import scala.collection.JavaConverters._

val c = Class.forName("org.apache.spark.sql.internal.SQLConf$")
val mod = c.getField("MODULE$").get(null)
val entries = c.getMethod("getConfigEntries").invoke(mod).asInstanceOf[java.util.Collection[AnyRef]].asScala.toSeq

def call(o: AnyRef, name: String): String = {
  val m = o.getClass.getMethod(name)
  m.setAccessible(true)
  String.valueOf(m.invoke(o))
}

entries
  .map(e => (call(e, "key"), call(e, "defaultValueString")))
  .filter(_._1.startsWith("spark.sql.legacy."))
  .sortBy(_._1)
  .foreach { case (k, d) => println("@@L\t" + k + "\t" + d) }
println("@@END")
System.exit(0)
