// SPDX-License-Identifier: Apache-2.0
// Run by gen_spark_sql_defaults.py inside a Spark release's spark-shell: prints every registered SQL config
// entry (`SQLConf.getConfigEntries`, internal and static ones included) with what `spark.conf.get` answers for
// it when it is unset. `getConfigEntries` is private[spark] (public bytecode), so it is called by reflection.
//
// The answer is Spark's own: `getConfString(key)` of a fresh `SQLConf`, which is
// `entry.stringConverter(entry.readFrom(reader))` (v4.3.0 `SQLConf.scala:9801-9808`), so a bytes or time entry
// answers its converted value (`spark.sql.broadcastTimeout` = `300000ms`), not its default's raw text.
//
// One line per entry: `@@E <key> <kind> <value> <static>`, tab-separated, where kind is
//   value    : the entry answers a value; value is its text
//   none     : an optional entry: `get` answers no value (`<undefined>` in `SET`)
//   fallback : the entry reads another SQL entry when unset; value is that entry's key (a fallback to an entry
//              SQLConf does not register, a core one such as `spark.shuffle.*`, answers that entry's default:
//              printed as a value)
//   function : the default is computed (`createWithDefaultFunction`); value is what it computed here
// and static is `true` for a `StaticSQLConf` key.
//
// Then, after one query has run (so the objects a running session loads have registered their entries), one
// line per Spark config entry that is not a SQL one (`ConfigEntry.knownConfigs`, private, read by reflection,
// minus the keys `SQLConf.containsConfigKey` knows): `@@C <key>`. Spark refuses those at runtime
// (`RuntimeConfig.requireNonStaticConf`: `CANNOT_MODIFY_CONFIG`).
import scala.collection.JavaConverters._

val c = Class.forName("org.apache.spark.sql.internal.SQLConf$")
val mod = c.getField("MODULE$").get(null)
val entries = c.getMethod("getConfigEntries").invoke(mod).asInstanceOf[java.util.Collection[AnyRef]].asScala.toSeq
val isStatic = c.getMethod("isStaticConfigKey", classOf[String])
val fresh = new org.apache.spark.sql.internal.SQLConf()

def call(o: AnyRef, name: String): AnyRef = {
  val m = o.getClass.getMethod(name)
  m.setAccessible(true)
  m.invoke(o)
}

def field(o: AnyRef, name: String): AnyRef = {
  val f = o.getClass.getDeclaredField(name)
  f.setAccessible(true)
  f.get(o)
}

val sqlKeys = entries.map(e => String.valueOf(call(e, "key"))).toSet

// What `spark.conf.get(key)` answers unset: null is no value.
def answer(key: String): (String, String) =
  try {
    val v = fresh.getConfString(key)
    if (v == null) ("none", "") else ("value", v)
  } catch { case _: java.util.NoSuchElementException => ("none", "") }

def clean(s: String): String =
  s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")

entries
  .map { e =>
    val key = String.valueOf(call(e, "key"))
    val (kind, value) = e.getClass.getSimpleName match {
      case "FallbackConfigEntry" =>
        val toKey = String.valueOf(call(field(e, "fallback"), "key"))
        if (sqlKeys.contains(toKey)) ("fallback", toKey) else answer(key)
      case "ConfigEntryWithDefaultFunction" => ("function", answer(key)._2)
      case _ => answer(key)
    }
    (key, kind, value, isStatic.invoke(mod, key).toString)
  }
  .sortBy(_._1)
  .foreach { case (k, kind, v, s) => println("@@E\t" + k + "\t" + kind + "\t" + clean(v) + "\t" + s) }
spark.range(1).collect()
val contains = c.getMethod("containsConfigKey", classOf[String])
val known = Class.forName("org.apache.spark.internal.config.ConfigEntry$")
val knownMod = known.getField("MODULE$").get(null)
val knownField = known.getDeclaredFields.find(_.getName.endsWith("knownConfigs")).get
knownField.setAccessible(true)
knownField.get(knownMod).asInstanceOf[java.util.Map[String, AnyRef]].keySet.asScala.toSeq
  .filterNot(k => contains.invoke(mod, k).asInstanceOf[Boolean])
  .sorted
  .foreach(k => println("@@C\t" + k))
println("@@END")
System.exit(0)
