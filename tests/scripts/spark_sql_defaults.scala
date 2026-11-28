// SPDX-License-Identifier: Apache-2.0
// Run by gen_spark_sql_defaults.py inside a Spark release's spark-shell: prints every registered SQL config
// entry (`SQLConf.getConfigEntries`, internal and static ones included) with what `spark.conf.get` answers for
// it when it is unset. `getConfigEntries` is private[spark] (public bytecode), so it is called by reflection.
//
// One line per entry: `@@E <key> <kind> <value> <static>`, tab-separated, where kind is
//   value    : the entry has a default; value is its text (`defaultValueString`)
//   none     : an optional entry: `get` answers no value (`<undefined>` in `SET`)
//   fallback : the entry reads another SQL entry when unset; value is that entry's key (a fallback to an entry
//              SQLConf does not register, a core one such as `spark.shuffle.*`, is printed as that entry's
//              default: `spark.conf.get` answers it, while a get of the core key itself is SQL_CONF_NOT_FOUND)
//   function : the default is computed (`createWithDefaultFunction`); value is what it computed here
// and static is `true` for a `StaticSQLConf` key.
import scala.collection.JavaConverters._

val c = Class.forName("org.apache.spark.sql.internal.SQLConf$")
val mod = c.getField("MODULE$").get(null)
val entries = c.getMethod("getConfigEntries").invoke(mod).asInstanceOf[java.util.Collection[AnyRef]].asScala.toSeq
val isStatic = c.getMethod("isStaticConfigKey", classOf[String])

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

// The answer of an entry SQLConf does not register, read through its fallbacks.
def resolved(e: AnyRef): (String, String) = e.getClass.getSimpleName match {
  case "FallbackConfigEntry" => resolved(field(e, "fallback"))
  case "OptionalConfigEntry" => ("none", "")
  case "ConfigEntryWithDefaultFunction" => ("function", String.valueOf(call(e, "defaultValueString")))
  case _ => ("value", String.valueOf(call(e, "defaultValueString")))
}

def clean(s: String): String =
  s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")

entries
  .map { e =>
    val key = String.valueOf(call(e, "key"))
    val cls = e.getClass.getSimpleName
    val (kind, value) = cls match {
      case "FallbackConfigEntry" =>
        val to = field(e, "fallback")
        val toKey = String.valueOf(call(to, "key"))
        if (sqlKeys.contains(toKey)) ("fallback", toKey) else resolved(to)
      case "OptionalConfigEntry" => ("none", "")
      case "ConfigEntryWithDefaultFunction" => ("function", String.valueOf(call(e, "defaultValueString")))
      case _ => ("value", String.valueOf(call(e, "defaultValueString")))
    }
    (key, kind, value, isStatic.invoke(mod, key).toString)
  }
  .sortBy(_._1)
  .foreach { case (k, kind, v, s) => println("@@E\t" + k + "\t" + kind + "\t" + clean(v) + "\t" + s) }
println("@@END")
System.exit(0)
