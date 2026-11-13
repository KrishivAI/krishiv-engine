<!-- SPDX-License-Identifier: Apache-2.0 -->
# Migrating JVM UDFs

Krishiv does not run the JVM. A Scala, Java or Hive UDF packaged in a JAR cannot be loaded, and a
registration that names one is refused. This guide shows what to use instead.

## What happens today

A statement that registers a function implemented by a JVM class fails at parse time with
`KRISHIV_UNSUPPORTED_FEATURE`, which names the feature it could not run:

```sql
CREATE TEMPORARY FUNCTION my_upper AS 'com.example.MyUpper' USING JAR 'my-udfs.jar';
-- KRISHIV_UNSUPPORTED_FEATURE: CREATE FUNCTION ... AS 'class' (a function implemented by a JVM class)
```

The statement is refused before anything runs, so nothing is registered and no JAR is read.

## Pick a replacement

Work down this list. The first option that fits your function is usually the simplest.

1. **A built-in function.** Most Hive and Spark-community UDFs have a Krishiv built-in with the same
   behaviour. The table below lists the common ones. Built-ins are the fastest option because they run in
   the native engine.
2. **A SQL-bodied function.** A function whose logic is a single SQL expression can be written as a
   temporary function. It is planned by the engine like any other expression:

   ```sql
   CREATE TEMPORARY FUNCTION add_one(x INT) RETURNS INT RETURN x + 1;
   SELECT add_one(41);  -- 42
   ```

3. **A Rust UDF.** A function that needs real code is written in Rust, compiled into the application, and
   registered on the session. See [Native Rust UDFs](../reference/rust-udfs.md).
4. **A Python UDF.** Not available yet. Python UDFs over Spark Connect are planned (P3 task T014). Until
   they ship, rewrite the logic as one of the options above.

## Common Hive and Spark-community UDFs

Each row names a Krishiv built-in with the same intent. Check the linked function page for the exact
argument types and null handling before you switch a query over.

| JVM UDF or idiom | Krishiv built-in | Notes |
|---|---|---|
| `NVL(a, b)` | [`nvl`](../reference/sql/functions/nvl.md) | Same as `coalesce(a, b)` for two arguments. |
| `NVL2(a, b, c)` | [`nvl2`](../reference/sql/functions/nvl2.md) | |
| `IF(cond, a, b)` | [`if`](../reference/sql/functions/if.md) | |
| `unix_timestamp(ts, fmt)` | [`unix_timestamp`](../reference/sql/functions/unix_timestamp.md) | The format string follows Spark's datetime patterns. |
| `from_unixtime(sec, fmt)` | [`from_unixtime`](../reference/sql/functions/from_unixtime.md) | |
| `to_date(s)` | [`to_date`](../reference/sql/functions/to_date.md) | |
| `date_format(ts, fmt)` | [`date_format`](../reference/sql/functions/date_format.md) | |
| `datediff(a, b)` | [`datediff`](../reference/sql/functions/datediff.md) | |
| `date_add`, `date_sub` | [`date_add`](../reference/sql/functions/date_add.md), [`date_sub`](../reference/sql/functions/date_sub.md) | |
| `months_between(a, b)` | [`months_between`](../reference/sql/functions/months_between.md) | |
| `from_utc_timestamp(ts, tz)` | [`from_utc_timestamp`](../reference/sql/functions/from_utc_timestamp.md) | |
| `regexp_extract(s, re, i)` | [`regexp_extract`](../reference/sql/functions/regexp_extract.md) | |
| `regexp_replace(s, re, r)` | [`regexp_replace`](../reference/sql/functions/regexp_replace.md) | |
| `s RLIKE re` | [`rlike`](../reference/sql/functions/rlike.md) | |
| `get_json_object(s, path)` | [`get_json_object`](../reference/sql/functions/get_json_object.md) | |
| `from_json`, `to_json` | [`from_json`](../reference/sql/functions/from_json.md), [`to_json`](../reference/sql/functions/to_json.md) | |
| `collect_list`, `collect_set` | [`collect_list`](../reference/sql/functions/collect_list.md), [`collect_set`](../reference/sql/functions/collect_set.md) | Aggregates. |
| `concat_ws(sep, ...)` | [`concat_ws`](../reference/sql/functions/concat_ws.md) | |
| `instr(s, sub)`, `locate(sub, s)` | [`instr`](../reference/sql/functions/instr.md), [`locate`](../reference/sql/functions/locate.md) | |
| `substr(s, pos, len)` | [`substr`](../reference/sql/functions/substr.md) | |
| `split(s, re)` | [`split`](../reference/sql/functions/split.md) | |
| `str_to_map(s)` | [`str_to_map`](../reference/sql/functions/str_to_map.md) | |
| `size(x)` | [`size`](../reference/sql/functions/size.md) | |
| `array_contains(a, v)` | [`array_contains`](../reference/sql/functions/array_contains.md) | |
| `map_keys`, `map_values` | [`map_keys`](../reference/sql/functions/map_keys.md), [`map_values`](../reference/sql/functions/map_values.md) | |
| `explode(x)` | [`explode`](../reference/sql/functions/explode.md) | A generator. |
| `pmod(a, b)` | [`pmod`](../reference/sql/functions/pmod.md) | |
| `hash(x)`, `xxhash64(x)` | [`hash`](../reference/sql/functions/hash.md), [`xxhash64`](../reference/sql/functions/xxhash64.md) | Spark's hash functions. |
| `md5(x)`, `sha2(x, bits)` | [`md5`](../reference/sql/functions/md5.md), [`sha2`](../reference/sql/functions/sha2.md) | |
| `percentile(x, p)`, `approx_count_distinct(x)` | [`percentile`](../reference/sql/functions/percentile.md), [`approx_count_distinct`](../reference/sql/functions/approx_count_distinct.md) | Aggregates. |

If a function you need is not in the table, check the [SQL function reference](../reference/sql/functions/index.md).
If it is missing there, rewrite it with option 2 or 3 above.

## Before you switch

- Run the old and new expression on a sample of your data and compare the results, including nulls and
  empty strings. Built-ins follow Spark's behaviour, but a hand-written UDF may have behaved differently.
- Replace each `CREATE FUNCTION ... USING JAR` statement in your migration scripts. The statement is refused
  wherever it runs, so leaving one in place stops the script.
- Keep the function name the same where you can, so existing queries need no edits.
