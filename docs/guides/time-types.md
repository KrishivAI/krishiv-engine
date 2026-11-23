---
description: "The TIME(p) type: turn it on, write and cast TIME values, the time functions, and what it cannot do yet."
---

# The TIME type

`TIME(p)` is a time of day with no date and no time zone. `p` is the number of fraction digits, from 0 to 9; `TIME`
alone is `TIME(6)`. It follows Spark 4.2.

## Turn it on

Spark keeps the written type off by default, and Krishiv does the same. With `spark.sql.timeType.enabled` off, a query
that writes `TIME(p)` fails with `UNSUPPORTED_TIME_TYPE`, although a `TIME` value that a function produces, such as
`current_time()`, still works. Set the key to `true` to write the type:

```bash
krishiv sql --conf spark.sql.timeType.enabled=true -o csv \
  -e "SELECT CAST(TIME'12:34:56.789' AS STRING) AS t, typeof(CAST('08:15:30' AS TIME(0))) AS type"
```

```text
t,type
12:34:56.789,time(0)
```

In Python or Scala set it on the session: `spark.conf.set("spark.sql.timeType.enabled", "true")`.

## Make and read values

```sql
SELECT CAST(make_time(9, 30, 15.5) AS STRING);           -- 09:30:15.5
SELECT hour(TIME'12:34:56'), minute(TIME'12:34:56'), second(TIME'12:34:56');   -- 12, 34, 56
SELECT CAST(TIME'12:00:00' + INTERVAL '1' HOUR AS STRING);   -- 13:00:00
SELECT time_diff('HOUR', TIME'08:00:00', TIME'17:30:00');    -- 9
SELECT CAST(time_trunc('HOUR', TIME'12:34:56') AS STRING);   -- 12:00:00
SELECT TIME'12:00:00' < TIME'13:00:00';                      -- true
```

The `TIME` functions are listed in the [function reference](../reference/sql/functions/index.md): `current_time`,
`make_time`, `time_diff`, `time_trunc`, `hour`, `minute`, `second` and the casts.

## Limits

- Cast a `TIME` to `STRING` to print it. In 0.3 `krishiv sql` prints a `TIME` column as the text `<Time64(ns)>` in
  every output format, so the examples above cast first.
- A `TIME` cannot be cast to or from `TIMESTAMP` or `DATE`; the cast fails with
  `DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION`, as in Spark 4.2.
