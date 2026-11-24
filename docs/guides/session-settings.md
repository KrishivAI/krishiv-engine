---
description: "Set the session time zone and ANSI mode, and see which Spark version Krishiv follows."
---

# Session settings: time zone, ANSI mode and Spark version

Two Spark settings change what a query returns, and Krishiv reads both under their Spark names. Set them on the session
with `spark.conf.set` or in SQL with `SET`; the [configuration reference](../reference/config/index.md) lists every key
with its `krishiv.` name.

## Session time zone

`spark.sql.session.timeZone` (`krishiv.sql.session.timeZone`) is a region ID such as `America/Los_Angeles` or a zone
offset. It decides how a timestamp is shown, how a string without a zone is read as a timestamp, and which day
`current_date()` returns. When it is unset, the local system time zone is used, as in Spark.

```python
spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
```

```sql
SET spark.sql.session.timeZone = America/Los_Angeles;
```

## ANSI mode

`spark.sql.ansi.enabled` (`krishiv.sql.ansi.enabled`) is `true` by default. In ANSI mode an invalid cast or an arithmetic
overflow raises an error. Set it to `false` and the same expression returns NULL, the Spark 3 behaviour:

```python
spark.conf.set("spark.sql.ansi.enabled", "false")
```

A SQL function keeps the ANSI mode of the session that created it; see
[known differences](../concepts/known-differences.md).

## Which Spark version

The Spark version Krishiv follows is listed in [Spark compatibility](../concepts/compatibility.md), under SQL. Where
Krishiv differs from that version, [known differences](../concepts/known-differences.md) says how.

## See also

- [Data sources](data-sources.md): read CSV and Parquet files, and use S3.
- [The TIME type](time-types.md): a time of day with no date and no time zone.
- [Migrate a PySpark job](pyspark-migration.md).
