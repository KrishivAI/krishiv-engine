---
description: "Define a metric view once with CREATE VIEW ... WITH METRICS, then read its measures with measure() under any grouping."
---

# Metric views

A metric view stores a business metric next to the dimensions you group it by, so every query that reads the metric
gets the same definition. You define the view once and read each measure with `measure()`. This is Spark 4.2's
metric view syntax and is available when the session's Spark profile is 4.2 or later.

## Define a view

The definition is YAML between `$$` markers:

```sql
CREATE TABLE src (region STRING, n INT) USING parquet;
INSERT INTO src VALUES ('a', 1), ('a', 2), ('b', 5);

CREATE VIEW mv WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: src
dimensions:
  - name: region
    expr: region
measures:
  - name: total
    expr: sum(n)
$$;
```

`source` names the table or view the metric is computed from. Each dimension and measure has a `name` and an `expr`;
a measure's `expr` is an aggregate.

## Read a measure

Wrap each measure in `measure()` and group by the dimensions:

```sql
SELECT region, measure(total) FROM mv GROUP BY region ORDER BY region;
```

```text
a  3
b  5
```

A different grouping uses the same definition: `SELECT measure(total) FROM mv` returns 8.

## Limits

- A read of a metric view needs an aggregate. `SELECT * FROM mv` reads the stored schema (`region`, `total`) but
  does not run, as in Spark; group by the dimensions and use `measure()`.
- `measure()` takes a measure's name. A dimension or a projection's alias is not a measure and is refused.
- A measure that contains a subquery or a SQL user-defined function is refused with
  `KRISHIV_UNSUPPORTED_FEATURE` when it is read.
- The reader needs the right to read the source table: a metric view does not run with its creator's rights.
- Releases before Spark 4.2 do not have the syntax. Under a 4.1 profile, `CREATE VIEW ... WITH METRICS` is a
  `PARSE_SYNTAX_ERROR`.
- A dimension is cast to its stored type when read. A source column that was widened after the view was created
  fails the read.

See [`measure`](../reference/sql/functions/measure.md) in the function reference.
