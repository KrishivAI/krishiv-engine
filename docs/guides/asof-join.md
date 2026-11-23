---
description: "ASOF JOIN: join each row to the nearest earlier (or later) row of another table, such as a trade to the quote before it."
---

# As-of joins

An as-of join matches each row of the left table to the one right-hand row that is nearest to it in time (or in another
ordered column; a forward join accepts fewer column types, see Limits) instead of an exact one. Use it to attach the latest quote to a trade, the price in force at an
order, or the last reading before an event. `ASOF JOIN` is Spark 4.3 syntax.

## Turn it on

Spark keeps `ASOF JOIN` off by default, and Krishiv does the same. With the switch off, the statement fails with
`UNSUPPORTED_FEATURE.ASOF_JOIN`:

```sql
SET spark.sql.join.asofJoin.enabled = true;
```

## Join a trade to the quote before it

```sql
CREATE TEMPORARY VIEW trades AS SELECT * FROM VALUES
  (1, 'AAPL', 10), (2, 'AAPL', 20), (3, 'MSFT', 15) AS trades(id, sym, ts);
CREATE TEMPORARY VIEW quotes AS SELECT * FROM VALUES
  ('AAPL', 5, 100.0), ('AAPL', 18, 101.0), ('MSFT', 16, 300.0) AS quotes(sym, ts, px);

SELECT t.id, t.ts, q.ts AS quote_ts, q.px
FROM trades t LEFT ASOF JOIN quotes q
  MATCH_CONDITION (t.ts >= q.ts)
  ON t.sym = q.sym
ORDER BY t.id;
```

```text
id,ts,quote_ts,px
1,10,5,100.0
2,20,18,101.0
3,15,NULL,NULL
```

`ON` holds the equality (and any other) condition that rows must meet; `MATCH_CONDITION` picks the nearest row among
those. Trade 3 has no MSFT quote at or before time 15, so a `LEFT` join keeps it with NULLs, and an inner `ASOF JOIN`
drops it.

## The match condition

`MATCH_CONDITION` is one comparison between a left expression and a right expression:

| Condition | Picks, among the right rows that pass `ON` |
|---|---|
| `t.ts >= q.ts` | the greatest `q.ts` at most `t.ts` (backward) |
| `t.ts > q.ts` | the greatest `q.ts` strictly before `t.ts` |
| `t.ts <= q.ts` | the least `q.ts` at least `t.ts` (forward) |
| `t.ts < q.ts` | the least `q.ts` strictly after `t.ts` |

The operands may be written either way round: `q.ts <= t.ts` is `t.ts >= q.ts`. When several right rows share the
chosen value, Spark leaves the pick open and so does Krishiv. A row whose match value is NULL matches nothing, and a NULL key matches nothing under `=`; under the null-safe `<=>`
in `ON`, NULL keys match each other.
Extra conditions in `ON` are applied before the nearest row is chosen, so a right row that fails them is skipped.

## Limits

- A forward join (`<=` or `<`) refuses a floating-point match column (`DOUBLE`, `FLOAT`), an integral or non-exact
  decimal one when ANSI mode is off, and a `TIMESTAMP` in a session time zone whose UTC offset can decrease. A backward
  join does not apply these checks.
- Only inner and `LEFT` (`LEFT OUTER`) as-of joins exist. `RIGHT`, `FULL`, `CROSS`, `NATURAL`, `SEMI` and `ANTI`
  before `ASOF` are a `PARSE_SYNTAX_ERROR`, as in Spark.
- `MATCH_CONDITION` is required: an as-of join without one is a `PARSE_SYNTAX_ERROR`.
- Spark Connect's `AsOfJoin` relation covers the backward and forward directions; its `nearest` direction and
  `tolerance` are not supported yet.
- The join runs as a sort-merge as-of join (`SortMergeAsOfJoin` in `EXPLAIN`), which sorts its inputs, so a large
  right side costs a sort.
