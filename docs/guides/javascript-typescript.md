---
description: "Query a Krishiv server from Node.js or TypeScript with the krishiv npm package: SQL, DataFrames, Arrow results and errors."
---

# JavaScript and TypeScript

The `krishiv` npm package (0.3.1 and later) runs SQL and DataFrame queries against a Krishiv server from Node.js 22 or
newer. It speaks Arrow Flight SQL on the server's Spark Connect port, so it needs no native binary and no JVM. Results
come back as Apache Arrow, or as plain objects. The embedded engine is not available from JavaScript yet; start a
server first (see [Deployment](deploy.md) and [Securing a single-node server](single-node-server-security.md)).

## Connect

```ts
import { Session, col, count } from "krishiv";

const spark = await Session.builder()
  .remote("krishiv://localhost:15002") // krishivs:// for TLS
  .token(process.env.KRISHIV_TOKEN!)
  .config("spark.sql.session.timeZone", "UTC")
  .getOrCreate();
```

The port is `krishiv.connect.port` (default 15002). The token is sent as a bearer token. Settings made with `config()` or
`spark.conf.set` and temporary views stay with the session between calls. Close the session when you are done:
`await spark.close()`, or declare it with `await using`.

## Query

`spark.sql(text)` and `spark.read.parquet|csv|json(path)` return a lazy `DataFrame`; nothing runs until an action.
A DataFrame builds SQL text, so it behaves the same on any backend.

```ts
const rows = await spark
  .sql("SELECT id, id % 3 AS bucket FROM range(100)")
  .filter(col("id").gt(10))
  .groupBy("bucket")
  .agg(count().alias("n"))
  .collect();
```

Available transformations: `select`, `filter`/`where`, `orderBy`, `limit`, `distinct`, `withColumn`, `groupBy().agg()`
and `join`. `toSQL()` shows the generated query. For anything else, write SQL.

`lit()` quotes strings for the default parser. If you set `spark.sql.parser.escapedStringLiterals` to `true`, a backslash
is kept as written, so build such text yourself rather than through `lit()`.

## Results

| Call | Returns |
|---|---|
| `df.toArrow()` | an Arrow `Table` |
| `df.stream()` | the record batches as they arrive |
| `df.collect()` | plain objects |

In `collect()`, 64-bit integers are `bigint`; pass `{ bigint: "number" }` or `{ bigint: "string" }` to change that, and
`{ decimal: "number" }` for decimals (strings by default, so nothing is rounded). Dates are `YYYY-MM-DD` strings and
timestamps `YYYY-MM-DDTHH:MM:SS.ffffffZ` strings. Lists and structs become arrays and objects. `toRows`, `toColumns`,
`toJSON` (safe for `bigint`) and `schemaToDDL` work on any Arrow table.

## Errors

A failed query throws a `KrishivError` carrying the Spark error class and SQLSTATE:

```ts
try {
  await spark.sql("SELECT 1/0").collect();
} catch (e) {
  if (e instanceof KrishivError) console.log(e.errorClass, e.sqlState); // DIVIDE_BY_ZERO 22012
}
```

Import `KrishivError` from `krishiv`. `1/0` fails with `DIVIDE_BY_ZERO` while ANSI mode is on (see
[Session settings](session-settings.md)).
