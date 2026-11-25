# krishiv

Krishiv for JavaScript and TypeScript: SQL and DataFrames over Apache Arrow, against a Krishiv server or embedded in Node.

```ts
import { Session, col, count } from "krishiv";

const spark = await Session.builder()
  .remote("krishiv://localhost:15002") // krishivs:// for TLS
  .token(process.env.KRISHIV_TOKEN!)
  .getOrCreate();

const rows = await spark
  .sql("SELECT id, id % 3 AS bucket FROM range(100)")
  .filter(col("id").gt(10))
  .groupBy("bucket")
  .agg(count().alias("n"))
  .collect();
```

## Remote mode

`Session.builder().remote("krishiv://host:port")` talks Arrow Flight SQL to `krishiv server` (the Spark Connect port,
`krishiv.connect.port`, default 15002). The token is sent as a bearer token. A session keeps its settings and temporary
views between calls (the server's session cookie). Remote mode needs no native binary.

## Results

- `df.toArrow()` an Apache Arrow `Table`; `df.stream()` the record batches as they arrive.
- `df.collect()` plain objects. 64-bit integers are `bigint`; pass `{ bigint: "number" | "string" }` to change that, and
  `{ decimal: "number" }` for decimals (strings by default, so nothing is rounded).
- `toRows`, `toColumns`, `toJSON` (bigint-safe), `schemaToDDL` work on any Arrow table.

## Errors

Every failure is a `KrishivError` with `errorClass` and `sqlState` (the Spark error class, for example
`DIVIDE_BY_ZERO` and `22012`).

## Embedded engine

`Session.builder().getOrCreate()` runs the engine in your process through the optional package
`@krishiv/engine-<os>-<arch>`. Without a binary for your platform it throws a `KrishivError`; use remote mode instead.

Node 20 or newer. Apache-2.0.
