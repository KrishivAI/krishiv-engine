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

`Session.builder().getOrCreate()` runs the engine in your process. npm installs the engine binary for your machine as
one optional package, at the same version as `krishiv`:

| Platform | Package |
|---|---|
| Linux x64, glibc 2.28+ | `@krishiv/engine-linux-x64-gnu` |
| Linux arm64, glibc 2.28+ | `@krishiv/engine-linux-arm64-gnu` |
| Linux x64, musl (Alpine) | `@krishiv/engine-linux-x64-musl` |
| macOS arm64, 11.0+ | `@krishiv/engine-darwin-arm64` |

The engine loads only when a session uses embedded mode, never at import. Without a binary for your platform (or
with `--omit=optional`), `getOrCreate()` throws a `KrishivError` with class `KRISHIV_NATIVE_ENGINE_UNAVAILABLE`; use
remote mode instead. Embedded mode runs on Node 22 and 24, Bun and Deno (`deno run --allow-all`, with the packages
installed in `node_modules`). To load a local build instead of the package, set `KRISHIV_NATIVE_ENGINE` to the path of
the `.node` file (`cargo build -p krishiv-node`, then copy `libkrishiv_node.so` to `krishiv.node`).

Node 22 or newer. Apache-2.0.
