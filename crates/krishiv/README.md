# krishiv

**Spark alternative in Rust: run Spark SQL and DataFrames embedded, vectorized on Arrow, no JVM.**

Krishiv runs inside your process: no cluster and no JVM. Queries follow Spark's SQL and DataFrame semantics and
error classes, execute vectorized on Apache Arrow, and return Arrow record batches.

```bash
cargo add krishiv
```

```rust
use std::sync::Arc;
use krishiv::arrow::array::{Int32Array, RecordBatch};
use krishiv::arrow::datatypes::{DataType, Field, Schema};
use krishiv::functions::col;
use krishiv::Session;

let session = Session::new()?;
let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![3, 1, 2]))]).unwrap();
session.register_batches("t", schema, vec![batch])?;

// DataFrames and SQL run on the same engine.
let top = session.table("t").order_by([col("a").desc()]).limit(2);
let rows: Vec<RecordBatch> = futures::executor::block_on(top.collect())?;
assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
# Ok::<(), krishiv::Error>(())
```

Actions are `async` and run on the session's own executor, so any runtime (or none, with
`futures::executor::block_on`) can await them.

## Speed

Benchmark results against Spark 4.2.0 on TPC-H-derived queries (not official TPC results) are published with their
method and numbers at <https://krishiv.ai/benchmarks>.

## Status

Alpha. The embedded batch core works; streaming, incremental views, Spark Connect and distributed execution are
still being built. There is no stable API yet. Python users: `pip install krishiv`.

## Links

- Website: <https://krishiv.ai>
- Documentation: <https://krishiv.ai/docs>
- Source and issues: <https://github.com/KrishivAI/krishiv-engine>
- Releases and changelog: <https://github.com/KrishivAI/krishiv-engine/releases>

Licensed under Apache-2.0. Apache Spark is a trademark of the Apache Software Foundation; Krishiv is not
affiliated with it.
