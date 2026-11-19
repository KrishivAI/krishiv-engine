---
description: "Krishiv: a Spark-compatible query engine, embedded."
---

**Version:** 0.2.0

# Module `krishiv`

Krishiv: a Spark-compatible query engine, embedded.

A [`Session`] runs Spark SQL ([`Session::sql`]) and DataFrames
([`DataFrame`], built with [`functions`] and [`Column`]) in this
process. Results are Arrow record batches ([`arrow`]). Errors are
[`Error`]s carrying Spark's error class and SQLSTATE.

```
use std::sync::Arc;
use krishiv::arrow::array::{Int32Array, RecordBatch};
use krishiv::arrow::datatypes::{DataType, Field, Schema};
use krishiv::functions::col;
use krishiv::Session;

let session = Session::new()?;
let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![3, 1, 2]))]).unwrap();
session.register_batches("t", schema, vec![batch])?;

let df = session.table("t").order_by([col("a").desc()]).limit(2);
let rows: Vec<RecordBatch> = futures::executor::block_on(df.collect())?;
assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
# Ok::<(), krishiv::Error>(())
```

Actions are `async` and run on the session's own executor, so any
runtime (or none, with `futures::executor::block_on`) can await them.

## Modules

## Module `arrow`

The Arrow types of results, at the version Krishiv is built with.

```rust
pub mod arrow { /* ... */ }
```

### Re-exports

#### Re-export `arrow_array`

Arrays and record batches (`arrow-array`).

```rust
pub use arrow_array as array;
```

#### Re-export `arrow_schema`

Schemas, fields and Arrow data types (`arrow-schema`).

```rust
pub use arrow_schema as datatypes;
```

## Types

### Type Alias `Result`

A result with Krishiv's [`Error`].

```rust
pub type Result<T, E = Error> = std::result::Result<T, E>;
```

## Re-exports

### Re-export `KrishivError`

```rust
pub use krishiv_common::error::KrishivError as Error;
```

### Re-export `AggregateProperties`

```rust
pub use krishiv_session::AggregateProperties;
```

### Re-export `AggregateUdf`

```rust
pub use krishiv_session::AggregateUdf;
```

### Re-export `ArgField`

```rust
pub use krishiv_session::ArgField;
```

### Re-export `CancelReason`

```rust
pub use krishiv_session::CancelReason;
```

### Re-export `Column`

```rust
pub use krishiv_session::Column;
```

### Re-export `CounterSample`

```rust
pub use krishiv_session::CounterSample;
```

### Re-export `DataFrame`

```rust
pub use krishiv_session::DataFrame;
```

### Re-export `DataFrameReader`

```rust
pub use krishiv_session::DataFrameReader;
```

### Re-export `DataFrameWriter`

```rust
pub use krishiv_session::DataFrameWriter;
```

### Re-export `EngineRuntime`

```rust
pub use krishiv_session::EngineRuntime;
```

### Re-export `EngineRuntimeBuilder`

```rust
pub use krishiv_session::EngineRuntimeBuilder;
```

### Re-export `Estimate`

```rust
pub use krishiv_session::Estimate;
```

### Re-export `ExplainMode`

```rust
pub use krishiv_session::ExplainMode;
```

### Re-export `ExplainVerbose`

```rust
pub use krishiv_session::ExplainVerbose;
```

### Re-export `GaugeSample`

```rust
pub use krishiv_session::GaugeSample;
```

### Re-export `GraphFormat`

```rust
pub use krishiv_session::GraphFormat;
```

### Re-export `GroupedData`

```rust
pub use krishiv_session::GroupedData;
```

### Re-export `GroupsAccumulator`

```rust
pub use krishiv_session::GroupsAccumulator;
```

### Re-export `HistogramBucket`

```rust
pub use krishiv_session::HistogramBucket;
```

### Re-export `HistogramSample`

```rust
pub use krishiv_session::HistogramSample;
```

### Re-export `InvokeContext`

```rust
pub use krishiv_session::InvokeContext;
```

### Re-export `JoinType`

```rust
pub use krishiv_session::JoinType;
```

### Re-export `MetricDict`

```rust
pub use krishiv_session::MetricDict;
```

### Re-export `MetricsSnapshot`

```rust
pub use krishiv_session::MetricsSnapshot;
```

### Re-export `NullHandling`

```rust
pub use krishiv_session::NullHandling;
```

### Re-export `OptimizerStep`

```rust
pub use krishiv_session::OptimizerStep;
```

### Re-export `QueryId`

```rust
pub use krishiv_session::QueryId;
```

### Re-export `QueryOptions`

```rust
pub use krishiv_session::QueryOptions;
```

### Re-export `QueryProfile`

```rust
pub use krishiv_session::QueryProfile;
```

### Re-export `RecordBatchStream`

```rust
pub use krishiv_session::RecordBatchStream;
```

### Re-export `ReturnField`

```rust
pub use krishiv_session::ReturnField;
```

### Re-export `RuntimeConf`

```rust
pub use krishiv_session::RuntimeConf;
```

### Re-export `ScalarUdf`

```rust
pub use krishiv_session::ScalarUdf;
```

### Re-export `Session`

```rust
pub use krishiv_session::Session;
```

### Re-export `SessionBuilder`

```rust
pub use krishiv_session::SessionBuilder;
```

### Re-export `Signature`

```rust
pub use krishiv_session::Signature;
```

### Re-export `StateField`

```rust
pub use krishiv_session::StateField;
```

### Re-export `StorageLevel`

```rust
pub use krishiv_session::StorageLevel;
```

### Re-export `TextOptions`

```rust
pub use krishiv_session::TextOptions;
```

### Re-export `UdfArg`

```rust
pub use krishiv_session::UdfArg;
```

### Re-export `UdfError`

```rust
pub use krishiv_session::UdfError;
```

### Re-export `UdfInputType`

```rust
pub use krishiv_session::UdfInputType;
```

### Re-export `UdfLibrary`

```rust
pub use krishiv_session::UdfLibrary;
```

### Re-export `UdfProperties`

```rust
pub use krishiv_session::UdfProperties;
```

### Re-export `agent`

```rust
pub use krishiv_session::agent;
```

### Re-export `dot_of_profile`

```rust
pub use krishiv_session::dot_of_profile;
```

### Re-export `functions`

```rust
pub use krishiv_session::functions;
```

### Re-export `mermaid_of_profile`

```rust
pub use krishiv_session::mermaid_of_profile;
```

### Re-export `register_build_features`

```rust
pub use krishiv_session::register_build_features;
```

### Re-export `DataType`

Spark SQL data types, for [`Column::cast`].

```rust
pub use krishiv_types::DataType;
```
