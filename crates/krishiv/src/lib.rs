// SPDX-License-Identifier: Apache-2.0
//! Krishiv: a Spark-compatible query engine, embedded.
//!
//! A [`Session`] runs Spark SQL ([`Session::sql`]) and DataFrames
//! ([`DataFrame`], built with [`functions`] and [`Column`]) in this
//! process. Results are Arrow record batches ([`arrow`]). Errors are
//! [`Error`]s carrying Spark's error class and SQLSTATE: match on
//! [`Error::error_class`] against [`ErrorClass`] variants, or compare
//! `error_class().name()` with Spark's class name.
//!
//! # Cargo features
//!
//! **Reserved features (no effect yet):** `kafka`, `iceberg`, `cdc`, `connect`, `flight-sql`, `mcp`,
//! `python`, `celeborn` and `mimalloc`. They are wired in 0.3.x.
//!
//! `kafka`, `iceberg`, `cdc`, `connect`, `flight-sql`, `mcp`, `python`, `wasm-udf`,
//! `celeborn` and `mimalloc` are reserved names: they enable no code in this crate
//! yet (features are additive, so they can gain code later without breaking
//! anyone who names them). The ones that do something are `delta`, `cloud`,
//! `collations-icu` (on by default), `ui`, `unstable-ir`, `tokio-console`, `telemetry-logs` and
//! `telemetry-otlp`; `full` turns on the reserved names and the optional backends together.
//!
//! ```
//! use std::sync::Arc;
//! use krishiv::arrow::array::{Int32Array, RecordBatch};
//! use krishiv::arrow::datatypes::{DataType, Field, Schema};
//! use krishiv::functions::col;
//! use krishiv::Session;
//!
//! let session = Session::new()?;
//! let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
//! let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![3, 1, 2]))]).unwrap();
//! session.register_batches("t", schema, vec![batch])?;
//!
//! let df = session.table("t").order_by([col("a").desc()]).limit(2);
//! let rows: Vec<RecordBatch> = futures::executor::block_on(df.collect())?;
//! assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
//! # Ok::<(), krishiv::Error>(())
//! ```
//!
//! Actions are `async` and run on the session's own executor, so any
//! runtime (or none, with `futures::executor::block_on`) can await them.

#![forbid(unsafe_code)]

// The crate README is a doc test: its example is compiled and run by `cargo test --doc`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub use krishiv_common::error::{ErrorClass, ErrorQueryContext, KrishivError as Error};
pub use krishiv_session::{
    AggregateProperties, AggregateUdf, ArgField, CancelReason, Column, CounterSample, DataFrame,
    DataFrameReader, DataFrameWriter, EngineRuntime, EngineRuntimeBuilder, Estimate, ExplainMode,
    ExplainVerbose, GaugeSample, GraphFormat, GroupedData, GroupsAccumulator, HistogramBucket,
    HistogramSample, InvokeContext, JoinType, MetricDict, MetricsSnapshot, NullHandling,
    OptimizerStep, QueryId, QueryOptions, QueryProfile, RecordBatchStream, ReturnField,
    RuntimeConf, ScalarUdf, Session, SessionBuilder, Signature, StateField, StorageLevel,
    TextOptions, UdfArg, UdfError, UdfInputType, UdfLibrary, UdfProperties, agent, dot_of_profile,
    functions, mermaid_of_profile, register_build_features,
};
/// Spark SQL data types, for [`Column::cast`].
pub use krishiv_types::DataType;

/// A result with Krishiv's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The Arrow types of results, at the version Krishiv is built with.
pub mod arrow {
    /// Arrays and record batches (`arrow-array`).
    pub use arrow_array as array;
    /// Schemas, fields and Arrow data types (`arrow-schema`).
    pub use arrow_schema as datatypes;
}

/// Admin and UI server (feature `ui`).
#[cfg(feature = "ui")]
pub use krishiv_server as server;

#[cfg(feature = "ui")]
mod console;
#[cfg(feature = "ui")]
pub use console::{Console, ConsoleError, ui, ui_on_port};

/// Starts the tokio-console subscriber when the `tokio-console` feature is
/// on and `KRISHIV_TOKIO_CONSOLE=1` (see `krishiv_exec::console`); a no-op
/// otherwise. The CLI calls this at startup; embedders call it before
/// building a session.
#[cfg(feature = "tokio-console")]
pub use krishiv_exec::maybe_init_tokio_console;
