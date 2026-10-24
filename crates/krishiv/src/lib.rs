// SPDX-License-Identifier: Apache-2.0
//! Krishiv: a Spark-compatible query engine, embedded (layer L7).
//!
//! A [`Session`] runs Spark SQL ([`Session::sql`]) and DataFrames
//! ([`DataFrame`], built with [`functions`] and [`Column`]) in this
//! process. Results are Arrow record batches ([`arrow`]). Errors are
//! [`Error`]s carrying Spark's error class and SQLSTATE.
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

pub use krishiv_common::error::KrishivError as Error;
pub use krishiv_session::{
    Column, DataFrame, DataFrameReader, DataFrameWriter, ExplainMode, ExplainVerbose, GraphFormat,
    GroupedData, JoinType, OptimizerStep, QueryProfile, RecordBatchStream, RuntimeConf, Session,
    SessionBuilder, TextOptions, dot_of_plan, dot_of_profile, functions, mermaid_of_plan,
    mermaid_of_profile,
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
