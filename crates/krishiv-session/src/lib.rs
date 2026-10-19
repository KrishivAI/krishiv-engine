// SPDX-License-Identifier: Apache-2.0
//! Krishiv sessions (layer L5): Spark SQL and DataFrames over the engine.
//!
//! A [`Session`] holds the configuration, the temporary tables and views,
//! and the engine that runs queries. [`Session::sql`] and the
//! [`DataFrame`] builders produce Krishiv's logical plan; actions analyze
//! it (Spark 4.2's rules), optimize it, plan it physically and run it on
//! the session's executor. Every API surface (the Rust facade `krishiv`,
//! the Python facade, the Spark Connect and Flight SQL servers) goes
//! through this one pipeline.
//!
//! ```
//! use std::sync::Arc;
//! use arrow_array::{Int32Array, RecordBatch};
//! use arrow_schema::{DataType, Field, Schema};
//! use krishiv_session::Session;
//! use krishiv_session::functions::col;
//!
//! let session = Session::new()?;
//! let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
//! let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![3, 1, 2]))]).unwrap();
//! session.register_batches("t", schema, vec![batch])?;
//!
//! let df = futures::executor::block_on(session.sql("SELECT a FROM t ORDER BY a"))?;
//! let rows = futures::executor::block_on(df.limit(2).collect())?;
//! assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
//! # Ok::<(), krishiv_common::error::KrishivError>(())
//! ```
//!
//! # Semantics
//! - Transformations only build the plan; analysis errors surface at the
//!   first action (Spark Connect), except for [`Session::sql`], which
//!   analyzes at once (Spark).
//! - Each query reads a snapshot of the configuration taken when it
//!   starts.
//! - Actions run on the session's executor, so they work from any async
//!   runtime. Dropping an action's future or result stream cancels its
//!   query.
//! - The logical plan IR is reachable only with the `unstable-ir` feature,
//!   outside the semver promise.

#![forbid(unsafe_code)]

mod catalog;
mod column;
mod conf;
mod dataframe;
pub mod functions;
mod session;

pub use column::Column;
pub use conf::RuntimeConf;
pub use dataframe::{
    DataFrame, DataFrameReader, DataFrameWriter, ExplainMode, GroupedData, JoinType,
};
pub use session::{RecordBatchStream, Session, SessionBuilder};
