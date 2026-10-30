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

mod access;
mod cache;
mod catalog;
mod column;
mod command;
mod conf;
mod dataframe;
mod execute_immediate;
mod explain;
mod explain_graph;
pub mod functions;
mod identifier_clause;
mod introspection;
mod profile;
mod session;
mod sql_functions;
mod store;
mod table_commands;
mod udfs;
mod variables;
mod window;

pub use access::{PathPrefixPolicy, StatementInfo};
pub use column::Column;
pub use conf::RuntimeConf;
pub use dataframe::{
    DataFrame, DataFrameReader, DataFrameWriter, DataFrameWriterV2, ExplainMode, GroupedData,
    JoinType, StorageLevel,
};
pub use explain_graph::{
    GraphFormat, dot_of_plan, dot_of_profile, mermaid_of_plan, mermaid_of_profile,
};
pub use krishiv_exec::profile::{QueryProfile, TextOptions};
pub use krishiv_exec::{
    CounterSample, GaugeSample, HistogramBucket, HistogramSample, MetricDict, MetricsSnapshot,
};
pub use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, AllowAll, DenyAll, Principal, PrincipalKind, Privilege,
    Securable, SecurableUse,
};
pub use krishiv_sql::{ExplainKind, StatementClass};
pub use krishiv_udf_sdk::{
    ArgField, InvokeContext, NullHandling, ReturnField, ScalarUdf, Signature, UdfArg, UdfError,
    UdfInputType, UdfLibrary, UdfProperties,
};
pub use profile::{ExplainVerbose, OptimizerStep, VERBOSE_FORMAT, VERBOSE_VERSION};
pub use session::{RecordBatchStream, Session, SessionBuilder};
/// Native Rust scalar UDFs: the [`ScalarUdf`] trait and its boundary types,
/// for [`Session::register_udf`].
pub use udfs::SessionFunctions;
pub use window::WindowSpec;
