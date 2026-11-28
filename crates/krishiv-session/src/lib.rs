// SPDX-License-Identifier: Apache-2.0
//! Krishiv sessions: Spark SQL and DataFrames over the engine.
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
mod action;
mod admin_groups;
mod admin_view;
mod admission;
mod analyze;
mod arrow_values;
pub mod auth;
mod cache;
mod call;
mod cancel;
mod carrier;
mod catalog;
pub mod catalog_decl;
pub mod catalog_registry;
mod char_write;
mod collation_defaults;
mod column;
mod column_stats;
mod command;
mod conf;
mod config_defaults;
mod dataframe;
mod delta;
pub mod egress;
pub mod events;
mod execute_immediate;
mod explain;
mod explain_graph;
pub mod functions;
pub mod grants;
mod history;
mod identifier_clause;
mod introspection;
mod introspection_functions;
#[cfg(test)]
mod lineage_tests;
#[cfg(feature = "unstable-ir")]
mod local_relation;
mod locals;
#[cfg(feature = "telemetry-logs")]
pub mod logging;
mod memory_table;
mod metric_view;
mod monitor;
mod nested_value;
mod observation;
#[cfg(feature = "telemetry-otlp")]
pub mod otlp;
#[cfg(feature = "unstable-ir")]
pub mod parameter_types;
mod partition_advice;
mod partition_keys;
mod partitions;
mod plan_round_trip;
mod profile;
mod profile_history;
pub use profile_history::{StoredProfile, read_history as read_profile_history};
mod information_schema;
mod python_udfs;
mod query_fold;
mod recursion;
pub mod resource_groups;
mod result_cache;
mod run_site;
mod runtime;
mod scripting;
mod session;
mod sessions;
pub mod spark_catalog;
mod sql_functions;
mod sql_path;
mod store;
mod store_catalog;
mod system;
mod system_catalogs;
mod system_groups;
mod table_commands;
mod telemetry;
mod trace_context;
mod udfs;
mod variable_substitution;
mod variables;
mod view_schema;
mod view_session_refs;
mod window;

pub use access::{
    IntersectionPolicy, PathPrefixPolicy, READ_ONLY_PRIVILEGES, ReadOnlyPolicy, StatementContext,
    StatementInfo, SystemOnlyPolicy,
};
pub use admin_groups::{ResourceGroupDocument, ResourceGroupInfo};
pub use admin_view::{QueryInfo, SessionInfo};
pub use admission::{AdmissionController, Permit, QueryRequest, Unlimited};
pub use analyze::{Analysis, AnalysisWarning, OutputColumn};
pub use config_defaults::{ConfigListing, ConfigSource};
pub use sessions::Transport;
pub use system::{capabilities, register_build_features, register_surface};

pub mod agent;

pub use cancel::CancelReason;
pub use column::Column;
pub use conf::{RuntimeConf, SparkConfValue};
pub use dataframe::{
    DataFrame, DataFrameReader, DataFrameWriter, DataFrameWriterV2, ExplainMode, GroupedData,
    JoinType, ObservedSink, StorageLevel,
};
pub use explain::Estimate;
pub use explain_graph::{
    GraphFormat, dot_of_plan, dot_of_profile, mermaid_of_plan, mermaid_of_profile,
};
pub use history::{QueryId, QueryOptions};
pub use krishiv_exec::profile::{QueryProfile, TextOptions};
pub use krishiv_exec::{
    CounterSample, GaugeSample, HistogramBucket, HistogramSample, MetricDict, MetricsSnapshot,
};
pub use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, AllowAll, BoxFuture, DenyAll, PolicyVersion, Principal,
    PrincipalKind, Privilege, Securable, SecurableUse,
};
pub use krishiv_sql::{ExplainKind, StatementClass};
pub use krishiv_udf_sdk::{
    AggregateProperties, AggregateUdf, ArgField, GroupsAccumulator, InvokeContext, NullHandling,
    ReturnField, ScalarUdf, Signature, StateField, UdfArg, UdfError, UdfInputType, UdfLibrary,
    UdfProperties,
};
pub use memory_table::{BatchSource, PlannedRead, SourcePlanner};
pub use observation::Observation;
pub use profile::{ExplainVerbose, OptimizerStep, VERBOSE_FORMAT, VERBOSE_VERSION};
#[cfg(feature = "unstable-ir")]
pub use python_udfs::PersistedFunctions;
pub use python_udfs::{
    ANONYMOUS_VERSION_BIT, PythonUdfDef, PythonUdfEntry, PythonUdfLibrary, PythonUdtfDef,
    PythonUdtfEntry,
};
pub use python_udfs::{MAX_PYTHON_SOURCE_CALLS, python_source_call, run_python_source_call};
#[cfg(feature = "unstable-ir")]
pub use python_udfs::{python_udf_call_name, python_udtf_call_name};
pub use runtime::{EngineRuntime, EngineRuntimeBuilder};
pub use session::{RecordBatchStream, Session, SessionBuilder};
#[cfg(feature = "unstable-ir")]
pub use session::{SqlOutcome, SqlQuery};
/// Native Rust scalar UDFs: the [`ScalarUdf`] trait and its boundary types,
/// for [`Session::register_udf`].
pub use udfs::SessionFunctions;
pub use window::WindowSpec;
