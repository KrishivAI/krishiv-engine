// SPDX-License-Identifier: Apache-2.0
//! Catalog procedures, `CALL cat.system.p(args)` (design P3-T027 PR-36e §2, §3): what a catalog
//! offers and the session runs. A procedure names its parameters (`IN` only, as Spark's
//! `validateParameterModes`), takes constant arguments the session has cast to the parameter
//! types, and returns its result rows. No plan node: a `CALL` is a command, never shipped to a
//! worker.

use std::fmt;

use arrow_array::RecordBatch;
use krishiv_common::error::KrishivError;

use crate::analyzer::BoxFuture;

/// The type of a procedure parameter: the types Iceberg's maintenance procedures take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProcedureType {
    /// `STRING`.
    String,
    /// `INT`.
    Int,
    /// `BIGINT`.
    Long,
    /// `BOOLEAN`.
    Boolean,
    /// `TIMESTAMP`.
    Timestamp,
    /// `MAP<STRING, STRING>`.
    StringMap,
    /// `ARRAY<BIGINT>`.
    LongArray,
    /// `ARRAY<STRING>`.
    StringArray,
}

impl ProcedureType {
    /// The SQL type an argument is cast to.
    #[must_use]
    pub const fn sql(self) -> &'static str {
        match self {
            Self::String => "STRING",
            Self::Int => "INT",
            Self::Long => "BIGINT",
            Self::Boolean => "BOOLEAN",
            Self::Timestamp => "TIMESTAMP",
            Self::StringMap => "MAP<STRING, STRING>",
            Self::LongArray => "ARRAY<BIGINT>",
            Self::StringArray => "ARRAY<STRING>",
        }
    }
}

/// One `IN` parameter of a procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcedureParameter {
    /// The parameter's name.
    pub name: &'static str,
    /// Its type.
    pub data_type: ProcedureType,
    /// Whether a call must give it; an optional one is `NULL` when missing.
    pub required: bool,
}

/// An argument, after the session cast it to its parameter's type.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ProcedureValue {
    /// `NULL`, or a missing optional parameter.
    Null,
    /// A string.
    String(String),
    /// An integer (`INT` and `BIGINT`).
    Long(i64),
    /// A boolean.
    Boolean(bool),
    /// A timestamp, in microseconds since the epoch (UTC).
    TimestampMicros(i64),
    /// A map of strings, in the map's order.
    StringMap(Vec<(String, String)>),
    /// An array of longs.
    LongArray(Vec<i64>),
    /// An array of strings.
    StringArray(Vec<String>),
}

/// Who runs a procedure, for the table rights and the commit identity.
#[derive(Debug, Clone, Default)]
pub struct ProcedureContext {
    /// The catalog's name as the statement wrote or resolved it.
    pub catalog: String,
}

/// A procedure of a catalog.
pub trait Procedure: fmt::Debug + Send + Sync {
    /// The lower-case name (`expire_snapshots`).
    fn name(&self) -> &str;

    /// The parameters, in positional order.
    fn parameters(&self) -> &[ProcedureParameter];

    /// The parameter names in the order Spark lists them as candidates for
    /// `UNRECOGNIZED_PARAMETER_NAME`: `parameterNamesSet.toSeq`, a Scala immutable `Set`'s order,
    /// which is insertion order up to four names and hash order beyond. Ties of edit distance
    /// keep this order. The default is the signature's order (right for four names or fewer).
    fn name_order(&self) -> Vec<&str> {
        self.parameters().iter().map(|p| p.name).collect()
    }

    /// Whether it changes the table it names, so the session drops what it cached of it
    /// (Spark's `modifyIcebergTable`, against `withIcebergTable`).
    fn modifies_table(&self) -> bool;

    /// The table the arguments name (`table`), for the session's cache refresh and access check.
    fn table(&self, args: &[ProcedureValue]) -> Option<String>;

    /// Runs the procedure with `args`, one per parameter, cast and with defaults filled in.
    ///
    /// # Errors
    /// The procedure's failure, with a Spark error class.
    fn call<'a>(
        &'a self,
        ctx: &'a ProcedureContext,
        args: Vec<ProcedureValue>,
    ) -> BoxFuture<'a, Result<RecordBatch, KrishivError>>;
}
