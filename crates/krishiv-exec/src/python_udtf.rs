// SPDX-License-Identifier: Apache-2.0
//! Python table functions (UDTFs, eval types 300, 301 and 302; design
//! P3-T085 §2 and §3.3 "UDTF analyze"): what a runtime needs beside the
//! relation frames to run one call site, and the `analyze` seam.
//!
//! **Running.** A UDTF call site is a relation call
//! ([`RelationCallSpec`](crate::python_relation::RelationCallSpec)) whose
//! [`udtf`](crate::python_relation::RelationCallSpec::udtf) is a
//! [`UdtfCall`]: its frames are batches of the call's argument columns
//! (Spark's `allInputs` in `EvalPythonUDTFExec`: each distinct argument
//! once, a `TABLE` argument as one struct column holding the table row with
//! its projected `PARTITION BY` values), then the end. One task is one
//! Spark task: `eval` per row (per batch for 302), `terminate` at each
//! `PARTITION BY` boundary and at the end.
//!
//! **Analysis.** A UDTF registered without a return type has a static
//! `analyze` method, which Spark runs at analysis time on the driver in a
//! Python worker (`UserDefinedPythonFunction.scala`,
//! `pyspark/sql/worker/analyze_udtf.py` at v4.2.0):
//! [`UdfRuntime::analyze_udtf`](crate::python_eval::UdfRuntime::analyze_udtf)
//! takes a [`UdtfAnalyzeSpec`] and gives a [`UdtfAnalysis`], whose pickled
//! `AnalyzeResult` the call's tasks then pass to the UDTF's `__init__`. A
//! failure is `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON`.

use std::sync::Arc;

/// One argument of a UDTF call: the frame column it reads, and its name
/// when it is a named argument (`f(x => 1)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdtfArg {
    /// The frame column (0-based) holding the argument's values.
    pub offset: usize,
    /// The argument's name, when named.
    pub name: Option<String>,
}

/// A UDTF call site's arguments and partitioning, as the Python side reads
/// them (`read_udtf`, `worker.py` L1191-1270 at v4.2.0).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UdtfCall {
    /// The arguments, in call order (positional ones first, as Spark's
    /// analyzer orders them).
    pub args: Vec<UdtfArg>,
    /// The fields of the `TABLE` argument's struct column that are
    /// projected `PARTITION BY` values: removed before `eval` sees the row,
    /// and compared between rows to find a partition's end.
    pub partition_child_indexes: Vec<usize>,
    /// The frame columns that are `TABLE` arguments.
    pub table_arg_offsets: Vec<usize>,
    /// `analyze`'s pickled `AnalyzeResult`, when the UDTF was analyzed.
    pub analyze_result: Option<Arc<[u8]>>,
    /// A `LATERAL` call's input columns (frame columns after the
    /// arguments): no argument reads them; each result row carries the
    /// values of the input row whose `eval` gave it (the last row's for
    /// `terminate`'s rows; nulls after no row), ahead of the UDTF's columns,
    /// as Spark's `EvalPythonUDTFExec` joins them. The call site's declared
    /// result starts with them.
    pub passthrough: Vec<usize>,
}

pub use krishiv_plan::udtf::{
    UdtfAnalysis, UdtfAnalyzeArg, UdtfAnalyzeSpec, UdtfOrdering, UdtfSelect,
};
