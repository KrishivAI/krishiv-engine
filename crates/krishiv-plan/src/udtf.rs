// SPDX-License-Identifier: Apache-2.0
//! Python table functions (UDTFs, eval types 300, 301 and 302; design
//! P3-T085 §2, §3.1 and §3.3 "UDTF analyze") as the analyzer sees them: a
//! registration ([`PythonUdtfEntry`]) and the `analyze` seam's data.
//!
//! **Registration.** A UDTF is a session's temporary table function (Spark's
//! `spark.udtf.register`, `UDTFRegistration.register` at v4.2.0): the
//! analyzer finds it through
//! [`TableFunctionRegistry::python_udtf`](crate::analyzer::TableFunctionRegistry::python_udtf)
//! by name and version, never by its callable. The eval type is the one the
//! registration carries: Spark's client picks 300 or 301 from
//! `spark.sql.execution.pythonUDTF.arrow.enabled` when the UDTF is made
//! (`pyspark/sql/udtf.py` `_create_py_udtf` at v4.2.0), so the engine never
//! re-reads the setting.
//!
//! **Analysis.** A UDTF registered without a return type has a static
//! `analyze` method, which Spark runs at analysis time on the driver in a
//! Python worker (`UserDefinedPythonFunction.scala`,
//! `pyspark/sql/worker/analyze_udtf.py` at v4.2.0):
//! [`TableFunctionRegistry::analyze_python_udtf`](crate::analyzer::TableFunctionRegistry::analyze_python_udtf)
//! takes a [`UdtfAnalyzeSpec`] and gives a [`UdtfAnalysis`], which the
//! analyzer applies to the call (its schema; `withSinglePartition`,
//! `partitionBy`, `orderBy` and `select` on the `TABLE` argument) and keeps
//! in the plan ([`PlanNode::PythonTableFunction`](crate::PlanNode)'s
//! `analyzed`), so a plan cached by its fingerprint never runs `analyze`
//! again. A failure is `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON`.

use std::sync::Arc;

use arrow_array::ArrayRef;
use krishiv_types::DataType;

use crate::PythonUdfRef;

/// A registered Python UDTF, as the analyzer resolves a call of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonUdtfEntry {
    /// The UDTF by name and registration version (a table kind: 300, 301,
    /// 302).
    pub udf: PythonUdfRef,
    /// The declared result (a struct), or `None` when the UDTF has an
    /// `analyze` method that gives it per call.
    pub return_type: Option<DataType>,
}

/// One argument of an `analyze` call (Spark's `AnalyzeArgument`).
#[derive(Debug, Clone, PartialEq)]
pub struct UdtfAnalyzeArg {
    /// The argument's type (a struct for a `TABLE` argument).
    pub data_type: DataType,
    /// The value of a constant argument: a one-row array of the type's
    /// Arrow form ([`DataType::to_arrow`]), its row null for a `NULL`
    /// constant; `None` for an argument that is not constant. The worker
    /// gives `analyze` the Python value Spark gives (`analyze_udtf.py`
    /// L76-82 at v4.2.0: the internal value through `fromInternal`).
    pub value: Option<ArrayRef>,
    /// Whether it is a `TABLE` argument.
    pub is_table: bool,
    /// Its name, when named.
    pub name: Option<String>,
}

/// What `analyze` is run for: a registered UDTF and one call's arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct UdtfAnalyzeSpec {
    /// The UDTF's canonical (lower-case) name.
    pub name: String,
    /// The registration version.
    pub version: u32,
    /// The call's arguments.
    pub args: Vec<UdtfAnalyzeArg>,
}

/// One `ORDER BY` column `analyze` asked for (Spark's `OrderingColumn`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdtfOrdering {
    /// The column (an expression's text, as Spark parses it).
    pub name: String,
    /// Ascending.
    pub ascending: bool,
    /// `overrideNullsFirst`: `None` keeps the direction's default.
    pub nulls_first: Option<bool>,
}

/// One column `analyze` selected from the `TABLE` argument (Spark's
/// `SelectedColumn`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdtfSelect {
    /// The column (an expression's text).
    pub name: String,
    /// Its alias (empty for none).
    pub alias: String,
}

/// `analyze`'s result (Spark's `AnalyzeResult`, as the JVM reads it back).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdtfAnalysis {
    /// The result schema: a struct.
    pub schema: DataType,
    /// The pickled `AnalyzeResult`, passed to the UDTF's `__init__`.
    pub pickled: Arc<[u8]>,
    /// `withSinglePartition`.
    pub with_single_partition: bool,
    /// `partitionBy`: the column expressions' text.
    pub partition_by: Vec<String>,
    /// `orderBy`.
    pub order_by: Vec<UdtfOrdering>,
    /// `select`.
    pub select: Vec<UdtfSelect>,
}
