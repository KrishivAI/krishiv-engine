// SPDX-License-Identifier: Apache-2.0
//! Table-valued function resolution: `PlanNode::TableFunction` in, a
//! resolved leaf out.

use std::sync::Arc;

use arrow_array::{Int64Array, StringArray};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{
    FailedFunctionCall, KrishivInvalidPartitionNumber, NonFoldableArgument, UnexpectedInputType,
    UnresolvableTableValuedFunction, WrongNumArgsWithoutSuggestion,
};
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_types::DataType;

use super::coerce::{ordinal, quoted_type};
use super::exprs::{Bound, pretty};
use super::{AnalysisError, TableFunctionKind, TableFunctionLookup};
use crate::{BoundColumn, Expr, LogicalPlan, Origin, PlanNode};

use super::resolve::{Env, Rel, Resolver};

const SPARK_DOC_ROOT: &str = "https://spark.apache.org/docs/latest";

/// Parameter names of `range`, in order.
const RANGE_PARAMS: [&str; 4] = ["start", "end", "step", "numSlices"];

impl Resolver<'_> {
    /// Resolves `PlanNode::TableFunction { name, args }` at `origin`.
    pub(super) fn table_function(
        &mut self,
        p: &Arc<LogicalPlan>,
        name: &str,
        args: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        match self.ctx.table_functions.lookup(name, self.ctx.profile) {
            TableFunctionLookup::NotFound => {
                // A SQL-bodied table function of the session or its catalog.
                let parts: Vec<String> = name.split('.').map(str::to_owned).collect();
                if let super::FunctionLookup::Sql(def) =
                    self.ctx.functions.lookup(&parts, self.ctx.profile)
                    && def.returns_table.is_some()
                {
                    return self.sql_table_call(p, &def, args, env);
                }
                Err(AnalysisError::new(
                    UnresolvableTableValuedFunction {
                        name: format!("`{name}`"),
                    },
                    o,
                ))
            }
            TableFunctionLookup::Found(TableFunctionKind::UnsupportedGenerator) => {
                Err(super::bind::unsupported("table-valued functions", o))
            }
            TableFunctionLookup::Found(TableFunctionKind::Range) => self.range(p, args, env),
            TableFunctionLookup::Found(TableFunctionKind::SqlKeywords) => {
                self.sql_keywords(p, args, env)
            }
        }
    }

    /// Resolves the `Range` leaf at `origin`.
    pub(super) fn range_leaf(
        &mut self,
        p: &Arc<LogicalPlan>,
        start: i64,
        end: i64,
        step: i64,
        num_slices: Option<u32>,
        env: &Arc<Env>,
    ) -> Rel {
        let _ = env;
        let o = p.origin;
        let output = BoundColumn {
            id: self.fresh(),
            name: "id".to_owned(),
            data_type: DataType::Long,
            nullable: false,
        };
        let node = PlanNode::Range {
            start,
            end,
            step,
            num_slices,
            output: output.clone(),
        };
        let qual: Arc<[String]> = Arc::from(Vec::new());
        let attr = super::resolve::Attr {
            col: output,
            qual,
            source: None,
        };
        self.wrap(
            Arc::new(LogicalPlan::new(node).with_origin(o)),
            o.plan_id,
            vec![attr],
            &[],
        )
    }

    fn range(
        &mut self,
        p: &Arc<LogicalPlan>,
        args: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let n = args.len();
        if !(1..=4).contains(&n) {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`range`".to_owned(),
                    expected_num: "[1, 2, 3, 4]".to_owned(),
                    actual_num: n.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                o,
            ));
        }
        // Binds each argument over no input, as inline-table cells are
        // bound: no enclosing query is visible here (a lateral reference
        // never reaches this point; `LATERAL` fails in lowering).
        let mut bounds: Vec<Bound> = Vec::with_capacity(n);
        for arg in args {
            let cx = Self::cx(&[], &[], env, "Range");
            let mut cx = cx;
            cx.outer = false;
            bounds.push(self.bind(arg, &cx)?);
        }
        let mut values: Vec<i64> = Vec::with_capacity(n);
        for (i, b) in bounds.iter().enumerate() {
            let param = if n == 1 { 1 } else { i };
            values.push(self.range_arg(b, param)?);
        }
        let (start, end, step, num_slices) = match &values[..] {
            [end] => (0, *end, 1, None),
            [start, end] => (*start, *end, 1, None),
            [start, end, step] => (*start, *end, *step, None),
            [start, end, step, slices] => {
                if *slices <= 0 {
                    return Err(AnalysisError::new(
                        KrishivInvalidPartitionNumber {
                            num_partitions: slices.to_string(),
                        },
                        o,
                    ));
                }
                let num_slices = self.slices_to_int(*slices, o)?;
                (*start, *end, *step, Some(num_slices))
            }
            _ => return Err(super::bind::unsupported("a plan shape", o)),
        };
        if step == 0 {
            return Err(AnalysisError::new(
                FailedFunctionCall {
                    func_name: "`range`".to_owned(),
                },
                o,
            ));
        }
        Ok(self.range_leaf(p, start, end, step, num_slices, env))
    }

    /// `numSlices` as Spark's `INT`: a value past `i32::MAX` is a
    /// `CAST_OVERFLOW` under ANSI, as Spark's cast to `INT` raises it (and
    /// wraps without ANSI, where a wrapped value that is not positive is
    /// an invalid partition number).
    fn slices_to_int(&self, slices: i64, origin: Origin) -> Result<u32, AnalysisError> {
        let zone = FixedOffset(0);
        let mode = if self.ctx.conf.ansi {
            KernelMode::Ansi
        } else {
            KernelMode::Legacy
        };
        let opts = CastOptions {
            mode,
            time_zone: &zone,
        };
        let input: arrow_array::ArrayRef = Arc::new(Int64Array::from(vec![slices]));
        let narrowed = cast(input.as_ref(), &ArrowType::Int32, &opts)
            .map_err(|e| AnalysisError::new(e, origin))?;
        let value = narrowed
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .filter(|a| arrow_array::Array::is_valid(*a, 0))
            .map(|a| a.value(0));
        match value.and_then(|v| u32::try_from(v).ok()).filter(|v| *v > 0) {
            Some(v) => Ok(v),
            None => Err(AnalysisError::new(
                KrishivInvalidPartitionNumber {
                    num_partitions: slices.to_string(),
                },
                origin,
            )),
        }
    }

    /// One `range` argument to a `BIGINT`: Spark's type check, then its
    /// fold and cast. `param` is the parameter position Spark reports.
    fn range_arg(&self, b: &Bound, param: usize) -> Result<i64, AnalysisError> {
        let zone = self.ctx.conf.session_zone;
        let sql = || format!("\"{}\"", pretty(&b.shown(), zone));
        let required = "\"BIGINT\"".to_owned();
        let unexpected = |ty: Option<DataType>| {
            AnalysisError::new(
                UnexpectedInputType {
                    param_index: ordinal(param),
                    function_name: "`range`".to_owned(),
                    required_type: required.clone(),
                    input_sql: sql(),
                    input_type: quoted_type(ty.as_ref()),
                },
                b.expr.origin,
            )
        };
        // The type check comes before folding: an array is an unexpected
        // type, not a non-foldable argument.
        let ty = b.ty.data_type.clone();
        match &ty {
            None => return Err(unexpected(None)),
            Some(
                DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
                | DataType::String,
            ) => {}
            Some(_) => return Err(unexpected(ty)),
        }
        let folded = self.fold(&b.expr, self.ctx.conf.ansi);
        match folded {
            None => Err(AnalysisError::new(
                NonFoldableArgument {
                    func_name: "`range`".to_owned(),
                    param_name: format!("`{}`", RANGE_PARAMS[param]),
                    param_type: "\"BIGINT\"".to_owned(),
                },
                b.expr.origin,
            )),
            Some(super::Constant::Error(e)) => {
                // A folding error Spark raises while typing the call
                // (overflow, malformed input) is raised as is.
                Err(AnalysisError::new(e, b.expr.origin))
            }
            Some(super::Constant::Integral(v) | super::Constant::Fractional(v)) => Ok(v),
            Some(super::Constant::String(s)) => self.cast_string_to_long(&s, &b.expr.origin, param),
            Some(_) => Err(unexpected(ty)),
        }
    }

    /// Casts a folded string to `BIGINT` with Spark's cast errors
    /// (`range('a')` is `CAST_INVALID_INPUT`).
    fn cast_string_to_long(
        &self,
        s: &str,
        at: &Origin,
        param: usize,
    ) -> Result<i64, AnalysisError> {
        let zone = FixedOffset(0);
        let mode = if self.ctx.conf.ansi {
            KernelMode::Ansi
        } else {
            KernelMode::Legacy
        };
        let opts = CastOptions {
            mode,
            time_zone: &zone,
        };
        let input: arrow_array::ArrayRef = Arc::new(StringArray::from(vec![s.to_owned()]));
        // The kernels render Spark's `CAST_INVALID_INPUT` exactly.
        match cast(input.as_ref(), &ArrowType::Int64, &opts) {
            Ok(a) => {
                if a.null_count() > 0 {
                    // Legacy mode turns malformed input to NULL, which a
                    // range argument rejects as an unexpected type.
                    let ty = Some(DataType::String);
                    // Rebuild the unexpected-type error without the bound.
                    return Err(AnalysisError::new(
                        UnexpectedInputType {
                            param_index: ordinal(param),
                            function_name: "`range`".to_owned(),
                            required_type: "\"BIGINT\"".to_owned(),
                            input_sql: format!("\"{s}\""),
                            input_type: quoted_type(ty.as_ref()),
                        },
                        *at,
                    ));
                }
                Ok(a.as_any()
                    .downcast_ref::<Int64Array>()
                    .map(|v| v.value(0))
                    .unwrap_or(0))
            }
            Err(e) => {
                // The kernels' error already carries Spark's parameters;
                // a hand-rolled parse would not match its text.
                Err(AnalysisError::new(e, *at))
            }
        }
    }

    fn sql_keywords(
        &mut self,
        p: &Arc<LogicalPlan>,
        args: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        if !args.is_empty() {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`sql_keywords`".to_owned(),
                    expected_num: "0".to_owned(),
                    actual_num: args.len().to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                o,
            ));
        }
        let Some(frontend) = &self.ctx.sql else {
            return Err(super::bind::unsupported(
                "sql_keywords without a SQL front end",
                o,
            ));
        };
        let keywords = frontend.keywords();
        // An inline relation over one row per keyword, as Spark's
        // `SqlKeywords` is: `keyword STRING NOT NULL`, `reserved BOOLEAN
        // NOT NULL`, in the parser's order.
        let mut rows: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(keywords.len());
        for (keyword, _reserved) in &keywords {
            rows.push(vec![
                Arc::new(
                    Expr::new(crate::ExprNode::Literal(crate::Literal::String(
                        keyword.clone(),
                    )))
                    .with_origin(o),
                ),
                Arc::new(
                    Expr::new(crate::ExprNode::Literal(crate::Literal::Boolean(
                        *_reserved,
                    )))
                    .with_origin(o),
                ),
            ]);
        }
        let columns = ["keyword".to_owned(), "reserved".to_owned()];
        // Resolve as an inline table would: bind each cell over no input.
        let mut bound: Vec<Vec<Bound>> = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut cells = Vec::with_capacity(row.len());
            for cell in row {
                let cx = Self::cx(&[], &[], env, "Project");
                cells.push(self.bind(cell, &cx)?);
            }
            bound.push(cells);
        }
        // Types are known: STRING and BOOLEAN, both not null.
        let types = [DataType::String, DataType::Boolean];
        let mut projections: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(bound.len());
        for row in &bound {
            let mut exprs = Vec::with_capacity(row.len());
            for (c, ty) in row.iter().zip(types.iter()) {
                let cast = self
                    .implicit_cast(c, super::coerce::cast_to(c.ty.data_type.as_ref(), Some(ty)));
                exprs.push(cast.expr);
            }
            projections.push(exprs);
        }
        let mut output = Vec::with_capacity(2);
        let attr_types = [
            super::catalog::ArgType::new(DataType::String, false),
            super::catalog::ArgType::new(DataType::Boolean, false),
        ];
        for (name, ty) in columns.iter().zip(attr_types.iter()) {
            output.push(super::resolve::Attr {
                col: BoundColumn {
                    id: self.fresh(),
                    name: name.clone(),
                    data_type: ty.data_type.clone().unwrap_or(DataType::Null),
                    nullable: false,
                },
                qual: Arc::from(Vec::new()),
                source: None,
            });
        }
        let one = Arc::new(LogicalPlan::new(PlanNode::OneRowRelation).with_origin(o));
        let base = self.wrap(Arc::clone(&one), o.plan_id, Vec::new(), &[]);
        if projections.len() == 1 {
            let exprs = projections.pop().unwrap_or_default();
            let mut aliased = Vec::with_capacity(exprs.len());
            let mut out = Vec::with_capacity(exprs.len());
            for (e, a) in exprs.into_iter().zip(output.iter()) {
                aliased.push(Arc::new(
                    Expr::new(crate::ExprNode::Resolved(crate::ResolvedExpr::Alias {
                        expr: e,
                        output: a.col.clone(),
                    }))
                    .with_origin(o),
                ));
                out.push(a.clone());
            }
            let node = PlanNode::Project {
                input: one,
                exprs: aliased,
            };
            return Ok(self.make(o, node, out, &[&base]));
        }
        let out_cols: Vec<BoundColumn> = output.iter().map(|a| a.col.clone()).collect();
        let node = PlanNode::Expand {
            input: one,
            projections,
            output: out_cols,
        };
        Ok(self.make(o, node, output, &[&base]))
    }
}

/// The slice of `range(start, end, step)` the task
/// `(partition_index, partition_count)` computes, with Spark's slice
/// boundaries in `i128` arithmetic so `range(LONG_MAX - 1, LONG_MAX, 1)`
/// returns one row without overflow.
pub fn range_slice(
    start: i64,
    end: i64,
    step: i64,
    partition_index: u32,
    partition_count: u32,
) -> Option<(i64, i64)> {
    debug_assert!(step != 0);
    debug_assert!(partition_count > 0);
    debug_assert!(partition_index < partition_count);
    let count = range_count(start, end, step);
    let (lo, hi) = slice_bounds(count, partition_index, partition_count);
    if lo >= hi {
        return None;
    }
    let lo_start = i128::from(start) + lo * i128::from(step);
    let hi_start = i128::from(start) + hi * i128::from(step);
    // `lo_start` is a value of the range, so it fits in `i64`; `hi_start`
    // is exclusive and may lie past `end` (or past `i64` when `end` is
    // `LONG_MAX` and `step > 1`), so it is clamped to `end`, which fits.
    let lo_start = i64::try_from(lo_start).ok()?;
    let hi_start = if step > 0 {
        if hi_start > i128::from(end) {
            end
        } else {
            i64::try_from(hi_start).unwrap_or(end)
        }
    } else if hi_start < i128::from(end) {
        end
    } else {
        i64::try_from(hi_start).unwrap_or(end)
    };
    Some((lo_start, hi_start))
}

/// The number of rows of `range(start, end, step)`.
pub fn range_count(start: i64, end: i64, step: i64) -> i128 {
    // Widened before negating: `-i64::MIN` does not fit `i64`. A zero step
    // has no rows (the analyzer and decoder reject it; this never divides).
    let (start, end, step) = (i128::from(start), i128::from(end), i128::from(step));
    let (span, stride) = if step > 0 {
        (end - start, step)
    } else if step < 0 {
        (start - end, -step)
    } else {
        return 0;
    };
    if span <= 0 {
        return 0;
    }
    (span + stride - 1) / stride
}

/// The `[lo, hi)` element indices of one slice of `count` rows.
fn slice_bounds(count: i128, partition_index: u32, partition_count: u32) -> (i128, i128) {
    let n = i128::from(partition_count);
    let i = i128::from(partition_index);
    (count * i / n, count * (i + 1) / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::testing::{block_on, context};
    use crate::analyzer::{
        AnalysisError, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf,
    };
    use crate::{ExprNode, Literal, LogicalPlan, Plan};

    fn ctx() -> crate::analyzer::AnalysisContext {
        context(
            MemoryCatalog::new("spark_catalog"),
            ResolutionConf::default(),
        )
    }

    fn table_function(name: &str, args: Vec<crate::Expr>) -> LogicalPlan {
        use std::sync::Arc;
        LogicalPlan::new(PlanNode::TableFunction {
            name: name.to_owned(),
            args: args.into_iter().map(Arc::new).collect(),
        })
    }

    fn lit_int(v: i32) -> crate::Expr {
        crate::Expr::new(ExprNode::Literal(Literal::Integer(v)))
    }

    fn lit_string(v: &str) -> crate::Expr {
        crate::Expr::new(ExprNode::Literal(Literal::String(v.to_owned())))
    }

    fn lit_null() -> crate::Expr {
        crate::Expr::new(ExprNode::Literal(Literal::Null(None)))
    }

    #[expect(clippy::unwrap_in_result, reason = "test helper panics on bad input")]
    fn run(plan: LogicalPlan) -> Result<crate::analyzer::AnalyzedPlan, AnalysisError> {
        let plan = Plan::try_new(plan).unwrap();
        let ctx = ctx();
        block_on(DefaultAnalyzer.analyze(&plan, &ctx))
    }

    fn class_of(r: &Result<crate::analyzer::AnalyzedPlan, AnalysisError>) -> String {
        match r {
            Ok(_) => "OK".to_owned(),
            Err(e) => e.error.error_class().name().to_owned(),
        }
    }

    fn params_of(
        r: &Result<crate::analyzer::AnalyzedPlan, AnalysisError>,
    ) -> Vec<(String, String)> {
        match r {
            Err(e) => e
                .error
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
            Ok(_) => Vec::new(),
        }
    }

    #[test]
    fn unknown_table_function_is_unresolvable() {
        let r = run(table_function("dummy", vec![lit_int(3)]));
        assert_eq!(class_of(&r), "UNRESOLVABLE_TABLE_VALUED_FUNCTION");
        let params = params_of(&r);
        assert!(
            params.iter().any(|(k, v)| k == "name" && v == "`dummy`"),
            "{params:?}"
        );
    }

    #[test]
    fn range_lookup_is_case_insensitive() {
        let r = run(table_function("RaNgE", vec![lit_int(2)]));
        assert_eq!(class_of(&r), "OK");
        let plan = r.unwrap();
        assert_eq!(plan.output.len(), 1);
        assert_eq!(plan.output[0].name, "id");
        assert_eq!(plan.output[0].data_type, krishiv_types::DataType::Long);
        assert!(!plan.output[0].nullable);
    }

    #[test]
    fn range_wrong_arity_names_expected_counts() {
        for n in [0, 5] {
            let args = (0..n).map(lit_int).collect();
            let r = run(table_function("range", args));
            assert_eq!(class_of(&r), "WRONG_NUM_ARGS.WITHOUT_SUGGESTION", "{n}");
            let params = params_of(&r);
            let get = |k: &str| {
                params
                    .iter()
                    .find(|(kk, _)| *kk == k)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default()
            };
            assert_eq!(get("functionName"), "`range`");
            assert_eq!(get("expectedNum"), "[1, 2, 3, 4]");
            assert_eq!(get("actualNum"), n.to_string());
        }
    }

    #[test]
    fn range_arguments_must_be_bigint() {
        // `range(1, null)` is an unexpected type, not a fold failure.
        let r = run(table_function("range", vec![lit_int(1), lit_null()]));
        assert_eq!(class_of(&r), "UNEXPECTED_INPUT_TYPE");
        // A boolean is an unexpected type as well (no `array()` needed:
        // `array` is not implemented, so it would fail as an unknown
        // function before the type check).
        let boolean = crate::Expr::new(ExprNode::Literal(Literal::Boolean(true)));
        let r = run(table_function("range", vec![boolean]));
        assert_eq!(class_of(&r), "UNEXPECTED_INPUT_TYPE");
    }

    #[test]
    fn range_string_cast_failure_is_cast_invalid_input() {
        let r = run(table_function("range", vec![lit_string("a")]));
        assert_eq!(class_of(&r), "CAST_INVALID_INPUT");
    }

    #[test]
    fn range_double_truncates_to_one_row() {
        let float = crate::Expr::new(ExprNode::Literal(Literal::Float(1.5f32)));
        let r = run(table_function("range", vec![float]));
        assert_eq!(class_of(&r), "OK");
    }

    #[test]
    fn range_step_zero_fails_preparing() {
        let r = run(table_function(
            "range",
            vec![lit_int(0), lit_int(5), lit_int(0)],
        ));
        assert_eq!(class_of(&r), "FAILED_FUNCTION_CALL");
        let params = params_of(&r);
        assert!(
            params
                .iter()
                .any(|(k, v)| k == "funcName" && v == "`range`")
        );
    }

    #[test]
    fn range_non_positive_slices_fail() {
        let r = run(table_function(
            "range",
            vec![lit_int(0), lit_int(10), lit_int(1), lit_int(0)],
        ));
        assert_eq!(class_of(&r), "KRISHIV_INVALID_PARTITION_NUMBER");
    }

    /// `numSlices` is Spark's `INT`: past `i32::MAX` it overflows the cast
    /// under ANSI; the largest `INT` is accepted.
    #[test]
    fn range_slices_beyond_int_overflow_the_cast() {
        let long = |v: i64| crate::Expr::new(ExprNode::Literal(Literal::Long(v)));
        let range = |slices: i64| {
            run(table_function(
                "range",
                vec![lit_int(0), lit_int(10), lit_int(1), long(slices)],
            ))
        };
        assert_eq!(class_of(&range(i64::from(i32::MAX))), "OK");
        let r = range(i64::from(i32::MAX) + 1);
        let want = if ResolutionConf::default().ansi {
            "CAST_OVERFLOW"
        } else {
            "KRISHIV_INVALID_PARTITION_NUMBER"
        };
        assert_eq!(class_of(&r), want);
        assert_eq!(
            class_of(&range(i64::MIN)),
            "KRISHIV_INVALID_PARTITION_NUMBER"
        );
    }

    /// The legacy NULL result of a malformed string argument reports the
    /// argument's own position.
    #[test]
    fn range_string_argument_position_is_reported() {
        let conf = ResolutionConf {
            ansi: false,
            ..ResolutionConf::default()
        };
        let plan =
            Plan::try_new(table_function("range", vec![lit_int(0), lit_string("x")])).unwrap();
        let r = block_on(
            DefaultAnalyzer.analyze(&plan, &context(MemoryCatalog::new("spark_catalog"), conf)),
        );
        assert_eq!(class_of(&r), "UNEXPECTED_INPUT_TYPE");
        assert!(
            params_of(&r)
                .iter()
                .any(|(k, v)| k == "paramIndex" && v == "second"),
            "{:?}",
            params_of(&r)
        );
    }

    /// The row count never divides by zero and survives `i64::MIN`.
    #[test]
    fn range_count_is_total() {
        assert_eq!(range_count(0, 10, 3), 4);
        assert_eq!(range_count(10, 0, -3), 4);
        assert_eq!(range_count(5, 1, 1), 0);
        assert_eq!(range_count(1, 5, -1), 0);
        // A zero step has no rows rather than a division by zero.
        assert_eq!(range_count(10, 0, 0), 0);
        assert_eq!(range_count(0, 10, 0), 0);
        // `-i64::MIN` is 2^63: one row from the top of the range.
        assert_eq!(range_count(i64::MAX, i64::MIN, i64::MIN), 2);
        assert_eq!(range_count(0, i64::MIN, i64::MIN), 1);
    }

    #[test]
    fn range_non_foldable_argument_names_end() {
        // A scalar subquery binds but never folds: Spark names `end` (as
        // in `range(0, (select 1))`).
        let inner = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![lit_int(1)]);
        let subquery = crate::Expr::subquery(crate::SubqueryKind::Scalar, inner);
        let r = run(table_function("range", vec![lit_int(0), subquery]));
        assert_eq!(class_of(&r), "NON_FOLDABLE_ARGUMENT");
        let params = params_of(&r);
        assert!(params.iter().any(|(k, v)| k == "paramName" && v == "`end`"));
    }

    #[test]
    fn range_negative_step_and_empty_ranges_resolve() {
        let r = run(table_function(
            "range",
            vec![lit_int(5), lit_int(1), lit_int(-1)],
        ));
        assert_eq!(class_of(&r), "OK");
        let r = run(table_function("range", vec![lit_int(5), lit_int(1)]));
        assert_eq!(class_of(&r), "OK");
    }

    #[test]
    fn generator_names_stay_unsupported() {
        for name in ["explode", "inline", "stack", "posexplode", "json_tuple"] {
            let r = run(table_function(name, vec![lit_int(1)]));
            assert_eq!(class_of(&r), "KRISHIV_UNSUPPORTED_FEATURE", "{name}");
        }
    }

    struct Keywords;
    impl crate::analyzer::SqlFrontend for Keywords {
        fn lower(
            &self,
            _query: &str,
            _source: crate::SourceId,
        ) -> Result<crate::Plan, crate::analyzer::AnalysisError> {
            unreachable!()
        }
        fn redact(&self, query: &str) -> krishiv_common::redact::RedactedSql {
            krishiv_common::redact::RedactedSql::opaque(query)
        }
        fn fingerprint(&self) -> u64 {
            1
        }
        fn keywords(&self) -> Vec<(String, bool)> {
            vec![
                ("ADD".to_owned(), false),
                ("AFTER".to_owned(), false),
                ("ALL".to_owned(), true),
            ]
        }
    }

    #[test]
    fn sql_keywords_resolves_to_an_inline_relation() {
        use std::sync::Arc;
        let plan = Plan::try_new(table_function("sql_keywords", Vec::new())).unwrap();
        let mut ctx = ctx();
        ctx.sql = Some(Arc::new(Keywords));
        let analyzed =
            block_on(DefaultAnalyzer.analyze(&plan, &ctx)).expect("sql_keywords resolves");
        assert_eq!(analyzed.output.len(), 2);
        assert_eq!(analyzed.output[0].name, "keyword");
        assert_eq!(analyzed.output[1].name, "reserved");
        // Wrong arity fails as a wrong number of arguments.
        let plan = Plan::try_new(table_function("sql_keywords", vec![lit_int(1)])).unwrap();
        let r = block_on(DefaultAnalyzer.analyze(&plan, &ctx));
        assert_eq!(
            r.unwrap_err().error.error_class().name(),
            "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
        );
    }

    #[test]
    fn slices_partition_the_range() {
        for (start, end, step) in [
            (0, 10, 1),
            (5, 1, -1),
            (0, 100, 3),
            (10, 0, -3),
            (0, 0, 1),
            (5, 5, -1),
            (i64::MIN, i64::MIN + 10, 1),
            (i64::MAX - 10, i64::MAX, 1),
            (i64::MAX - 1, i64::MAX, 1),
            (i64::MIN, i64::MIN + 1, 1),
        ] {
            for n in [1, 2, 3, 4, 7, 16] {
                let mut all = Vec::new();
                for i in 0..n {
                    if let Some((lo, hi)) = range_slice(start, end, step, i, n) {
                        assert!(step > 0 && lo < hi || step < 0 && lo > hi);
                        let mut v = lo;
                        while if step > 0 { v < hi } else { v > hi } {
                            all.push(v);
                            v = v.wrapping_add(step);
                        }
                    }
                }
                let mut expected = Vec::new();
                let mut v = start;
                while if step > 0 { v < end } else { v > end } {
                    expected.push(v);
                    v = v.wrapping_add(step);
                }
                assert_eq!(all, expected, "{start} {end} {step} {n}");
            }
        }
        // Near the limits without enumerating: the full range's slice
        // bounds still partition the count with no overflow.
        for (start, end, step) in [
            (i64::MIN, i64::MAX, 1),
            (i64::MAX, i64::MIN, -1),
            (i64::MIN, i64::MAX, 2),
        ] {
            let count = range_count(start, end, step);
            assert!(count > 0);
            for n in [1, 2, 4, 16] {
                let mut total = 0;
                let mut prev = 0;
                for i in 0..n {
                    let (lo, hi) = slice_bounds(count, i, n);
                    assert!(0 <= lo && lo <= hi && hi <= count);
                    assert!(i == 0 || lo == prev, "{start} {end} {step} {n} {i}");
                    prev = hi;
                    total += hi - lo;
                }
                assert_eq!(total, count);
                assert_eq!(prev, count);
            }
        }
    }
}
