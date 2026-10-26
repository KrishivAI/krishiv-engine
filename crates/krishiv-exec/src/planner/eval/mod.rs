// SPDX-License-Identifier: Apache-2.0
//! The expression evaluator: resolved IR expressions compiled once per plan
//! into trees of [`PhysExpr`] over the kernels, evaluated once per batch.
//!
//! # Semantics
//! - **Types** come from the analyzer: every operator's operands already
//!   have the types Spark coerces them to, and a function's result type is
//!   the one it resolved. A mismatch here is an engine bug
//!   (`INTERNAL_ERROR`), never a silent cast.
//! - **Overflow policy** (ADR 0023, P2-T021a): under ANSI, overflow and
//!   division by zero raise; in legacy mode integral `+ - * %` wrap as Java
//!   does (`JavaWrap`), decimal arithmetic and every `/` and `%` by zero
//!   return NULL (`NullOnOverflow` / `JavaWrap`). `Unchecked` is never used:
//!   it raises on a zero divisor where legacy Spark returns NULL.
//! - **Three-valued logic**: `AND`/`OR` follow Kleene's tables, comparisons
//!   with NULL are NULL, `<=>` is never NULL.
//! - **Laziness**: `CASE WHEN` evaluates a branch's condition only on the
//!   rows no earlier branch took, and its value only on the rows it takes,
//!   so a value that would fail (a division by zero under ANSI) fails only
//!   where Spark evaluates it.
//! - **Strings** are Spark's `STRING` carried as Arrow `Utf8View`; the
//!   string kernels take `Utf8`, so their inputs and outputs are converted
//!   at the boundary.

mod arith;
mod cast;
mod cond;
mod field;
mod func;
mod hof;
mod logic;
mod nested_cast;
mod strings;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, UInt32Array, new_null_array};
use arrow_schema::{DataType as ArrowType, SchemaRef};
use krishiv_common::config::{
    ANSI_ENABLED, Config, LEGACY_STATISTICAL_AGGREGATE, SESSION_TIME_ZONE,
};
use krishiv_kernels::KernelCtx;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr,
    ResolvedFunction,
};

use super::build::{internal, unsupported};
use crate::error::Result;
use crate::operators::{Predicate, ScalarExpr};
use crate::physical::ExprEnv;

/// A value of an expression over one batch: a column, or one value for
/// every row (a literal, or an expression of literals only).
#[derive(Clone, Debug)]
pub(crate) enum Value {
    /// One value per row.
    Array(ArrayRef),
    /// One value for every row, as a one-row array.
    Scalar(ArrayRef),
}

impl Value {
    /// One value per row of a `rows`-row batch.
    pub(crate) fn into_array(self, rows: usize) -> Result<ArrayRef> {
        match self {
            Self::Array(a) => Ok(a),
            Self::Scalar(s) => broadcast(&s, rows),
        }
    }

    pub(crate) fn array(&self) -> &ArrayRef {
        match self {
            Self::Array(a) | Self::Scalar(a) => a,
        }
    }

    pub(crate) fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

/// Repeats the one value of `s` `rows` times.
pub(crate) fn broadcast(s: &ArrayRef, rows: usize) -> Result<ArrayRef> {
    if s.is_null(0) {
        return Ok(new_null_array(s.data_type(), rows));
    }
    let zero = u32::try_from(rows)
        .map(|_| UInt32Array::from(vec![0_u32; rows]))
        .map_err(|_| internal("batch too large to broadcast a literal"))?;
    arrow_select::take::take(s.as_ref(), &zero, None).map_err(|e| internal(&e.to_string()))
}

/// `e` over the rows of `batch` selected by `mask`, as a column of every
/// row (NULL where not selected): `e` sees only the selected rows, so it
/// raises only for them. `partition` is the executing partition's index
/// (see [`PhysExpr::eval`]).
pub(crate) fn eval_where(
    e: &dyn PhysExpr,
    batch: &RecordBatch,
    mask: &arrow_buffer::BooleanBuffer,
    partition: u32,
) -> Result<ArrayRef> {
    let rows = batch.num_rows();
    let selected = mask.count_set_bits();
    if selected == 0 {
        return Ok(new_null_array(e.data_type(), rows));
    }
    let chosen =
        arrow_select::filter::filter_record_batch(batch, &BooleanArray::new(mask.clone(), None))
            .map_err(|err| internal(&err.to_string()))?;
    let values = e.eval(&chosen, partition)?.into_array(selected)?;
    scatter(values.as_ref(), mask)
}

/// Spreads `values` (one per set bit of `mask`, in order) over `mask.len()`
/// rows, NULL where the mask is unset. One `take` with NULL indices: a
/// per-row `interleave` of the values and a NULL went through Arrow's generic
/// row-at-a-time `MutableArrayData` path for booleans, dominating filters with
/// a few `AND`ed predicates.
fn scatter(values: &dyn Array, mask: &arrow_buffer::BooleanBuffer) -> Result<ArrayRef> {
    if u32::try_from(mask.len()).is_err() {
        return Err(internal("batch too large to scatter a masked evaluation"));
    }
    // Unset rows point at row 0 (any in-bounds row; their index is NULL).
    // `next` counts set bits of a mask shorter than `u32::MAX`: no overflow.
    let mut next = 0_u32;
    let indices: Vec<u32> = mask
        .iter()
        .map(|on| {
            let at = if on { next } else { 0 };
            next += u32::from(on);
            at
        })
        .collect();
    if usize::try_from(next).ok() != Some(values.len()) {
        return Err(internal("a masked evaluation returned the wrong row count"));
    }
    let indices = UInt32Array::new(
        indices.into(),
        Some(arrow_buffer::NullBuffer::new(mask.clone())),
    );
    arrow_select::take::take(values, &indices, None).map_err(|err| internal(&err.to_string()))
}

/// A compiled expression.
pub(crate) trait PhysExpr: Send + Sync + fmt::Debug {
    /// The Arrow type of its values.
    fn data_type(&self) -> &ArrowType;
    /// Whether a value may be NULL.
    fn nullable(&self) -> bool;
    /// Its values over `batch` in the executing partition `partition`
    /// (Spark's partition index): a stateful function (`rand`, `uuid`,
    /// `monotonically_increasing_id`, ...) owns one value stream per
    /// partition, and this evaluates the next values of that stream. The
    /// same query runs each partition's stream from its start, so one
    /// partition's values never depend on another's rows.
    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value>;
}

/// What every expression of one query is evaluated with.
#[derive(Debug, Clone)]
pub(crate) struct EvalEnv {
    /// `spark.sql.ansi.enabled`.
    pub(crate) ansi: bool,
    /// `spark.sql.legacy.statisticalAggregate`.
    pub(crate) legacy_statistical_aggregate: bool,
    /// The session time zone as a fixed offset in seconds, when it is one
    /// (`UTC`, `+hh:mm`); region zones need the tz database (P2-T021b).
    pub(crate) zone: Option<i32>,
    /// The kernels' dispatch target.
    pub(crate) kernels: KernelCtx,
    /// The query's clock (microseconds since the epoch, UTC), read once at
    /// planning: `current_date` and `now()` see one instant per query.
    pub(crate) query_start_micros: i64,
    /// The session user (`current_user()`).
    pub(crate) user: Arc<str>,
    /// Counts the unseeded random binds of one plan instantiation (see
    /// [`BindOptions::seed_salt`](krishiv_functions::scalar::BindOptions::seed_salt)).
    pub(crate) seed_salt: Arc<AtomicU64>,
}

impl EvalEnv {
    pub(crate) fn new(config: &Config, user: Arc<str>) -> Self {
        let ansi = config
            .get(&ANSI_ENABLED)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(true);
        let legacy_statistical_aggregate = config
            .get(&LEGACY_STATISTICAL_AGGREGATE)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(false);
        let zone = config
            .get(&SESSION_TIME_ZONE)
            .and_then(|v| v.as_str())
            .and_then(|z| krishiv_kernels::cast::FixedOffset::parse(z).map(|o| o.0));
        let query_start_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_micros()).ok())
            .unwrap_or_default();
        Self {
            ansi,
            legacy_statistical_aggregate,
            zone,
            kernels: KernelCtx::new(),
            query_start_micros,
            user,
            seed_salt: Arc::new(AtomicU64::new(0)),
        }
    }

    /// The environment a plan fixed, with this worker's kernels. Lambda
    /// variables match by stable identity, so no name-resolution flag is
    /// plan data.
    pub(crate) fn from_plan(env: &ExprEnv) -> Self {
        Self {
            ansi: env.ansi,
            legacy_statistical_aggregate: env.legacy_statistical_aggregate,
            zone: env.zone,
            kernels: KernelCtx::new(),
            query_start_micros: env.query_start_micros,
            user: Arc::clone(&env.user),
            seed_salt: Arc::new(AtomicU64::new(0)),
        }
    }

    /// The environment as plan data.
    pub(crate) fn to_plan(&self) -> ExprEnv {
        ExprEnv {
            ansi: self.ansi,
            legacy_statistical_aggregate: self.legacy_statistical_aggregate,
            zone: self.zone,
            query_start_micros: self.query_start_micros,
            user: Arc::clone(&self.user),
        }
    }
}

/// The input an expression is compiled against.
#[derive(Clone, Copy)]
pub(crate) struct Input<'a> {
    /// The attribute id of each input column.
    pub(crate) layout: &'a [AttrId],
    /// The input batches' schema.
    pub(crate) schema: &'a SchemaRef,
    pub(crate) env: &'a EvalEnv,
    /// The values of the uncorrelated subqueries, by the attribute
    /// expressions read them as (`SubqueryId::attr`).
    pub(crate) subqueries: &'a [(AttrId, Arc<dyn PhysExpr>)],
}

/// A column of `input`'s batch: the layout position's field. A lambda
/// variable reads such a column of the flattened batch the higher-order
/// call compiled around the body (see `hof`).
fn column(input: Input<'_>, c: &BoundColumn) -> Result<Arc<dyn PhysExpr>> {
    let index = input
        .layout
        .iter()
        .position(|a| *a == c.id)
        .ok_or_else(|| internal(&format!("attribute #{} is not an input column", c.id.0)))?;
    let field = input.schema.field(index);
    Ok(Arc::new(Column {
        index,
        data_type: field.data_type().clone(),
        nullable: field.is_nullable(),
    }))
}

/// Compiles `e` over `input`.
pub(crate) fn compile(e: &Expr, input: Input<'_>) -> Result<Arc<dyn PhysExpr>> {
    match &e.node {
        ExprNode::Literal(l) => Ok(Arc::new(Lit::new(l)?)),
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            if let Some((_, v)) = input.subqueries.iter().find(|(a, _)| *a == c.id) {
                return Ok(Arc::clone(v));
            }
            column(input, c)
        }
        // A lambda variable: a column of the flattened batch the
        // higher-order call compiled around the body (see `hof`).
        ExprNode::Resolved(ResolvedExpr::LambdaVar(c)) => column(input, c),
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => compile(expr, input),
        ExprNode::Resolved(ResolvedExpr::Field {
            base,
            ordinal,
            data_type,
            nullable,
            ..
        }) => field::compile(
            compile(base, input)?,
            *ordinal,
            &data_type.to_arrow(),
            *nullable,
        ),
        ExprNode::Cast { expr, to, mode } => cast::compile(compile(expr, input)?, to, *mode, input),
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Scalar => {
            if let FunctionId::Registry { name, .. } = &f.id
                && hof::is_supported(name)
            {
                return hof::compile(name, f, input, e);
            }
            let args = f
                .args
                .iter()
                .map(|a| compile(a, input))
                .collect::<Result<Vec<_>>>()?;
            match &f.id {
                FunctionId::Builtin(b) => builtin(*b, f, args, input, e),
                FunctionId::Registry { name, .. } => {
                    if name == "coalesce" {
                        cond::coalesce(f, args)
                    } else if name == "nullif" || name == "nullifzero" {
                        cond::null_if(f, args, input)
                    } else if name == "nanvl" {
                        cond::nan_vl(f, args, input)
                    } else {
                        func::compile(name, f, args, input, e)
                    }
                }
                _ => Err(unsupported_expr(e)),
            }
        }
        // Includes subqueries: the planner replaces those it runs by
        // columns ([`Input::subqueries`]).
        _ => Err(unsupported_expr(e)),
    }
}

fn builtin(
    b: Builtin,
    f: &ResolvedFunction,
    mut args: Vec<Arc<dyn PhysExpr>>,
    input: Input<'_>,
    e: &Expr,
) -> Result<Arc<dyn PhysExpr>> {
    let out = f.data_type.to_arrow();
    let env = input.env;
    Ok(match (b, args.len()) {
        (Builtin::And | Builtin::Or, 2) => {
            let r = args.pop().ok_or_else(arity)?;
            let l = args.pop().ok_or_else(arity)?;
            logic::and_or(b == Builtin::And, l, r)
        }
        (Builtin::Not, 1) => logic::not(args.pop().ok_or_else(arity)?),
        (Builtin::IsNull | Builtin::IsNotNull, 1) => {
            logic::is_null(b == Builtin::IsNull, args.pop().ok_or_else(arity)?)
        }
        (
            Builtin::Eq
            | Builtin::EqNullSafe
            | Builtin::Lt
            | Builtin::LtEq
            | Builtin::Gt
            | Builtin::GtEq,
            2,
        ) => {
            let r = args.pop().ok_or_else(arity)?;
            let l = args.pop().ok_or_else(arity)?;
            logic::compare(b, l, r, env)?
        }
        (
            Builtin::Add
            | Builtin::Subtract
            | Builtin::Multiply
            | Builtin::Divide
            | Builtin::Remainder,
            2,
        ) => {
            let r = args.pop().ok_or_else(arity)?;
            let l = args.pop().ok_or_else(arity)?;
            arith::binary(b, l, r, out, f.nullable, env)?
        }
        (Builtin::IntegralDivide, 2) => {
            let r = args.pop().ok_or_else(arity)?;
            let l = args.pop().ok_or_else(arity)?;
            arith::integral_divide(l, r, env)?
        }
        (Builtin::Negative, 1) => arith::negative(args.pop().ok_or_else(arity)?, env)?,
        (Builtin::In, n) if n >= 2 => cond::in_list(args, env)?,
        (Builtin::Between, 3) => {
            let hi = args.pop().ok_or_else(arity)?;
            let lo = args.pop().ok_or_else(arity)?;
            let v = args.pop().ok_or_else(arity)?;
            let ge = logic::compare(Builtin::GtEq, Arc::clone(&v), lo, env)?;
            let le = logic::compare(Builtin::LtEq, v, hi, env)?;
            logic::and_or(true, ge, le)
        }
        (Builtin::Like, 2 | 3) => strings::like(args)?,
        // `rlike(a, b)` is Spark's `RLike(a, b)`: the `regexp_like` kernel
        // over the same arguments, reporting Spark's `rlike` name.
        (Builtin::RLike, 2) => func::compile("rlike", f, args, input, e)?,
        (Builtin::When, n) if n >= 2 => cond::when(args, out)?,
        (Builtin::Concat, _) => strings::concat(args, f.nullable)?,
        _ => {
            return Err(unsupported(&format!(
                "the operator {} with {} arguments in physical plans",
                b.name(),
                args.len()
            )));
        }
    })
}

/// Kleene `l AND r`.
pub(crate) fn and(l: Arc<dyn PhysExpr>, r: Arc<dyn PhysExpr>) -> Arc<dyn PhysExpr> {
    logic::and_or(true, l, r)
}

fn arity() -> krishiv_common::error::KrishivError {
    internal("operator arity")
}

pub(crate) fn unsupported_expr(e: &Expr) -> krishiv_common::error::KrishivError {
    unsupported(&format!("the expression {e} in physical plans"))
}

/// An input column.
#[derive(Debug)]
struct Column {
    index: usize,
    data_type: ArrowType,
    nullable: bool,
}

impl PhysExpr for Column {
    fn data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, _partition: u32) -> Result<Value> {
        batch
            .columns()
            .get(self.index)
            .map(|c| Value::Array(Arc::clone(c)))
            .ok_or_else(|| internal("input column out of range"))
    }
}

/// A literal.
#[derive(Debug)]
pub(crate) struct Lit {
    value: ArrayRef,
}

impl Lit {
    pub(crate) fn new(l: &Literal) -> Result<Self> {
        Ok(Self {
            value: literal_array(l)?,
        })
    }
}

impl PhysExpr for Lit {
    fn data_type(&self) -> &ArrowType {
        self.value.data_type()
    }

    fn nullable(&self) -> bool {
        // An untyped NULL is a `NullArray`, whose nulls are logical only.
        self.value.logical_null_count() > 0
    }

    fn eval(&self, _batch: &RecordBatch, _partition: u32) -> Result<Value> {
        Ok(Value::Scalar(Arc::clone(&self.value)))
    }
}

/// A literal as a one-row array of its type's Arrow form.
pub(crate) fn literal_array(l: &Literal) -> Result<ArrayRef> {
    use arrow_array::{
        BinaryViewArray, BooleanArray as B, Date32Array, Decimal128Array, Float32Array,
        Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, MapArray, NullArray,
        StringArray, StringViewArray, StructArray, TimestampMicrosecondArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::Field;
    use krishiv_types::DataType;
    Ok(match l {
        Literal::Null(None) => Arc::new(NullArray::new(1)),
        Literal::Null(Some(t)) => new_null_array(&t.to_arrow(), 1),
        Literal::Boolean(v) => Arc::new(B::from(vec![*v])),
        Literal::Byte(v) => Arc::new(Int8Array::from(vec![*v])),
        Literal::Short(v) => Arc::new(Int16Array::from(vec![*v])),
        Literal::Integer(v) => Arc::new(Int32Array::from(vec![*v])),
        Literal::Long(v) => Arc::new(Int64Array::from(vec![*v])),
        Literal::Float(v) => Arc::new(Float32Array::from(vec![*v])),
        Literal::Double(v) => Arc::new(Float64Array::from(vec![*v])),
        Literal::Decimal {
            unscaled,
            data_type,
        } => {
            let ArrowType::Decimal128(p, s) = DataType::Decimal(*data_type).to_arrow() else {
                return Err(internal("decimal literal type"));
            };
            Arc::new(
                Decimal128Array::from(vec![*unscaled])
                    .with_precision_and_scale(p, s)
                    .map_err(|e| internal(&e.to_string()))?,
            )
        }
        Literal::String(v) => Arc::new(StringViewArray::from(vec![v.as_str()])),
        Literal::Binary(v) => Arc::new(BinaryViewArray::from(vec![v.as_slice()])),
        Literal::Date(v) => Arc::new(Date32Array::from(vec![*v])),
        Literal::Timestamp(v) => Arc::new(
            TimestampMicrosecondArray::from(vec![*v])
                .with_data_type(DataType::Timestamp.to_arrow()),
        ),
        Literal::TimestampNtz(v) => Arc::new(TimestampMicrosecondArray::from(vec![*v])),
        // A one-row string map, as the JSON options bind.
        Literal::Map(pairs) => {
            let keys: ArrayRef = Arc::new(StringArray::from(
                pairs
                    .iter()
                    .map(|(k, _)| Some(k.as_str()))
                    .collect::<Vec<_>>(),
            ));
            let values: ArrayRef = Arc::new(StringArray::from(
                pairs
                    .iter()
                    .map(|(_, v)| Some(v.as_str()))
                    .collect::<Vec<_>>(),
            ));
            let entries = StructArray::new(
                vec![
                    Field::new("key", ArrowType::Utf8, false),
                    Field::new("value", ArrowType::Utf8, true),
                ]
                .into(),
                vec![keys, values],
                None,
            );
            let len = i32::try_from(pairs.len())
                .map_err(|_| internal("a map literal past the 32-bit offset limit"))?;
            Arc::new(
                MapArray::try_new(
                    Arc::new(Field::new("entries", entries.data_type().clone(), false)),
                    OffsetBuffer::new(ScalarBuffer::from(vec![0, len])),
                    entries,
                    None,
                    false,
                )
                .map_err(|e| internal(&e.to_string()))?,
            )
        }
        _ => return Err(unsupported("interval literals in physical plans")),
    })
}

/// A compiled expression as the operators' [`Predicate`] and
/// [`ScalarExpr`].
#[derive(Debug, Clone)]
pub(crate) struct Compiled(pub(crate) Arc<dyn PhysExpr>);

impl Compiled {
    /// The values over `batch` in `partition`. When evaluation fails, the error is the
    /// first failing row's, as Spark's row-at-a-time evaluation raises it:
    /// a batch evaluates sub-expressions one after another (and each lazy
    /// `CASE` branch over its own rows), so the first error found need not
    /// be the first row's. Only the failing path pays for the search.
    fn values(&self, batch: &RecordBatch, partition: u32) -> Result<ArrayRef> {
        let rows = batch.num_rows();
        match self
            .0
            .eval(batch, partition)
            .and_then(|v| v.into_array(rows))
        {
            Ok(v) => Ok(v),
            Err(e) if rows > 1 => {
                for i in 0..rows {
                    self.0.eval(&batch.slice(i, 1), partition)?;
                }
                Err(e)
            }
            Err(e) => Err(e),
        }
    }
}

impl Predicate for Compiled {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        <Self as Predicate>::evaluate_in(self, batch, 0)
    }

    fn evaluate_in(&self, batch: &RecordBatch, partition: u32) -> Result<BooleanArray> {
        let v = self.values(batch, partition)?;
        v.as_boolean_opt()
            .cloned()
            .ok_or_else(|| internal("a predicate is not boolean"))
    }
}

impl ScalarExpr for Compiled {
    fn data_type(&self) -> ArrowType {
        self.0.data_type().clone()
    }

    fn nullable(&self) -> bool {
        self.0.nullable()
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        <Self as ScalarExpr>::evaluate_in(self, batch, 0)
    }

    fn evaluate_in(&self, batch: &RecordBatch, partition: u32) -> Result<ArrayRef> {
        self.values(batch, partition)
    }
}

#[cfg(test)]
mod tests {
    use krishiv_plan::Literal;
    use krishiv_types::DataType;

    use super::{Lit, PhysExpr as _};

    #[test]
    fn a_null_literal_is_nullable_typed_or_not() {
        assert!(Lit::new(&Literal::Null(None)).unwrap().nullable());
        assert!(
            Lit::new(&Literal::Null(Some(DataType::Integer)))
                .unwrap()
                .nullable()
        );
        assert!(!Lit::new(&Literal::Integer(1)).unwrap().nullable());
    }

    #[test]
    fn a_map_literal_is_a_one_row_string_map() {
        use arrow_array::Array;
        use arrow_array::cast::AsArray;

        let array = super::literal_array(&Literal::Map(vec![
            ("mode".to_owned(), "FAILFAST".to_owned()),
            ("timestampFormat".to_owned(), "dd/MM/yyyy".to_owned()),
        ]))
        .unwrap();
        assert_eq!(array.len(), 1);
        let map = array.as_map();
        assert_eq!((map.value_length(0), map.is_null(0)), (2, false));
        let entries = map.entries();
        let keys: Vec<Option<&str>> = entries.column(0).as_string::<i32>().iter().collect();
        let values: Vec<Option<&str>> = entries.column(1).as_string::<i32>().iter().collect();
        assert_eq!(keys, [Some("mode"), Some("timestampFormat")]);
        assert_eq!(values, [Some("FAILFAST"), Some("dd/MM/yyyy")]);
    }

    /// `scatter` puts the i-th value at the i-th set row, NULL elsewhere,
    /// keeping the values' own NULLs, for booleans and strings alike.
    #[test]
    fn scatter_spreads_values_over_the_mask() {
        use arrow_array::cast::AsArray;
        use arrow_array::{Array, BooleanArray, StringArray};
        use arrow_buffer::BooleanBuffer;

        let mask = BooleanBuffer::from(vec![false, true, true, false, true, false]);
        let bools = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let out = super::scatter(&bools, &mask).unwrap();
        let expected = [None, Some(true), None, None, Some(false), None];
        assert_eq!(out.as_boolean().iter().collect::<Vec<_>>(), expected);
        let strings = StringArray::from(vec!["a", "b", "c"]);
        let out = super::scatter(&strings, &mask).unwrap();
        let expected = [None, Some("a"), Some("b"), None, Some("c"), None];
        assert_eq!(out.as_string::<i32>().iter().collect::<Vec<_>>(), expected);
        // A value count that does not match the mask is an internal error.
        assert!(super::scatter(&strings.slice(0, 2), &mask).is_err());
        let none = BooleanBuffer::new_unset(4);
        let out = super::scatter(&BooleanArray::from(Vec::<bool>::new()), &none).unwrap();
        assert_eq!((out.len(), out.null_count()), (4, 4));
    }
}
