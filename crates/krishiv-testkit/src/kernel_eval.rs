// SPDX-License-Identifier: Apache-2.0
//! A kernel-level evaluator for literal-only expressions (P2-T021a): runs the
//! function-matrix cells of the arithmetic, comparison and cast operators on
//! `krishiv-kernels` before the SQL engine exists.
//!
//! It parses the cell's `SELECT <expr>` with `krishiv-sql`, types it with
//! the subset of Spark 4.2's implicit casts these operators need (ANSI and
//! legacy; the analyzer proper is a separate component), and evaluates it on
//! the kernels in four operand shapes: one-row arrays, a left and a right
//! scalar, and 67-row arrays whose odd rows are NULL. Every shape must agree
//! with Spark's typed result (exact bits, decimal scale, the result type) or
//! Spark's error (class and message). Cells Spark rejects during analysis,
//! and constructs this evaluator does not type, are reported as not run with
//! the reason.

mod datetime;
mod strings;
mod zone;

use std::sync::Arc;

pub use zone::JiffZone;

use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType, UInt32Type,
};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BinaryArray, BooleanArray, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, NullArray, PrimitiveArray,
    StringArray, cast::AsArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, TimeUnit};
use krishiv_kernels::cast::{CastMode, CastOptions, TimeZoneRules, cast};
use krishiv_kernels::compare::{
    self, BoolOperand, BytesOperand, CompareKind, Equal, Greater, GreaterEq, Less, LessEq,
    NotEqual, Validity,
};
use krishiv_kernels::decimal::{
    DecimalOp, DecimalPlan, decimal_arith, decimal_integral_divide, decimal_unary, result_type,
};
use krishiv_kernels::integer::{self, SparkIntegral, UnaryOp};
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, KernelError, Mul, NullOnOverflow, Operand, Sub,
    float,
};
use krishiv_sql::ast::{self, BinaryOp as Bop, ExprKind, Literal, NumKind};
use serde_json::{Value, json};

use crate::fn_matrix::{Expected, MatrixCase, Mode};

/// Functions (`SHOW FUNCTIONS` names) whose matrix cells this evaluator runs.
pub const FUNCTIONS: &[&str] = &[
    "+",
    "-",
    "*",
    "/",
    "%",
    "div",
    "mod",
    "abs",
    "negative",
    "positive",
    "try_add",
    "try_subtract",
    "try_multiply",
    "try_divide",
    "try_mod",
    "=",
    "==",
    "<=>",
    "<",
    "<=",
    ">",
    ">=",
    "!=",
    "<>",
    "cast",
    "int",
    "bigint",
    "smallint",
    "tinyint",
    "double",
    "float",
    "decimal",
    "date",
    "timestamp",
    "string",
    // P2-T023a: string functions.
    "upper",
    "ucase",
    "lower",
    "lcase",
    "length",
    "char_length",
    "character_length",
    "trim",
    "ltrim",
    "rtrim",
    "btrim",
    "startswith",
    "endswith",
    "contains",
    "replace",
    "lpad",
    "rpad",
    "substring",
    "substr",
    "concat",
    "||",
    "like",
    "split",
    // P2-T025a: datetime functions.
    "date_add",
    "dateadd",
    "date_sub",
    "datediff",
    "date_diff",
    "year",
    "month",
    "day",
    "dayofmonth",
    "extract",
    "date_part",
    "current_date",
    "curdate",
    "current_timestamp",
    "now",
    "to_date",
    "to_timestamp",
];

/// The outcome of one cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellOutcome {
    /// Every operand shape matched Spark.
    Pass,
    /// It ran and differed from Spark.
    Fail(String),
    /// Not run, with the reason (analysis-time rejections and constructs
    /// outside the kernels of this task).
    NotRun(String),
}

/// Why a cell is not evaluated.
type Skip = String;

/// A typed expression.
#[derive(Debug, Clone)]
enum Node {
    /// A one-row literal of its type (`Null` for `NULL`).
    Lit(ArrayRef),
    /// A cast (explicit or implicit).
    Cast(Box<Node>, DataType, CastMode),
    /// Arithmetic on coerced operands; `ty` is the result type.
    Arith {
        op: AOp,
        l: Box<Node>,
        r: Box<Node>,
        ty: DataType,
        try_: bool,
    },
    /// A comparison (`None`: `<=>`).
    Cmp(Option<CompareKind>, Box<Node>, Box<Node>),
    /// Unary minus or `abs`.
    Unary(UnaryOp, Box<Node>),
    /// A string function (`strings`) with its result type.
    Str(strings::StrFn, Vec<Node>, DataType),
    /// An ANSI interval literal.
    Interval(datetime::IntervalVal),
    /// A datetime function (`datetime`) with its result type.
    Dt(datetime::DtFn, Vec<Node>, DataType),
}

/// Binary arithmetic operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    IntDiv,
}

impl Node {
    fn ty(&self) -> DataType {
        match self {
            Self::Lit(a) => a.data_type().clone(),
            Self::Cast(_, t, _)
            | Self::Arith { ty: t, .. }
            | Self::Str(_, _, t)
            | Self::Dt(_, _, t) => t.clone(),
            Self::Interval(datetime::IntervalVal::YearMonth(_)) => {
                DataType::Interval(arrow_schema::IntervalUnit::YearMonth)
            }
            Self::Interval(datetime::IntervalVal::DayTime { .. }) => {
                DataType::Duration(TimeUnit::Microsecond)
            }
            Self::Cmp(..) => DataType::Boolean,
            Self::Unary(_, c) => c.ty(),
        }
    }

    fn integral_literal(&self) -> Option<i64> {
        match self {
            Self::Lit(a) if a.is_valid(0) => match a.data_type() {
                DataType::Int8 => Some(a.as_primitive::<Int8Type>().value(0).into()),
                DataType::Int16 => Some(a.as_primitive::<Int16Type>().value(0).into()),
                DataType::Int32 => Some(a.as_primitive::<Int32Type>().value(0).into()),
                DataType::Int64 => Some(a.as_primitive::<Int64Type>().value(0)),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Spark's `DataType.simpleString`.
#[must_use]
pub fn simple_string(t: &DataType) -> String {
    match t {
        DataType::Null => "void".to_owned(),
        DataType::Boolean => "boolean".to_owned(),
        DataType::Int8 => "tinyint".to_owned(),
        DataType::Int16 => "smallint".to_owned(),
        DataType::Int32 => "int".to_owned(),
        DataType::Int64 => "bigint".to_owned(),
        DataType::Float32 => "float".to_owned(),
        DataType::Float64 => "double".to_owned(),
        DataType::Decimal128(p, s) => format!("decimal({p},{s})"),
        DataType::Utf8 => "string".to_owned(),
        DataType::Binary => "binary".to_owned(),
        DataType::List(f) => format!("array<{}>", simple_string(f.data_type())),
        DataType::Date32 => "date".to_owned(),
        DataType::Interval(_) => "interval year to month".to_owned(),
        DataType::Duration(_) => "interval day to second".to_owned(),
        DataType::Timestamp(_, Some(_)) => "timestamp".to_owned(),
        DataType::Timestamp(_, None) => "timestamp_ntz".to_owned(),
        other => other.to_string(),
    }
}

fn is_integral(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    )
}

fn is_numeric(t: &DataType) -> bool {
    is_integral(t)
        || matches!(
            t,
            DataType::Float32 | DataType::Float64 | DataType::Decimal128(..)
        )
}

fn rank(t: &DataType) -> u8 {
    match t {
        DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 => 3,
        _ => 4,
    }
}

const LTZ: DataType = DataType::Timestamp(TimeUnit::Microsecond, None);

fn ltz() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
}

/// Typing context.
#[derive(Debug, Clone, Copy)]
struct Binder<'z> {
    ansi: bool,
    /// `spark.sql.decimalOperations.allowPrecisionLoss`.
    allow_loss: bool,
    /// Typing a `try_*` operator: its implicit casts and arithmetic return
    /// NULL instead of raising.
    try_: bool,
    /// The session time zone.
    zone: &'z dyn TimeZoneRules,
    /// The query-start instant (UTC microseconds) when clock functions may
    /// be bound; `None` where a recorded result would depend on it.
    clock: Option<i64>,
}

impl Binder<'_> {
    fn cast_mode(self) -> CastMode {
        match (self.try_, self.ansi) {
            (true, _) => CastMode::Try,
            (false, true) => CastMode::Ansi,
            (false, false) => CastMode::Legacy,
        }
    }

    fn implicit(self, n: Node, to: &DataType) -> Node {
        if n.ty() == *to {
            return n;
        }
        if let Node::Lit(a) = &n
            && a.data_type() == &DataType::Null
        {
            return Node::Lit(arrow_array::new_null_array(to, 1));
        }
        Node::Cast(Box::new(n), to.clone(), self.cast_mode())
    }

    fn bind(self, e: &ast::Expr) -> Result<Node, Skip> {
        match &e.kind {
            ExprKind::Nested(inner) => self.bind(inner),
            ExprKind::Literal(l) => literal(l, self.zone),
            ExprKind::Cast {
                expr,
                data_type,
                try_cast,
            } => {
                let to = spec_type(data_type)?;
                let child = self.bind(expr)?;
                let mode = if *try_cast {
                    CastMode::Try
                } else {
                    self.cast_mode()
                };
                Ok(self.explicit_cast(child, &to, mode))
            }
            ExprKind::Unary(op, inner) => {
                let child = self.bind(inner)?;
                match op {
                    ast::UnaryOp::Minus => self.unary(UnaryOp::Negate, child),
                    ast::UnaryOp::Plus => self.unary_plus(child),
                    _ => Err("unary operator outside this task".to_owned()),
                }
            }
            ExprKind::Binary(l, op, r) => {
                let (l, r) = (self.bind(l)?, self.bind(r)?);
                self.binary(*op, l, r)
            }
            ExprKind::Function(f) => self.function(f),
            ExprKind::Like {
                expr,
                pattern,
                escape,
                negated,
                case_insensitive,
            } => self.like_expr(
                expr,
                pattern,
                escape.as_deref(),
                *negated,
                *case_insensitive,
            ),
            _ => Err("expression kind outside this task".to_owned()),
        }
    }

    fn explicit_cast(self, child: Node, to: &DataType, mode: CastMode) -> Node {
        if let Node::Lit(a) = &child
            && a.data_type() == &DataType::Null
        {
            return Node::Lit(arrow_array::new_null_array(to, 1));
        }
        Node::Cast(Box::new(child), to.clone(), mode)
    }

    fn function(self, f: &ast::Function) -> Result<Node, Skip> {
        let name = f
            .name
            .parts
            .last()
            .map(|p| p.value.to_ascii_lowercase())
            .unwrap_or_default();
        let args: Vec<&ast::Expr> = f.args.iter().map(|a| &a.expr).collect();
        if let Some(node) = self.string_function(&name, &args) {
            return node;
        }
        if let Some(node) = self.datetime_function(&name, &args) {
            return node;
        }
        let cast_to = match name.as_str() {
            "int" => Some(DataType::Int32),
            "bigint" => Some(DataType::Int64),
            "smallint" => Some(DataType::Int16),
            "tinyint" => Some(DataType::Int8),
            "double" => Some(DataType::Float64),
            "float" => Some(DataType::Float32),
            "decimal" => Some(DataType::Decimal128(10, 0)),
            "date" | "to_date" => Some(DataType::Date32),
            "timestamp" | "to_timestamp" => Some(ltz()),
            "string" => Some(DataType::Utf8),
            _ => None,
        };
        let try_bin = |op: AOp| -> Result<Node, Skip> {
            let [l, r] = args.as_slice() else {
                return Err("try_* arity".to_owned());
            };
            // Spark 4.2 evaluates the operator itself (and the implicit
            // casts it adds) in TRY mode; its argument expressions raise.
            let (l, r) = (self.bind(l)?, self.bind(r)?);
            Self { try_: true, ..self }.arith(op, l, r)
        };
        match (name.as_str(), args.as_slice()) {
            (_, [a]) if cast_to.is_some() => {
                let mut child = self.bind(a)?;
                // `to_timestamp` reads a boolean or binary as its string; `to_date`
                // reads anything but a string, date or timestamp so (`to_date(1)`
                // is the malformed date `'1'`).
                let via_string = match (name.as_str(), child.ty()) {
                    ("to_timestamp", t) => matches!(t, DataType::Boolean | DataType::Binary),
                    ("to_date", t) => !matches!(
                        t,
                        DataType::Null
                            | DataType::Utf8
                            | DataType::Date32
                            | DataType::Timestamp(..)
                    ),
                    _ => false,
                };
                if via_string {
                    child = self.implicit(child, &DataType::Utf8);
                }
                let to = cast_to.unwrap_or(DataType::Null);
                Ok(self.explicit_cast(child, &to, self.cast_mode()))
            }
            ("abs", [a]) => self.unary(UnaryOp::Abs, self.bind(a)?),
            ("negative", [a]) => self.unary(UnaryOp::Negate, self.bind(a)?),
            ("positive", [a]) => self.unary_plus(self.bind(a)?),
            ("mod", [l, r]) => self.arith(AOp::Rem, self.bind(l)?, self.bind(r)?),
            ("try_add", _) => try_bin(AOp::Add),
            ("try_subtract", _) => try_bin(AOp::Sub),
            ("try_multiply", _) => try_bin(AOp::Mul),
            ("try_divide", _) => try_bin(AOp::Div),
            ("try_mod", _) => try_bin(AOp::Rem),
            _ => Err(format!("function `{name}` with {} arguments", args.len())),
        }
    }

    fn unary_operand(self, child: Node) -> Result<Node, Skip> {
        Ok(match child.ty() {
            DataType::Utf8 | DataType::Null => self.implicit(child, &DataType::Float64),
            t if is_numeric(&t) => child,
            t => return Err(format!("unary operator on {}", simple_string(&t))),
        })
    }

    fn unary(self, op: UnaryOp, child: Node) -> Result<Node, Skip> {
        Ok(Node::Unary(op, Box::new(self.unary_operand(child)?)))
    }

    fn unary_plus(self, child: Node) -> Result<Node, Skip> {
        self.unary_operand(child)
    }

    fn binary(self, op: Bop, l: Node, r: Node) -> Result<Node, Skip> {
        let a = |o| self.arith(o, l.clone(), r.clone());
        let c = |k| self.compare(Some(k), l.clone(), r.clone());
        match op {
            Bop::Plus => a(AOp::Add),
            Bop::Minus => a(AOp::Sub),
            Bop::Multiply => a(AOp::Mul),
            Bop::Divide => a(AOp::Div),
            Bop::Modulo => a(AOp::Rem),
            Bop::IntegerDivide => a(AOp::IntDiv),
            Bop::Eq => c(CompareKind::Equal),
            Bop::NotEq => c(CompareKind::NotEqual),
            Bop::Lt => c(CompareKind::Less),
            Bop::LtEq => c(CompareKind::LessEq),
            Bop::Gt => c(CompareKind::Greater),
            Bop::GtEq => c(CompareKind::GreaterEq),
            Bop::NullSafeEq => self.compare(None, l, r),
            Bop::Concat => self.concat(vec![l, r]),
            _ => Err("binary operator outside this task".to_owned()),
        }
    }

    /// An integral operand as a decimal (`DecimalType.fromLiteral` for a
    /// literal, else `forType`).
    fn as_decimal(n: &Node) -> DataType {
        // `DecimalType.fromLiteral` covers short, int and long literals;
        // a tinyint literal takes `forType`, decimal(3,0).
        if let Some(v) = n.integral_literal().filter(|_| n.ty() != DataType::Int8) {
            let digits = u8::try_from(v.unsigned_abs().max(1).ilog10() + 1).unwrap_or(19);
            return DataType::Decimal128(digits, 0);
        }
        let p = match n.ty() {
            DataType::Int8 => 3,
            DataType::Int16 => 5,
            DataType::Int32 => 10,
            _ => 20,
        };
        DataType::Decimal128(p, 0)
    }

    /// Spark's numeric widening for arithmetic and comparisons; `None` when
    /// both operands stay as they are (decimals keep their own types).
    fn wider(self, l: &Node, r: &Node) -> (DataType, DataType) {
        let (lt, rt) = (l.ty(), r.ty());
        let f64t = DataType::Float64;
        match (&lt, &rt) {
            (DataType::Float64, _)
            | (_, DataType::Float64)
            | (DataType::Float32, DataType::Decimal128(..))
            | (DataType::Decimal128(..), DataType::Float32) => (f64t.clone(), f64t),
            (DataType::Float32, DataType::Float32)
            | (DataType::Decimal128(..), DataType::Decimal128(..)) => (lt, rt),
            (DataType::Float32, _) | (_, DataType::Float32) => {
                let t = if self.ansi { f64t } else { DataType::Float32 };
                (t.clone(), t)
            }
            (DataType::Decimal128(..), _) => (lt, Self::as_decimal(r)),
            (_, DataType::Decimal128(..)) => (Self::as_decimal(l), rt),
            _ => {
                let t = if rank(&lt) >= rank(&rt) { lt } else { rt };
                (t.clone(), t)
            }
        }
    }

    /// Resolves NULL and string operands of a numeric operator.
    fn numeric_operands(self, op: Option<AOp>, l: Node, r: Node) -> Result<(Node, Node), Skip> {
        let (lt, rt) = (l.ty(), r.ty());
        let pick = |s: &DataType, other: &DataType| -> Result<DataType, Skip> {
            match (s, other) {
                // `NumericType.defaultConcreteType`: double (`div` takes bigint).
                (DataType::Null, DataType::Null) => Ok(if op == Some(AOp::IntDiv) {
                    DataType::Int64
                } else {
                    DataType::Float64
                }),
                (DataType::Null, o) => Ok(o.clone()),
                (DataType::Utf8, o) if is_numeric(o) => {
                    if op == Some(AOp::IntDiv) && !self.ansi {
                        return Err("analysis: string div in legacy mode".to_owned());
                    }
                    Ok(if self.ansi && is_integral(o) {
                        DataType::Int64
                    } else {
                        DataType::Float64
                    })
                }
                (DataType::Utf8, DataType::Null) => Ok(DataType::Float64),
                (t, _) if is_numeric(t) => Ok(t.clone()),
                (t, _) => Err(format!("analysis: arithmetic on {}", simple_string(t))),
            }
        };
        let (lt2, rt2) = (pick(&lt, &rt)?, pick(&rt, &lt)?);
        Ok((self.implicit(l, &lt2), self.implicit(r, &rt2)))
    }

    fn arith(self, op: AOp, l: Node, r: Node) -> Result<Node, Skip> {
        if let Some(node) = self.temporal_arith(op, &l, &r) {
            return node;
        }
        let (l, r) = self.numeric_operands(Some(op), l, r)?;
        let (lt, rt) = (l.ty(), r.ty());
        let any_float = |a: &DataType, b: &DataType| {
            matches!(a, DataType::Float32 | DataType::Float64)
                || matches!(b, DataType::Float32 | DataType::Float64)
        };
        let any_dec =
            matches!(lt, DataType::Decimal128(..)) || matches!(rt, DataType::Decimal128(..));
        let (to_l, to_r) = match op {
            // `/` is defined on double and decimal only.
            AOp::Div if any_float(&lt, &rt) || !any_dec => (DataType::Float64, DataType::Float64),
            AOp::IntDiv if any_float(&lt, &rt) => {
                return Err("analysis: div on floating point".to_owned());
            }
            AOp::IntDiv if !any_dec => (DataType::Int64, DataType::Int64),
            _ => self.wider(&l, &r),
        };
        let (l, r) = (self.implicit(l, &to_l), self.implicit(r, &to_r));
        let ty = match (&to_l, &to_r) {
            _ if op == AOp::IntDiv => DataType::Int64,
            (DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)) => {
                let dop = match op {
                    AOp::Add => DecimalOp::Add,
                    AOp::Sub => DecimalOp::Subtract,
                    AOp::Mul => DecimalOp::Multiply,
                    AOp::Div => DecimalOp::Divide,
                    _ => DecimalOp::Remainder,
                };
                let (p, s) = result_type(dop, (*p1, *s1), (*p2, *s2), self.allow_loss)
                    .map_err(|e| format!("decimal result type: {e}"))?;
                DataType::Decimal128(p, s)
            }
            (t, _) => t.clone(),
        };
        Ok(Node::Arith {
            op,
            l: Box::new(l),
            r: Box::new(r),
            ty,
            try_: self.try_,
        })
    }

    fn compare(self, kind: Option<CompareKind>, l: Node, r: Node) -> Result<Node, Skip> {
        let (lt, rt) = (l.ty(), r.ty());
        let (l, r) = match (&lt, &rt) {
            (DataType::Null, DataType::Null) => (l, r),
            (DataType::Null, t) | (t, DataType::Null) => {
                let t = t.clone();
                (self.implicit(l, &t), self.implicit(r, &t))
            }
            (DataType::Utf8, o) | (o, DataType::Utf8) if is_numeric(o) => {
                // Legacy: the string takes the other side's type, except a
                // decimal (both become double, SPARK-22469).
                let t = if !self.ansi && !matches!(o, DataType::Decimal128(..)) {
                    o.clone()
                } else if self.ansi && is_integral(o) {
                    DataType::Int64
                } else {
                    DataType::Float64
                };
                let (l, r) = (self.implicit(l, &t), self.implicit(r, &t));
                let (wl, wr) = self.wider(&l, &r);
                (self.implicit(l, &wl), self.implicit(r, &wr))
            }
            (a, b) if is_numeric(a) && is_numeric(b) => {
                let (wl, wr) = self.wider(&l, &r);
                let (wl, wr) = match (&wl, &wr) {
                    (DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)) => {
                        let s = (*s1).max(*s2);
                        let int = (p1.saturating_sub(s1.unsigned_abs()))
                            .max(p2.saturating_sub(s2.unsigned_abs()));
                        let t =
                            DataType::Decimal128(int.saturating_add(s.unsigned_abs()).min(38), s);
                        (t.clone(), t)
                    }
                    _ => (wl, wr),
                };
                (self.implicit(l, &wl), self.implicit(r, &wr))
            }
            (a, b) if a == b => (l, r),
            // Legacy mode compares a boolean with a number as that number.
            (DataType::Boolean, o) | (o, DataType::Boolean) if is_numeric(o) && !self.ansi => {
                let o = o.clone();
                (self.implicit(l, &o), self.implicit(r, &o))
            }
            (a, b) => {
                return Err(format!(
                    "analysis: comparison of {} with {}",
                    simple_string(a),
                    simple_string(b)
                ));
            }
        };
        Ok(Node::Cmp(kind, Box::new(l), Box::new(r)))
    }
}

/// A literal's type and one-row value (Spark's `Literal` typing).
fn literal(l: &Literal, zone: &dyn TimeZoneRules) -> Result<Node, Skip> {
    let one = |a: ArrayRef| Ok(Node::Lit(a));
    match l {
        Literal::Null => one(Arc::new(NullArray::new(1))),
        Literal::Boolean(b) => one(Arc::new(BooleanArray::from(vec![*b]))),
        Literal::String(s) => one(Arc::new(StringArray::from(vec![s.as_str()]))),
        Literal::Number { text, kind } => number(text, *kind),
        Literal::Typed { type_name, value } => {
            match type_name.value.to_ascii_lowercase().as_str() {
                "date" => cast_lit(value, &DataType::Date32, zone),
                "x" => {
                    let bytes = (0..value.len())
                        .step_by(2)
                        .map(|i| u8::from_str_radix(value.get(i..i + 2).unwrap_or("zz"), 16))
                        .collect::<Result<Vec<u8>, _>>()
                        .map_err(|_| "hex literal".to_owned())?;
                    one(Arc::new(BinaryArray::from(vec![bytes.as_slice()])))
                }
                "timestamp" | "timestamp_ltz" => cast_lit(value, &ltz(), zone),
                "timestamp_ntz" => cast_lit(value, &LTZ, zone),
                other => Err(format!("{other} literal")),
            }
        }
        Literal::Interval(iv) => datetime::interval_literal(iv),
        _ => Err("literal outside this task".to_owned()),
    }
}

fn cast_lit(value: &str, to: &DataType, zone: &dyn TimeZoneRules) -> Result<Node, Skip> {
    let s = StringArray::from(vec![value]);
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: zone,
    };
    cast(&s, to, &opts)
        .map(Node::Lit)
        .map_err(|e| format!("literal: {e}"))
}

fn number(text: &str, kind: NumKind) -> Result<Node, Skip> {
    let lit = |a: ArrayRef| Ok(Node::Lit(a));
    let bad = || format!("numeric literal {text}");
    match kind {
        NumKind::Int => match (text.parse::<i32>(), text.parse::<i64>()) {
            (Ok(v), _) => lit(Arc::new(Int32Array::from(vec![v]))),
            (_, Ok(v)) => lit(Arc::new(Int64Array::from(vec![v]))),
            _ => decimal_lit(text),
        },
        NumKind::Long => lit(Arc::new(Int64Array::from(vec![
            text.parse::<i64>().map_err(|_| bad())?,
        ]))),
        NumKind::Short => lit(Arc::new(Int16Array::from(vec![
            text.parse::<i16>().map_err(|_| bad())?,
        ]))),
        NumKind::Byte => lit(Arc::new(Int8Array::from(vec![
            text.parse::<i8>().map_err(|_| bad())?,
        ]))),
        NumKind::Double | NumKind::Exponent => lit(Arc::new(Float64Array::from(vec![
            text.parse::<f64>().map_err(|_| bad())?,
        ]))),
        NumKind::Float => lit(Arc::new(Float32Array::from(vec![
            text.parse::<f32>().map_err(|_| bad())?,
        ]))),
        NumKind::Decimal | NumKind::BigDecimal => decimal_lit(text),
        _ => Err(bad()),
    }
}

/// A decimal literal: Java `BigDecimal` precision and scale; a negative
/// scale (`5e36BD`) becomes scale 0, as Spark does without
/// `allowNegativeScaleOfDecimal`.
fn decimal_lit(text: &str) -> Result<Node, Skip> {
    let bad = || format!("decimal literal {text}");
    let (neg, body) = text.strip_prefix('-').map_or((false, text), |d| (true, d));
    let (mant, exp) = match body.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().map_err(|_| bad())?),
        None => (body, 0),
    };
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut digits = format!("{int}{frac}").trim_start_matches('0').to_owned();
    let mut scale = i32::try_from(frac.len())
        .map_err(|_| bad())?
        .checked_sub(exp)
        .ok_or_else(bad)?;
    if scale < 0 {
        let pad = usize::try_from(scale.unsigned_abs()).map_err(|_| bad())?;
        if digits.len().saturating_add(pad) > 38 {
            return Err(bad());
        }
        digits.push_str(&"0".repeat(pad));
        scale = 0;
    }
    let scale = u8::try_from(scale).map_err(|_| bad())?;
    let precision = u8::try_from(digits.len().max(1))
        .map_err(|_| bad())?
        .max(scale);
    if precision > 38 {
        return Err(bad());
    }
    let v: i128 = if digits.is_empty() {
        0
    } else {
        digits.parse().map_err(|_| bad())?
    };
    let a = Decimal128Array::from(vec![if neg { -v } else { v }])
        .with_precision_and_scale(precision, i8::try_from(scale).map_err(|_| bad())?)
        .map_err(|e| e.to_string())?;
    Ok(Node::Lit(Arc::new(a)))
}

/// A Spark type name as written in `CAST(... AS type)`.
fn spec_type(t: &ast::DataTypeSpec) -> Result<DataType, Skip> {
    let ast::DataTypeKind::Primitive { name, args } = &t.kind else {
        return Err("cast to a nested type".to_owned());
    };
    let arg = |i: usize, d: u8| -> Result<u8, Skip> {
        args.get(i).map_or(Ok(d), |a| {
            a.trim().parse().map_err(|_| "type argument".to_owned())
        })
    };
    Ok(match name.as_str() {
        "int" | "integer" => DataType::Int32,
        "bigint" | "long" => DataType::Int64,
        "smallint" | "short" => DataType::Int16,
        "tinyint" | "byte" => DataType::Int8,
        "double" => DataType::Float64,
        "float" | "real" => DataType::Float32,
        "decimal" | "dec" | "numeric" => {
            DataType::Decimal128(arg(0, 10)?, i8::try_from(arg(1, 0)?).map_err(|_| "scale")?)
        }
        "string" => DataType::Utf8,
        "date" => DataType::Date32,
        "timestamp" | "timestamp_ltz" => ltz(),
        "timestamp_ntz" => LTZ,
        "boolean" => DataType::Boolean,
        "binary" => DataType::Binary,
        other => return Err(format!("cast to {other}")),
    })
}

/// How operands are fed to a binary kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// One-row arrays.
    Arrays,
    /// The left operand as a scalar.
    LeftScalar,
    /// The right operand as a scalar.
    RightScalar,
    /// 67-row arrays, odd rows NULL.
    Padded,
}

const PADDED_ROWS: usize = 67;

/// An evaluation failure: a kernel error (compared with Spark's) or an
/// evaluator limitation.
#[derive(Debug)]
enum EvalError {
    Kernel(KernelError),
    Other(String),
}

impl EvalError {
    fn other(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<KernelError> for EvalError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}

struct Evaluator<'z> {
    ctx: KernelCtx,
    /// The session time zone.
    zone: &'z dyn TimeZoneRules,
    /// The query-start instant (UTC microseconds) for clock functions.
    now: i64,
    ansi: bool,
    shape: Shape,
    /// Set when a `BINARY` -> `STRING` cast met bytes that are not UTF-8:
    /// Krishiv's STRING is valid UTF-8 (documented divergence, ADR 0023).
    invalid_utf8: std::cell::Cell<bool>,
    /// Set when ANSI day arithmetic on dates overflowed: Spark wraps there,
    /// Krishiv raises (documented divergence, ADR 0023).
    day_overflow: std::cell::Cell<bool>,
}

/// A scalar of a one-row primitive array.
fn scalar<T: ArrowPrimitiveType>(a: &dyn Array) -> Option<T::Native> {
    let a = a.as_primitive::<T>();
    a.is_valid(0).then(|| a.value(0))
}

impl Evaluator<'_> {
    fn leaf(&self, a: &ArrayRef) -> Result<ArrayRef, EvalError> {
        if self.shape != Shape::Padded {
            return Ok(Arc::clone(a));
        }
        let idx = PrimitiveArray::<UInt32Type>::from(vec![0_u32; PADDED_ROWS]);
        let taken = arrow_select::take::take(a.as_ref(), &idx, None)
            .map_err(|e| EvalError::other(e.to_string()))?;
        if a.data_type() == &DataType::Null {
            return Ok(taken);
        }
        let mask = NullBuffer::from(
            (0..PADDED_ROWS)
                .map(|i| i % 2 == 0 && a.is_valid(0))
                .collect::<Vec<_>>(),
        );
        let data = taken.to_data().into_builder().nulls(Some(mask)).build();
        data.map(arrow_array::make_array)
            .map_err(|e| EvalError::other(e.to_string()))
    }

    fn eval(&self, n: &Node) -> Result<ArrayRef, EvalError> {
        match n {
            Node::Lit(a) => self.leaf(a),
            Node::Cast(c, to, mode) => {
                let v = self.eval(c)?;
                if v.data_type() == &DataType::Binary
                    && v.as_binary::<i32>()
                        .iter()
                        .flatten()
                        .any(|b| std::str::from_utf8(b).is_err())
                {
                    self.invalid_utf8.set(true);
                }
                Ok(cast(
                    &v,
                    to,
                    &CastOptions {
                        mode: *mode,
                        time_zone: self.zone,
                    },
                )?)
            }
            Node::Unary(op, c) => {
                let v = self.eval(c)?;
                self.unary(*op, &v)
            }
            Node::Str(f, args, _) => self.string_fn(f, args),
            Node::Arith { op, l, r, ty, try_ } => {
                let (a, b) = (self.eval(l)?, self.eval(r)?);
                self.arith(*op, &a, &b, ty, *try_)
            }
            Node::Cmp(kind, l, r) => {
                let (a, b) = (self.eval(l)?, self.eval(r)?);
                self.compare(*kind, &a, &b)
            }
            Node::Interval(v) => self.interval_array(*v),
            Node::Dt(f, args, _) => self.datetime_fn(*f, args),
        }
    }

    fn unary(&self, op: UnaryOp, a: &ArrayRef) -> Result<ArrayRef, EvalError> {
        macro_rules! int {
            ($t:ty) => {{
                let a = a.as_primitive::<$t>();
                Arc::new(if self.ansi {
                    integer::unary::<$t, AnsiError>(op, a, &self.ctx)?
                } else {
                    integer::unary::<$t, JavaWrap>(op, a, &self.ctx)?
                }) as ArrayRef
            }};
        }
        Ok(match a.data_type() {
            DataType::Int8 => int!(Int8Type),
            DataType::Int16 => int!(Int16Type),
            DataType::Int32 => int!(Int32Type),
            DataType::Int64 => int!(Int64Type),
            DataType::Float32 => Arc::new(float::unary(op, a.as_primitive::<Float32Type>())?),
            DataType::Float64 => Arc::new(float::unary(op, a.as_primitive::<Float64Type>())?),
            DataType::Decimal128(..) => {
                Arc::new(decimal_unary(op, a.as_primitive::<Decimal128Type>())?)
            }
            other => return Err(EvalError::other(format!("unary on {other}"))),
        })
    }

    /// The operands of a primitive binary kernel, in this evaluator's shape.
    fn operands<'a, T: ArrowPrimitiveType>(
        &self,
        a: &'a dyn Array,
        b: &'a dyn Array,
    ) -> (Operand<'a, T>, Operand<'a, T>) {
        let (pa, pb) = (a.as_primitive::<T>(), b.as_primitive::<T>());
        match self.shape {
            Shape::LeftScalar => (Operand::Scalar(scalar::<T>(a)), Operand::Array(pb)),
            Shape::RightScalar => (Operand::Array(pa), Operand::Scalar(scalar::<T>(b))),
            Shape::Arrays | Shape::Padded => (Operand::Array(pa), Operand::Array(pb)),
        }
    }

    fn arith(
        &self,
        op: AOp,
        a: &ArrayRef,
        b: &ArrayRef,
        ty: &DataType,
        try_: bool,
    ) -> Result<ArrayRef, EvalError> {
        let ctx = &self.ctx;
        // try_*: NULL on overflow and zero divisors; legacy: integral
        // `+ - *` wrap, zero divisors are NULL.
        macro_rules! int {
            ($t:ty) => {{
                let (l, r) = self.operands::<$t>(a.as_ref(), b.as_ref());
                Arc::new(int_op::<$t>(op, l, r, ctx, try_, self.ansi)?) as ArrayRef
            }};
        }
        macro_rules! flt {
            ($t:ty) => {{
                let (l, r) = self.operands::<$t>(a.as_ref(), b.as_ref());
                let out = match op {
                    AOp::Add => float::binary(ArithOp::Add, l, r, ctx),
                    AOp::Sub => float::binary(ArithOp::Subtract, l, r, ctx),
                    AOp::Mul => float::binary(ArithOp::Multiply, l, r, ctx),
                    AOp::Div if self.ansi && !try_ => float::divide::<$t, AnsiError>(l, r, ctx),
                    AOp::Div => float::divide::<$t, JavaWrap>(l, r, ctx),
                    AOp::Rem if self.ansi && !try_ => float::remainder::<$t, AnsiError>(l, r, ctx),
                    _ => float::remainder::<$t, JavaWrap>(l, r, ctx),
                }?;
                Arc::new(out) as ArrayRef
            }};
        }
        Ok(match (op, a.data_type(), b.data_type()) {
            (AOp::IntDiv, DataType::Int64, _) => {
                let (l, r) = self.operands::<Int64Type>(a.as_ref(), b.as_ref());
                Arc::new(if try_ {
                    integer::integral_divide::<NullOnOverflow>(l, r, ctx)
                } else if self.ansi {
                    integer::integral_divide::<AnsiError>(l, r, ctx)
                } else {
                    integer::integral_divide::<JavaWrap>(l, r, ctx)
                }?)
            }
            (_, DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)) => {
                let (lt, rt) = ((*p1, *s1), (*p2, *s2));
                let (l, r) = self.operands::<Decimal128Type>(a.as_ref(), b.as_ref());
                let ansi = self.ansi && !try_;
                if op == AOp::IntDiv {
                    return Ok(Arc::new(if ansi {
                        decimal_integral_divide::<AnsiError>(l, r, lt, rt, ctx)
                    } else {
                        decimal_integral_divide::<NullOnOverflow>(l, r, lt, rt, ctx)
                    }?));
                }
                let DataType::Decimal128(p, s) = ty else {
                    return Err(EvalError::other("decimal type".to_owned()));
                };
                let dop = match op {
                    AOp::Add => DecimalOp::Add,
                    AOp::Sub => DecimalOp::Subtract,
                    AOp::Mul => DecimalOp::Multiply,
                    AOp::Div => DecimalOp::Divide,
                    _ => DecimalOp::Remainder,
                };
                let plan = DecimalPlan::new(dop, lt, rt, (*p, *s))?;
                Arc::new(if ansi {
                    decimal_arith::<AnsiError>(l, r, &plan, ctx)
                } else {
                    decimal_arith::<NullOnOverflow>(l, r, &plan, ctx)
                }?)
            }
            (_, DataType::Int8, _) => int!(Int8Type),
            (_, DataType::Int16, _) => int!(Int16Type),
            (_, DataType::Int32, _) => int!(Int32Type),
            (_, DataType::Int64, _) => int!(Int64Type),
            (_, DataType::Float32, _) => flt!(Float32Type),
            (_, DataType::Float64, _) => flt!(Float64Type),
            (_, other, _) => {
                return Err(EvalError::other(format!("arithmetic on {other}")));
            }
        })
    }

    fn compare(
        &self,
        kind: Option<CompareKind>,
        a: &ArrayRef,
        b: &ArrayRef,
    ) -> Result<ArrayRef, EvalError> {
        let k = kind.unwrap_or(CompareKind::Equal);
        macro_rules! prim {
            ($t:ty) => {{
                let (l, r) = self.operands::<$t>(a.as_ref(), b.as_ref());
                match k {
                    CompareKind::Equal => compare::compare::<$t, Equal>(l, r, &self.ctx),
                    CompareKind::NotEqual => compare::compare::<$t, NotEqual>(l, r, &self.ctx),
                    CompareKind::Less => compare::compare::<$t, Less>(l, r, &self.ctx),
                    CompareKind::LessEq => compare::compare::<$t, LessEq>(l, r, &self.ctx),
                    CompareKind::Greater => compare::compare::<$t, Greater>(l, r, &self.ctx),
                    CompareKind::GreaterEq => compare::compare::<$t, GreaterEq>(l, r, &self.ctx),
                }?
            }};
        }
        macro_rules! bytes {
            ($l:expr, $r:expr) => {{
                let (l, r) = ($l, $r);
                match k {
                    CompareKind::Equal => compare::compare_bytes::<_, Equal>(l, r),
                    CompareKind::NotEqual => compare::compare_bytes::<_, NotEqual>(l, r),
                    CompareKind::Less => compare::compare_bytes::<_, Less>(l, r),
                    CompareKind::LessEq => compare::compare_bytes::<_, LessEq>(l, r),
                    CompareKind::Greater => compare::compare_bytes::<_, Greater>(l, r),
                    CompareKind::GreaterEq => compare::compare_bytes::<_, GreaterEq>(l, r),
                }?
            }};
        }
        let eq = match a.data_type() {
            DataType::Int8 => prim!(Int8Type),
            DataType::Int16 => prim!(Int16Type),
            DataType::Int32 => prim!(Int32Type),
            DataType::Int64 => prim!(Int64Type),
            DataType::Float32 => prim!(Float32Type),
            DataType::Float64 => prim!(Float64Type),
            DataType::Decimal128(..) => prim!(Decimal128Type),
            DataType::Date32 => prim!(Date32Type),
            DataType::Timestamp(..) => prim!(TimestampMicrosecondType),
            DataType::Utf8 => {
                let (sa, sb) = (a.as_string::<i32>(), b.as_string::<i32>());
                let s = first_bytes;
                match self.shape {
                    Shape::LeftScalar => {
                        bytes!(BytesOperand::Scalar(s(sa)), BytesOperand::Array(sb))
                    }
                    Shape::RightScalar => {
                        bytes!(BytesOperand::Array(sa), BytesOperand::Scalar(s(sb)))
                    }
                    _ => bytes!(BytesOperand::Array(sa), BytesOperand::Array(sb)),
                }
            }
            DataType::Binary => {
                bytes!(
                    BytesOperand::Array(a.as_binary::<i32>()),
                    BytesOperand::Array(b.as_binary::<i32>())
                )
            }
            DataType::Boolean => {
                let (l, r) = (
                    BoolOperand::Array(a.as_boolean()),
                    BoolOperand::Array(b.as_boolean()),
                );
                match k {
                    CompareKind::Equal => compare::compare_bool::<Equal>(l, r),
                    CompareKind::NotEqual => compare::compare_bool::<NotEqual>(l, r),
                    CompareKind::Less => compare::compare_bool::<Less>(l, r),
                    CompareKind::LessEq => compare::compare_bool::<LessEq>(l, r),
                    CompareKind::Greater => compare::compare_bool::<Greater>(l, r),
                    CompareKind::GreaterEq => compare::compare_bool::<GreaterEq>(l, r),
                }?
            }
            DataType::Null => BooleanArray::new_null(a.len()),
            other => return Err(EvalError::other(format!("comparison of {other}"))),
        };
        if kind.is_some() {
            return Ok(Arc::new(eq));
        }
        let (ls, rs) = (
            self.shape == Shape::LeftScalar,
            self.shape == Shape::RightScalar,
        );
        Ok(Arc::new(compare::null_safe(
            &eq,
            validity(a, ls),
            validity(b, rs),
        )?))
    }
}

/// Integral `+ - * %` under the policy of the mode.
fn int_op<T: SparkIntegral>(
    op: AOp,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
    try_: bool,
    ansi: bool,
) -> Result<PrimitiveArray<T>, EvalError> {
    macro_rules! pol {
        ($p:ty) => {
            match op {
                AOp::Add => integer::binary::<T, Add, $p>(l, r, ctx),
                AOp::Sub => integer::binary::<T, Sub, $p>(l, r, ctx),
                AOp::Mul => integer::binary::<T, Mul, $p>(l, r, ctx),
                AOp::Rem => integer::remainder::<T, $p>(l, r, ctx),
                _ => return Err(EvalError::other("integral op".to_owned())),
            }
        };
    }
    Ok(if try_ {
        pol!(NullOnOverflow)
    } else if ansi {
        pol!(AnsiError)
    } else {
        pol!(JavaWrap)
    }?)
}

fn first_bytes(x: &StringArray) -> Option<&[u8]> {
    x.is_valid(0).then(|| x.value(0).as_bytes())
}

fn validity(x: &ArrayRef, scalar_side: bool) -> Validity<'_> {
    // A `NullArray` has no validity bitmap: every row is (logically) NULL.
    if x.data_type() == &DataType::Null {
        Validity::Scalar { null: true }
    } else if scalar_side {
        Validity::Scalar { null: x.is_null(0) }
    } else {
        Validity::Array(x.nulls())
    }
}

/// The single projected expression of a literal-only `SELECT`.
fn select_expr(q: &ast::Query) -> Option<&ast::Expr> {
    let ast::SetExprKind::Select(s) = &q.body.kind else {
        return None;
    };
    match (s.projection.as_slice(), &s.from, &s.selection) {
        ([item], None, None) if q.order_by.is_empty() && q.limit.is_none() => Some(&item.expr),
        _ => None,
    }
}

/// Typed-cell comparison (manifest `typed_cell_format`). NaNs compare equal
/// whatever their payload.
fn cell_matches(a: &dyn Array, i: usize, cell: &Value) -> bool {
    if a.is_null(i) {
        return cell.is_null();
    }
    let bits = |key: &str| {
        cell.get(key)
            .and_then(Value::as_str)
            .and_then(|h| u64::from_str_radix(h, 16).ok())
    };
    match a.data_type() {
        DataType::Boolean => cell.as_bool() == Some(a.as_boolean().value(i)),
        DataType::Int8 => cell.as_i64() == Some(a.as_primitive::<Int8Type>().value(i).into()),
        DataType::Int16 => cell.as_i64() == Some(a.as_primitive::<Int16Type>().value(i).into()),
        DataType::Int32 => cell.as_i64() == Some(a.as_primitive::<Int32Type>().value(i).into()),
        DataType::Int64 => cell.as_i64() == Some(a.as_primitive::<Int64Type>().value(i)),
        DataType::Float64 => {
            let v = a.as_primitive::<Float64Type>().value(i);
            bits("f64")
                .is_some_and(|b| b == v.to_bits() || (v.is_nan() && f64::from_bits(b).is_nan()))
        }
        DataType::Float32 => {
            let v = a.as_primitive::<Float32Type>().value(i);
            bits("f32")
                .and_then(|b| u32::try_from(b).ok())
                .is_some_and(|b| b == v.to_bits() || (v.is_nan() && f32::from_bits(b).is_nan()))
        }
        DataType::Decimal128(_, s) => {
            let v = a.as_primitive::<Decimal128Type>().value(i);
            cell.get("unscaled").and_then(Value::as_str) == Some(v.to_string().as_str())
                && cell.get("scale").and_then(Value::as_i64) == Some(i64::from(*s))
        }
        DataType::Date32 => {
            cell.get("days").and_then(Value::as_i64)
                == Some(a.as_primitive::<Date32Type>().value(i).into())
        }
        DataType::Timestamp(_, z) => {
            let key = if z.is_some() { "ts_us" } else { "ntz_us" };
            cell.get(key).and_then(Value::as_i64)
                == Some(a.as_primitive::<TimestampMicrosecondType>().value(i))
        }
        DataType::Utf8 => cell.as_str() == Some(a.as_string::<i32>().value(i)),
        DataType::List(_) => {
            let list = a.as_list::<i32>().value(i);
            cell.as_array().is_some_and(|cells| {
                cells.len() == list.len()
                    && cells
                        .iter()
                        .enumerate()
                        .all(|(j, c)| cell_matches(list.as_ref(), j, c))
            })
        }
        DataType::Binary => {
            let hex: String = a
                .as_binary::<i32>()
                .value(i)
                .iter()
                .map(|b| format!("{b:02X}"))
                .collect();
            cell.get("hex")
                .and_then(Value::as_str)
                .is_some_and(|h| h.eq_ignore_ascii_case(&hex))
        }
        _ => false,
    }
}

/// Runs one matrix cell on the kernels.
#[must_use]
pub fn run_cell(case: &MatrixCase) -> CellOutcome {
    if !FUNCTIONS.contains(&case.function.as_str()) {
        return CellOutcome::NotRun("function outside P2-T021a".to_owned());
    }
    let (schema, expected_rows, expected_error) = match &case.expected {
        Expected::Error { error, .. } if error.phase != "execution" => {
            return CellOutcome::NotRun(format!("analysis error {} (analyzer)", error.error_class));
        }
        Expected::Error { schema, error } => (schema.as_ref(), None, Some(error)),
        Expected::Rows {
            schema, typed_rows, ..
        } => (Some(schema), typed_rows.as_ref(), None),
        // The clock functions: only the result type is recorded.
        Expected::Nondeterministic { schema } => (Some(schema), None, None),
    };
    let schema_only = matches!(case.expected, Expected::Nondeterministic { .. });
    let zone_name = case
        .conf
        .get("spark.sql.session.timeZone")
        .map_or("UTC", String::as_str);
    let Some(zone) = JiffZone::new(zone_name) else {
        return CellOutcome::NotRun(format!("time zone {zone_name}"));
    };
    let opts = krishiv_sql::ParserOptions::from_conf(|k| case.conf.get(k).map(String::as_str));
    let Ok(q) = krishiv_sql::parse_query(&case.sql, &opts) else {
        return CellOutcome::NotRun("does not parse".to_owned());
    };
    let Some(expr) = select_expr(&q) else {
        return CellOutcome::NotRun("not a single-expression SELECT".to_owned());
    };
    let ansi = case.mode == Mode::AnsiOn;
    let node = match (Binder {
        ansi,
        allow_loss: true,
        try_: false,
        zone: &zone,
        clock: schema_only.then_some(datetime::NOW),
    })
    .bind(expr)
    {
        Ok(n) => n,
        Err(why) => return CellOutcome::NotRun(why),
    };
    if matches!(node.ty(), DataType::Interval(_) | DataType::Duration(_)) {
        return CellOutcome::NotRun("interval result".to_owned());
    }
    if let Some(field) = schema.and_then(|s| s.first())
        && field.data_type != simple_string(&node.ty())
    {
        return CellOutcome::Fail(format!(
            "type {} vs Spark {}",
            simple_string(&node.ty()),
            field.data_type
        ));
    }
    if schema_only {
        return CellOutcome::Pass;
    }
    for shape in [
        Shape::Arrays,
        Shape::LeftScalar,
        Shape::RightScalar,
        Shape::Padded,
    ] {
        let ev = Evaluator {
            ctx: KernelCtx::new(),
            zone: &zone,
            now: datetime::NOW,
            ansi,
            shape,
            invalid_utf8: std::cell::Cell::new(false),
            day_overflow: std::cell::Cell::new(false),
        };
        let got = ev.eval(&node);
        let outcome = match (got, expected_rows, expected_error) {
            (Err(EvalError::Other(why)), ..) => return CellOutcome::NotRun(why),
            (Err(EvalError::Kernel(e)), ..)
                if e.error_class()
                    == krishiv_common::error::ErrorClass::KrishivUnsupportedFeature =>
            {
                return CellOutcome::NotRun(format!("kernel: {}", e.error().message()));
            }
            (Err(EvalError::Kernel(e)), _, Some(want)) => {
                // Unstable Spark errors (unclassified, legacy) only need to
                // fail on the same row.
                let same = e.row() == Some(0)
                    && (!want.stable
                        || (e.error_class().name() == want.error_class
                            && e.to_string() == want.message));
                if same {
                    None
                } else {
                    Some(format!("{shape:?}: error {e} vs Spark {}", want.message))
                }
            }
            (Err(EvalError::Kernel(e)), ..) => Some(format!("{shape:?}: unexpected error {e}")),
            (Ok(_), _, Some(want)) => Some(format!(
                "{shape:?}: no error, Spark raised {}",
                want.error_class
            )),
            (Ok(a), Some(rows), None) => {
                let cell = rows.first().and_then(|r| r.first()).unwrap_or(&Value::Null);
                // Padded rows: every even row is the value; every odd row
                // (NULL inputs) is NULL, or true for `<=>` (NULL <=> NULL).
                let want_len = if shape == Shape::Padded {
                    PADDED_ROWS
                } else {
                    1
                };
                let null_safe = matches!(node, Node::Cmp(None, ..));
                let odd = if null_safe { json!(true) } else { Value::Null };
                let is_null = |i| a.data_type() == &DataType::Null || a.is_null(i);
                let rows_ok = a.len() == want_len
                    && (0..a.len()).all(|i| match (i % 2, odd.is_null()) {
                        (0, _) => cell_matches(a.as_ref(), i, cell),
                        (_, true) => is_null(i),
                        _ => cell_matches(a.as_ref(), i, &odd),
                    });
                (!rows_ok).then(|| {
                    format!(
                        "{shape:?}: value {:?} vs Spark {cell}",
                        a.slice(0, 1).to_data()
                    )
                })
            }
            (Ok(_), None, None) => return CellOutcome::NotRun("no typed rows".to_owned()),
        };
        if let Some(why) = outcome {
            if ev.invalid_utf8.get() {
                return CellOutcome::NotRun(
                    "documented divergence (ADR 0023): STRING holds valid UTF-8 only".to_owned(),
                );
            }
            if ev.day_overflow.get() && expected_rows.is_some() {
                return CellOutcome::NotRun(
                    "documented divergence (ADR 0023): ANSI date day arithmetic raises where Spark wraps"
                        .to_owned(),
                );
            }
            return CellOutcome::Fail(why);
        }
    }
    CellOutcome::Pass
}

/// Expression-level results of one golden file (see [`run_golden`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GoldenTally {
    /// Statements in the file (`SET` included).
    pub statements: usize,
    /// Literal-only `SELECT`s every expression of which was evaluated.
    pub evaluated: usize,
    /// Of those, matched Spark's recorded rows (types and rendered values)
    /// or error class.
    pub passed: usize,
    /// Mismatches: `(query, why)`.
    pub failed: Vec<(String, String)>,
    /// Statements that need the SQL engine (tables, DDL, other operators),
    /// with the reason.
    pub needs_engine: Vec<(String, String)>,
}

/// One rendered cell (`HiveResult.toHiveString` for these types).
fn render(a: &ArrayRef, zone: &dyn TimeZoneRules) -> Result<String, EvalError> {
    if a.is_null(0) || a.data_type() == &DataType::Null {
        return Ok("NULL".to_owned());
    }
    if let DataType::List(_) = a.data_type() {
        // `HiveResult`: `["a","b"]`.
        let items = a.as_list::<i32>().value(0);
        let items: Vec<String> = items
            .as_string::<i32>()
            .iter()
            .map(|s| s.map_or_else(|| "null".to_owned(), |s| format!("\"{s}\"")))
            .collect();
        return Ok(format!("[{}]", items.join(",")));
    }
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: zone,
    };
    let s = cast(a.as_ref(), &DataType::Utf8, &opts)?;
    Ok(s.as_string::<i32>().value(0).to_owned())
}

/// Evaluates the literal-only `SELECT`s of a golden result file on the
/// kernels and compares them with Spark's recorded output: rendered rows
/// and column types, or the error class. `SET` statements update the
/// session (`spark.sql.ansi.enabled`, `allowPrecisionLoss`); everything
/// else needs the SQL engine and is listed.
///
/// # Errors
/// A malformed result file.
pub fn run_golden(text: &str, ansi: bool) -> Result<GoldenTally, crate::golden::parse::ParseError> {
    use crate::golden::parse::{Expected as GoldenExpected, parse_results};
    let mut t = GoldenTally::default();
    let (mut ansi, mut allow_loss) = (ansi, true);
    // SQLQueryTestSuite runs in America/Los_Angeles; `None` is a zone
    // this harness cannot resolve (an offset, `LOCAL`, an unknown name).
    let mut zone = JiffZone::new("America/Los_Angeles");
    for st in parse_results(text)? {
        t.statements += 1;
        let q = st.query.trim();
        if let Some(rest) = q.strip_prefix("set ").or_else(|| q.strip_prefix("SET ")) {
            if let Some(name) = rest
                .trim()
                .strip_prefix("time zone ")
                .or_else(|| rest.trim().strip_prefix("TIME ZONE "))
            {
                zone = JiffZone::new(name.trim().trim_matches('\''));
                continue;
            }
            let (k, v) = rest.split_once('=').unwrap_or((rest, ""));
            let on = v.trim().eq_ignore_ascii_case("true");
            match k.trim() {
                "spark.sql.ansi.enabled" => ansi = on,
                "spark.sql.decimalOperations.allowPrecisionLoss" => allow_loss = on,
                "spark.sql.session.timeZone" => zone = JiffZone::new(v.trim()),
                _ => {}
            }
            continue;
        }
        let parsed = krishiv_sql::parse_query(q, &krishiv_sql::ParserOptions::default());
        let items = match &parsed {
            Ok(query) => match &query.body.kind {
                ast::SetExprKind::Select(s)
                    if s.from.is_none()
                        && s.selection.is_none()
                        && query.order_by.is_empty()
                        && query.limit.is_none() =>
                {
                    &s.projection
                }
                _ => {
                    t.needs_engine
                        .push((q.to_owned(), "not a FROM-less SELECT".to_owned()));
                    continue;
                }
            },
            Err(_) => {
                t.needs_engine.push((
                    q.to_owned(),
                    "statement outside the query parser".to_owned(),
                ));
                continue;
            }
        };
        // Errors Spark raises while analyzing (or parsing) the query are the
        // analyzer's to reproduce, as in the matrix.
        if let GoldenExpected::Error(want) = &st.expected
            && (want.exception.ends_with("AnalysisException") || want.is_parse_error())
        {
            t.needs_engine
                .push((q.to_owned(), "analysis error (analyzer)".to_owned()));
            continue;
        }
        let Some(zone) = zone.as_ref() else {
            t.needs_engine.push((
                q.to_owned(),
                "a session time zone this harness cannot resolve".to_owned(),
            ));
            continue;
        };
        let binder = Binder {
            ansi,
            allow_loss,
            try_: false,
            zone,
            clock: None,
        };
        let nodes: Result<Vec<Node>, Skip> = items.iter().map(|i| binder.bind(&i.expr)).collect();
        let nodes = match nodes {
            Ok(n) => n,
            Err(why) => {
                t.needs_engine.push((q.to_owned(), why));
                continue;
            }
        };
        if nodes
            .iter()
            .any(|n| matches!(n.ty(), DataType::Interval(_) | DataType::Duration(_)))
        {
            t.needs_engine
                .push((q.to_owned(), "an interval result".to_owned()));
            continue;
        }
        let ev = Evaluator {
            ctx: KernelCtx::new(),
            zone,
            now: datetime::NOW,
            ansi,
            shape: Shape::Arrays,
            invalid_utf8: std::cell::Cell::new(false),
            day_overflow: std::cell::Cell::new(false),
        };
        let values: Result<Vec<ArrayRef>, EvalError> = nodes.iter().map(|n| ev.eval(n)).collect();
        if ev.day_overflow.get() && matches!(&st.expected, GoldenExpected::Rows(_)) {
            t.needs_engine.push((
                q.to_owned(),
                "documented divergence (ADR 0023): ANSI date day arithmetic raises where Spark wraps"
                    .to_owned(),
            ));
            continue;
        }
        let types: Vec<String> = nodes.iter().map(|n| simple_string(&n.ty())).collect();
        let verdict = match (values, &st.expected) {
            (Err(EvalError::Other(why)), _) => {
                t.needs_engine.push((q.to_owned(), why));
                continue;
            }
            (Err(EvalError::Kernel(e)), _)
                if e.error_class()
                    == krishiv_common::error::ErrorClass::KrishivUnsupportedFeature =>
            {
                t.needs_engine.push((q.to_owned(), e.to_string()));
                continue;
            }
            (Err(EvalError::Kernel(e)), GoldenExpected::Error(want)) => {
                (want.error_class.as_deref() == Some(e.error_class().name()))
                    .then_some(())
                    .ok_or_else(|| {
                        format!(
                            "error {} vs Spark {:?}",
                            e.error_class().name(),
                            want.error_class
                        )
                    })
            }
            (Err(EvalError::Kernel(e)), GoldenExpected::Rows(_)) => {
                Err(format!("unexpected error {e}"))
            }
            (Ok(_), GoldenExpected::Error(want)) => {
                Err(format!("no error, Spark raised {:?}", want.error_class))
            }
            (Ok(vals), GoldenExpected::Rows(out)) => {
                let row: Result<Vec<String>, EvalError> =
                    vals.iter().map(|v| render(v, zone)).collect();
                let row = row.map_err(|_| "render".to_owned()).map(|r| r.join("\t"));
                let typed = schema_types_match(&st.schema, &types);
                match row {
                    Ok(r) if r == *out && typed => Ok(()),
                    Ok(r) => Err(format!(
                        "row `{r}` types {types:?} vs Spark `{out}` {}",
                        st.schema
                    )),
                    Err(e) => Err(e),
                }
            }
        };
        t.evaluated += 1;
        match verdict {
            Ok(()) => t.passed += 1,
            Err(why) => t.failed.push((q.to_owned(), why)),
        }
    }
    Ok(t)
}

/// Whether Spark's `struct<name:type,...>` has the column types `types`
/// (names may contain any character, so the types are found as `:type,`
/// separators in order, the last one at the end).
fn schema_types_match(schema: &str, types: &[String]) -> bool {
    let Some(body) = schema
        .strip_prefix("struct<")
        .and_then(|s| s.strip_suffix('>'))
    else {
        return false;
    };
    let mut pos = 0_usize;
    for (i, t) in types.iter().enumerate() {
        if i.saturating_add(1) == types.len() {
            let tail = format!(":{t}");
            return body.ends_with(&tail) && body.len().saturating_sub(tail.len()) >= pos;
        }
        let sep = format!(":{t},");
        match body.get(pos..).and_then(|b| b.find(&sep)) {
            Some(p) => pos = pos.saturating_add(p).saturating_add(sep.len()),
            None => return false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;
    use crate::fn_matrix::{Field, SparkError};

    fn rows(ty: &str, cell: Value) -> Expected {
        Expected::Rows {
            schema: vec![Field {
                name: "c".to_owned(),
                data_type: ty.to_owned(),
                nullable: true,
            }],
            rows: vec![],
            render: vec![],
            ordered: false,
            truncated: false,
            typed_rows: Some(vec![vec![cell]]),
            typed_unavailable: None,
        }
    }

    fn error(class: &str, message: &str, phase: &str) -> Expected {
        Expected::Error {
            schema: None,
            error: SparkError {
                error_class: class.to_owned(),
                sqlstate: None,
                message: message.to_owned(),
                message_parameters: BTreeMap::new(),
                phase: phase.to_owned(),
                stable: true,
            },
        }
    }

    fn cell(function: &str, sql: &str, mode: Mode, expected: Expected) -> CellOutcome {
        run_cell(&MatrixCase {
            id: "x#1/ansi_on".to_owned(),
            base_id: "x#1".to_owned(),
            function: function.to_owned(),
            family: "predicate".to_owned(),
            kind: "seed0".to_owned(),
            mode,
            sql: sql.to_owned(),
            conf: BTreeMap::new(),
            expected,
        })
    }

    #[test]
    fn shapes_cover_every_operand_kind() {
        let on = Mode::AnsiOn;
        let pass = |f, sql, e| assert_eq!(cell(f, sql, on, e), CellOutcome::Pass, "{sql}");
        pass("<", "SELECT 'a' < 'b'", rows("boolean", json!(true)));
        pass("=", "SELECT X'01' = X'02'", rows("boolean", json!(false)));
        pass("<=", "SELECT true <= false", rows("boolean", json!(false)));
        pass(
            ">",
            "SELECT TIMESTAMP'2024-01-02 00:00:00.5' > TIMESTAMP'2024-01-01 00:00:00'",
            rows("boolean", json!(true)),
        );
        pass(
            "=",
            "SELECT TIMESTAMP_NTZ'2024-01-01 00:00:00' = TIMESTAMP_NTZ'2024-01-01 00:00:00'",
            rows("boolean", json!(true)),
        );
        pass("<=>", "SELECT 2 <=> NULL", rows("boolean", json!(false)));
        pass(
            "string",
            "SELECT string(TIMESTAMP_NTZ'2024-01-01 00:00:00')",
            rows("string", json!("2024-01-01 00:00:00")),
        );
        pass(
            "timestamp",
            "SELECT timestamp(DATE'2024-01-01')",
            rows(
                "timestamp",
                json!({"ts_us": 1_704_067_200_000_000_i64, "tz": "UTC"}),
            ),
        );
        pass("+", "SELECT 1Y + 2S", rows("smallint", json!(3)));
        pass(
            "+",
            "SELECT 1Y + 1.5BD",
            rows("decimal(5,1)", json!({"unscaled": "25", "scale": 1})),
        );
        pass(
            "-",
            "SELECT -(1.5F)",
            rows("float", json!({"f32": "bfc00000"})),
        );
        pass(
            "*",
            "SELECT 99999999999999999999 * 1",
            rows(
                "decimal(22,0)",
                json!({"unscaled": "99999999999999999999", "scale": 0}),
            ),
        );
        pass(
            "cast",
            "SELECT CAST('1' AS BIGINT) + CAST(1 AS DECIMAL)",
            rows("decimal(21,0)", json!({"unscaled": "2", "scale": 0})),
        );
        pass(
            "cast",
            "SELECT CAST(1 AS NUMERIC(5, 2))",
            rows("decimal(5,2)", json!({"unscaled": "100", "scale": 2})),
        );
        pass(
            "cast",
            "SELECT CAST(1 AS REAL) = 1L",
            rows("boolean", json!(true)),
        );
        pass("=", "SELECT 1.5 = 1", rows("boolean", json!(false)));
        pass(
            "abs",
            "SELECT abs(1L) + 1e0",
            rows("double", json!({"f64": "4000000000000000"})),
        );
    }

    #[test]
    fn failures_and_reasons() {
        let (on, off) = (Mode::AnsiOn, Mode::AnsiOff);
        let fail = |o: CellOutcome| assert!(matches!(o, CellOutcome::Fail(_)), "{o:?}");
        let not_run = |o: CellOutcome| assert!(matches!(o, CellOutcome::NotRun(_)), "{o:?}");
        fail(cell("+", "SELECT 1 + 1", on, rows("bigint", json!(2))));
        fail(cell("+", "SELECT 1 + 1", on, rows("int", json!(3))));
        fail(cell(
            "+",
            "SELECT 1 + 1",
            on,
            error("DIVIDE_BY_ZERO", "x", "execution"),
        ));
        fail(cell(
            "div",
            "SELECT 1 div 0",
            on,
            rows("bigint", Value::Null),
        ));
        fail(cell(
            "div",
            "SELECT 1 div 0",
            on,
            error("DIVIDE_BY_ZERO", "other text", "execution"),
        ));
        not_run(cell(
            "sin",
            "SELECT sin(1)",
            on,
            rows("double", Value::Null),
        ));
        not_run(cell("+", "SELECT 1 + true", on, error("X", "", "analysis")));
        not_run(cell(
            "+",
            "SELECT 1 + rand()",
            on,
            Expected::Nondeterministic { schema: vec![] },
        ));
        not_run(cell("+", "SELECT 1 +", on, rows("int", json!(1))));
        not_run(cell("+", "SELECT a + 1 FROM t", on, rows("int", json!(1))));
        not_run(cell("+", "SELECT 1 + sin(1)", on, rows("double", json!(1))));
        not_run(cell("+", "SELECT 1 + rand()", on, rows("double", json!(1))));
        not_run(cell("+", "SELECT '1' div 2", off, rows("bigint", json!(0))));
        not_run(cell("+", "SELECT 1.5D div 2", on, rows("bigint", json!(0))));
        not_run(cell("+", "SELECT ~1", on, rows("int", json!(-2))));
        not_run(cell("+", "SELECT 1 | 2", on, rows("int", json!(3))));
        not_run(cell("+", "SELECT X'01' + 1", on, rows("int", json!(3))));
        not_run(cell("-", "SELECT -X'01'", on, rows("int", json!(3))));
        not_run(cell(
            "cast",
            "SELECT CAST(1 AS ARRAY<INT>)",
            on,
            rows("int", json!(3)),
        ));
        not_run(cell(
            "cast",
            "SELECT CAST(1 AS INTERVAL DAY)",
            on,
            rows("int", json!(3)),
        ));
        not_run(cell(
            "cast",
            "SELECT CAST(1 AS DECIMAL(x))",
            on,
            rows("int", json!(3)),
        ));
        not_run(cell(
            "cast",
            "SELECT CAST(X'01' AS INT)",
            on,
            rows("int", json!(3)),
        ));
        not_run(cell(
            "try_add",
            "SELECT try_add(1)",
            on,
            rows("int", json!(3)),
        ));
        not_run(cell(
            "=",
            "SELECT DATE'2024-01-01' = X'01'",
            on,
            rows("boolean", json!(3)),
        ));
        not_run(cell(
            "=",
            "SELECT TIME'01:00:00' = 1",
            on,
            rows("boolean", json!(3)),
        ));
        not_run(cell(
            "=",
            "SELECT TIMESTAMP'2024-01-01' = 1",
            on,
            rows("boolean", json!(3)),
        ));
        not_run(cell(
            "=",
            "SELECT 1e400BD = 1",
            on,
            rows("boolean", json!(3)),
        ));
        not_run(cell(
            "=",
            "SELECT 1S = 99999S",
            on,
            rows("boolean", json!(3)),
        ));
        for sql in [
            "SELECT 1e2000000000BD = 1",
            "SELECT 1e-2147483648BD = 1",
            "SELECT TIMESTAMP'2024-01-01 99999999999999:00:00' = 1",
            "SELECT TIMESTAMP'2024-01-01 25:00:00' = 1",
            "SELECT TIMESTAMP'1000000-01-01 00:00:00' = 1",
        ] {
            not_run(cell("=", sql, on, rows("boolean", json!(3))));
        }
        not_run(cell(
            "cast",
            "SELECT CAST('a' AS BINARY)",
            on,
            rows("binary", json!({"hex": "61"})),
        ));
        let pass = |f, sql, m, e| assert_eq!(cell(f, sql, m, e), CellOutcome::Pass, "{sql}");
        pass("+", "SELECT NULL + NULL", on, rows("double", Value::Null));
        pass(
            "div",
            "SELECT NULL div NULL",
            on,
            rows("bigint", Value::Null),
        );
        pass(
            "=",
            "SELECT '1.21' = 1.2",
            off,
            rows("boolean", json!(false)),
        );
        pass(
            "=",
            "SELECT X'0A' = X'0A'",
            on,
            rows("boolean", json!(true)),
        );
        pass(
            "+",
            "SELECT 1Y + 1.5BD",
            on,
            rows("decimal(5,1)", json!({"unscaled": "25", "scale": 1})),
        );
        // An unstable Spark error only needs to fail on the same row.
        let mut unstable = error("UNCLASSIFIED.ArithmeticException", "x", "execution");
        if let Expected::Error { error, .. } = &mut unstable {
            error.stable = false;
        }
        pass("div", "SELECT 1 div 0", on, unstable);
        let mut tz = MatrixCase {
            id: "x#2/ansi_on".to_owned(),
            base_id: "x#2".to_owned(),
            function: "+".to_owned(),
            family: "math".to_owned(),
            kind: "seed0".to_owned(),
            mode: on,
            sql: "SELECT 1 + 1".to_owned(),
            conf: BTreeMap::new(),
            expected: rows("int", json!(2)),
        };
        tz.conf.insert(
            "spark.sql.session.timeZone".to_owned(),
            "Mars/Olympus".to_owned(),
        );
        not_run(run_cell(&tz));
    }

    #[test]
    fn string_functions_run_in_every_shape() {
        let on = Mode::AnsiOn;
        let hex = |h: &str| json!({ "hex": h });
        for (f, sql, ty, want) in [
            ("upper", "SELECT upper('abc')", "string", json!("ABC")),
            ("length", "SELECT length(X'0102')", "int", json!(2)),
            (
                "trim",
                "SELECT trim(LEADING 'x' FROM 'xxa')",
                "string",
                json!("a"),
            ),
            ("||", "SELECT 'a' || 1 || 'b'", "string", json!("a1b")),
            ("concat", "SELECT concat()", "string", json!("")),
            (
                "concat",
                "SELECT concat(X'01', X'02')",
                "binary",
                hex("0102"),
            ),
            (
                "lpad",
                "SELECT lpad(X'01', 3, X'02')",
                "binary",
                hex("020201"),
            ),
            (
                "substr",
                "SELECT substr(X'AABBCC', 2)",
                "binary",
                hex("BBCC"),
            ),
            (
                "substring",
                "SELECT substring('Spark SQL' FROM 5 FOR 1)",
                "string",
                json!("k"),
            ),
            (
                "replace",
                "SELECT replace('abc', 'b')",
                "string",
                json!("ac"),
            ),
            (
                "startswith",
                "SELECT startswith(X'AB01', X'AB')",
                "boolean",
                json!(true),
            ),
            (
                "like",
                "SELECT 'abc' NOT LIKE 'a%'",
                "boolean",
                json!(false),
            ),
            (
                "like",
                "SELECT like('a%c', 'a$%c', '$')",
                "boolean",
                json!(true),
            ),
            (
                "split",
                "SELECT split('a,b', ',', 1)",
                "array<string>",
                json!(["a,b"]),
            ),
            (
                "split",
                "SELECT split('a.b', '\\\\.')",
                "array<string>",
                json!(["a", "b"]),
            ),
            (
                "split",
                "SELECT split('a', NULL)",
                "array<string>",
                Value::Null,
            ),
        ] {
            let outcome = cell(f, sql, on, rows(ty, want));
            assert_eq!(outcome, CellOutcome::Pass, "{sql}");
        }
        let escape = error("INVALID_FORMAT.ESC_AT_THE_END", "other text", "execution");
        let fails = [
            ("upper", "SELECT upper('a')", rows("string", json!("a"))),
            ("like", "SELECT 'a' LIKE 'a'", rows("boolean", json!(false))),
            (
                "split",
                "SELECT split('a,b', ',')",
                rows("array<string>", json!(["a"])),
            ),
            ("like", "SELECT 'a' LIKE 'a\\\\'", escape),
        ];
        for (f, sql, want) in fails {
            let outcome = cell(f, sql, on, want);
            assert!(
                matches!(outcome, CellOutcome::Fail(_)),
                "{sql}: {outcome:?}"
            );
        }
        for (f, sql) in [
            ("split", "SELECT split('a', '[ab]')"),
            ("like", "SELECT 'a' ILIKE 'A'"),
            ("like", "SELECT 'a' LIKE 'a' ESCAPE 'ab'"),
            ("upper", "SELECT upper()"),
            ("upper", "SELECT upper(INTERVAL '1' DAY)"),
            ("upper", "SELECT upper(array(1))"),
            ("upper", "SELECT upper(split('a', ','))"),
            ("substr", "SELECT substr('a', split('a', ','))"),
            ("split", "SELECT split('a', CAST(1 / 0 AS STRING))"),
            ("replace", "SELECT replace('a')"),
            ("like", "SELECT like('a', 'a', NULL)"),
            ("split", "SELECT split('a')"),
            ("substr", "SELECT substr('a', ARRAY(1))"),
        ] {
            let outcome = cell(f, sql, on, rows("string", json!("a")));
            assert!(
                matches!(outcome, CellOutcome::NotRun(_)),
                "{sql}: {outcome:?}"
            );
        }
    }

    #[test]
    fn datetime_functions_run_in_every_shape() {
        let on = Mode::AnsiOn;
        let day = |d: i64| json!({ "days": d });
        let micros = |m: i64| json!({ "ts_us": m, "tz": "UTC" });
        for (f, sql, ty, want) in [
            (
                "date_add",
                "SELECT date_add('2016-07-30', 1)",
                "date",
                day(17_013),
            ),
            (
                "date_sub",
                "SELECT date_sub(DATE'2016-07-30', 1S)",
                "date",
                day(17_011),
            ),
            (
                "datediff",
                "SELECT datediff(DATE'2024-02-29', '2009-07-30')",
                "int",
                json!(5_327),
            ),
            (
                "year",
                "SELECT year(TIMESTAMP'2024-02-29 12:34:56')",
                "int",
                json!(2024),
            ),
            (
                "extract",
                "SELECT extract(DOY FROM DATE'2024-03-10')",
                "int",
                json!(70),
            ),
            (
                "date_part",
                "SELECT date_part('week', DATE'2024-03-10')",
                "int",
                json!(10),
            ),
            (
                "extract",
                "SELECT extract(SECOND FROM TIMESTAMP'2024-03-10 13:45:56.789012')",
                "decimal(8,6)",
                json!({"unscaled": "56789012", "scale": 6}),
            ),
            (
                "+",
                "SELECT DATE'2020-01-31' + INTERVAL '1' MONTH",
                "date",
                day(18_321),
            ),
            (
                "-",
                "SELECT DATE'2020-01-01' - interval 90 days",
                "date",
                day(18_262 - 90),
            ),
            (
                "+",
                "SELECT INTERVAL 14 MONTHS + DATE'2020-01-01'",
                "date",
                day(18_687),
            ),
            ("+", "SELECT DATE'2020-01-01' + 2", "date", day(18_264)),
            (
                "+",
                "SELECT DATE'2020-01-01' + INTERVAL 1 DAY 2 HOURS 3 MINUTES 4.5 SECONDS",
                "timestamp",
                micros(1_577_930_584_500_000),
            ),
            (
                "to_date",
                "SELECT to_date(' 2020-01-01 10:00:00')",
                "date",
                day(18_262),
            ),
            (
                "to_timestamp",
                "SELECT to_timestamp('2020-01-01 10:00:00+02:00')",
                "timestamp",
                micros(1_577_865_600_000_000),
            ),
            (
                "try_add",
                "SELECT try_add(DATE'9999-12-31', INTERVAL 2147483647 MONTH)",
                "date",
                Value::Null,
            ),
        ] {
            let outcome = cell(f, sql, on, rows(ty, want));
            assert_eq!(outcome, CellOutcome::Pass, "{sql}");
        }
        // Day arithmetic past the `int` range: Spark wraps; Krishiv wraps with
        // ANSI off and raises with ANSI on (reported as a documented divergence).
        let sql = "SELECT date_add('2016-07-30', 2147483647)";
        let wrapped = rows("date", day(-2_147_466_637));
        assert_eq!(
            cell("date_add", sql, Mode::AnsiOff, wrapped.clone()),
            CellOutcome::Pass
        );
        let ansi = cell("date_add", sql, on, wrapped);
        assert!(
            matches!(&ansi, CellOutcome::NotRun(w) if w.contains("ADR 0023")),
            "{ansi:?}"
        );
        // The clock functions only check the result type.
        let nd = |ty: &str| Expected::Nondeterministic {
            schema: vec![Field {
                name: "c".to_owned(),
                data_type: ty.to_owned(),
                nullable: false,
            }],
        };
        for (f, sql, ty, pass) in [
            ("current_date", "SELECT current_date()", "date", true),
            ("now", "SELECT now()", "timestamp", true),
            ("current_date", "SELECT current_date()", "timestamp", false),
        ] {
            let outcome = cell(f, sql, on, nd(ty));
            assert_eq!(outcome == CellOutcome::Pass, pass, "{sql}: {outcome:?}");
        }
        let fail = |o: CellOutcome| assert!(matches!(o, CellOutcome::Fail(_)), "{o:?}");
        fail(cell(
            "date_add",
            "SELECT date_add(DATE'2020-01-01', 1)",
            on,
            rows("date", day(0)),
        ));
        let v = rows("date", day(0));
        for (f, sql) in [
            ("to_date", "SELECT to_date('2020', 'yyyy')"),
            ("extract", "SELECT extract(century FROM DATE'2020-01-01')"),
            ("extract", "SELECT extract(YEAR FROM INTERVAL '1' MONTH)"),
            ("+", "SELECT DATE'2020-01-01' * INTERVAL '1' DAY"),
            ("+", "SELECT INTERVAL '1' DAY - DATE'2020-01-01'"),
            ("+", "SELECT INTERVAL '1' DAY + INTERVAL '1' DAY"),
            ("+", "SELECT DATE'2020-01-01' + INTERVAL '1' YEAR '1' DAY"),
            ("+", "SELECT DATE'2020-01-01' + INTERVAL '1.5' DAY"),
            (
                "+",
                "SELECT DATE'2020-01-01' + INTERVAL '99999999999999999999999999999999' DAY",
            ),
            ("+", "SELECT DATE'2020-01-01' + INTERVAL '2147483648' MONTH"),
            (
                "+",
                "SELECT DATE'2020-01-01' + INTERVAL '1 2' DAY TO MINUTE",
            ),
            ("date_add", "SELECT date_add(1.5D, 1)"),
            ("date_add", "SELECT date_add(DATE'2020-01-01', 1L)"),
            ("current_date", "SELECT current_date(1)"),
        ] {
            let outcome = cell(f, sql, on, v.clone());
            assert!(
                matches!(outcome, CellOutcome::NotRun(_)),
                "{sql}: {outcome:?}"
            );
        }
    }

    #[test]
    fn golden_statements() {
        let text = "-- Automatically generated by SQLQueryTestSuite\n\
                    -- !query\nset spark.sql.ansi.enabled=false\n-- !query schema\nstruct<key:string,value:string>\n-- !query output\nspark.sql.ansi.enabled\tfalse\n\n\n\
                    -- !query\nselect 1 div 0\n-- !query schema\nstruct<(1 div 0):bigint>\n-- !query output\nNULL\n\n\n\
                    -- !query\nset spark.sql.ansi.enabled=true\n-- !query schema\nstruct<>\n-- !query output\n\n\n\n\
                    -- !query\nselect 1 div 0\n-- !query schema\nstruct<>\n-- !query output\norg.apache.spark.SparkArithmeticException\n{\"errorClass\" : \"DIVIDE_BY_ZERO\", \"sqlState\" : \"22012\"}\n\n\n\
                    -- !query\nselect 1, 2.5D\n-- !query schema\nstruct<1:int,2.5:double>\n-- !query output\n1\t2.5\n\n\n\
                    -- !query\nselect 1 + 1\n-- !query schema\nstruct<(1 + 1):int>\n-- !query output\n3\n\n\n\
                    -- !query\nselect 1 div 0\n-- !query schema\nstruct<>\n-- !query output\norg.apache.spark.SparkArithmeticException\n{\"errorClass\" : \"ARITHMETIC_OVERFLOW\"}\n\n\n\
                    -- !query\nselect 1 + 1\n-- !query schema\nstruct<>\n-- !query output\norg.apache.spark.SparkArithmeticException\n{\"errorClass\" : \"ARITHMETIC_OVERFLOW\"}\n\n\n\
                    -- !query\nselect 1 + 2147483647\n-- !query schema\nstruct<>\n-- !query output\n3\n\n\n\
                    -- !query\nselect sin(1)\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                    -- !query\nselect CAST(X'01' AS INT)\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                    -- !query\nselect 1 + rand()\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                    -- !query\nselect 1 limit 0\n-- !query schema\nstruct<1:int>\n-- !query output\n\n\n\n\
                    -- !query\nselect * from t\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                    -- !query\nshow tables\n-- !query schema\nstruct<>\n-- !query output\n1\n";
        let t = run_golden(text, true).unwrap();
        assert_eq!((t.statements, t.evaluated, t.passed), (15, 7, 3), "{t:?}");
        assert_eq!(t.failed.len(), 4, "{t:?}");
        assert_eq!(t.needs_engine.len(), 6, "{t:?}");
        assert!(schema_types_match(
            "struct<a,b:int,c:double>",
            &["int".to_owned(), "double".to_owned()]
        ));
        assert!(!schema_types_match("struct<a:int>", &[]));
        assert!(!schema_types_match("nope", &["int".to_owned()]));
        assert!(!schema_types_match(
            "struct<a:int,b:int>",
            &["bigint".to_owned(), "int".to_owned()]
        ));
    }
}
