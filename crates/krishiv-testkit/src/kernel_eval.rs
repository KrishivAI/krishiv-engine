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

use std::sync::Arc;

use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType, UInt32Type,
};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BinaryArray, BooleanArray, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, NullArray, PrimitiveArray,
    StringArray, TimestampMicrosecondArray, cast::AsArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, TimeUnit};
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
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
use serde_json::Value;

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
            Self::Cast(_, t, _) | Self::Arith { ty: t, .. } => t.clone(),
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
        DataType::Date32 => "date".to_owned(),
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
struct Binder {
    ansi: bool,
    /// Inside `try_*`: casts and arithmetic return NULL instead of raising.
    try_: bool,
}

impl Binder {
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
            ExprKind::Literal(l) => literal(l),
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
        let cast_to = match name.as_str() {
            "int" => Some(DataType::Int32),
            "bigint" => Some(DataType::Int64),
            "smallint" => Some(DataType::Int16),
            "tinyint" => Some(DataType::Int8),
            "double" => Some(DataType::Float64),
            "float" => Some(DataType::Float32),
            "decimal" => Some(DataType::Decimal128(10, 0)),
            "date" | "to_date" => Some(DataType::Date32),
            "timestamp" => Some(ltz()),
            "string" => Some(DataType::Utf8),
            _ => None,
        };
        let try_bin = |op: AOp| -> Result<Node, Skip> {
            let [l, r] = args.as_slice() else {
                return Err("try_* arity".to_owned());
            };
            let inner = Self { try_: true, ..self };
            let (l, r) = (inner.bind(l)?, inner.bind(r)?);
            inner.arith(op, l, r)
        };
        match (name.as_str(), args.as_slice()) {
            (_, [a]) if cast_to.is_some() => {
                let child = self.bind(a)?;
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
            _ => Err("binary operator outside this task".to_owned()),
        }
    }

    /// An integral operand as a decimal (`DecimalType.fromLiteral` for a
    /// literal, else `forType`).
    fn as_decimal(n: &Node) -> DataType {
        if let Some(v) = n.integral_literal() {
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
                (DataType::Null, DataType::Null) => Ok(DataType::Int32),
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
                let (p, s) = result_type(dop, (*p1, *s1), (*p2, *s2), true)
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
                let t = if !self.ansi {
                    o.clone()
                } else if is_integral(o) {
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
fn literal(l: &Literal) -> Result<Node, Skip> {
    let one = |a: ArrayRef| Ok(Node::Lit(a));
    match l {
        Literal::Null => one(Arc::new(NullArray::new(1))),
        Literal::Boolean(b) => one(Arc::new(BooleanArray::from(vec![*b]))),
        Literal::String(s) => one(Arc::new(StringArray::from(vec![s.as_str()]))),
        Literal::Number { text, kind } => number(text, *kind),
        Literal::Typed { type_name, value } => {
            match type_name.value.to_ascii_lowercase().as_str() {
                "date" => cast_lit(value, &DataType::Date32),
                "x" => {
                    let bytes = (0..value.len())
                        .step_by(2)
                        .map(|i| u8::from_str_radix(value.get(i..i + 2).unwrap_or("zz"), 16))
                        .collect::<Result<Vec<u8>, _>>()
                        .map_err(|_| "hex literal".to_owned())?;
                    one(Arc::new(BinaryArray::from(vec![bytes.as_slice()])))
                }
                "timestamp" | "timestamp_ltz" => timestamp_lit(value, ltz()),
                "timestamp_ntz" => timestamp_lit(value, LTZ),
                other => Err(format!("{other} literal")),
            }
        }
        _ => Err("interval literal".to_owned()),
    }
}

fn cast_lit(value: &str, to: &DataType) -> Result<Node, Skip> {
    let s = StringArray::from(vec![value]);
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: &FixedOffset(0),
    };
    cast(&s, to, &opts)
        .map(Node::Lit)
        .map_err(|e| format!("literal: {e}"))
}

/// `yyyy-MM-dd HH:mm:ss[.ffffff]` in UTC (the matrix's session zone).
fn timestamp_lit(value: &str, ty: DataType) -> Result<Node, Skip> {
    let (date, time) = value.split_once(' ').ok_or("timestamp literal shape")?;
    let day = cast_lit(date, &DataType::Date32)?;
    let Node::Lit(day) = day else {
        return Err("timestamp literal".into());
    };
    let days = i64::from(day.as_primitive::<Date32Type>().value(0));
    let (hms, frac) = time.split_once('.').unwrap_or((time, "0"));
    let parts: Vec<i64> = hms
        .split(':')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| "timestamp time")?;
    let [h, m, s] = parts.as_slice() else {
        return Err("timestamp time".into());
    };
    let frac: i64 = format!("{frac:0<6}")
        .get(..6)
        .unwrap_or("0")
        .parse()
        .map_err(|_| "fraction")?;
    let micros = ((days * 24 + h) * 60 + m) * 60_000_000 + s * 1_000_000 + frac;
    let a = TimestampMicrosecondArray::from(vec![micros]);
    let a: ArrayRef = match ty {
        DataType::Timestamp(_, Some(z)) => Arc::new(a.with_timezone(z)),
        _ => Arc::new(a),
    };
    Ok(Node::Lit(a))
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

/// A decimal literal: Java `BigDecimal` precision and scale.
fn decimal_lit(text: &str) -> Result<Node, Skip> {
    let (neg, digits) = text.strip_prefix('-').map_or((false, text), |d| (true, d));
    if digits.contains(['e', 'E']) {
        return Err(format!("decimal literal {text}"));
    }
    let (int, frac) = digits.split_once('.').unwrap_or((digits, ""));
    let all = format!("{int}{frac}");
    let significant = all.trim_start_matches('0');
    let scale = u8::try_from(frac.len()).map_err(|_| "scale")?;
    let precision = u8::try_from(significant.len().max(1))
        .map_err(|_| "precision")?
        .max(scale);
    if precision > 38 {
        return Err(format!("decimal literal {text}"));
    }
    let v: i128 = if significant.is_empty() {
        0
    } else {
        significant.parse().map_err(|_| "digits")?
    };
    let a = Decimal128Array::from(vec![if neg { -v } else { v }])
        .with_precision_and_scale(precision, i8::try_from(scale).map_err(|_| "scale")?)
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

struct Evaluator {
    ctx: KernelCtx,
    ansi: bool,
    shape: Shape,
}

/// A scalar of a one-row primitive array.
fn scalar<T: ArrowPrimitiveType>(a: &dyn Array) -> Option<T::Native> {
    let a = a.as_primitive::<T>();
    a.is_valid(0).then(|| a.value(0))
}

impl Evaluator {
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
                Ok(cast(
                    &v,
                    to,
                    &CastOptions {
                        mode: *mode,
                        time_zone: &FixedOffset(0),
                    },
                )?)
            }
            Node::Unary(op, c) => {
                let v = self.eval(c)?;
                self.unary(*op, &v)
            }
            Node::Arith { op, l, r, ty, try_ } => {
                let (a, b) = (self.eval(l)?, self.eval(r)?);
                self.arith(*op, &a, &b, ty, *try_)
            }
            Node::Cmp(kind, l, r) => {
                let (a, b) = (self.eval(l)?, self.eval(r)?);
                self.compare(*kind, &a, &b)
            }
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
        Expected::Nondeterministic { .. } => {
            return CellOutcome::NotRun("nondeterministic".to_owned());
        }
    };
    if case
        .conf
        .get("spark.sql.session.timeZone")
        .is_some_and(|z| z != "UTC")
    {
        return CellOutcome::NotRun("session time zone other than UTC".to_owned());
    }
    let opts = krishiv_sql::ParserOptions::from_conf(|k| case.conf.get(k).map(String::as_str));
    let Ok(q) = krishiv_sql::parse_query(&case.sql, &opts) else {
        return CellOutcome::NotRun("does not parse".to_owned());
    };
    let Some(expr) = select_expr(&q) else {
        return CellOutcome::NotRun("not a single-expression SELECT".to_owned());
    };
    let ansi = case.mode == Mode::AnsiOn;
    let node = match (Binder { ansi, try_: false }).bind(expr) {
        Ok(n) => n,
        Err(why) => return CellOutcome::NotRun(why),
    };
    if let Some(field) = schema.and_then(|s| s.first())
        && field.data_type != simple_string(&node.ty())
    {
        return CellOutcome::Fail(format!(
            "type {} vs Spark {}",
            simple_string(&node.ty()),
            field.data_type
        ));
    }
    for shape in [
        Shape::Arrays,
        Shape::LeftScalar,
        Shape::RightScalar,
        Shape::Padded,
    ] {
        let ev = Evaluator {
            ctx: KernelCtx::new(),
            ansi,
            shape,
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
                let same = e.error_class().name() == want.error_class
                    && e.to_string() == want.message
                    && e.row() == Some(0);
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
                let rows_ok = (0..a.len())
                    .step_by(2)
                    .all(|i| cell_matches(a.as_ref(), i, cell));
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
            return CellOutcome::Fail(why);
        }
    }
    CellOutcome::Pass
}
