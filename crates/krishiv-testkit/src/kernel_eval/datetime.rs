// SPDX-License-Identifier: Apache-2.0
//! The datetime functions of the kernel evaluator (P2-T025a): `date_add`,
//! `date_sub`, `datediff`, `year`/`month`/`day`/`dayofmonth`, `extract` /
//! `date_part`, `current_date`/`current_timestamp`, `to_date`/`to_timestamp`
//! (the default pattern: the string casts) and `date`/`timestamp` ± ANSI
//! interval literals.

use std::sync::Arc;

use arrow_array::types::{
    Date32Type, Int32Type, Int64Type, IntervalYearMonthType, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, Date32Array, DurationMicrosecondArray, Int32Array, Int64Array,
    IntervalYearMonthArray, PrimitiveArray, TimestampMicrosecondArray, cast::AsArray,
};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use krishiv_kernels::Operand;
use krishiv_kernels::cast::{CastMode, CastOptions, cast};
use krishiv_kernels::datetime::{
    DatePart, current_date, date_add, date_add_months, date_diff, date_sub, extract_date,
    extract_timestamp, timestamp_add_day_time, timestamp_add_months,
};
use krishiv_kernels::policy::OverflowMode;
use krishiv_sql::ast::{self, ExprKind};

use super::{AOp, Binder, EvalError, Evaluator, Node, Shape, Skip, ltz, simple_string};

const MICROS_PER_DAY: i64 = 86_400_000_000;

/// The query-start instant matrix cells run clock functions at
/// (2026-10-18 12:34:56.789012 UTC).
pub(super) const NOW: i64 = 1_792_326_896_789_012;

/// An ANSI interval literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IntervalVal {
    /// Months (`INTERVAL '3' MONTH`, `'1-2' YEAR TO MONTH`).
    YearMonth(i32),
    /// Microseconds; `day_only` when the interval is `DAY` alone (Spark adds
    /// it to a date as days and keeps a date).
    DayTime { micros: i64, day_only: bool },
}

/// A datetime function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DtFn {
    /// `date_add`/`date + int`; `try_`.
    DateAdd(bool),
    /// `date_sub`/`date - int`; `try_`.
    DateSub(bool),
    DateDiff,
    Extract(DatePart),
    CurrentDate,
    CurrentTimestamp,
    /// `date +/- year-month interval`; `negate`, `try_`.
    DateMonths(bool, bool),
    /// `date +/- DAY interval`.
    DateDays(bool, bool),
    /// `timestamp +/- year-month interval`: `negate`, `try_`, `ntz`.
    TsMonths(bool, bool, bool),
    /// `timestamp +/- day-time interval`: `negate`, `try_`.
    TsMicros(bool, bool),
    /// `date +/- day-time interval` in hours or smaller: the date becomes a
    /// timestamp in the session zone first.
    DateMicros(bool, bool),
}

fn is_interval(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Interval(IntervalUnit::YearMonth) | DataType::Duration(TimeUnit::Microsecond)
    )
}

fn is_timestamp(t: &DataType) -> bool {
    matches!(t, DataType::Timestamp(TimeUnit::Microsecond, _))
}

/// `[+-]n` as an integer.
fn integer(text: &str) -> Option<i128> {
    let t = text.trim();
    let (neg, digits) = match t.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let v: i128 = digits
        .parse()
        .ok()
        .filter(|_| !digits.starts_with(['+', '-']))?;
    Some(if neg { -v } else { v })
}

/// `s[.f]` seconds (at most 6 fraction digits) as microseconds.
fn seconds(text: &str) -> Option<i128> {
    let (whole, frac) = text.split_once('.').unwrap_or((text, ""));
    if frac.len() > 6 || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let micros: i128 = format!("{frac:0<6}").parse().ok()?;
    let neg = whole.trim().starts_with('-');
    let secs = integer(whole)?.checked_mul(1_000_000)?;
    secs.checked_add(if neg { -micros } else { micros })
}

fn year_month(months: i128) -> Result<IntervalVal, Skip> {
    i32::try_from(months)
        .map(IntervalVal::YearMonth)
        .map_err(|_| "interval overflow".to_owned())
}

fn day_time(micros: i128, day_only: bool) -> Result<IntervalVal, Skip> {
    i64::try_from(micros)
        .map(|micros| IntervalVal::DayTime { micros, day_only })
        .map_err(|_| "interval overflow".to_owned())
}

/// The value of one `value unit` pair as months or microseconds.
fn unit_value(v: &ast::IntervalValue, unit: &str) -> Result<(bool, i128), Skip> {
    let text = if v.negative {
        format!("-{}", v.value)
    } else {
        v.value.clone()
    };
    let bad = || format!("interval value {text} {unit}");
    let unit = unit.trim_end_matches('s');
    let (year_month, factor) = match unit {
        "year" => (true, 12),
        "month" => (true, 1),
        "day" => (false, i128::from(MICROS_PER_DAY)),
        "hour" => (false, 3_600_000_000),
        "minute" => (false, 60_000_000),
        "millisecond" => (false, 1_000),
        "microsecond" | "second" => (false, 1),
        other => return Err(format!("interval unit {other}")),
    };
    let x = if unit == "second" {
        seconds(&text)
    } else {
        integer(&text).and_then(|n| n.checked_mul(factor))
    };
    Ok((year_month, x.ok_or_else(bad)?))
}

/// An interval literal (`INTERVAL '90' DAY`, `interval 3 months`,
/// `INTERVAL '1 02:03:04' DAY TO SECOND`) as a node.
pub(super) fn interval_literal(iv: &ast::Interval) -> Result<Node, Skip> {
    if iv.trailing.is_some() {
        return Err("interval literal with a trailing interval".to_owned());
    }
    let val = match &iv.range {
        Some((_, from, to)) => return Err(format!("interval {from} TO {to}")),
        None => {
            let (mut months, mut micros, mut family) = (0_i128, 0_i128, None);
            for (v, unit) in &iv.units {
                let (is_ym, x) = unit_value(v, unit)?;
                if family.is_some_and(|f| f != is_ym) {
                    return Err("interval mixing year-month and day-time fields".to_owned());
                }
                family = Some(is_ym);
                let total = if is_ym { &mut months } else { &mut micros };
                *total = total.checked_add(x).ok_or("interval overflow")?;
            }
            match family {
                Some(true) => year_month(months)?,
                Some(false) => day_time(
                    micros,
                    matches!(iv.units.as_slice(), [(_, u)] if u.trim_end_matches('s') == "day"),
                )?,
                None => return Err("empty interval".to_owned()),
            }
        }
    };
    Ok(Node::Interval(val))
}

impl Binder<'_> {
    /// A `DATE` argument: dates as they are; timestamps, strings and NULL
    /// are cast.
    fn date_arg(self, n: Node) -> Result<Node, Skip> {
        match n.ty() {
            DataType::Date32 => Ok(n),
            DataType::Null | DataType::Utf8 | DataType::Timestamp(..) => {
                Ok(self.implicit(n, &DataType::Date32))
            }
            t => Err(format!("{} as a date argument", simple_string(&t))),
        }
    }

    /// An `int` argument.
    fn day_arg(self, n: Node) -> Result<Node, Skip> {
        match n.ty() {
            DataType::Int32 => Ok(n),
            DataType::Null | DataType::Utf8 | DataType::Int8 | DataType::Int16 => {
                Ok(self.implicit(n, &DataType::Int32))
            }
            t => Err(format!("{} as a day count", simple_string(&t))),
        }
    }

    fn dt(f: DtFn, args: Vec<Node>, ty: DataType) -> Node {
        Node::Dt(f, args, ty)
    }

    /// The datetime function `name` applied to `args`, or `None` when it is
    /// not one of them.
    pub(super) fn datetime_function(
        self,
        name: &str,
        args: &[&ast::Expr],
    ) -> Option<Result<Node, Skip>> {
        Some(match name {
            "date_add" | "dateadd" | "date_sub" | "datediff" | "date_diff" => {
                self.day_function(name, args)
            }
            "year" | "month" | "day" | "dayofmonth" => self.field_function(name, args),
            "extract" | "date_part" => self.extract_function(args),
            "current_date" | "curdate" | "current_timestamp" | "now" => {
                self.clock_function(name, args)
            }
            _ => return None,
        })
    }

    fn day_function(self, name: &str, args: &[&ast::Expr]) -> Result<Node, Skip> {
        let [a, b] = args else {
            return Err(format!("{name} with {} arguments", args.len()));
        };
        if name.contains("diff") {
            let (end, start) = (self.bind(a)?, self.bind(b)?);
            let (end, start) = (self.date_arg(end)?, self.date_arg(start)?);
            return Ok(Self::dt(DtFn::DateDiff, vec![end, start], DataType::Int32));
        }
        let (start, days) = (self.bind(a)?, self.bind(b)?);
        let f = if name == "date_sub" {
            DtFn::DateSub(self.try_)
        } else {
            DtFn::DateAdd(self.try_)
        };
        Ok(Self::dt(
            f,
            vec![self.date_arg(start)?, self.day_arg(days)?],
            DataType::Date32,
        ))
    }

    fn field_function(self, name: &str, args: &[&ast::Expr]) -> Result<Node, Skip> {
        let [a] = args else {
            return Err(format!("{name} with {} arguments", args.len()));
        };
        let part = if name == "year" {
            DatePart::Year
        } else if name == "month" {
            DatePart::Month
        } else {
            DatePart::Day
        };
        let d = self.date_arg(self.bind(a)?)?;
        Ok(Self::dt(DtFn::Extract(part), vec![d], DataType::Int32))
    }

    /// `extract(field FROM x)` / `date_part('field', x)`.
    fn extract_function(self, args: &[&ast::Expr]) -> Result<Node, Skip> {
        let [field, source] = args else {
            return Err(format!("extract with {} arguments", args.len()));
        };
        let name = match &field.kind {
            ExprKind::Keyword(k) => k.value.clone(),
            ExprKind::Literal(ast::Literal::String(s)) => s.clone(),
            _ => return Err("a non-constant extract field".to_owned()),
        };
        let part =
            DatePart::parse(&name).ok_or_else(|| format!("analysis: extract field {name}"))?;
        let x = self.bind(source)?;
        // A string is read as a date, except for the time fields (`HOUR`,
        // ...), which read a timestamp.
        let target = if matches!(part, DatePart::Hour | DatePart::Minute | DatePart::Second) {
            ltz()
        } else {
            DataType::Date32
        };
        let x = match x.ty() {
            DataType::Date32 | DataType::Timestamp(..) => x,
            DataType::Null | DataType::Utf8 => self.implicit(x, &target),
            t => return Err(format!("{} as an extract source", simple_string(&t))),
        };
        let ty = part.result_type();
        Ok(Self::dt(DtFn::Extract(part), vec![x], ty))
    }

    fn clock_function(self, name: &str, args: &[&ast::Expr]) -> Result<Node, Skip> {
        if !args.is_empty() {
            return Err(format!("{name} with arguments"));
        }
        if self.clock.is_none() {
            return Err("depends on the query clock".to_owned());
        }
        Ok(if name.contains("date") {
            Self::dt(DtFn::CurrentDate, vec![], DataType::Date32)
        } else {
            Self::dt(DtFn::CurrentTimestamp, vec![], ltz())
        })
    }

    /// `l op r` where an operand is a date, a timestamp or an interval;
    /// `None` for plain numeric arithmetic.
    pub(super) fn temporal_arith(self, op: AOp, l: &Node, r: &Node) -> Option<Result<Node, Skip>> {
        let (lt, rt) = (l.ty(), r.ty());
        let temporal = |t: &DataType| matches!(t, DataType::Date32) || is_timestamp(t);
        if !(temporal(&lt) || temporal(&rt) || is_interval(&lt) || is_interval(&rt)) {
            return None;
        }
        let negate = match op {
            AOp::Add => false,
            AOp::Sub => true,
            _ => {
                return Some(Err(
                    "interval `*`, `/`, `%` or `div` (interval arithmetic)".to_owned()
                ));
            }
        };
        // `date + int`, `date - int` and `int + date` are `date_add`/`date_sub`.
        let small_int =
            |t: &DataType| matches!(t, DataType::Int8 | DataType::Int16 | DataType::Int32);
        let days = match (&lt, &rt) {
            (DataType::Date32, t) if small_int(t) => Some((l, r)),
            (t, DataType::Date32) if small_int(t) && !negate => Some((r, l)),
            _ => None,
        };
        if let Some((d, n)) = days {
            let f = if negate {
                DtFn::DateSub(self.try_)
            } else {
                DtFn::DateAdd(self.try_)
            };
            let n = self.implicit(n.clone(), &DataType::Int32);
            return Some(Ok(Self::dt(f, vec![d.clone(), n], DataType::Date32)));
        }
        // `interval + temporal` commutes; `interval - temporal` does not exist.
        let (base, iv) = if temporal(&lt) && is_interval(&rt) {
            (l, r)
        } else if is_interval(&lt) && temporal(&rt) && !negate {
            (r, l)
        } else if is_interval(&lt) && is_interval(&rt) {
            return Some(Err("interval +/- interval (interval arithmetic)".to_owned()));
        } else {
            return Some(Err("analysis: arithmetic on date".to_owned()));
        };
        let day_only = matches!(
            iv,
            Node::Interval(IntervalVal::DayTime { day_only: true, .. })
        );
        let try_ = self.try_;
        let (f, ty) = match (base.ty(), iv.ty()) {
            (DataType::Date32, DataType::Interval(_)) => {
                (DtFn::DateMonths(negate, try_), DataType::Date32)
            }
            (DataType::Date32, _) if day_only => (DtFn::DateDays(negate, try_), DataType::Date32),
            (DataType::Date32, _) => (DtFn::DateMicros(negate, try_), ltz()),
            (t, DataType::Interval(_)) => {
                let ntz = matches!(t, DataType::Timestamp(_, None));
                (DtFn::TsMonths(negate, try_, ntz), t)
            }
            (t, _) => (DtFn::TsMicros(negate, try_), t),
        };
        Some(Ok(Self::dt(f, vec![base.clone(), iv.clone()], ty)))
    }
}

impl Evaluator<'_> {
    /// Argument `i` of `n` as a scalar operand in this shape (`LeftScalar`:
    /// the first; `RightScalar`: all but the first).
    fn dt_scalar(&self, i: usize, n: usize) -> bool {
        n > 1
            && match self.shape {
                Shape::LeftScalar => i == 0,
                Shape::RightScalar => i > 0,
                Shape::Arrays | Shape::Padded => false,
            }
    }

    fn operand<'a, T: arrow_array::ArrowPrimitiveType>(
        &self,
        a: &'a PrimitiveArray<T>,
        i: usize,
        n: usize,
    ) -> Operand<'a, T> {
        if self.dt_scalar(i, n) {
            Operand::Scalar(a.is_valid(0).then(|| a.value(0)))
        } else {
            Operand::Array(a)
        }
    }

    /// The overflow policy of day arithmetic on dates: ANSI raises, `try_*`
    /// gives NULL, legacy wraps like Java's `int`.
    fn day_mode(&self, try_: bool) -> OverflowMode {
        match (try_, self.ansi) {
            (true, _) => OverflowMode::NullOnOverflow,
            (false, true) => OverflowMode::AnsiError,
            (false, false) => OverflowMode::JavaWrap,
        }
    }

    /// Boxes a day-arithmetic result, noting an ANSI overflow: Spark wraps
    /// there, Krishiv raises (a documented divergence, ADR 0023).
    fn day_overflow<A: Array + 'static>(
        &self,
        r: Result<A, krishiv_kernels::KernelError>,
    ) -> Result<ArrayRef, EvalError> {
        if r.as_ref().is_err_and(|e| {
            e.error_class() == krishiv_common::error::ErrorClass::ArithmeticOverflow
        }) {
            self.day_overflow.set(true);
        }
        Ok(Arc::new(r?))
    }

    /// The interval literal as an array in this evaluator's shape.
    pub(super) fn interval_array(&self, v: IntervalVal) -> Result<ArrayRef, EvalError> {
        let a: ArrayRef = match v {
            IntervalVal::YearMonth(m) => Arc::new(IntervalYearMonthArray::from(vec![m])),
            IntervalVal::DayTime { micros, .. } => {
                Arc::new(DurationMicrosecondArray::from(vec![micros]))
            }
        };
        self.leaf(&a)
    }

    /// Evaluates datetime function `f` over `args`.
    pub(super) fn datetime_fn(&self, f: DtFn, args: &[Node]) -> Result<ArrayRef, EvalError> {
        let v = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<Result<Vec<_>, _>>()?;
        let n = v.len();
        let ctx = &self.ctx;
        let date = |i: usize| v[i].as_primitive::<Date32Type>();
        let ts = |i: usize| v[i].as_primitive::<TimestampMicrosecondType>();
        let zone = |i: usize| -> Option<&dyn krishiv_kernels::cast::TimeZoneRules> {
            match v[i].data_type() {
                DataType::Timestamp(_, Some(_)) => Some(self.zone),
                _ => None,
            }
        };
        Ok(match f {
            DtFn::DateAdd(try_) | DtFn::DateSub(try_) => {
                let days = v[1].as_primitive::<Int32Type>();
                let (s, d) = (self.operand(date(0), 0, n), self.operand(days, 1, n));
                let mode = self.day_mode(try_);
                self.day_overflow(if matches!(f, DtFn::DateAdd(_)) {
                    date_add(s, d, mode, ctx)
                } else {
                    date_sub(s, d, mode, ctx)
                })?
            }
            DtFn::DateDiff => {
                let (e, s) = (self.operand(date(0), 0, n), self.operand(date(1), 1, n));
                self.day_overflow(date_diff(e, s, self.day_mode(false), ctx))?
            }
            DtFn::Extract(part) => match v[0].data_type() {
                DataType::Date32 => extract_date(part, date(0))?,
                _ => extract_timestamp(part, ts(0), zone(0))?,
            },
            DtFn::CurrentDate => {
                let d = current_date(self.now, self.zone)?;
                self.per_selected_row(Arc::new(Date32Array::from(vec![d])))?
            }
            DtFn::CurrentTimestamp => self.per_selected_row(Arc::new(
                TimestampMicrosecondArray::from(vec![self.now]).with_timezone("UTC"),
            ))?,
            DtFn::DateMonths(negate, try_) => {
                let months = v[1].as_primitive::<IntervalYearMonthType>();
                let months = Int32Array::new(months.values().clone(), months.nulls().cloned());
                Arc::new(date_add_months(
                    self.operand(date(0), 0, n),
                    self.operand(&months, 1, n),
                    negate,
                    try_,
                    ctx,
                )?)
            }
            DtFn::DateDays(negate, try_) => {
                let micros = v[1].as_primitive::<arrow_array::types::DurationMicrosecondType>();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "Spark's `.toInt` on whole days"
                )]
                let days: Int32Array = micros.unary(|m| (m / MICROS_PER_DAY) as i32);
                let (s, d) = (self.operand(date(0), 0, n), self.operand(&days, 1, n));
                let mode = self.day_mode(try_);
                self.day_overflow(if negate {
                    date_sub(s, d, mode, ctx)
                } else {
                    date_add(s, d, mode, ctx)
                })?
            }
            DtFn::TsMonths(negate, try_, _) => {
                let months = v[1].as_primitive::<IntervalYearMonthType>();
                let months = Int32Array::new(months.values().clone(), months.nulls().cloned());
                let out = timestamp_add_months(
                    self.operand(ts(0), 0, n),
                    self.operand(&months, 1, n),
                    negate,
                    zone(0),
                    try_,
                )?;
                Arc::new(out.with_timezone_opt(timezone_of(&v[0])))
            }
            DtFn::TsMicros(negate, try_) => {
                let micros = v[1].as_primitive::<arrow_array::types::DurationMicrosecondType>();
                let micros = Int64Array::new(micros.values().clone(), micros.nulls().cloned());
                let out = timestamp_add_day_time(
                    self.operand(ts(0), 0, n),
                    self.operand(&micros, 1, n),
                    negate,
                    zone(0),
                    try_,
                )?;
                Arc::new(out.with_timezone_opt(timezone_of(&v[0])))
            }
            DtFn::DateMicros(negate, try_) => {
                let start = cast(
                    &v[0],
                    &ltz(),
                    &CastOptions {
                        mode: CastMode::Ansi,
                        time_zone: self.zone,
                    },
                )?;
                let micros = v[1].as_primitive::<arrow_array::types::DurationMicrosecondType>();
                let micros: PrimitiveArray<Int64Type> =
                    Int64Array::new(micros.values().clone(), micros.nulls().cloned());
                let out = timestamp_add_day_time(
                    self.operand(start.as_primitive::<TimestampMicrosecondType>(), 0, n),
                    self.operand(&micros, 1, n),
                    negate,
                    Some(self.zone),
                    try_,
                )?;
                Arc::new(out.with_timezone("UTC"))
            }
        })
    }
}

/// The zone name of a timestamp array's carrier.
fn timezone_of(a: &ArrayRef) -> Option<Arc<str>> {
    match a.data_type() {
        DataType::Timestamp(_, z) => z.clone(),
        _ => None,
    }
}
