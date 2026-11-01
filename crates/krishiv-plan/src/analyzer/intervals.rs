// SPDX-License-Identifier: Apache-2.0
//! `+ - * /` over intervals, dates and timestamps: Spark's
//! `BinaryArithmeticWithDatetimeResolver`, case by case, its fall-throughs
//! included, so an error reads as Spark's does (`1 + INTERVAL '1' DAY` is a
//! bad `TimestampAddInterval`, and so is `INTERVAL '1' MONTH + INTERVAL '1' DAY`).
//!
//! Each rewrite has its Spark name in [`Bound::shown`]. Calendar intervals are
//! not handled yet.

use std::sync::Arc;

use krishiv_types::{DataType, DayTimeField};

use super::AnalysisError;
use super::bind::unsupported;
use super::coerce::{self, Mismatch};
use super::exprs::{Bound, Cx, arg_type};
use super::resolve::Resolver;
use crate::{
    Builtin, Expr, ExprNode, FunctionId, FunctionKind, Origin, ResolvedExpr, ResolvedFunction,
};

/// How the resolver sees an operand.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Null,
    String,
    Date,
    Timestamp,
    YearMonth,
    /// `INTERVAL DAY`.
    Days,
    /// Any other day-time interval.
    DayTime,
    Calendar,
    Other,
}

fn kind(b: &Bound) -> Kind {
    match &b.ty.data_type {
        None | Some(DataType::Null) => Kind::Null,
        Some(DataType::String) => Kind::String,
        Some(DataType::Date) => Kind::Date,
        Some(DataType::Timestamp | DataType::TimestampNtz) => Kind::Timestamp,
        Some(DataType::YearMonthInterval { .. }) => Kind::YearMonth,
        Some(DataType::DayTimeInterval {
            start: DayTimeField::Day,
            end: DayTimeField::Day,
        }) => Kind::Days,
        Some(DataType::DayTimeInterval { .. }) => Kind::DayTime,
        Some(DataType::CalendarInterval) => Kind::Calendar,
        Some(_) => Kind::Other,
    }
}

const fn day_time(k: Kind) -> bool {
    matches!(k, Kind::Days | Kind::DayTime)
}

const fn ansi(k: Kind) -> bool {
    matches!(k, Kind::YearMonth | Kind::Days | Kind::DayTime)
}

fn numeric(b: &Bound) -> bool {
    matches!(
        b.ty.data_type,
        Some(
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
        )
    )
}

/// The required type of a timestamp operand, as Spark prints it.
const ANY_TIMESTAMP: &str = "\"(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)\"";

/// A resolved call of operator `id`.
fn call(
    id: Builtin,
    args: Vec<Arc<Expr>>,
    data_type: DataType,
    nullable: bool,
    origin: Origin,
) -> Arc<Expr> {
    let f = ResolvedFunction {
        id: FunctionId::Builtin(id),
        kind: FunctionKind::Scalar,
        args,
        distinct: false,
        filter: None,
        within_group: Vec::new(),
        data_type,
        nullable,
        deterministic: true,
    };
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(origin))
}

/// The shown form of `id(kids)`, over the operands as written.
fn shown(id: Builtin, kids: [&Bound; 2], origin: Origin) -> Arc<Expr> {
    call(
        id,
        kids.iter().map(|k| k.shown()).collect(),
        DataType::Boolean,
        true,
        origin,
    )
}

/// A bound `expr` of type `data_type`, with the markers of `kids`.
fn bound(expr: Arc<Expr>, data_type: DataType, shown: Arc<Expr>, kids: &[Bound]) -> Bound {
    let nullable = kids.iter().any(|k| k.ty.nullable);
    let mut b = Bound {
        expr,
        ty: arg_type(data_type, nullable),
        agg: None,
        win: None,
        outer: None,
        lateral: Vec::new(),
        shown: Some(shown),
    };
    b.carry_markers(kids);
    b
}

impl Resolver<'_> {
    /// `l name r` when an interval, or a date or timestamp difference, is
    /// involved; `None` when the operator's ordinary path applies.
    pub(super) fn interval_operator(
        &mut self,
        name: &str,
        origin: Origin,
        kids: &[Bound],
        cx: &Cx<'_>,
    ) -> Result<Option<Bound>, AnalysisError> {
        if let ("-", [e]) = (name, kids) {
            return Ok(match kind(e) {
                Kind::Calendar => return Err(calendar(origin)),
                k if ansi(k) => {
                    let t = e.ty.data_type.clone().unwrap_or(DataType::Null);
                    let shown = call(
                        Builtin::Negative,
                        vec![e.shown()],
                        t.clone(),
                        e.ty.nullable,
                        origin,
                    );
                    Some(bound(
                        negative(e, origin),
                        t,
                        shown,
                        std::slice::from_ref(e),
                    ))
                }
                _ => None,
            });
        }
        if let ("positive", [e]) = (name, kids)
            && ansi(kind(e))
        {
            let mut b = e.clone();
            b.shown = Some(Arc::new(
                Expr::function("positive", vec![Expr::clone(&e.shown())]).with_origin(origin),
            ));
            return Ok(Some(b));
        }
        let [l, r] = kids else {
            return Ok(None);
        };
        let (lk, rk) = (kind(l), kind(r));
        if let Some(op) = try_operator(name) {
            let datetimes = op == "-"
                && matches!(lk, Kind::Date | Kind::Timestamp)
                && matches!(rk, Kind::Date | Kind::Timestamp);
            if !ansi(lk) && !ansi(rk) && !datetimes {
                return Ok(None);
            }
            let Some(inner) = self.interval_operator(op, origin, kids, cx)? else {
                return Ok(None);
            };
            let t = inner.ty.data_type.clone().unwrap_or(DataType::Null);
            let expr = call(
                Builtin::TryEval,
                vec![Arc::clone(&inner.expr)],
                t.clone(),
                true,
                origin,
            );
            let shown = Arc::new(
                Expr::function(name, vec![Expr::clone(&l.shown()), Expr::clone(&r.shown())])
                    .with_origin(origin),
            );
            let mut b = bound(expr, t, shown, kids);
            b.ty.nullable = true;
            return Ok(Some(b));
        }
        match name {
            "+" => self.interval_add(origin, l, r, lk, rk, cx),
            "-" => self.interval_subtract(origin, l, r, lk, rk, cx),
            "*" => match (lk, rk) {
                (Kind::Calendar, _) | (_, Kind::Calendar) => Err(calendar(origin)),
                (a, _) if ansi(a) => self
                    .interval_scale(Builtin::Multiply, origin, l, r)
                    .map(Some),
                (_, b) if ansi(b) => self
                    .interval_scale(Builtin::Multiply, origin, r, l)
                    .map(Some),
                _ => Ok(None),
            },
            "/" => match lk {
                Kind::Calendar => Err(calendar(origin)),
                a if ansi(a) => self.interval_scale(Builtin::Divide, origin, l, r).map(Some),
                _ => Ok(None),
            },
            _ => Ok(None),
        }
    }

    fn interval_add(
        &mut self,
        origin: Origin,
        l: &Bound,
        r: &Bound,
        lk: Kind,
        rk: Kind,
        cx: &Cx<'_>,
    ) -> Result<Option<Bound>, AnalysisError> {
        Ok(Some(match (lk, rk) {
            (Kind::Calendar, _) | (_, Kind::Calendar) => return Err(calendar(origin)),
            (Kind::Date, Kind::Days) => self.date_add_days(origin, l, r, false, cx)?,
            (Kind::Days, Kind::Date) => self.date_add_days(origin, r, l, false, cx)?,
            (Kind::Date, Kind::DayTime) => {
                self.date_add_time(origin, l, r, Builtin::TimestampAddInterval)
            }
            (Kind::DayTime, Kind::Date) => {
                self.date_add_time(origin, r, l, Builtin::TimestampAddInterval)
            }
            (Kind::Date, Kind::YearMonth) => add_months(Builtin::DateAddYmInterval, origin, l, r),
            (Kind::YearMonth, Kind::Date) => add_months(Builtin::DateAddYmInterval, origin, r, l),
            (Kind::Timestamp, Kind::YearMonth) => {
                add_months(Builtin::TimestampAddYmInterval, origin, l, r)
            }
            (Kind::YearMonth, Kind::Timestamp) => {
                add_months(Builtin::TimestampAddYmInterval, origin, r, l)
            }
            (a, b) if (day_time(a) && day_time(b)) || (a == Kind::YearMonth && b == a) => {
                same_family(Builtin::Add, origin, l, r)
            }
            (Kind::Null, b) if ansi(b) => self.null_operand(Builtin::Add, origin, l, r),
            (a, Kind::Null) if ansi(a) => self.null_operand(Builtin::Add, origin, l, r),
            // `DATE + NULL` reads the NULL as a day-time interval.
            (Kind::Date, Kind::Null) => self.date_add_null(origin, l, r),
            (Kind::Null, Kind::Date) => self.date_add_null(origin, r, l),
            // The operator form takes no string for the days, as the
            // `date_add` function does.
            (Kind::Date, Kind::String) => return Err(self.string_days("date_add", origin, l, r)),
            (Kind::Timestamp, Kind::Null) => self.timestamp_add_null(origin, l, r),
            (Kind::Null, Kind::Timestamp) => self.timestamp_add_null(origin, r, l),
            (_, b) if day_time(b) => self.timestamp_add(origin, l, r, false)?,
            (a, _) if day_time(a) => self.timestamp_add(origin, r, l, false)?,
            _ => return Ok(None),
        }))
    }

    fn interval_subtract(
        &mut self,
        origin: Origin,
        l: &Bound,
        r: &Bound,
        lk: Kind,
        rk: Kind,
        cx: &Cx<'_>,
    ) -> Result<Option<Bound>, AnalysisError> {
        Ok(Some(match (lk, rk) {
            (Kind::Calendar, _) | (_, Kind::Calendar) => return Err(calendar(origin)),
            (Kind::Date, Kind::Days) => self.date_add_days(origin, l, r, true, cx)?,
            (Kind::Date, Kind::DayTime) => {
                let ts = self.implicit_cast(l, Some(DataType::Timestamp));
                let negated = negative(r, origin);
                let expr = call(
                    Builtin::TimestampAddInterval,
                    vec![Arc::clone(&ts.expr), negated],
                    DataType::Timestamp,
                    l.ty.nullable || r.ty.nullable,
                    origin,
                );
                bound(
                    expr,
                    DataType::Timestamp,
                    shown(Builtin::DatetimeSub, [l, r], origin),
                    &[l.clone(), r.clone()],
                )
            }
            (Kind::Date | Kind::Timestamp, Kind::YearMonth) => {
                let (id, t) = if lk == Kind::Date {
                    (Builtin::DateAddYmInterval, DataType::Date)
                } else {
                    (
                        Builtin::TimestampAddYmInterval,
                        l.ty.data_type.clone().unwrap_or(DataType::Timestamp),
                    )
                };
                let expr = call(
                    id,
                    vec![Arc::clone(&l.expr), negative(r, origin)],
                    t.clone(),
                    l.ty.nullable || r.ty.nullable,
                    origin,
                );
                bound(
                    expr,
                    t,
                    shown(Builtin::DatetimeSub, [l, r], origin),
                    &[l.clone(), r.clone()],
                )
            }
            (a, b) if (day_time(a) && day_time(b)) || (a == Kind::YearMonth && b == a) => {
                same_family(Builtin::Subtract, origin, l, r)
            }
            (Kind::Null, b) if ansi(b) => self.null_operand(Builtin::Subtract, origin, l, r),
            (a, Kind::Null) if ansi(a) => self.null_operand(Builtin::Subtract, origin, l, r),
            (_, b) if day_time(b) => self.timestamp_add(origin, l, r, true)?,
            (Kind::Timestamp, _) | (_, Kind::Timestamp) => {
                self.timestamp_difference(origin, l, r)?
            }
            // Only ANSI reads the string as a date.
            (Kind::Date, Kind::String) if !self.ctx.conf.ansi => {
                return Err(self.string_days("date_sub", origin, l, r));
            }
            (_, Kind::Date) | (Kind::Date, Kind::Null | Kind::String) => {
                self.date_difference(origin, l, r)?
            }
            _ => return Ok(None),
        }))
    }

    /// A string where `date_add` or `date_sub` takes its days.
    fn string_days(&self, function: &str, origin: Origin, l: &Bound, r: &Bound) -> AnalysisError {
        let name = Arc::new(
            Expr::function(
                function,
                vec![Expr::clone(&l.shown()), Expr::clone(&r.shown())],
            )
            .with_origin(origin),
        );
        let ty = if self.ctx.conf.ansi {
            // Spark has cast the string to the date's type by then.
            DataType::Date
        } else {
            DataType::String
        };
        let m = Mismatch::Unexpected {
            arg: 1,
            param: 1,
            ty: Some(ty),
            required: "(\"INT\" or \"SMALLINT\" or \"TINYINT\")".to_owned(),
        };
        self.mismatch(m, &name, &[l.clone(), r.clone()], function, origin)
    }

    /// `date_add(d, extractansiintervaldays(i))`, or with the days negated.
    fn date_add_days(
        &mut self,
        origin: Origin,
        date: &Bound,
        days: &Bound,
        negate: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let nullable = days.ty.nullable;
        let mut extract = call(
            Builtin::ExtractAnsiIntervalDays,
            vec![Arc::clone(&days.expr)],
            DataType::Integer,
            nullable,
            origin,
        );
        let mut shown_days = Arc::new(
            Expr::function("extractansiintervaldays", vec![Expr::clone(&days.shown())])
                .with_origin(origin),
        );
        if negate {
            extract = call(
                Builtin::Negative,
                vec![extract],
                DataType::Integer,
                nullable,
                origin,
            );
            shown_days = call(
                Builtin::Negative,
                vec![shown_days],
                DataType::Integer,
                nullable,
                origin,
            );
        }
        let rewritten = Expr::function(
            "date_add",
            vec![Expr::clone(&date.expr), Expr::clone(&extract)],
        )
        .with_origin(origin);
        let mut b = self.expr(&Arc::new(rewritten), cx)?;
        b.carry_markers(&[date.clone(), days.clone()]);
        b.shown = Some(Arc::new(
            Expr::function(
                "date_add",
                vec![Expr::clone(&date.shown()), Expr::clone(&shown_days)],
            )
            .with_origin(origin),
        ));
        Ok(b)
    }

    /// `date + time`, as a timestamp.
    fn date_add_time(&mut self, origin: Origin, date: &Bound, i: &Bound, id: Builtin) -> Bound {
        let ts = self.implicit_cast(date, Some(DataType::Timestamp));
        let expr = call(
            id,
            vec![Arc::clone(&ts.expr), Arc::clone(&i.expr)],
            DataType::Timestamp,
            date.ty.nullable || i.ty.nullable,
            origin,
        );
        bound(
            expr,
            DataType::Timestamp,
            shown(id, [date, i], origin),
            &[date.clone(), i.clone()],
        )
    }

    /// `DATE + NULL`: a day-time interval of no value, so a `TIMESTAMP`.
    fn date_add_null(&mut self, origin: Origin, date: &Bound, null: &Bound) -> Bound {
        let i = self.implicit_cast(null, Some(DataType::DEFAULT_DAY_TIME_INTERVAL));
        let ts = self.implicit_cast(date, Some(DataType::Timestamp));
        let expr = call(
            Builtin::TimestampAddInterval,
            vec![Arc::clone(&ts.expr), Arc::clone(&i.expr)],
            DataType::Timestamp,
            true,
            origin,
        );
        bound(
            expr,
            DataType::Timestamp,
            shown(Builtin::TimestampAddInterval, [date, null], origin),
            &[date.clone(), null.clone()],
        )
    }

    /// `TIMESTAMP + NULL`: a day-time interval of no value.
    fn timestamp_add_null(&mut self, origin: Origin, ts: &Bound, null: &Bound) -> Bound {
        let i = self.implicit_cast(null, Some(DataType::DEFAULT_DAY_TIME_INTERVAL));
        let t = ts.ty.data_type.clone().unwrap_or(DataType::Timestamp);
        let expr = call(
            Builtin::TimestampAddInterval,
            vec![Arc::clone(&ts.expr), Arc::clone(&i.expr)],
            t.clone(),
            true,
            origin,
        );
        bound(
            expr,
            t,
            shown(Builtin::TimestampAddInterval, [ts, null], origin),
            &[ts.clone(), null.clone()],
        )
    }

    /// `Cast(TimestampAddInterval(l, r), l's type)`: the fall-through for a
    /// day-time interval. Only a timestamp, or a string read as one, takes it.
    fn timestamp_add(
        &mut self,
        origin: Origin,
        l: &Bound,
        i: &Bound,
        negate: bool,
    ) -> Result<Bound, AnalysisError> {
        let nullable = l.ty.nullable || i.ty.nullable;
        let id = if negate {
            Builtin::DatetimeSub
        } else {
            Builtin::TimestampAddInterval
        };
        let name = shown(id, [l, i], origin);
        let kids = [l.clone(), i.clone()];
        match kind(l) {
            Kind::Timestamp | Kind::String => {}
            _ => {
                // The operands of the rewrite: a subtraction negates the
                // interval first.
                let error_name = if negate {
                    call(
                        Builtin::TimestampAddInterval,
                        vec![l.shown(), negative_shown(i, origin)],
                        DataType::Boolean,
                        true,
                        origin,
                    )
                } else {
                    Arc::clone(&name)
                };
                let m = Mismatch::Unexpected {
                    arg: 0,
                    param: 0,
                    ty: l.ty.data_type.clone(),
                    required: ANY_TIMESTAMP.to_owned(),
                };
                return Err(self.mismatch(m, &error_name, &kids, "+", origin));
            }
        }
        let ts_type = match kind(l) {
            Kind::String => DataType::Timestamp,
            _ => l.ty.data_type.clone().unwrap_or(DataType::Timestamp),
        };
        let start = self.implicit_cast(l, Some(ts_type.clone()));
        let interval = if negate {
            negative(i, origin)
        } else {
            Arc::clone(&i.expr)
        };
        let add = call(
            Builtin::TimestampAddInterval,
            vec![Arc::clone(&start.expr), interval],
            ts_type.clone(),
            nullable,
            origin,
        );
        if kind(l) == Kind::String {
            let bounded = bound(add, ts_type, Arc::clone(&name), &kids);
            let mut back = self.implicit_cast(&bounded, Some(DataType::String));
            back.shown = Some(name);
            return Ok(back);
        }
        Ok(bound(add, ts_type, name, &kids))
    }

    /// `l - r` of timestamps (`SubtractTimestamps`); a date or a string reads
    /// as a timestamp, a timestamp without time zone only beside its own kind.
    fn timestamp_difference(
        &mut self,
        origin: Origin,
        l: &Bound,
        r: &Bound,
    ) -> Result<Bound, AnalysisError> {
        let kids = [l.clone(), r.clone()];
        let name = shown(Builtin::Subtract, [l, r], origin);
        for (arg, side) in kids.iter().enumerate() {
            // Only ANSI reads a string as a timestamp.
            let refused = kind(side) == Kind::String && !self.ctx.conf.ansi;
            if refused
                || matches!(
                    kind(side),
                    Kind::Other | Kind::YearMonth | Kind::Days | Kind::DayTime | Kind::Calendar
                )
            {
                let m = Mismatch::Unexpected {
                    arg,
                    param: arg,
                    ty: side.ty.data_type.clone(),
                    required: ANY_TIMESTAMP.to_owned(),
                };
                return Err(self.mismatch(m, &name, &kids, "-", origin));
            }
        }
        let both_ntz = kids
            .iter()
            .all(|k| matches!(k.ty.data_type, Some(DataType::TimestampNtz)));
        let t = if both_ntz {
            DataType::TimestampNtz
        } else {
            DataType::Timestamp
        };
        let (a, b) = (
            self.implicit_cast(l, Some(t.clone())),
            self.implicit_cast(r, Some(t)),
        );
        let out = DataType::DEFAULT_DAY_TIME_INTERVAL;
        let expr = call(
            Builtin::SubtractTimestamps,
            vec![Arc::clone(&a.expr), Arc::clone(&b.expr)],
            out.clone(),
            l.ty.nullable || r.ty.nullable,
            origin,
        );
        Ok(bound(expr, out, name, &kids))
    }

    /// `l - r` of dates (`SubtractDates`), an `INTERVAL DAY`; a string reads
    /// as a date.
    fn date_difference(
        &mut self,
        origin: Origin,
        l: &Bound,
        r: &Bound,
    ) -> Result<Bound, AnalysisError> {
        let kids = [l.clone(), r.clone()];
        let name = shown(Builtin::Subtract, [l, r], origin);
        for (arg, side) in kids.iter().enumerate() {
            if !matches!(kind(side), Kind::Date | Kind::String | Kind::Null) {
                let m = Mismatch::Unexpected {
                    arg,
                    param: arg,
                    ty: side.ty.data_type.clone(),
                    required: "\"DATE\"".to_owned(),
                };
                return Err(self.mismatch(m, &name, &kids, "-", origin));
            }
        }
        let (a, b) = (
            self.implicit_cast(l, Some(DataType::Date)),
            self.implicit_cast(r, Some(DataType::Date)),
        );
        let out = DataType::DayTimeInterval {
            start: DayTimeField::Day,
            end: DayTimeField::Day,
        };
        let expr = call(
            Builtin::SubtractDates,
            vec![Arc::clone(&a.expr), Arc::clone(&b.expr)],
            out.clone(),
            l.ty.nullable || r.ty.nullable,
            origin,
        );
        Ok(bound(expr, out, name, &kids))
    }

    /// `NULL op interval`: the NULL takes the interval's type.
    fn null_operand(&mut self, id: Builtin, origin: Origin, l: &Bound, r: &Bound) -> Bound {
        let t = if kind(l) == Kind::Null {
            r.ty.data_type.clone()
        } else {
            l.ty.data_type.clone()
        };
        let (a, b) = (
            self.implicit_cast(l, t.clone()),
            self.implicit_cast(r, t.clone()),
        );
        let t = t.unwrap_or(DataType::Null);
        let expr = call(
            id,
            vec![Arc::clone(&a.expr), Arc::clone(&b.expr)],
            t.clone(),
            true,
            origin,
        );
        bound(expr, t, shown(id, [l, r], origin), &[l.clone(), r.clone()])
    }

    /// `interval * number` or `interval / number`, of the default fields.
    fn interval_scale(
        &mut self,
        id: Builtin,
        origin: Origin,
        interval: &Bound,
        by: &Bound,
    ) -> Result<Bound, AnalysisError> {
        let name = shown(id, [interval, by], origin);
        let kids = [interval.clone(), by.clone()];
        let out = if kind(interval) == Kind::YearMonth {
            DataType::DEFAULT_YEAR_MONTH_INTERVAL
        } else {
            DataType::DEFAULT_DAY_TIME_INTERVAL
        };
        let number = if numeric(by) {
            by.clone()
        } else if matches!(kind(by), Kind::String | Kind::Null) {
            self.implicit_cast(by, Some(DataType::Double))
        } else {
            let m = Mismatch::Unexpected {
                arg: 1,
                param: 1,
                ty: by.ty.data_type.clone(),
                required: "\"NUMERIC\"".to_owned(),
            };
            return Err(self.mismatch(m, &name, &kids, id.name(), origin));
        };
        let expr = call(
            id,
            vec![Arc::clone(&interval.expr), Arc::clone(&number.expr)],
            out.clone(),
            interval.ty.nullable || number.ty.nullable,
            origin,
        );
        Ok(bound(expr, out, name, &kids))
    }
}

/// The operator a `try_*` function evaluates.
fn try_operator(name: &str) -> Option<&'static str> {
    Some(match name {
        "try_add" => "+",
        "try_subtract" => "-",
        "try_multiply" => "*",
        "try_divide" => "/",
        _ => return None,
    })
}

fn calendar(origin: Origin) -> AnalysisError {
    unsupported("calendar interval arithmetic", origin)
}

fn negative(i: &Bound, origin: Origin) -> Arc<Expr> {
    call(
        Builtin::Negative,
        vec![Arc::clone(&i.expr)],
        i.ty.data_type.clone().unwrap_or(DataType::Null),
        i.ty.nullable,
        origin,
    )
}

fn negative_shown(i: &Bound, origin: Origin) -> Arc<Expr> {
    call(
        Builtin::Negative,
        vec![i.shown()],
        i.ty.data_type.clone().unwrap_or(DataType::Null),
        i.ty.nullable,
        origin,
    )
}

/// `start + months`; the result has `start`'s type.
fn add_months(id: Builtin, origin: Origin, start: &Bound, i: &Bound) -> Bound {
    let t = start.ty.data_type.clone().unwrap_or(DataType::Date);
    let expr = call(
        id,
        vec![Arc::clone(&start.expr), Arc::clone(&i.expr)],
        t.clone(),
        start.ty.nullable || i.ty.nullable,
        origin,
    );
    bound(
        expr,
        t,
        shown(id, [start, i], origin),
        &[start.clone(), i.clone()],
    )
}

/// `l op r` of one family; the fields widen, the values need no change.
fn same_family(id: Builtin, origin: Origin, l: &Bound, r: &Bound) -> Bound {
    let t = match (&l.ty.data_type, &r.ty.data_type) {
        (Some(a), Some(b)) => coerce::widen_intervals(a, b),
        _ => None,
    }
    .or_else(|| l.ty.data_type.clone())
    .unwrap_or(DataType::Null);
    let expr = call(
        id,
        vec![Arc::clone(&l.expr), Arc::clone(&r.expr)],
        t.clone(),
        l.ty.nullable || r.ty.nullable,
        origin,
    );
    bound(expr, t, shown(id, [l, r], origin), &[l.clone(), r.clone()])
}
