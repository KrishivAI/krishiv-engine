// SPDX-License-Identifier: Apache-2.0
//! `date_format` of a `TIME` (Spark 4.3's `DateFormatClass` over a `TimeType`): the time printed
//! through `TimeFormatter` over its nanoseconds, a pattern with a date or zone field the time
//! cannot fill raising `INVALID_PARAMETER_VALUE.PATTERN` (`formatTimeWithError`).

use std::sync::Arc;

use arrow_array::types::Time64NanosecondType;
use arrow_array::{Array, ArrayRef, StringArray};
use arrow_schema::DataType;
use krishiv_common::error::{InvalidParameterValuePattern, KrishivError};
use krishiv_kernels::datetime::TimePattern;
use krishiv_kernels::datetime_pattern::Purpose;

use super::super::Binding;
use super::super::args::{Inputs, internal};
use super::{PatternArg, PatternCache, reserve};

/// The binding of `date_format(time, pattern)` for a compiled pattern argument.
pub(super) fn binding(pat: PatternArg) -> Binding {
    Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        let times = x.array::<Time64NanosecondType>(0)?;
        let literal = match &pat {
            PatternArg::Null => {
                return Ok(Arc::new(StringArray::new_null(x.rows())) as ArrayRef);
            }
            PatternArg::Literal(p) => Some(p),
            PatternArg::Column => None,
        };
        let fmts = if literal.is_some() {
            None
        } else {
            Some(x.strings(1)?)
        };
        let mut cache = PatternCache::new(Purpose::FormatTime, false);
        let mut out = reserve::<Option<String>>(x.rows())?;
        for i in 0..x.rows() {
            if times.is_null(i) || fmts.as_ref().is_some_and(|f| f.is_null(i)) {
                out.push(None);
                continue;
            }
            let row = |e| krishiv_kernels::KernelError::with_row(Some(i), e);
            let p = match (literal, &fmts) {
                (Some(p), _) => p,
                (None, Some(f)) => cache.get(f.value(i)).map_err(row)?,
                (None, None) => {
                    return Err(internal("date_format: no pattern".to_owned()));
                }
            };
            out.push(Some(format_time(p, times.value(i)).map_err(row)?));
        }
        Ok(Arc::new(StringArray::from_iter(out.iter().map(|s| s.as_deref()))) as ArrayRef)
    })
}

/// `nanos` (a `TIME`) printed with `p`, as Spark 4.3's `DateFormatClass.formatTimeWithError`.
fn format_time(p: &TimePattern, nanos: i64) -> Result<String, KrishivError> {
    let printed = match p {
        TimePattern::Corrected(p) => p.format_time_of_day(nanos)?,
        TimePattern::Legacy(_) => None,
    };
    printed.ok_or_else(|| {
        let pattern = match p {
            TimePattern::Corrected(p) => p.pattern().to_owned(),
            TimePattern::Legacy(l) => l.pattern().to_owned(),
        };
        InvalidParameterValuePattern {
            function_name: "`date_format`".to_owned(),
            parameter: "`format`".to_owned(),
            value: format!("'{}'", pattern.replace('\\', "\\\\").replace('\'', "\\'")),
        }
        .into()
    })
}
