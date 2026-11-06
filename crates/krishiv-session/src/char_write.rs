// SPDX-License-Identifier: Apache-2.0
//! The write side of `CHAR(n)` / `VARCHAR(n)` columns (Spark's `CharVarcharUtils.stringLengthCheck`):
//! a stored value longer than `n` code points loses trailing spaces down to `n`; if it is still
//! longer the write fails with `EXCEED_LIMIT_LENGTH`. Nothing is padded on write (the read side
//! pads, `krishiv-plan`'s `char_varchar`). The check is an `if(...)` over the value whose failing
//! branch is a tagged `raise_error`, which [`map_error`] turns into the error class once the write
//! has failed, so the check needs no kernel of its own.

use krishiv_common::error::{ErrorClass, ExceedLimitLength, KrishivError};
use krishiv_plan::analyzer::ResolutionConf;

use crate::column::Column;
use crate::functions::{call_function, lit};
use crate::store::StoreColumn;

/// Marks the `raise_error` message of a failed check; the limit follows.
const TAG: &str = "\u{1}krishiv.exceed_limit_length:";

/// The `n` of a simple `char(n)` / `varchar(n)` catalog string (nested ones are not checked).
fn limit(declared: &str) -> Option<i32> {
    let d = declared.trim().to_ascii_lowercase();
    let inner = d
        .strip_prefix("char(")
        .or_else(|| d.strip_prefix("varchar("))?
        .strip_suffix(')')?;
    inner.trim().parse().ok().filter(|n| *n >= 0)
}

/// `value` as stored in a column declared `declared`: trimmed to the limit or failing the write.
pub(crate) fn checked(value: Column, column: &StoreColumn, conf: &ResolutionConf) -> Column {
    let declared = column
        .char_varchar
        .as_deref()
        .filter(|_| !conf.char_varchar_as_string);
    let Some(n) = declared.and_then(limit) else {
        return value;
    };
    let tail = || call_function("substring", [value.clone(), lit(n + 1)]);
    let over = call_function("length", [tail()])
        .gt(lit(0))
        .and(call_function("trim", [tail()]).not_eq(lit("")));
    call_function(
        "if",
        [
            over,
            call_function("raise_error", [lit(format!("{TAG}{n}"))]),
            call_function("substring", [value, lit(1), lit(n)]),
        ],
    )
}

/// A failed check as `EXCEED_LIMIT_LENGTH`; any other error as it is.
pub(crate) fn map_error(e: KrishivError) -> KrishivError {
    if e.error_class() != ErrorClass::UserRaisedException {
        return e;
    }
    let text = e.to_string();
    match text.split_once(TAG).map(|(_, rest)| {
        rest.chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
    }) {
        Some(limit) if !limit.is_empty() => ExceedLimitLength { limit }.into(),
        _ => e,
    }
}

#[cfg(test)]
mod tests {
    use super::limit;

    #[test]
    fn only_simple_char_and_varchar_have_a_limit() {
        assert_eq!(limit("char(5)"), Some(5));
        assert_eq!(limit("VARCHAR(10)"), Some(10));
        assert_eq!(limit("array<varchar(2)>"), None);
        assert_eq!(limit("string"), None);
    }
}
