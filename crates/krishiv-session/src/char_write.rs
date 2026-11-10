// SPDX-License-Identifier: Apache-2.0
//! The write side of `CHAR(n)` / `VARCHAR(n)` columns (Spark's `CharVarcharUtils.stringLengthCheck`):
//! a stored value longer than `n` code points loses trailing spaces down to `n`; if it is still
//! longer the write fails with `EXCEED_LIMIT_LENGTH`. A top-level `CHAR` is not padded on write (the
//! read side pads, `krishiv-plan`'s `char_varchar`); inside arrays, maps and structs it is, as Spark. The check is an `if(...)` over the value whose failing
//! branch is a tagged `raise_error`, which [`map_error`] turns into the error class once the write
//! has failed, so the check needs no kernel of its own.

use krishiv_common::error::{
    ErrorClass, ExceedLimitLength, KrishivError, KrishivUnsupportedFeature,
};
use krishiv_plan::Expr;
use krishiv_plan::analyzer::{AnalyzedPlan, DeclaredType, ResolutionConf};
use krishiv_types::DataType;

use crate::column::Column;
use crate::functions::{call_function, col, lit};
use crate::store::StoreColumn;

/// Marks the `raise_error` message of a failed check; the limit follows.
const TAG: &str = "\u{1}krishiv.exceed_limit_length:";

/// `value` as stored in a column declared `declared` (`CharVarcharUtils.stringLengthCheck`): trimmed to the limit or
/// failing the write. A `CHAR` inside an array, map or struct is padded too (`charTypeWriteSideCheck`); a top-level
/// one is left to the read side, which pads it.
pub(crate) fn checked(
    value: Column,
    column: &StoreColumn,
    conf: &ResolutionConf,
) -> Result<Column, KrishivError> {
    let Some(declared) = column
        .char_varchar
        .as_deref()
        .filter(|_| !conf.char_varchar_as_string)
    else {
        return Ok(value);
    };
    Ok(match DeclaredType::parse(declared) {
        DeclaredType::Char(n) | DeclaredType::Varchar(n) => leaf(value, n, false),
        DeclaredType::Other => value,
        nested => match nested_check(value, &nested, &column.data_type, 0) {
            Some(c) => c,
            None => {
                return Err(KrishivUnsupportedFeature {
                    feature: format!("writes to a column declared {declared}"),
                }
                .into());
            }
        },
    })
}

/// The check of one string: past the limit only trailing spaces may be cut, else the write fails.
fn leaf(value: Column, n: i32, pad: bool) -> Column {
    // At the largest length no string can be longer (a value holds fewer than 2^31 characters).
    let Some(past) = n.checked_add(1) else {
        return value;
    };
    let tail = || call_function("substring", [value.clone(), lit(past)]);
    let over = call_function("length", [tail()])
        .gt(lit(0))
        .and(call_function("trim", [tail()]).not_eq(lit("")));
    let cut = call_function("substring", [value.clone(), lit(1), lit(n)]);
    let stored = if pad {
        call_function("rpad", [cut, lit(n), lit(" ")])
    } else {
        cut
    };
    call_function(
        "if",
        [
            over,
            call_function("raise_error", [lit(format!("{TAG}{n}"))]),
            stored,
        ],
    )
}

/// `value` of type `ty` checked as `declared` says, recursively; `None` when `declared` does not follow `ty`.
/// `depth` names the lambda variables apart.
fn nested_check(
    value: Column,
    declared: &DeclaredType,
    ty: &DataType,
    depth: usize,
) -> Option<Column> {
    if !declared.is_limited() {
        return Some(value);
    }
    match (declared, ty) {
        (DeclaredType::Char(n), DataType::String) => Some(leaf(value, *n, true)),
        (DeclaredType::Varchar(n), DataType::String) => Some(leaf(value, *n, false)),
        (DeclaredType::Struct(fields), DataType::Struct(st))
            if fields.len() == st.fields().len() =>
        {
            let mut args = Vec::with_capacity(fields.len() * 2);
            for (d, f) in fields.iter().zip(st.fields().iter()) {
                args.push(lit(f.name()));
                args.push(nested_check(
                    value.clone().get_field(f.name()),
                    d,
                    f.data_type(),
                    depth,
                )?);
            }
            let is_null = call_function("isnull", [value]);
            let built = call_function("named_struct", args);
            Some(call_function(
                "if",
                [
                    is_null,
                    Column::new(Expr::literal(krishiv_plan::Literal::Null(Some(ty.clone())))),
                    built,
                ],
            ))
        }
        (DeclaredType::Array(e), DataType::Array(a)) => {
            array_check(value, e, a.element_type(), depth)
        }
        (DeclaredType::Map(k, v), DataType::Map(m)) => {
            let keys = array_check(
                call_function("map_keys", [value.clone()]),
                k,
                m.key_type(),
                depth,
            )?;
            let values = array_check(
                call_function("map_values", [value]),
                v,
                m.value_type(),
                depth,
            )?;
            Some(call_function("map_from_arrays", [keys, values]))
        }
        _ => None,
    }
}

/// `transform(arr, x -> <x checked>)`; `arr` itself when no element is limited.
fn array_check(
    arr: Column,
    declared: &DeclaredType,
    element: &DataType,
    depth: usize,
) -> Option<Column> {
    if !declared.is_limited() {
        return Some(arr);
    }
    let name = format!("__krishiv_x{depth}");
    let body = nested_check(col(&name), declared, element, depth + 1)?;
    let lambda = Column::new(Expr::lambda(vec![name], body.expr));
    Some(call_function("transform", [arr, lambda]))
}

/// The stored columns of a table or view made from a query: its output columns, each keeping the
/// `CHAR(n)` / `VARCHAR(n)` it was declared with and its comment (Spark carries both on the attribute); `nullable`
/// overrides the query's nullability (CTAS columns are nullable).
pub(crate) fn output_columns(a: &AnalyzedPlan, nullable: Option<bool>) -> Vec<StoreColumn> {
    a.output
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut column = StoreColumn::plain(
                c.name.clone(),
                krishiv_plan::without_generated_marker(&c.data_type),
                nullable.unwrap_or(c.nullable),
            );
            column.char_varchar = a.output_declared.get(i).cloned().flatten();
            column.comment = a.output_comments.get(i).cloned().flatten();
            column
        })
        .collect()
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
    use krishiv_plan::analyzer::DeclaredType as D;

    #[test]
    fn declared_types_parse() {
        assert_eq!(D::parse("char(5)"), D::Char(5));
        assert_eq!(D::parse("VARCHAR(10)"), D::Varchar(10));
        assert_eq!(D::parse("string"), D::Other);
        assert_eq!(
            D::parse("array<varchar(2)>"),
            D::Array(Box::new(D::Varchar(2)))
        );
        assert_eq!(
            D::parse("map<char(1),struct<a:int,b:array<varchar(3)>,c:decimal(10,2)>>"),
            D::Map(
                Box::new(D::Char(1)),
                Box::new(D::Struct(vec![
                    D::Other,
                    D::Array(Box::new(D::Varchar(3))),
                    D::Other
                ]))
            )
        );
        assert!(!D::parse("array<int>").is_limited());
        assert!(D::parse("array<char(1)>").is_limited());
    }
}
