// SPDX-License-Identifier: Apache-2.0
//! `substring(c, 1, n) IN (...)` answers as Spark 4.2.0 does, for multi-byte characters, short and
//! empty values, NULL rows and a NULL element. The expected vectors are from a real Spark 4.2.0
//! run over the same values (PySpark, `local[1]`).
#![cfg(test)]
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

const VALUES: [&str; 11] = [
    "13-xyz",
    "31",
    "1",
    "",
    "ab",
    "\u{e9}1x",
    "\u{e9}\u{e9}",
    "a",
    "13",
    "\u{1F600}x",
    "abc-long-value-here",
];

fn run(list: &str, n: u32) -> R<Vec<Option<bool>>> {
    let s = Session::new()?;
    let rows: Vec<String> = VALUES
        .iter()
        .enumerate()
        .map(|(i, v)| format!("({i}, '{v}')"))
        .chain(std::iter::once(format!("({}, NULL)", VALUES.len())))
        .collect();
    let q = format!(
        "select substring(c, 1, {n}) in ({list}) from values {} as t(i, c) order by i",
        rows.join(", ")
    );
    let batches = block_on(async { s.sql(&q).await?.collect().await })?;
    let mut out = Vec::new();
    for b in &batches {
        let a = b.column(0).as_boolean();
        out.extend((0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i))));
    }
    Ok(out)
}

fn t(v: bool) -> Option<bool> {
    Some(v)
}

#[test]
fn a_leading_substring_in_list_matches_spark() -> R {
    let list = "'13','\u{e9}1','1','zzz','abc'";
    assert_eq!(
        run(list, 2)?,
        [
            t(true),
            t(false),
            t(true),
            t(false),
            t(false),
            t(true),
            t(false),
            t(false),
            t(true),
            t(false),
            t(false),
            None
        ]
    );
    assert_eq!(
        run(list, 3)?,
        [
            t(false),
            t(false),
            t(true),
            t(false),
            t(false),
            t(false),
            t(false),
            t(false),
            t(true),
            t(false),
            t(true),
            None
        ]
    );
    let list = "'13','\u{1F600}x','ab'";
    assert_eq!(
        run(list, 2)?,
        [
            t(true),
            t(false),
            t(false),
            t(false),
            t(true),
            t(false),
            t(false),
            t(false),
            t(true),
            t(true),
            t(true),
            None
        ]
    );
    let list = "'a','', 'ab'";
    assert_eq!(
        run(list, 3)?,
        [
            t(false),
            t(false),
            t(false),
            t(true),
            t(true),
            t(false),
            t(false),
            t(true),
            t(false),
            t(false),
            t(false),
            None
        ]
    );
    Ok(())
}

#[test]
fn a_null_element_turns_the_misses_into_null() -> R {
    let got = run("'13', NULL", 2)?;
    let want = [
        t(true),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        t(true),
        None,
        None,
        None,
    ];
    assert_eq!(got, want);
    assert_eq!(
        run("'13', NULL", 3)?,
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            t(true),
            None,
            None,
            None
        ]
    );
    Ok(())
}
