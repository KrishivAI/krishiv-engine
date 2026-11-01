// SPDX-License-Identifier: Apache-2.0
//! `to_char` and `to_varchar` over a date or timestamp are `date_format` to
//! Spark (`ToCharacterBuilder`): the same text and the same column name.
//! Every value below is what a Spark 4.2.0 run returned.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
    s
}

/// The single row's cells as text (`NULL` for a null) and the column names.
fn row(s: &Session, sql: &str) -> (Vec<String>, Vec<String>) {
    let df = block_on(s.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let b = &batches[0];
    let cells = (0..b.num_columns())
        .map(|c| {
            let col = b.column(c);
            if col.is_null(0) {
                "NULL".to_owned()
            } else if let Some(a) = col.as_string_opt::<i32>() {
                a.value(0).to_owned()
            } else if let Some(a) = col.as_string_view_opt() {
                a.value(0).to_owned()
            } else {
                format!("<{}>", col.data_type())
            }
        })
        .collect();
    let names = b
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    (cells, names)
}

#[test]
fn datetime_to_char_is_date_format() {
    let s = session();
    let (cells, names) = row(
        &s,
        "SELECT to_char(timestamp'2020-03-04 05:06:07.89', 'yyyy-MM-dd HH:mm:ss.SS'), \
         to_varchar(date'2021-12-31', 'EEEE, MMM d'), \
         to_char(timestamp_ntz'2020-03-04 05:06:07', 'yy/M/d'), \
         to_char(cast(null as timestamp), 'yyyy'), \
         to_char(timestamp'2020-03-04 05:06:07', cast(null as string))",
    );
    assert_eq!(
        cells,
        [
            "2020-03-04 05:06:07.89",
            "Friday, Dec 31",
            "20/3/4",
            "NULL",
            "NULL"
        ]
    );
    assert_eq!(names[1], "date_format(DATE '2021-12-31', EEEE, MMM d)");
}

#[test]
fn the_column_is_named_date_format() {
    let s = session();
    let (_, names) = row(
        &s,
        "SELECT to_char(timestamp'2020-03-04 05:06:07', 'yyyy') AS a, \
         to_char(timestamp'2020-03-04 05:06:07', 'yyyy')",
    );
    assert_eq!(
        names,
        ["a", "date_format(TIMESTAMP '2020-03-04 05:06:07', yyyy)"]
    );
}
