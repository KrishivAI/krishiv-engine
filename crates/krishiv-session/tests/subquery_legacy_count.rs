// SPDX-License-Identifier: Apache-2.0
//! Spark's legacy count-bug flags for correlated subqueries:
//! `spark.sql.optimizer.decorrelateExistsSubqueryLegacyIncorrectCountHandling.enabled`
//! (`EXISTS` / `IN` over a correlated global aggregate) and
//! `spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled`
//! (a grouped correlated scalar subquery). Views and expected rows are those
//! of Spark 4.2.0's `exists-count-bug.sql`, `in-count-bug.sql` and
//! `scalar-subquery-count-bug.sql` goldens; each statement also runs with the
//! flag off, where the answer is the correct one.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        other => format!("<{other}>"),
    }
}

fn run(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    b.columns()
                        .iter()
                        .map(|c| cell(c.as_ref(), r))
                        .collect::<Vec<_>>()
                        .join("|")
                })
            })
            .collect();
        out.sort();
        Ok(out)
    })
}

fn rows(v: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = v.iter().map(|r| (*r).to_owned()).collect();
    out.sort();
    out
}

const EXISTS_FLAG: &str =
    "spark.sql.optimizer.decorrelateExistsSubqueryLegacyIncorrectCountHandling.enabled";
const SCALAR_FLAG: &str =
    "spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled";

fn session() -> R<Session> {
    let s = Session::new()?;
    for v in [
        "create temporary view t1(c1, c2) as values (0, 1), (1, 2)",
        "create temporary view t2(c1, c2) as values (0, 2), (0, 3)",
        "create temporary view l(a, b) as values \
         (1, 2.0), (1, 2.0), (2, 1.0), (2, 1.0), (3, 3.0), (null, null), (null, 5.0), (6, null)",
        "create temporary view r(c, d) as values \
         (2, 3.0), (2, 3.0), (3, 2.0), (4, 1.0), (null, null), (null, 5.0), (6, null)",
        "create temporary view null_view(a, b) as select cast(null as int), cast(null as int)",
    ] {
        run(&s, v)?;
    }
    Ok(s)
}

/// `sql`'s rows with `flag` off (`correct`) and on (`legacy`).
fn both(s: &Session, flag: &str, sql: &str, correct: &[&str], legacy: &[&str]) -> R {
    run(s, &format!("set {flag} = false"))?;
    assert_eq!(run(s, sql)?, rows(correct), "{sql}");
    run(s, &format!("set {flag} = true"))?;
    assert_eq!(run(s, sql)?, rows(legacy), "{sql} (legacy)");
    run(s, &format!("set {flag} = false"))?;
    Ok(())
}

#[test]
fn exists_over_a_correlated_count_keeps_only_matching_rows_under_the_legacy_flag() -> R {
    let s = session()?;
    let q = "select * from t1 where exists (select count(*) from t2 where t2.c1 = t1.c1)";
    both(&s, EXISTS_FLAG, q, &["0|1", "1|2"], &["0|1"])?;
    let q = "select * from t1 where not exists (select count(*) from t2 where t2.c1 = t1.c1)";
    both(&s, EXISTS_FLAG, q, &[], &["1|2"])?;
    let q = "select *, exists (select count(*) from t2 where t2.c1 = t1.c1) from t1";
    both(
        &s,
        EXISTS_FLAG,
        q,
        &["0|1|true", "1|2|true"],
        &["0|1|true", "1|2|false"],
    )?;
    let q = "select *, not exists (select count(*) from t2 where t2.c1 = t1.c1) from t1";
    both(
        &s,
        EXISTS_FLAG,
        q,
        &["0|1|false", "1|2|false"],
        &["0|1|false", "1|2|true"],
    )?;
    // A correlated conjunct with no inner column leaves the aggregate global.
    let q = "select * from t1 where exists (select count(*) from t2 where t1.c1 = 100)";
    both(&s, EXISTS_FLAG, q, &["0|1", "1|2"], &[])?;
    Ok(())
}

#[test]
fn in_over_a_correlated_count_matches_no_row_where_nothing_matched_under_the_legacy_flag() -> R {
    let s = session()?;
    let q = "select * from t1 where c1 in (select count(*) + 1 from t2 where t2.c1 = t1.c1)";
    both(&s, EXISTS_FLAG, q, &["1|2"], &[])?;
    let q = "select *, c1 not in (select count(*) + 1 from t2 where t2.c1 = t1.c1) from t1";
    both(
        &s,
        EXISTS_FLAG,
        q,
        &["0|1|true", "1|2|false"],
        &["0|1|true", "1|2|true"],
    )?;
    // A `HAVING` over the correlated count: without the flag the shape is
    // not decorrelated yet (`KRISHIV_UNSUPPORTED_FEATURE`); with it, the
    // grouped count is never 0, so `IN` keeps no row and `NOT IN` all.
    run(&s, &format!("set {EXISTS_FLAG} = true"))?;
    let q = "select * from t1 where c1 in \
             (select 1 from t2 where t2.c1 = t1.c1 having count(*) = 0)";
    assert_eq!(run(&s, q)?, rows(&[]), "{q}");
    let q = "select * from t1 where c1 not in \
             (select 1 from t2 where t2.c1 = t1.c1 having count(*) = 0)";
    assert_eq!(run(&s, q)?, rows(&["0|1", "1|2"]), "{q}");
    Ok(())
}

#[test]
fn a_grouped_correlated_count_reads_zero_where_nothing_matched_under_the_legacy_flag() -> R {
    let s = session()?;
    let correct = [
        "1|NULL",
        "1|NULL",
        "2|2",
        "2|2",
        "3|1",
        "6|1",
        "NULL|NULL",
        "NULL|NULL",
    ];
    let legacy = ["1|0", "1|0", "2|2", "2|2", "3|1", "6|1", "NULL|0", "NULL|0"];
    let q = "select a, (select count(*) from r where l.a = r.c group by c) from l";
    both(&s, SCALAR_FLAG, q, &correct, &legacy)?;
    let q = "select a, (select count(*) from r where l.a = r.c group by 'constant') from l";
    both(&s, SCALAR_FLAG, q, &correct, &legacy)?;
    // A global count reads 0 either way.
    let q = "select a, (select count(*) from r where l.a = r.c) from l";
    both(&s, SCALAR_FLAG, q, &legacy, &legacy)?;
    Ok(())
}

/// A correlated global aggregate whose value is an expression of a call, or
/// whose `HAVING` reads another call than the value: each call takes its
/// value over no rows (a count's 0) where nothing matched, as Spark 4.2.0
/// answers (`scalar-subquery-count-bug.sql` #5, #11, #13, #15 and
/// `in-count-bug.sql` #8, #9, with the legacy flags off).
#[test]
fn a_having_or_an_expression_over_correlated_calls_sees_zero_counts() -> R {
    let s = session()?;
    let q = "select a, (select (count(*)) is null from r where l.a = r.c) from l";
    let want = [
        "1|false",
        "1|false",
        "2|false",
        "2|false",
        "3|false",
        "6|false",
        "NULL|false",
        "NULL|false",
    ];
    assert_eq!(run(&s, q)?, rows(&want), "{q}");
    let zeros = ["0", "0", "0", "0", "0", "0", "0", "0"];
    for q in [
        "select (select count(null_view.a) as result from null_view \
         where null_view.a = l.a having count(*) > -1) from l",
        "select (select count(null_view.a) as result from null_view \
         inner join r on r.c = null_view.a and r.c is not null \
         where null_view.a = l.a having count(*) > -1) from l",
        "select (select count(f.a) as result from \
         (select a, b from null_view intersect select c, d from r where c is not null) as f \
         where f.a = l.a having count(*) > -1) from l",
    ] {
        assert_eq!(run(&s, q)?, rows(&zeros), "{q}");
    }
    let q = "select * from t1 where c1 in \
             (select 1 from t2 where t2.c1 = t1.c1 having count(*) = 0)";
    assert_eq!(run(&s, q)?, rows(&["1|2"]), "{q}");
    let q = "select * from t1 where c1 not in \
             (select 1 from t2 where t2.c1 = t1.c1 having count(*) = 0)";
    assert_eq!(run(&s, q)?, rows(&["0|1"]), "{q}");
    Ok(())
}

/// Under the legacy flags Spark answers or raises in shapes the legacy
/// rewrite here does not reproduce (a projection between the correlated
/// filter and the aggregate, a `LIMIT` over the correlation, a correlation
/// that is not an equality under an aggregate, a loosely grouped scalar
/// subquery): they are refused, never answered the non-legacy way.
#[test]
fn legacy_shapes_not_reproduced_are_refused() -> R {
    let s = session()?;
    run(&s, &format!("set {EXISTS_FLAG} = true"))?;
    for q in [
        "select * from t1 where exists \
         (select count(*) from (select * from t2 where t2.c1 = t1.c1) s)",
        "select * from t1 where exists (select * from t2 where t2.c1 = t1.c1 limit 1)",
        "select * from t1 where c1 in (select count(*) from t2 where t2.c1 > t1.c1)",
    ] {
        assert!(run(&s, q).is_err(), "{q}");
    }
    run(&s, &format!("set {EXISTS_FLAG} = false"))?;
    run(&s, &format!("set {SCALAR_FLAG} = true"))?;
    let q = "select a, (select count(*) from r where r.c > l.a group by c) from l";
    assert!(run(&s, q).is_err(), "{q}");
    // Grouped by a key the filter fixes (`d = 1`): one group per outer row
    // at most, but the single join gives NULL where the legacy rule's count
    // over no rows is 0.
    let q = "select a, (select count(*) from r where r.c = l.a and r.d = 1 group by r.d) from l";
    assert!(run(&s, q).is_err(), "{q}");
    run(&s, &format!("set {SCALAR_FLAG} = false"))?;
    // Several calls split into one subquery each over the input: with a
    // nondeterministic filter they would read different rows.
    let q = "select a, (select count(*) = count(d) from r where r.c = l.a and rand() < 0.5) from l";
    assert!(run(&s, q).is_err(), "{q}");
    // The value is read again by the HAVING: a nondeterministic value
    // would be drawn twice.
    let q =
        "select a, (select count(*) + rand() as v from r where r.c = l.a having v < 0.5) from l";
    assert!(run(&s, q).is_err(), "{q}");
    Ok(())
}
