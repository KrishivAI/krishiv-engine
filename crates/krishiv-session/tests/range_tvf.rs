// SPDX-License-Identifier: Apache-2.0
//! `range` and `sql_keywords` end to end.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

fn collect_sorted(session: &Session, sql: &str) -> Vec<i64> {
    use arrow_array::cast::AsArray as _;
    use arrow_array::types::Int64Type;
    let df = block_on(session.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for b in batches {
        out.extend(
            b.column(0)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied(),
        );
    }
    out.sort_unstable();
    out
}

fn class_of(sql: &str) -> String {
    let session = session();
    match block_on(session.sql(sql)) {
        Ok(df) => match block_on(df.collect()) {
            Ok(_) => "OK".to_owned(),
            Err(e) => e.error_class().name().to_owned(),
        },
        Err(e) => e.error_class().name().to_owned(),
    }
}

#[test]
fn range_end_counts_from_zero() {
    let session = session();
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(5)"),
        vec![0, 1, 2, 3, 4]
    );
}

#[test]
fn range_start_end_step_and_slices() {
    let session = session();
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(5, 10)"),
        vec![5, 6, 7, 8, 9]
    );
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(0, 10, 2)"),
        vec![0, 2, 4, 6, 8]
    );
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(0, 10, 1, 200)"),
        (0..10).collect::<Vec<_>>()
    );
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(1, 10, 3, 2)"),
        vec![1, 4, 7]
    );
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(5, 1)"),
        Vec::<i64>::new()
    );
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(1, 10, 3)"),
        vec![1, 4, 7]
    );
}

#[test]
fn range_mixed_case_and_alias() {
    let session = session();
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM RaNgE(2)"),
        vec![0, 1]
    );
    assert_eq!(
        collect_sorted(&session, "SELECT i FROM range(0, 2) t(i)"),
        vec![0, 1]
    );
}

#[test]
fn range_folded_expression_and_double_truncation() {
    let session = session();
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(6 + cos(3))"),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(
        collect_sorted(&session, "SELECT * FROM range(1.5)"),
        vec![0]
    );
}

#[test]
fn range_errors_match_spark() {
    assert_eq!(
        class_of("SELECT * FROM dummy(3)"),
        "UNRESOLVABLE_TABLE_VALUED_FUNCTION"
    );
    assert_eq!(
        class_of("SELECT * FROM range(1, 2, 3, 4, 5)"),
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
    );
    assert_eq!(
        class_of("SELECT * FROM range(1, null)"),
        "UNEXPECTED_INPUT_TYPE"
    );
    assert_eq!(
        class_of("SELECT * FROM range(0, 5, 0)"),
        "FAILED_FUNCTION_CALL"
    );
    assert_eq!(
        class_of("SELECT * FROM range(0, (select 1))"),
        "NON_FOLDABLE_ARGUMENT"
    );
    assert_eq!(class_of("SELECT * FROM range('a')"), "CAST_INVALID_INPUT");
    assert_eq!(
        class_of("SELECT * FROM range(0, 10, 1, 5000000000)"),
        "CAST_OVERFLOW"
    );
}

#[test]
fn range_long_max_boundary_returns_one_row() {
    let session = session();
    assert_eq!(
        collect_sorted(
            &session,
            "SELECT * FROM range(9223372036854775806, 9223372036854775807, 1)"
        ),
        vec![9223372036854775806]
    );
}

#[test]
fn session_range_matches_sql_range() {
    let session = session();
    let from_sql = collect_sorted(&session, "SELECT * FROM range(1, 10, 3, 2)");
    let df = session.range(1, 10, 3, Some(2));
    let batches = block_on(df.collect()).unwrap();
    let mut got = Vec::new();
    for b in batches {
        use arrow_array::cast::AsArray as _;
        use arrow_array::types::Int64Type;
        got.extend(
            b.column(0)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied(),
        );
    }
    got.sort_unstable();
    assert_eq!(got, from_sql);
    assert_eq!(from_sql, vec![1, 4, 7]);
    // Without `numSlices` the task's parallelism decides; the rows match.
    let df = session.range(0, 5, 1, None);
    let batches = block_on(df.collect()).unwrap();
    let mut got = Vec::new();
    for b in batches {
        use arrow_array::cast::AsArray as _;
        use arrow_array::types::Int64Type;
        got.extend(
            b.column(0)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied(),
        );
    }
    got.sort_unstable();
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
}

#[test]
fn session_range_invalid_inputs_fail_on_first_action() {
    let session = session();
    let err = block_on(session.range(0, 5, 0, None).collect()).unwrap_err();
    assert_eq!(err.error_class().name(), "FAILED_FUNCTION_CALL");
    let err = block_on(session.range(0, 5, 1, Some(0)).collect()).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_INVALID_PARTITION_NUMBER");
}

#[test]
fn sql_keywords_first_rows_match_spark() {
    let session = session();
    let df = block_on(session.sql("SELECT * FROM sql_keywords()")).unwrap();
    let batches = block_on(df.collect()).unwrap();
    assert!(!batches.is_empty());
    use arrow_array::cast::AsArray as _;
    use arrow_schema::DataType as ArrowType;
    let mut keywords = Vec::new();
    for b in &batches {
        let rs = b.column(1).as_boolean();
        for i in 0..b.num_rows() {
            let k = match b.column(0).data_type() {
                ArrowType::Utf8View => b.column(0).as_string_view().value(i).to_owned(),
                _ => b.column(0).as_string::<i32>().value(i).to_owned(),
            };
            keywords.push((k, rs.value(i)));
        }
    }
    assert!(keywords.len() > 400);
    assert_eq!(keywords[0], ("ADD".to_owned(), false));
    assert_eq!(keywords[1], ("AFTER".to_owned(), false));
    // All reserved flags are false without enforced keywords.
    assert!(keywords.iter().all(|(_, r)| !r));
}

/// Two DataFrame-API ranges in one plan are two distinct leaves: the
/// analyzer gives each its own attribute id, so a join of them (which would
/// see duplicate ids otherwise) runs and matches by value.
#[test]
fn two_dataframe_ranges_join_with_distinct_attributes() {
    use arrow_array::cast::AsArray as _;
    use arrow_array::types::Int64Type;
    use krishiv_session::JoinType;
    use krishiv_session::functions::col;
    let session = session();
    let left = session.range(0, 4, 1, None).alias("l");
    let right = session.range(2, 6, 1, None).alias("r");
    let df = left.join(&right, col("l.id").eq(col("r.id")), JoinType::Inner);
    let mut got = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        got.extend(
            b.column(0)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied(),
        );
    }
    got.sort_unstable();
    assert_eq!(got, vec![2, 3]);
}

/// `LIMIT` over a multi-slice range returns the first rows in slice order,
/// as Spark's limit takes partitions in order, however the drivers race.
#[test]
fn limit_over_range_takes_the_first_rows() {
    let session = session();
    for _ in 0..50 {
        assert_eq!(
            collect_sorted(&session, "SELECT * FROM (SELECT * FROM range(10) LIMIT 5)"),
            vec![0, 1, 2, 3, 4]
        );
    }
}

/// The first column's values in the order the query returned them.
fn collect_in_order(session: &Session, sql: &str) -> Vec<i64> {
    use arrow_array::cast::AsArray as _;
    use arrow_array::types::Int64Type;
    let df = block_on(session.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect()
}

/// A multi-slice range collected without `ORDER BY` (through filters and
/// projections) returns its rows in slice order, as Spark collects
/// partitions in order, however the drivers race, also while other queries
/// load the executor.
#[test]
fn range_collects_in_slice_order_under_load() {
    // Eight compute threads whatever the machine: several drivers read the
    // slices and race, so without the ordered read the rows come back
    // reordered (on a one-slot executor a single driver would read them
    // in order by itself).
    let executor = krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig {
        compute_threads: 8,
        io_threads: 1,
        blocking_threads: 8,
        io_blocking_threads: 2,
    })
    .unwrap();
    let session = Session::builder().executor(executor).build().unwrap();
    let all: Vec<i64> = (0..64).collect();
    let doubled: Vec<i64> = (0..64).filter(|i| i % 3 != 0).map(|i| i * 2).collect();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let session = session.clone();
            let (all, doubled) = (&all, &doubled);
            scope.spawn(move || {
                for _ in 0..50 {
                    assert_eq!(
                        &collect_in_order(&session, "SELECT * FROM range(0, 64, 1, 8)"),
                        all
                    );
                    assert_eq!(
                        &collect_in_order(
                            &session,
                            "SELECT id * 2 FROM range(0, 64, 1, 8) WHERE id % 3 != 0"
                        ),
                        doubled
                    );
                    assert_eq!(
                        collect_in_order(&session, "SELECT * FROM range(3)"),
                        [0, 1, 2]
                    );
                }
            });
        }
    });
}
