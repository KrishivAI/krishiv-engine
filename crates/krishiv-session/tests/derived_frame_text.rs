// SPDX-License-Identifier: Apache-2.0
//! A DataFrame derived from a `sql` frame (`limit`, `filter`, ...) runs under the text and class of
//! the statement it came from, so `system.queries` lists the SQL of a console editor's run.
#![cfg(test)]

use arrow_array::{Array, StringArray};
use futures::executor::block_on;
use krishiv_session::Session;

fn texts(s: &Session, like: &str) -> Vec<(Option<String>, Option<String>)> {
    let batches = block_on(
        block_on(s.sql(&format!(
            "SELECT CAST(query_text AS STRING), CAST(statement_class AS STRING) FROM system.queries \
             WHERE query_text LIKE '{like}'"
        )))
        .unwrap()
        .collect(),
    )
    .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let t = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let c = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((
                (!t.is_null(i)).then(|| t.value(i).to_owned()),
                (!c.is_null(i)).then(|| c.value(i).to_owned()),
            ));
        }
    }
    out
}

#[test]
fn a_derived_frame_run_shows_the_statement_it_came_from() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT id FROM range(0, 50) /*derived-marker*/")).unwrap();
    let limited = df.limit(5);
    let rows: usize = block_on(limited.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5);
    let got = texts(&s, "SELECT id FROM range%derived-marker%");
    assert_eq!(
        got,
        [(
            Some("SELECT id FROM range(0, 50) /*derived-marker*/".to_owned()),
            Some("READ".to_owned())
        )],
        "the derived frame's run has the statement's text and class"
    );
}

#[cfg(feature = "unstable-ir")]
#[test]
fn a_sql_relation_run_shows_its_text_and_class() {
    use krishiv_plan::{LogicalPlan, Plan, PlanNode, SqlArgs};
    let s = Session::new().unwrap();
    let plan = Plan::try_new(LogicalPlan::new(PlanNode::Sql {
        query: "SELECT id FROM range(0, 5) WHERE id >= 0 /*relation-marker*/".to_owned(),
        args: SqlArgs::None,
    }))
    .unwrap();
    let df = s.from_plan(&plan);
    let rows: usize = block_on(df.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5);
    let got = texts(&s, "SELECT id FROM range%relation-marker%");
    assert_eq!(got.len(), 1, "one row: {got:?}");
    assert_eq!(got[0].1.as_deref(), Some("READ"));
    let text = got[0].0.as_deref().expect("the relation's run has a text");
    assert!(text.contains("relation-marker"), "{text}");
}

/// What a Spark Connect client sends for `spark.sql(...).filter(...)` or `.count()`: its own nodes over
/// the relation. The run is listed under the statement it reads, as an embedded derived frame's is.
#[cfg(feature = "unstable-ir")]
#[test]
fn a_plan_over_sql_relations_shows_the_first_ones_text() {
    use krishiv_plan::{JoinCondition, JoinKind, LogicalPlan, Plan, PlanNode, SqlArgs};
    let sql = |query: &str| {
        LogicalPlan::new(PlanNode::Sql {
            query: query.to_owned(),
            args: SqlArgs::None,
        })
    };
    let s = Session::new().unwrap();
    let plan = Plan::try_new(sql("SELECT id FROM range(0, 5) /*left-marker*/").join(
        sql("SELECT id AS other FROM range(0, 2) /*right-marker*/"),
        JoinKind::Cross,
        JoinCondition::None,
    ))
    .unwrap();
    let rows: usize = block_on(s.from_plan(&plan).collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 10);
    let got = texts(&s, "SELECT id FROM range%left-marker%");
    assert_eq!(got.len(), 1, "one row: {got:?}");
    assert_eq!(got[0].1.as_deref(), Some("READ"));
    assert!(texts(&s, "SELECT id AS other%right-marker%").is_empty());
}

/// A plan whose subtrees are shared (each level a union of the level below with itself: 41 nodes,
/// 2^40 paths) is searched for its `spark.sql` relation once per node, not once per path.
#[cfg(feature = "unstable-ir")]
#[test]
fn a_plan_of_shared_subtrees_is_searched_once_per_node() {
    use krishiv_plan::{LogicalPlan, Plan, PlanNode};
    use std::sync::Arc;
    let s = Session::new().unwrap();
    let mut level = Arc::new(LogicalPlan::table(["absent_table"]));
    for _ in 0..40 {
        level = Arc::new(LogicalPlan::new(PlanNode::Union {
            inputs: vec![Arc::clone(&level), Arc::clone(&level)],
            by_name: false,
            allow_missing_columns: false,
        }));
    }
    let plan = Plan::try_new(level).unwrap();
    // Building the frame is the search; the frame is never analyzed or run.
    let _frame = s.from_plan(&plan);
}

/// A plan that also reads a frame the host holds (Spark Connect's cached relation, a checkpoint) is
/// listed under its own `spark.sql` relation, not under a statement of the frame spliced in.
#[cfg(feature = "unstable-ir")]
#[test]
fn a_plan_over_a_held_frame_shows_its_own_relations_text() {
    use krishiv_plan::{JoinCondition, JoinKind, LogicalPlan, Plan, PlanNode, SqlArgs};
    use krishiv_session::QueryOptions;
    let s = Session::new().unwrap();
    let held = block_on(s.sql("SELECT id AS other FROM range(0, 2) /*held-marker*/")).unwrap();
    let plan = Plan::try_new(LogicalPlan::table(["held"]).join(
        LogicalPlan::new(PlanNode::Sql {
            query: "SELECT id FROM range(0, 5) /*over-marker*/".to_owned(),
            args: SqlArgs::None,
        }),
        JoinKind::Cross,
        JoinCondition::None,
    ))
    .unwrap();
    let frames = [(vec!["held".to_owned()], held)];
    let df = block_on(s.from_plan_over(&plan, QueryOptions::new(), &frames)).unwrap();
    let rows: usize = block_on(df.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 10);
    let got = texts(&s, "SELECT id FROM range%over-marker%");
    assert_eq!(got.len(), 1, "one row: {got:?}");
    assert_eq!(got[0].1.as_deref(), Some("READ"));
    assert!(texts(&s, "SELECT id AS other%held-marker%").is_empty());
}

/// The text of a `spark.sql` relation is masked like a statement's: the credentials of a URL in
/// it never reach `system.queries`.
#[cfg(feature = "unstable-ir")]
#[test]
fn a_sql_relations_secret_is_masked_in_its_listed_text() {
    use krishiv_plan::{LogicalPlan, Plan, PlanNode, SqlArgs};
    let s = Session::new().unwrap();
    let plan = Plan::try_new(LogicalPlan::new(PlanNode::Sql {
        query: "SELECT 's3://key-id:hunter2-secret@bucket/p' AS u /*masked-marker*/".to_owned(),
        args: SqlArgs::None,
    }))
    .unwrap();
    let rows: usize = block_on(s.from_plan(&plan).collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 1);
    let got = texts(&s, "SELECT _s3://key-id:%masked-marker%");
    assert_eq!(got.len(), 1, "one row: {got:?}");
    let text = got[0].0.as_deref().expect("the relation's run has a text");
    assert!(text.contains("bucket/p"), "{text}");
    assert!(!text.contains("hunter2-secret"), "{text}");
}

#[test]
fn a_frame_no_sql_started_still_has_no_text() {
    let s = Session::new().unwrap();
    let df = s.range(0, 10, 1, None).limit(3);
    block_on(df.collect()).unwrap();
    let rows = block_on(
        block_on(s.sql(
            "SELECT count(*) FROM system.queries WHERE query_text IS NULL \
             AND NOT (state = 'running')",
        ))
        .unwrap()
        .collect(),
    )
    .unwrap();
    let n = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, 1, "the range frame's run keeps a null text");
}

/// The single string column of `sql`'s result, in order.
fn strings(s: &Session, sql: &str) -> Vec<Option<String>> {
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let t = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((!t.is_null(i)).then(|| t.value(i).to_owned()));
        }
    }
    out
}

/// What the web console's SQL editor does: the statement with a request tag, a row cap over it, the schema of
/// the capped frame, then its rows. The one run is listed with the statement's text and its tag.
#[test]
fn a_tagged_statement_read_through_a_row_cap_is_one_run_with_its_text_and_tag() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql_with(
        "SELECT id FROM range(0, 50) /*editor-marker*/",
        krishiv_session::QueryOptions::new().tags(["ui-run-1"]),
    ))
    .unwrap();
    let capped = df.limit(10);
    assert_eq!(block_on(capped.schema()).unwrap().fields().len(), 1);
    block_on(capped.collect()).unwrap();
    let got = strings(
        &s,
        "SELECT CAST(query_text AS STRING) FROM system.queries WHERE tags['ui-run-1'] IS NOT NULL",
    );
    assert_eq!(
        got,
        [Some(
            "SELECT id FROM range(0, 50) /*editor-marker*/".to_owned()
        )]
    );
}

/// A frame derived from a derived frame still runs under the first statement's text, and every run of it is
/// listed with that text.
#[test]
fn a_chain_of_derived_frames_keeps_the_text_on_every_run() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT id FROM range(0, 50) /*chain-marker*/")).unwrap();
    let chained = df.filter_expr("id > 40").unwrap().limit(3);
    block_on(chained.collect()).unwrap();
    block_on(chained.collect()).unwrap();
    let got = strings(
        &s,
        "SELECT CAST(query_text AS STRING) FROM system.queries \
         WHERE query_text LIKE 'SELECT id FROM range%chain-marker%'",
    );
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(
        got.iter()
            .all(|t| t.as_deref() == Some("SELECT id FROM range(0, 50) /*chain-marker*/")),
        "{got:?}"
    );
}

/// The text and class of every run that carries the request tag `tag`, the runs with a text first.
fn tagged(s: &Session, tag: &str) -> Vec<Option<String>> {
    let text = strings(
        s,
        &format!(
            "SELECT CAST(query_text AS STRING) FROM system.queries WHERE tags['{tag}'] IS NOT NULL"
        ),
    );
    let class = strings(
        s,
        &format!(
            "SELECT CAST(statement_class AS STRING) FROM system.queries WHERE tags['{tag}'] IS NOT NULL"
        ),
    );
    let mut rows: Vec<Option<String>> = text
        .into_iter()
        .zip(class)
        .map(|(t, c)| {
            assert_eq!(t.is_some(), c.is_some(), "a text and a class come together");
            Some(format!("{} | {}", t?, c?))
        })
        .collect();
    rows.sort_by_key(Option::is_none);
    rows
}

/// The editor's sequence over a write: the run that wrote is the one listed with the statement's text and
/// class. The capped read of the write's empty result is a run of its own and carries neither.
#[test]
fn a_write_run_through_a_row_cap_is_listed_once_with_its_text_on_the_write() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE zt (a INT) USING parquet")).unwrap();
    let df = block_on(s.sql_with(
        "INSERT INTO zt VALUES (1), (2)",
        krishiv_session::QueryOptions::new().tags(["ui-run-9"]),
    ))
    .unwrap();
    let write = Some("INSERT INTO zt VALUES (1), (2) | WRITE".to_owned());
    assert_eq!(
        tagged(&s, "ui-run-9"),
        std::slice::from_ref(&write),
        "the write's own row"
    );
    let capped = df.limit(10);
    assert_eq!(block_on(capped.schema()).unwrap().fields().len(), 0);
    block_on(capped.collect()).unwrap();
    assert_eq!(tagged(&s, "ui-run-9"), [write.clone(), None]);
    // A plain collect of the statement's frame is the same trivial read.
    block_on(df.collect()).unwrap();
    assert_eq!(tagged(&s, "ui-run-9"), [write, None, None]);
    // A command that runs no query keeps its text on the read of its result, as before.
    let ddl = block_on(s.sql_with(
        "CREATE TABLE zu (a INT) USING parquet",
        krishiv_session::QueryOptions::new().tags(["ui-run-10"]),
    ))
    .unwrap();
    block_on(ddl.limit(10).collect()).unwrap();
    assert_eq!(
        tagged(&s, "ui-run-10"),
        [Some(
            "CREATE TABLE zu (a INT) USING parquet | DDL".to_owned()
        )]
    );
}

/// A write that fails while it runs is listed once: the run that failed, with the statement's text and class.
/// The statement gets no second, refused row beside it.
#[test]
fn a_write_that_fails_while_it_runs_is_listed_once_with_its_text() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE zf (a INT) USING parquet")).unwrap();
    let failed = block_on(s.sql_with(
        "INSERT INTO zf SELECT CAST('x' AS INT)",
        krishiv_session::QueryOptions::new().tags(["ui-run-12"]),
    ));
    assert!(failed.is_err(), "the conversion fails while the write runs");
    assert_eq!(
        tagged(&s, "ui-run-12"),
        [Some(
            "INSERT INTO zf SELECT CAST('x' AS INT) | WRITE".to_owned()
        )]
    );
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(state AS STRING) FROM system.queries WHERE tags['ui-run-12'] IS NOT NULL"
        ),
        [Some("failed".to_owned())]
    );
    // A statement refused before any query of it began still has its refused row, with the text.
    let refused = block_on(s.sql_with(
        "INSERT INTO no_such_table VALUES (1)",
        krishiv_session::QueryOptions::new().tags(["ui-run-13"]),
    ));
    assert!(refused.is_err());
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(query_text AS STRING) FROM system.queries WHERE tags['ui-run-13'] IS NOT NULL"
        ),
        [Some("INSERT INTO no_such_table VALUES (1)".to_owned())]
    );
}

/// A script from the editor: the script's own row (the one a cancel addresses) has the script's text and
/// class, and each statement the script runs is a row with that statement's own text and class. The read of
/// the script's result is a run of its own, listed under the script's text and class.
#[test]
fn a_script_is_listed_with_its_text_and_each_statement_with_its_own() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE zs (a INT) USING parquet")).unwrap();
    let script = "BEGIN INSERT INTO zs VALUES (1); SELECT a FROM zs; SELECT a + 1 FROM zs; END";
    let df = block_on(s.sql_with(
        script,
        krishiv_session::QueryOptions::new().tags(["ui-run-11"]),
    ))
    .unwrap();
    let mut rows = tagged(&s, "ui-run-11");
    rows.sort();
    let listed = vec![
        Some(format!("{script} | WRITE")),
        Some("INSERT INTO zs VALUES (1) | WRITE".to_owned()),
        Some("SELECT a + 1 FROM zs | READ".to_owned()),
        Some("SELECT a FROM zs | READ".to_owned()),
    ];
    assert_eq!(rows, listed);
    block_on(df.limit(10).collect()).unwrap();
    let mut rows = tagged(&s, "ui-run-11");
    rows.sort();
    // The read of the script's result (its last query's rows) is listed under the script's text and class.
    let mut with_read = vec![Some(format!("{script} | WRITE"))];
    with_read.extend(listed);
    with_read.sort();
    assert_eq!(rows, with_read);
}

/// The rows of the runs tagged `tag`, as `text | class | state` (`-` for an absent text or class), sorted.
fn listed(s: &Session, tag: &str) -> Vec<String> {
    let sql = format!(
        "SELECT CAST(query_text AS STRING), CAST(statement_class AS STRING), CAST(state AS STRING) FROM system.queries WHERE tags['{tag}'] IS NOT NULL"
    );
    let batches = block_on(block_on(s.sql(&sql)).unwrap().collect()).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let column = |i: usize| b.column(i).as_any().downcast_ref::<StringArray>().unwrap();
        let value = |c: usize, i: usize| {
            if column(c).is_null(i) {
                "-"
            } else {
                column(c).value(i)
            }
        };
        for i in 0..b.num_rows() {
            rows.push(format!(
                "{} | {} | {}",
                value(0, i),
                value(1, i),
                value(2, i)
            ));
        }
    }
    rows.sort();
    rows
}

/// Runs `sql` under the request tag `tag`; whether it succeeded.
fn run_tagged(s: &Session, sql: &str, tag: &str) -> bool {
    block_on(s.sql_with(sql, krishiv_session::QueryOptions::new().tags([tag])))
        .inspect_err(|e| eprintln!("{sql}: {e}"))
        .is_ok()
}

/// The queries a script runs outside a SQL statement of it (a `DECLARE` default, a `SET`, a condition, the
/// query of a `FOR`) are listed under the text of that part of the script: the declaration and the assignment
/// as session statements, the condition and the loop's query as reads. None is listed under a `SELECT` the
/// interpreter put together, or under the whole script.
#[test]
fn the_queries_of_a_scripts_variables_conditions_and_loops_are_listed_under_their_own_text() {
    let s = Session::new().unwrap();
    let script = "BEGIN DECLARE n INT DEFAULT 1 + 1; SET n = n + 1; \
                  IF n > 2 THEN SELECT n; END IF; \
                  FOR r AS SELECT id FROM range(0, 2) DO SET n = n + r.id; END FOR; END";
    assert!(run_tagged(&s, script, "ui-run-21"));
    assert_eq!(
        listed(&s, "ui-run-21"),
        [
            format!("{script} | SESSION | finished"),
            "DECLARE n INT DEFAULT 1 + 1 | SESSION | finished".to_owned(),
            "SELECT id FROM range(0, 2) | READ | finished".to_owned(),
            "SELECT n | READ | finished".to_owned(),
            "SET n = n + 1 | SESSION | finished".to_owned(),
            "SET n = n + r.id | SESSION | finished".to_owned(),
            "SET n = n + r.id | SESSION | finished".to_owned(),
            "n > 2 | READ | finished".to_owned(),
        ]
    );
}

/// A script that fails in one of its statements: the script's row and the statement's row are the failed
/// ones, the statements before it finished, and there is no refused row beside them.
#[test]
fn a_failed_script_is_listed_once_and_its_failed_statement_once() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE zg (a INT) USING parquet")).unwrap();
    let script = "BEGIN SELECT a FROM zg; INSERT INTO zg SELECT CAST('x' AS INT); END";
    assert!(!run_tagged(&s, script, "ui-run-22"));
    assert_eq!(
        listed(&s, "ui-run-22"),
        [
            format!("{script} | WRITE | failed"),
            "INSERT INTO zg SELECT CAST('x' AS INT) | WRITE | failed".to_owned(),
            "SELECT a FROM zg | READ | finished".to_owned(),
        ]
    );
}

/// `SET VAR` typed alone: the query that evaluates the value is the statement's run, listed with the
/// statement's text and class; a value that fails while it is evaluated is that one row, failed.
#[test]
fn a_variable_assignment_is_one_run_with_the_statements_text() {
    let s = Session::new().unwrap();
    block_on(s.sql("DECLARE VARIABLE pv INT DEFAULT 0")).unwrap();
    let set = "SET VAR pv = (SELECT max(id) FROM range(0, 5))";
    assert!(run_tagged(&s, set, "ui-run-23"));
    assert_eq!(
        listed(&s, "ui-run-23"),
        [format!("{set} | SESSION | finished")]
    );
    let many = "SET VAR pv = (SELECT id FROM range(0, 5))";
    assert!(!run_tagged(&s, many, "ui-run-24"));
    assert_eq!(
        listed(&s, "ui-run-24"),
        [format!("{many} | SESSION | failed")]
    );
    let declare = "DECLARE VARIABLE pw INT DEFAULT 3 + 4";
    assert!(run_tagged(&s, declare, "ui-run-25"));
    assert_eq!(
        listed(&s, "ui-run-25"),
        [format!("{declare} | SESSION | finished")]
    );
}

/// The row form of `SET VAR` runs its query once: one listed run, finished. A row query with more than
/// one row is that run, finished, and the refused assignment beside it.
#[test]
fn a_row_assignment_is_one_run() {
    let s = Session::new().unwrap();
    block_on(s.sql("DECLARE VARIABLE ra INT DEFAULT 0")).unwrap();
    block_on(s.sql("DECLARE VARIABLE rb STRING")).unwrap();
    let set = "SET VAR (ra, rb) = (SELECT 7, 'x')";
    assert!(run_tagged(&s, set, "ui-run-29"));
    assert_eq!(
        listed(&s, "ui-run-29"),
        [format!("{set} | SESSION | finished")]
    );
    let none = "SET VAR (ra, rb) = (SELECT 1, 'y' WHERE false)";
    assert!(run_tagged(&s, none, "ui-run-30"));
    assert_eq!(
        listed(&s, "ui-run-30"),
        [format!("{none} | SESSION | finished")]
    );
    let many = "SET VAR (ra, rb) = (SELECT id, 'z' FROM range(0, 3))";
    assert!(!run_tagged(&s, many, "ui-run-31"));
    assert_eq!(
        listed(&s, "ui-run-31"),
        [
            format!("{many} | SESSION | failed"),
            format!("{many} | SESSION | finished"),
        ]
    );
}

/// A row assignment whose cast to a target's type fails, as Spark 4.2 answers (the casts are part of the
/// analyzed plan, over the row query and under `LIMIT 2`; the row count is looked at after the run).
#[test]
fn a_row_assignment_whose_cast_fails_answers_as_spark_does() {
    let s = Session::new().unwrap();
    block_on(s.sql("DECLARE VARIABLE rc INT DEFAULT 5")).unwrap();
    let class = |sql: &str| {
        block_on(s.sql(sql))
            .err()
            .map(|e| e.error_class().name().to_owned())
    };
    let rc = || {
        let batches = block_on(block_on(s.sql("SELECT rc")).unwrap().collect()).unwrap();
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap()
            .clone();
        (!column.is_null(0)).then(|| column.value(0))
    };
    // A value no cast takes to INT fails at analysis, whatever the row count; the target keeps its value.
    assert_eq!(
        class("SET VAR (rc) = (SELECT array(1) WHERE false)").as_deref(),
        Some("DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION")
    );
    assert_eq!(rc(), Some(5));
    // A cast that fails on a row that is read comes before the row count.
    for sql in [
        "SET VAR (rc) = (SELECT col1 FROM VALUES ('x'), ('y'))",
        "SET VAR (rc) = (SELECT col1 FROM VALUES ('x'))",
    ] {
        assert_eq!(class(sql).as_deref(), Some("CAST_INVALID_INPUT"), "{sql}");
    }
    assert_eq!(rc(), Some(5));
    // The third row of a sorted row query is never cast: two rows are already too many.
    for sql in [
        "SET VAR (rc) = (SELECT col1 FROM VALUES ('2'), ('1'), ('x') ORDER BY col1)",
        "SET VAR (rc) = (SELECT col1 FROM VALUES (1), (2))",
    ] {
        assert_eq!(
            class(sql).as_deref(),
            Some("ROW_SUBQUERY_TOO_MANY_ROWS"),
            "{sql}"
        );
    }
    // No row sets the target to NULL, though the row's value would not cast.
    assert_eq!(
        class("SET VAR (rc) = (SELECT col1 FROM VALUES ('x') WHERE false)"),
        None
    );
    assert_eq!(rc(), None);
    assert_eq!(
        class("SET VAR (rc) = (SELECT col1 FROM VALUES ('3'))"),
        None
    );
    assert_eq!(rc(), Some(3));
}

/// A statement of a script refused before any query of it began (a missing table, an unknown variable)
/// is listed as refused under its own text and class, beside the failed script.
#[test]
fn a_script_statement_refused_before_it_ran_is_listed_with_its_text_and_class() {
    let s = Session::new().unwrap();
    let script = "BEGIN SELECT 1; SELECT a FROM zs_missing; END";
    assert!(!run_tagged(&s, script, "ui-run-32"));
    assert_eq!(
        listed(&s, "ui-run-32"),
        [
            format!("{script} | READ | failed"),
            "SELECT 1 | READ | finished".to_owned(),
            "SELECT a FROM zs_missing | READ | failed".to_owned(),
        ]
    );
    let script = "BEGIN SET zs_no_such = 1; END";
    assert!(!run_tagged(&s, script, "ui-run-33"));
    assert_eq!(
        listed(&s, "ui-run-33"),
        [
            format!("{script} | SESSION | failed"),
            "SET zs_no_such = 1 | SESSION | failed".to_owned(),
        ]
    );
}

/// `EXECUTE IMMEDIATE` as a statement of a script: the query that evaluates its string and what the named
/// statement runs are listed under the script statement's text and class. A script inside an
/// `EXECUTE IMMEDIATE` is refused (as Spark refuses it): its row is the failed one, beside the finished
/// query that evaluated the string.
#[test]
fn execute_immediate_in_a_script_is_listed_under_its_own_text() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE zh (a INT) USING parquet")).unwrap();
    let write = "EXECUTE IMMEDIATE 'INSERT INTO zh VALUES (1)'";
    let read = "EXECUTE IMMEDIATE 'SELECT a FROM zh'";
    let outer = format!("BEGIN {write}; {read}; END");
    assert!(run_tagged(&s, &outer, "ui-run-26"));
    assert_eq!(
        listed(&s, "ui-run-26"),
        [
            format!("{outer} | WRITE | finished"),
            format!("{write} | WRITE | finished"),
            format!("{write} | WRITE | finished"),
            format!("{read} | READ | finished"),
            format!("{read} | READ | finished"),
        ]
    );
    let inner = "EXECUTE IMMEDIATE 'BEGIN INSERT INTO zh VALUES (2); SELECT a FROM zh; END'";
    assert!(!run_tagged(&s, inner, "ui-run-27"));
    assert_eq!(
        listed(&s, "ui-run-27"),
        [
            format!("{inner} | WRITE | failed"),
            format!("{inner} | WRITE | finished"),
        ],
        "the query that evaluated the string finished; the refused statement is the failed row, with its class"
    );
}

/// A statement refused before any query of it began is listed with its class once it parsed: a read
/// of a missing table, a write to one, a statement of the session. One that did not parse has no class.
#[test]
fn a_refused_statement_is_listed_with_its_class_once_it_parsed() {
    let s = Session::new().unwrap();
    let cases = [
        ("SELECT a FROM zr_missing", "READ"),
        ("INSERT INTO zr_missing VALUES (1)", "WRITE"),
        ("DROP TABLE zr_missing", "DDL"),
        ("SELECT 1 +", "-"),
    ];
    for (i, (sql, class)) in cases.iter().enumerate() {
        let tag = format!("ui-run-28-{i}");
        assert!(!run_tagged(&s, sql, &tag), "{sql}");
        assert_eq!(listed(&s, &tag), [format!("{sql} | {class} | failed")]);
    }
}
