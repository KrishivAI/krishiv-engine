// SPDX-License-Identifier: Apache-2.0
//! SQL scripts through `Session::sql` (design note T196, unit 2b). The values and error classes are
//! Spark 4.2.0's (`tests/spark-golden/v4.2.0/scripting`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Result<Vec<Vec<String>>, (String, String)> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .map_err(|e| (e.error_class().name().to_owned(), e.to_string()))?;
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            "NULL".to_owned()
                        } else {
                            match c.data_type() {
                                ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                                ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                                ArrowType::Int32 => c
                                    .as_primitive::<arrow_array::types::Int32Type>()
                                    .value(i)
                                    .to_string(),
                                ArrowType::Int64 => c
                                    .as_primitive::<arrow_array::types::Int64Type>()
                                    .value(i)
                                    .to_string(),
                                ArrowType::Boolean => c.as_boolean().value(i).to_string(),
                                other => panic!("unexpected {other}"),
                            }
                        }
                    })
                    .collect(),
            );
        }
    }
    Ok(out)
}

fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    run(s, sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"))
}

fn class(s: &Session, sql: &str) -> String {
    run(s, sql).unwrap_err().0
}

fn ok(v: &[&[&str]]) -> Vec<Vec<String>> {
    v.iter()
        .map(|r| r.iter().map(|c| (*c).to_owned()).collect())
        .collect()
}

#[test]
fn the_result_is_the_last_query_that_ran() {
    let s = Session::new().unwrap();
    assert_eq!(rows(&s, "BEGIN END"), ok(&[]));
    assert_eq!(
        rows(&s, "BEGIN SELECT 1 AS a; SELECT 2 AS b; END"),
        ok(&[&["2"]])
    );
    // A command after a query leaves the query's rows the result.
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE x INT DEFAULT 2; SELECT 1 AS a; SET x = 3; END"
        ),
        ok(&[&["1"]])
    );
    assert_eq!(
        rows(
            &s,
            "BEGIN NOT ATOMIC SELECT 1 AS a; CREATE TEMPORARY VIEW sv AS SELECT 5 AS c; END"
        ),
        ok(&[&["1"]])
    );
    // Session effects stay after the script.
    assert_eq!(rows(&s, "SELECT c FROM sv"), ok(&[&["5"]]));
    // No query: no result.
    assert_eq!(
        rows(&s, "BEGIN CREATE TEMPORARY VIEW w AS SELECT 1 AS a; END"),
        ok(&[])
    );
    // Several rows keep their order.
    assert_eq!(
        rows(
            &s,
            "BEGIN SELECT * FROM VALUES (1, 'a'), (2, 'b') AS t(i, n); END"
        )
        .len(),
        2
    );
}

#[test]
fn locals_are_declared_assigned_and_scoped() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE x INT DEFAULT 1; DECLARE b INT DEFAULT x * 10; SET x = x + 1; SELECT x, b; END"
        ),
        ok(&[&["2", "10"]])
    );
    // A multiple declaration shares type and default; the type may come from the default.
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE a, b INT DEFAULT 4; DECLARE c DEFAULT 5L; SELECT a + b, c; END"
        ),
        ok(&[&["8", "5"]])
    );
    // An inner declaration shadows and then vanishes; defaults read the outer one.
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE x INT DEFAULT 1; BEGIN DECLARE x INT DEFAULT x + 1; SET x = x * 100; END; SELECT x; END"
        ),
        ok(&[&["1"]])
    );
    // Labels qualify, case does not matter.
    assert_eq!(
        rows(
            &s,
            "BEGIN lbl: BEGIN DECLARE x INT DEFAULT 4; BEGIN DECLARE x INT DEFAULT 9; SET lbl.x = 5; SELECT x, LBL.x; END; END lbl; END"
        ),
        ok(&[&["9", "5"]])
    );
    // Locals are gone after the script.
    assert_eq!(
        class(&s, "SELECT x"),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
}

#[test]
fn locals_and_session_variables_meet() {
    let s = Session::new().unwrap();
    rows(&s, "DECLARE sv INT DEFAULT 7");
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE sv INT DEFAULT 1; SET session.sv = 8; SELECT sv, session.sv; END"
        ),
        ok(&[&["1", "8"]])
    );
    assert_eq!(rows(&s, "SELECT sv"), ok(&[&["8"]]));
    // `SET x = DEFAULT` re-evaluates the default.
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE x INT DEFAULT 7; SET x = 9; SET x = DEFAULT; SELECT x; END"
        ),
        ok(&[&["7"]])
    );
    // A row assignment, and several targets assigned from the old values.
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE a INT; DECLARE b STRING; SET (a, b) = (SELECT 1, 'z'); SELECT a, b; END"
        ),
        ok(&[&["1", "z"]])
    );
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE a INT DEFAULT 1; DECLARE b INT DEFAULT 2; SET a = b, b = a; SELECT a, b; END"
        ),
        ok(&[&["2", "1"]])
    );
}

#[test]
fn control_flow() {
    let s = Session::new().unwrap();
    let flow = |body: &str| {
        rows(
            &s,
            &format!(
                "BEGIN DECLARE i INT DEFAULT 0; DECLARE s INT DEFAULT 0; {body} SELECT i, s; END"
            ),
        )
    };
    assert_eq!(
        flow("IF i = 0 THEN SET s = 1; ELSEIF i = 1 THEN SET s = 2; ELSE SET s = 3; END IF;"),
        ok(&[&["0", "1"]])
    );
    assert_eq!(
        flow(
            "SET i = 5; IF i = 0 THEN SET s = 1; ELSEIF i = 1 THEN SET s = 2; ELSE SET s = 3; END IF;"
        ),
        ok(&[&["5", "3"]])
    );
    assert_eq!(
        flow("WHILE i < 4 DO SET i = i + 1; SET s = s + i; END WHILE;"),
        ok(&[&["4", "10"]])
    );
    assert_eq!(
        flow("REPEAT SET i = i + 1; UNTIL i >= 3 END REPEAT;"),
        ok(&[&["3", "0"]])
    );
    assert_eq!(
        flow("l: LOOP SET i = i + 1; IF i = 3 THEN LEAVE l; END IF; END LOOP l;"),
        ok(&[&["3", "0"]])
    );
    assert_eq!(
        flow(
            "SET i = 2; CASE i WHEN 1 THEN SET s = 10; WHEN 2 THEN SET s = 20; ELSE SET s = 30; END CASE;"
        ),
        ok(&[&["2", "20"]])
    );
    assert_eq!(
        flow("CASE WHEN i > 0 THEN SET s = 1; WHEN i = 0 THEN SET s = 2; END CASE;"),
        ok(&[&["0", "2"]])
    );
    // No match and no ELSE does nothing.
    assert_eq!(
        flow("CASE i WHEN 9 THEN SET s = 1; END CASE;"),
        ok(&[&["0", "0"]])
    );
    assert_eq!(
        flow("SET i = 1; CASE NULL WHEN NULL THEN SET s = 1; ELSE SET s = 2; END CASE;"),
        ok(&[&["1", "2"]])
    );
    // ITERATE skips the rest of the pass, also from an inner loop to an outer label; LEAVE exits a
    // labelled compound.
    assert_eq!(
        flow(
            "o: WHILE i < 3 DO SET i = i + 1; WHILE TRUE DO ITERATE o; END WHILE; SET s = 99; END WHILE o;"
        ),
        ok(&[&["3", "0"]])
    );
    assert_eq!(
        flow("c: BEGIN SET s = 1; LEAVE c; SET s = 2; END c; SET i = 7;"),
        ok(&[&["7", "1"]])
    );
}

#[test]
fn leaves_see_locals_and_commands_work() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    rows(&s, "CREATE TABLE t (a INT, b STRING) USING parquet");
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE k INT DEFAULT 3; INSERT INTO t VALUES (k, 'x'), (k + 1, 'y'); SELECT sum(a) FROM t WHERE a >= k; END"
        ),
        ok(&[&["7"]])
    );
    // A condition over a table.
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE n INT DEFAULT 0; WHILE (SELECT count(*) FROM t) < 5 DO INSERT INTO t VALUES (9, 'z'); SET n = n + 1; END WHILE; SELECT n; END"
        ),
        ok(&[&["3"]])
    );
    // EXECUTE IMMEDIATE takes locals as arguments; its text does not see them.
    assert_eq!(
        rows(
            &s,
            "BEGIN DECLARE k INT DEFAULT 6; EXECUTE IMMEDIATE 'SELECT ? + 1' USING k; END"
        ),
        ok(&[&["7"]])
    );
    assert_eq!(
        class(
            &s,
            "BEGIN DECLARE k INT DEFAULT 6; EXECUTE IMMEDIATE 'SELECT k + 1'; END"
        ),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
}

#[test]
fn errors_follow_spark() {
    let s = Session::new().unwrap();
    assert_eq!(class(&s, "BEGIN SET nope = 1; END"), "UNRESOLVED_VARIABLE");
    assert_eq!(
        class(&s, "BEGIN SET spark.sql.ansi.enabled = false; END"),
        "UNRESOLVED_VARIABLE"
    );
    assert_eq!(
        class(&s, "BEGIN DECLARE x INT; DECLARE x INT; END"),
        "VARIABLE_ALREADY_EXISTS"
    );
    assert_eq!(
        class(&s, "BEGIN DECLARE a, A INT; END"),
        "DUPLICATE_VARIABLE_NAME_INSIDE_DECLARE"
    );
    assert_eq!(
        class(&s, "BEGIN DECLARE x INT; SET x = 1, X = 2; END"),
        "DUPLICATE_ASSIGNMENTS"
    );
    assert_eq!(
        class(&s, "BEGIN DECLARE x INT DEFAULT 'abc'; END"),
        "CAST_INVALID_INPUT"
    );
    assert_eq!(
        class(&s, "BEGIN DECLARE x INT DEFAULT (SELECT 1); END"),
        "INVALID_DEFAULT_VALUE.SUBQUERY_EXPRESSION"
    );
    assert_eq!(
        class(&s, "BEGIN IF 1 THEN SELECT 1; END IF; END"),
        "INVALID_BOOLEAN_STATEMENT"
    );
    assert_eq!(
        class(&s, "BEGIN IF NULL THEN SELECT 1; END IF; END"),
        "INVALID_BOOLEAN_STATEMENT"
    );
    // NULL is false.
    assert_eq!(
        rows(
            &s,
            "BEGIN IF CAST(NULL AS BOOLEAN) THEN SELECT 1; ELSE SELECT 2; END IF; END"
        ),
        ok(&[&["2"]])
    );
    assert_eq!(
        class(&s, "BEGIN DROP TEMPORARY VARIABLE v; END"),
        "UNSUPPORTED_FEATURE.SQL_SCRIPTING_DROP_TEMPORARY_VARIABLE"
    );
    assert_eq!(
        class(&s, "BEGIN DECLARE OR REPLACE x INT; END"),
        "INVALID_VARIABLE_DECLARATION.REPLACE_LOCAL_VARIABLE"
    );
    assert_eq!(
        class(&s, "BEGIN DECLARE a.x INT; END"),
        "INVALID_VARIABLE_DECLARATION.QUALIFIED_LOCAL_VARIABLE"
    );
    // An unhandled error fails the script even after an earlier result.
    assert_eq!(
        class(&s, "BEGIN SELECT 1; SELECT 1 / 0; END"),
        "DIVIDE_BY_ZERO"
    );
    // Handlers and FOR wait for unit 3.
    assert_eq!(
        class(
            &s,
            "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 1; SELECT 1 / 0; END"
        ),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn dropping_the_future_stops_a_script() {
    let s = Session::new().unwrap();
    rows(&s, "DECLARE counter INT DEFAULT 0");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    // Run the endless script for a short while, then drop it.
    runtime.block_on(async {
        let work =
            s.sql("BEGIN WHILE TRUE DO SET session.counter = session.counter + 1; END WHILE; END");
        let timeout = tokio::time::sleep(std::time::Duration::from_millis(300));
        tokio::pin!(work);
        tokio::pin!(timeout);
        futures::future::select(work, timeout).await;
    });
    let first = rows(&s, "SELECT counter");
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert_eq!(
        rows(&s, "SELECT counter"),
        first,
        "the script kept running after it was dropped"
    );
}
