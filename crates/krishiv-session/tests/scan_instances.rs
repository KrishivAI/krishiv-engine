// SPDX-License-Identifier: Apache-2.0
//! A table read several times in one query is one shared scan in the logical
//! plan; each instance of it reads only the columns its own parent reads
//! (study 38 lever A: TPC-H q21's `EXISTS` leg read four `lineitem` columns
//! and used two).
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::{Array, Int64Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};

fn write_parquet(s: &Session, path: &str, sql: &str) {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.write().parquet(path)).unwrap();
}

/// The column list of every scan line of a plan, sorted per scan.
fn scan_columns(plan: &str) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = plan
        .lines()
        .filter(|l| l.contains("Scan "))
        .map(|l| {
            let l = l.trim_end();
            let open = l.find('[').unwrap();
            let close = open + l[open..].find(']').unwrap();
            let mut cols: Vec<String> = l[open + 1..close]
                .split(", ")
                .filter(|c| !c.is_empty())
                .map(|c| c.split('#').next().unwrap().to_owned())
                .collect();
            cols.sort();
            cols
        })
        .collect();
    out.sort();
    out
}

fn longs(batches: &[RecordBatch]) -> Vec<i64> {
    let mut v: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            (0..c.len()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect();
    v.sort_unstable();
    v
}

fn run(s: &Session, sql: &str) -> (Vec<i64>, String) {
    let df = block_on(s.sql(sql)).unwrap();
    let plan = block_on(df.explain(ExplainMode::Simple)).unwrap();
    (longs(&block_on(df.collect()).unwrap()), plan)
}

fn table(s: &Session) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t").display().to_string();
    write_parquet(
        s,
        &path,
        "SELECT id % 10 AS k, id AS a, id * 2 AS b, id * 3 AS c, CAST(id AS STRING) AS d \
         FROM range(0, 100, 1, 2)",
    );
    (dir, path)
}

#[test]
fn each_side_of_a_self_join_reads_its_own_columns() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    let (rows, plan) = run(
        &s,
        &format!(
            "SELECT x.a + y.b FROM parquet.`{path}` x JOIN parquet.`{path}` y \
             ON x.k = y.k WHERE x.c < 30 AND y.a < 20"
        ),
    );
    // x: a < 10 (c < 30); y: a < 20; k = a % 10, so each x row meets two y rows.
    let mut want: Vec<i64> = (0..10_i64)
        .flat_map(|xa| [xa, xa + 10].map(|ya| xa + ya * 2))
        .collect();
    want.sort_unstable();
    assert_eq!(rows, want);
    assert_eq!(
        scan_columns(&plan),
        vec![
            vec!["a".to_owned(), "b".to_owned(), "k".to_owned()],
            vec!["a".to_owned(), "c".to_owned(), "k".to_owned()],
        ],
        "{plan}"
    );
}

#[test]
fn an_exists_leg_reads_only_its_keys() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    // q21's shape: the outer read uses `d`, the EXISTS leg only `k` and `b`.
    let (rows, plan) = run(
        &s,
        &format!(
            "SELECT l1.a FROM parquet.`{path}` l1 WHERE length(l1.d) = 1 AND EXISTS \
             (SELECT * FROM parquet.`{path}` l2 WHERE l2.k = l1.k AND l2.b <> l1.b)"
        ),
    );
    assert_eq!(rows, (0..10).collect::<Vec<i64>>());
    let cols = scan_columns(&plan);
    assert!(
        cols.iter().any(|c| c == &["b".to_owned(), "k".to_owned()]),
        "{plan}"
    );
    assert!(cols.iter().all(|c| !c.contains(&"c".to_owned())), "{plan}");
}

#[test]
fn a_self_join_of_json_reading_the_corrupt_record_column_still_runs() {
    // Each instance reads its own columns, so the instance under a `Shift`
    // reads its key and `_corrupt_record`; the corrupt-record-only check must
    // still see that its key is referenced above (review M2).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.json");
    std::fs::write(
        &path,
        "{\"a\": 1, \"k\": 1}\n{\"a\": 2, \"k\": 1}\n{not json\n",
    )
    .unwrap();
    let path = path.display().to_string();
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!(
        "SELECT count(x._corrupt_record), count(*) FROM json.`{path}` x \
         JOIN json.`{path}` y ON x.a = y.k WHERE y._corrupt_record IS NULL"
    )))
    .unwrap();
    let batches = block_on(df.collect()).unwrap();
    let b = &batches[0];
    let col = |i: usize| {
        b.column(i)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    };
    // x.a = 1 meets both y rows with k = 1; x's corrupt column is null there.
    assert_eq!((col(0), col(1)), (0, 2));
}

#[test]
fn the_switch_off_reads_the_union_and_returns_the_same_rows() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    let q = format!(
        "SELECT x.a + y.b FROM parquet.`{path}` x JOIN parquet.`{path}` y \
         ON x.k = y.k WHERE x.c < 30 AND y.a < 20"
    );
    let (on_rows, on_plan) = run(&s, &q);
    s.conf()
        .set("krishiv.sql.optimizer.narrowScanInstances", "false")
        .unwrap();
    let (off_rows, off_plan) = run(&s, &q);
    assert_eq!(on_rows, off_rows);
    assert_ne!(
        scan_columns(&on_plan),
        scan_columns(&off_plan),
        "{off_plan}"
    );
}

#[test]
fn a_union_and_a_cte_used_twice_return_the_same_rows_either_way() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    let qs = [
        format!(
            "SELECT a FROM parquet.`{path}` WHERE c < 30 UNION ALL \
             SELECT b FROM parquet.`{path}` WHERE a < 5"
        ),
        format!(
            "WITH t AS (SELECT * FROM parquet.`{path}`) \
             SELECT x.a + y.c FROM t x JOIN t y ON x.k = y.k WHERE x.b < 40"
        ),
        format!(
            "SELECT l.a FROM parquet.`{path}` l WHERE l.b > (SELECT avg(m.b) \
             FROM parquet.`{path}` m WHERE m.k = l.k)"
        ),
    ];
    for q in &qs {
        s.conf()
            .set("krishiv.sql.optimizer.narrowScanInstances", "true")
            .unwrap();
        let (on, _) = run(&s, q);
        s.conf()
            .set("krishiv.sql.optimizer.narrowScanInstances", "false")
            .unwrap();
        let (off, _) = run(&s, q);
        assert!(!on.is_empty());
        assert_eq!(on, off, "{q}");
    }
}

#[test]
fn a_self_join_of_csv_returns_the_same_rows_either_way() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.csv");
    std::fs::write(&path, "1,1,10,100\n1,2,20,200\n2,3,30,300\n").unwrap();
    let path = path.display().to_string();
    let s = Session::new().unwrap();
    let q = format!(
        "SELECT CAST(x._c1 AS BIGINT) + CAST(y._c2 AS BIGINT) FROM csv.`{path}` x JOIN csv.`{path}` y \
         ON x._c0 = y._c0 WHERE CAST(y._c3 AS BIGINT) > 100"
    );
    s.conf()
        .set("krishiv.sql.optimizer.narrowScanInstances", "true")
        .unwrap();
    let (on, _) = run(&s, &q);
    s.conf()
        .set("krishiv.sql.optimizer.narrowScanInstances", "false")
        .unwrap();
    let (off, _) = run(&s, &q);
    assert_eq!(on, vec![21, 22, 33]);
    assert_eq!(on, off);
}
