// SPDX-License-Identifier: Apache-2.0
//! A table read several times in one query is one shared scan in the logical plan; with
//! `krishiv.exec.scan.instanceProjection` each instance reads only the columns its own parent
//! reads (study 38 lever A: TPC-H q21's `EXISTS` leg read four `lineitem` columns and used two).
//! The rows are the same with the switch off. The first three tests are ported from perf-2's
//! `p3/perf2-scan-instance-columns` (the optimizer version of the same lever).
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::RecordBatch;
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};

const KEY: &str = "krishiv.exec.scan.instanceProjection";

/// The column names of every scan line of a plan, sorted per scan, scans sorted.
fn scan_columns(plan: &str) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = plan
        .lines()
        .filter(|l| l.contains("Scan "))
        .map(|l| {
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

fn cols(names: &[&str]) -> Vec<String> {
    names.iter().map(|&n| n.to_owned()).collect()
}

/// Every row as text, sorted.
fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let opts = FormatOptions::default().with_null("null");
    let mut out = Vec::new();
    for b in batches {
        let fs: Vec<ArrayFormatter<'_>> = b
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).unwrap())
            .collect();
        for i in 0..b.num_rows() {
            let cells: Vec<String> = fs.iter().map(|f| f.value(i).to_string()).collect();
            out.push(cells.join(","));
        }
    }
    out.sort();
    out
}

/// The rows and the plan of `sql` with the switch at `on`.
fn run_with(s: &Session, sql: &str, on: bool) -> (Vec<String>, String) {
    s.conf()
        .set(KEY, if on { "true" } else { "false" })
        .unwrap();
    let df = block_on(s.sql(sql)).unwrap();
    let plan = block_on(df.explain(ExplainMode::Simple)).unwrap();
    (rows(&block_on(df.collect()).unwrap()), plan)
}

/// `sql` with the switch on and off: equal rows, no plan reading more columns with it on.
/// Returns the rows and the plan with the switch on.
fn both(s: &Session, sql: &str) -> (Vec<String>, String) {
    let (on, plan_on) = run_with(s, sql, true);
    let (off, plan_off) = run_with(s, sql, false);
    assert_eq!(on, off, "{plan_on}\n{plan_off}");
    let width = |p: &str| scan_columns(p).iter().map(Vec::len).sum::<usize>();
    assert!(width(&plan_on) <= width(&plan_off), "{plan_on}\n{plan_off}");
    (on, plan_on)
}

/// A Parquet table of 100 rows: `k = id % 10`, `a = id`, `b = 2 id`, `c = 3 id`, `d` the
/// id as text, `arr = [id, id + 1]`.
fn table(s: &Session) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t").display().to_string();
    let df = block_on(s.sql(
        "SELECT id % 10 AS k, id AS a, id * 2 AS b, id * 3 AS c, CAST(id AS STRING) AS d, \
         array(id, id + 1) AS arr FROM range(0, 100, 1, 2)",
    ))
    .unwrap();
    block_on(df.write().parquet(&path)).unwrap();
    (dir, path)
}

#[test]
fn each_side_of_a_self_join_reads_its_own_columns() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    let (got, plan) = both(
        &s,
        &format!(
            "SELECT x.a + y.b FROM parquet.`{path}` x JOIN parquet.`{path}` y \
             ON x.k = y.k WHERE x.c < 30 AND y.a < 20"
        ),
    );
    // x: a < 10 (c < 30); y: a < 20; k = a % 10, so each x row meets two y rows.
    let mut want: Vec<String> = (0..10_i64)
        .flat_map(|xa| [xa, xa + 10].map(|ya| (xa + ya * 2).to_string()))
        .collect();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(
        scan_columns(&plan),
        vec![cols(&["a", "b", "k"]), cols(&["a", "c", "k"])],
        "{plan}"
    );
}

#[test]
fn an_exists_leg_reads_only_its_keys() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    // q21's shape: the outer read uses `d`, the EXISTS leg only `k` and `b`.
    let (got, plan) = both(
        &s,
        &format!(
            "SELECT l1.a FROM parquet.`{path}` l1 WHERE length(l1.d) = 1 AND EXISTS \
             (SELECT * FROM parquet.`{path}` l2 WHERE l2.k = l1.k AND l2.b <> l1.b)"
        ),
    );
    let mut want: Vec<String> = (0..10).map(|i: i64| i.to_string()).collect();
    want.sort();
    assert_eq!(got, want);
    let scans = scan_columns(&plan);
    assert!(scans.contains(&cols(&["b", "k"])), "{plan}");
    assert!(scans.iter().all(|c| !c.contains(&"c".to_owned())), "{plan}");
}

/// Review finding 3 (dc91ede0f), finding 2 (5724f9c20): an `EXISTS` under an `OR` plans
/// as a mark (Existence) join, which adds its mark column above the sides; the leg still
/// reads only its keys.
#[test]
fn a_mark_join_leg_reads_only_its_keys() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    let (got, plan) = both(
        &s,
        &format!(
            "SELECT x.a FROM parquet.`{path}` x WHERE EXISTS \
             (SELECT 1 FROM parquet.`{path}` y WHERE y.k = x.k AND y.b > x.c) OR x.d = '5'"
        ),
    );
    assert!(plan.contains("Existence"), "{plan}");
    // EXISTS: a row of the same k with 2 a' > 3 a, a' = a + 10 m <= 99, so a <= 59 (60
    // rows); `d = '5'` adds a = 5, already in.
    assert_eq!(got.len(), 60, "{plan}");
    assert!(scan_columns(&plan).contains(&cols(&["b", "k"])), "{plan}");
}

#[test]
fn a_self_join_of_json_reading_the_corrupt_record_column_still_runs() {
    // The instance under the `Shift` reads its key and `_corrupt_record`; the
    // corrupt-record-only check must still see that its key is referenced above.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.json");
    std::fs::write(
        &path,
        "{\"a\": 1, \"k\": 1}\n{\"a\": 2, \"k\": 1}\n{not json\n",
    )
    .unwrap();
    let path = path.display().to_string();
    let s = Session::new().unwrap();
    let (got, _) = both(
        &s,
        &format!(
            "SELECT count(x._corrupt_record), count(*) FROM json.`{path}` x \
             JOIN json.`{path}` y ON x.a = y.k WHERE y._corrupt_record IS NULL"
        ),
    );
    // x.a = 1 meets both y rows with k = 1; x's corrupt column is null there.
    assert_eq!(got, vec!["0,2".to_owned()]);
}

/// Review finding 2: an instance whose parent reads only `_corrupt_record` would parse
/// nothing else; it keeps the shared scan's columns, and the rows are as with the switch off.
#[test]
fn an_instance_reading_only_the_corrupt_record_column_keeps_the_shared_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("j.json");
    std::fs::write(&path, "{\"x\": 1}\n{\"x\": 2}\n{not json\n").unwrap();
    let path = path.display().to_string();
    let s = Session::new().unwrap();
    let (got, plan) = both(
        &s,
        &format!("SELECT a.x, b._corrupt_record FROM json.`{path}` a CROSS JOIN json.`{path}` b"),
    );
    assert_eq!(got.len(), 9, "{got:?}");
    assert_eq!(
        got.iter().filter(|r| r.ends_with(",{not json")).count(),
        3,
        "{got:?}"
    );
    // Neither instance reads `_corrupt_record` alone: `b` keeps the shared columns.
    let scans = scan_columns(&plan);
    assert_eq!(scans.len(), 2, "{plan}");
    assert!(scans.iter().all(|c| c.contains(&"x".to_owned())), "{plan}");
    assert!(scans.contains(&cols(&["_corrupt_record", "x"])), "{plan}");
}

/// Review finding 3: the demand stops at a node that reads columns other than its
/// parent's (a window, a `ROLLUP`'s `Expand`). Each query joins one such instance with a
/// plain one; the plan builds, the rows are as with the switch off, and the instance under
/// the node still reads every column the node uses.
///
/// The window cases give the plain side exactly the window's columns (`k`, `b`), so
/// optimizer pruning adds no projection under the window that would set its own demand
/// (review of 5724f9c20, finding 1): with the Window arms forwarding the demand, both fail.
#[test]
fn the_demand_stops_at_a_window_and_an_expand() {
    let s = Session::new().unwrap();
    let (_dir, path) = table(&s);
    let window = format!(
        "SELECT x.b, w.r FROM parquet.`{path}` x JOIN \
         (SELECT k, rank() OVER (PARTITION BY k ORDER BY b) AS r FROM parquet.`{path}`) w \
         ON x.k = w.k WHERE x.b < 40"
    );
    let cases = [
        // A Filter over the window: x has a < 20 (20 rows), each meets the two w rows of
        // its k ranked 1 and 2.
        (
            format!("{window} AND w.r <= 2"),
            cols(&["b", "k"]),
            Some(40),
        ),
        // The window alone: each x row meets the ten w rows of its k.
        (window, cols(&["b", "k"]), Some(200)),
        // A regression check only, not a guard of the Generate stop: the analyzer puts
        // `Project [k, arr AS _generate_arg]` under every `Generate`, and that projection
        // sets the demand, so SQL never reaches the stop itself.
        (
            format!(
                "SELECT x.a, g.e FROM parquet.`{path}` x JOIN \
                 (SELECT k, explode(arr) AS e FROM parquet.`{path}`) g \
                 ON x.k = g.k WHERE x.a < 3 AND g.e < 25"
            ),
            cols(&["arr", "k"]),
            None,
        ),
        // The `Expand` of a `ROLLUP` reads `k`, `b` and `c`.
        (
            format!(
                "SELECT x.a, r.s FROM parquet.`{path}` x JOIN \
                 (SELECT k, b, sum(c) AS s FROM parquet.`{path}` GROUP BY ROLLUP(k, b)) r \
                 ON x.k = r.k WHERE x.a < 3 AND r.b IS NULL"
            ),
            cols(&["b", "c", "k"]),
            None,
        ),
    ];
    for (sql, used, count) in cases {
        let (got, plan) = both(&s, &sql);
        assert!(!got.is_empty(), "{sql}\n{plan}");
        if let Some(n) = count {
            assert_eq!(got.len(), n, "{sql}\n{plan}");
        }
        let scans = scan_columns(&plan);
        assert!(
            scans.iter().any(|c| used.iter().all(|u| c.contains(u))),
            "an instance lost a column of {used:?}:\n{plan}"
        );
    }
}
