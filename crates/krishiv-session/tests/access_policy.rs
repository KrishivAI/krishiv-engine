// SPDX-License-Identifier: Apache-2.0
//! Access-policy enforcement on every statement kind (merge-blocking):
//!
//! * the deny-all matrix: every read (SQL, DataFrame, temporary views,
//!   nested views, reads by path, CTEs) and every write and DDL statement
//!   (`df.write`, `CREATE TABLE ... AS SELECT`, `DROP TABLE`,
//!   `CREATE DATABASE`, DML, ...) fails with
//!   `KRISHIV_INSUFFICIENT_PERMISSIONS` and leaves no file or catalog
//!   change;
//! * the inverse matrix: everything but `Modify` allowed permits every read
//!   and denies every write;
//! * a filtered reader cannot write; denials do not reveal existence;
//! * `PathPrefixPolicy` keeps reads and writes under its roots.
#![cfg(test)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion, RowColumnPolicy};
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, DenyAll, PathPrefixPolicy, Principal, PrincipalKind,
    Privilege, Securable, Session,
};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

/// Decides by privilege; `deny_all` switches to denying everything (the
/// version changes with it, as a policy's must).
struct Policy {
    deny_all: AtomicBool,
    version: AtomicU64,
    decide: fn(&Securable, Privilege) -> AccessDecision,
}

impl Policy {
    fn new(decide: fn(&Securable, Privilege) -> AccessDecision) -> Arc<Self> {
        Arc::new(Self {
            deny_all: AtomicBool::new(false),
            version: AtomicU64::new(1),
            decide,
        })
    }

    fn deny_everything(&self) {
        self.deny_all.store(true, Ordering::SeqCst);
        self.version.fetch_add(1, Ordering::SeqCst);
    }
}

impl AccessPolicy for Policy {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(self.version.load(Ordering::SeqCst))
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = if self.deny_all.load(Ordering::SeqCst) {
            deny()
        } else {
            (self.decide)(securable, privilege)
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

fn deny() -> AccessDecision {
    AccessDecision::Deny {
        reason: "the reason stays private".to_owned(),
    }
}

fn allow(_: &Securable, _: Privilege) -> AccessDecision {
    AccessDecision::Allow
}

fn all_but_modify(_: &Securable, p: Privilege) -> AccessDecision {
    if p == Privilege::Modify {
        deny()
    } else {
        AccessDecision::Allow
    }
}

fn filtered_reads(_: &Securable, p: Privilege) -> AccessDecision {
    if p == Privilege::Select {
        AccessDecision::AllowWith(RowColumnPolicy::default())
    } else {
        AccessDecision::Allow
    }
}

/// A session under `policy`, with the table `t` (column `a`, three rows).
fn session(policy: Arc<dyn AccessPolicy>) -> Session {
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(policy)
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

/// Every file and directory under `dir`, with file sizes.
fn tree(dir: &Path) -> BTreeSet<(PathBuf, u64)> {
    let mut out = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let e = e.unwrap();
            let meta = e.metadata().unwrap();
            if meta.is_dir() {
                stack.push(e.path());
            }
            out.insert((e.path(), if meta.is_dir() { 0 } else { meta.len() }));
        }
    }
    out
}

fn rows(s: &Session, sql: &str) -> Result<usize, KrishivError> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        Ok(batches.iter().map(RecordBatch::num_rows).sum())
    })
}

fn class(r: Result<impl Sized, KrishivError>) -> String {
    match r {
        Ok(_) => "OK".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    }
}

/// The fixture: `dir/data` holds Parquet files of `t`; temporary views
/// `v1` over `t`, `v2` over `v1`, `f` over the files.
struct Fixture {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    data: String,
}

fn fixture(s: &Session) -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let data = dir.join("data").to_str().unwrap().to_owned();
    block_on(s.table("t").write().parquet(&data)).unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW v1 AS SELECT a FROM t")).unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW v2 AS SELECT * FROM v1")).unwrap();
    Fixture {
        _tmp: tmp,
        dir,
        data,
    }
}

/// The read cases: name and outcome.
fn reads(s: &Session, f: &Fixture) -> Vec<(&'static str, String)> {
    vec![
        ("SQL query", class(rows(s, "SELECT * FROM t"))),
        (
            "DataFrame plan",
            class(block_on(s.table("t").select(["a"]).collect())),
        ),
        (
            "temporary view over a table",
            class(rows(s, "SELECT * FROM v1")),
        ),
        ("nested temporary views", class(rows(s, "SELECT * FROM v2"))),
        (
            "parquet.`path`",
            class(rows(s, &format!("SELECT * FROM parquet.`{}`", f.data))),
        ),
        (
            "spark.read.parquet",
            class(block_on(s.read().parquet([&f.data]).collect())),
        ),
        (
            "CTE",
            class(rows(s, "WITH c AS (SELECT * FROM v2) SELECT * FROM c")),
        ),
        (
            "temporary view creation",
            class(block_on(
                s.sql("CREATE TEMPORARY VIEW v3 AS SELECT * FROM t"),
            )),
        ),
    ]
}

/// The write and DDL cases: name and outcome.
fn writes(s: &Session, f: &Fixture) -> Vec<(&'static str, String)> {
    let out = f.dir.join("out").to_str().unwrap().to_owned();
    let sql = |q: &str| class(block_on(s.sql(q)));
    let df = s.sql("SELECT 1 AS a");
    let df = block_on(df).unwrap();
    vec![
        (
            "df.write overwrite (new path)",
            class(block_on(df.write().mode("overwrite").parquet(&out))),
        ),
        (
            "df.write overwrite (existing path)",
            class(block_on(df.write().mode("overwrite").parquet(&f.data))),
        ),
        (
            "df.write append",
            class(block_on(df.write().mode("append").parquet(&f.data))),
        ),
        (
            "CREATE TABLE AS SELECT",
            sql("CREATE TABLE c AS SELECT * FROM t"),
        ),
        ("INSERT", sql("INSERT INTO t VALUES (4)")),
        (
            "INSERT INTO IDENTIFIER",
            sql("INSERT INTO IDENTIFIER('t') VALUES (4)"),
        ),
        (
            "INSERT OVERWRITE",
            sql("INSERT OVERWRITE t SELECT * FROM t"),
        ),
        ("DELETE", sql("DELETE FROM t WHERE a = 1")),
        ("UPDATE", sql("UPDATE t SET a = 0")),
        (
            "MERGE",
            sql("MERGE INTO t USING t AS s ON t.a = s.a WHEN MATCHED THEN DELETE"),
        ),
        ("TRUNCATE", sql("TRUNCATE TABLE t")),
        ("ANALYZE TABLE", sql("ANALYZE TABLE t COMPUTE STATISTICS")),
    ]
}

/// DDL and admin cases (not writes: the inverse matrix allows `Create`
/// and `Drop`).
fn ddl(s: &Session) -> Vec<(&'static str, String)> {
    let sql = |q: &str| class(block_on(s.sql(q)));
    vec![
        ("DROP TABLE", sql("DROP TABLE t")),
        ("DROP TABLE IDENTIFIER", sql("DROP TABLE IDENTIFIER('t')")),
        ("CREATE DATABASE", sql("CREATE DATABASE d")),
        ("CREATE TABLE", sql("CREATE TABLE c (a INT)")),
        ("ALTER TABLE", sql("ALTER TABLE t ADD COLUMNS (b INT)")),
        ("CREATE VIEW", sql("CREATE VIEW default.pv AS SELECT 1")),
        (
            "CREATE GLOBAL TEMPORARY VIEW",
            sql("CREATE GLOBAL TEMPORARY VIEW g AS SELECT 1"),
        ),
        ("DROP DATABASE", sql("DROP DATABASE default")),
        ("SHOW TABLES", sql("SHOW TABLES")),
        ("DESCRIBE TABLE", sql("DESCRIBE TABLE t")),
        ("DESCRIBE QUERY", sql("DESCRIBE QUERY SELECT * FROM t")),
        ("DESCRIBE NAMESPACE", sql("DESCRIBE NAMESPACE default")),
        ("SHOW VIEWS", sql("SHOW VIEWS")),
        ("SHOW COLUMNS", sql("SHOW COLUMNS IN t")),
        ("SHOW NAMESPACES", sql("SHOW NAMESPACES")),
        ("SHOW TBLPROPERTIES", sql("SHOW TBLPROPERTIES t")),
        ("SHOW CREATE TABLE", sql("SHOW CREATE TABLE t")),
        ("CLEAR CACHE", sql("CLEAR CACHE")),
        ("ADD JAR", sql("ADD JAR '/tmp/x.jar'")),
    ]
}

fn assert_all(cases: &[(&str, String)], expected: &str) {
    let wrong: Vec<String> = cases
        .iter()
        .filter(|(_, got)| got != expected)
        .map(|(name, got)| format!("{name}: {got}"))
        .collect();
    assert!(wrong.is_empty(), "expected {expected}: {wrong:?}");
}

#[test]
fn deny_all_matrix() {
    let policy = Policy::new(allow);
    let s = session(Arc::clone(&policy) as Arc<dyn AccessPolicy>);
    let f = fixture(&s);
    let files = tree(&f.dir);
    let tables = s.temp_table_names();
    policy.deny_everything();

    assert_all(&reads(&s, &f), DENIED);
    assert_all(&writes(&s, &f), DENIED);
    assert_all(&ddl(&s), DENIED);
    // `IDENTIFIER('t')` is evaluated first, so the read is checked as `t`.
    assert_eq!(
        class(rows(&s, "SELECT * FROM IDENTIFIER('t')")),
        "KRISHIV_INSUFFICIENT_PERMISSIONS"
    );
    // Session-state statements need no privilege.
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();

    assert_eq!(tree(&f.dir), files, "no file was written or removed");
    assert_eq!(s.temp_table_names(), tables, "no catalog change");
}

#[test]
fn deny_all_policy_type_denies_from_the_start() {
    let s = session(Arc::new(DenyAll));
    assert_eq!(class(rows(&s, "SELECT * FROM t")), DENIED);
    assert_eq!(class(block_on(s.sql("DROP TABLE t"))), DENIED);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("o");
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    assert_eq!(
        class(block_on(df.write().parquet(out.to_str().unwrap()))),
        DENIED
    );
    assert!(!out.exists());
    // The policy's reason never reaches the user.
    let e = block_on(s.sql("DROP TABLE t")).unwrap_err();
    assert!(!e.to_string().contains("deny-all"), "{e}");
}

#[test]
fn inverse_matrix_reads_pass_writes_fail() {
    let s = session(Policy::new(all_but_modify));
    let f = {
        // The fixture writes files: build it under a permissive twin, then
        // point this session at the same files.
        let setup = session(Arc::new(AllowAll));
        let f = fixture(&setup);
        block_on(s.sql("CREATE TEMPORARY VIEW v1 AS SELECT a FROM t")).unwrap();
        block_on(s.sql("CREATE TEMPORARY VIEW v2 AS SELECT * FROM v1")).unwrap();
        f
    };
    let files = tree(&f.dir);
    let read = reads(&s, &f);
    assert_all(&read, "OK");
    assert_all(&writes(&s, &f), DENIED);
    assert_eq!(tree(&f.dir), files, "no file was written or removed");
    // `Drop` and `Create` are allowed: DDL passes the checks and runs (here
    // the name is a temporary view, which `DROP TABLE` drops).
    let ddl = ddl(&s);
    let drop = ddl.iter().find(|(n, _)| *n == "DROP TABLE").unwrap();
    assert_eq!(drop.1, "OK");
}

#[test]
fn a_filtered_reader_cannot_write() {
    let s = session(Policy::new(filtered_reads));
    // Reads are filtered: writes to the table and to paths are denied,
    // although the policy allows `Modify`.
    assert_eq!(class(block_on(s.sql("INSERT INTO t VALUES (4)"))), DENIED);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("o");
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    assert_eq!(
        class(block_on(df.write().parquet(out.to_str().unwrap()))),
        DENIED
    );
    assert!(!out.exists());
}

#[test]
fn denials_do_not_reveal_existence() {
    let with_t = session(Arc::new(DenyAll));
    let without_t = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(DenyAll))
        .build()
        .unwrap();
    for sql in [
        "DROP TABLE t",
        "INSERT INTO t VALUES (1)",
        "SELECT * FROM t",
        "DESCRIBE TABLE t",
        "SHOW COLUMNS IN t",
        "SHOW TBLPROPERTIES t",
        "SHOW CREATE TABLE t",
    ] {
        let a = block_on(with_t.sql(sql)).unwrap_err().to_string();
        let b = block_on(without_t.sql(sql)).unwrap_err().to_string();
        assert_eq!(a, b, "{sql}");
        assert!(a.contains("[KRISHIV_INSUFFICIENT_PERMISSIONS]"), "{a}");
    }
    // A path that exists and one that does not.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("p");
    let df = block_on(with_t.sql("SELECT 1 AS a")).unwrap();
    let missing = block_on(df.write().mode("append").parquet(p.to_str().unwrap()))
        .unwrap_err()
        .to_string();
    std::fs::create_dir(&p).unwrap();
    let existing = block_on(df.write().mode("append").parquet(p.to_str().unwrap()))
        .unwrap_err()
        .to_string();
    assert_eq!(missing, existing);
}

#[test]
fn names_are_checked_as_written_and_folded() {
    use std::sync::Mutex;
    struct Record(Mutex<Vec<String>>);
    impl AccessPolicy for Record {
        fn version(&self) -> PolicyVersion {
            PolicyVersion(0)
        }
        fn check<'a>(
            &'a self,
            _: &'a Principal,
            securable: &'a Securable,
            privilege: Privilege,
        ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
            self.0
                .lock()
                .unwrap()
                .push(format!("{privilege} {securable}"));
            Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
        }
    }
    let record = Arc::new(Record(Mutex::new(Vec::new())));
    let s = session(Arc::clone(&record) as Arc<dyn AccessPolicy>);
    let asked = |sql: &str| {
        record.0.lock().unwrap().clear();
        let _ = block_on(s.sql(sql));
        record.0.lock().unwrap().clone()
    };
    assert_eq!(
        asked("DROP TABLE Db.T"),
        [
            "SELECT table `spark_catalog`.`Db`.`T`",
            "DROP table `spark_catalog`.`Db`.`T`",
            "SELECT table `spark_catalog`.`db`.`t`",
            "DROP table `spark_catalog`.`db`.`t`",
        ]
    );
    // Every change asks about the read first (a filtered reader may not
    // change the securable).
    let cwd = std::env::current_dir().unwrap();
    let loc = format!("path 'file:{}/loc'", cwd.display());
    assert_eq!(
        asked("CREATE TABLE x.y USING parquet LOCATION 'loc' AS SELECT 1"),
        [
            "SELECT namespace `spark_catalog`.`x`".to_owned(),
            "CREATE namespace `spark_catalog`.`x`".to_owned(),
            "SELECT table `spark_catalog`.`x`.`y`".to_owned(),
            "MODIFY table `spark_catalog`.`x`.`y`".to_owned(),
            format!("SELECT {loc}"),
            format!("MODIFY {loc}"),
        ]
    );
    assert_eq!(
        asked("CREATE DATABASE IF NOT EXISTS n"),
        [
            "SELECT namespace `spark_catalog`",
            "CREATE namespace `spark_catalog`"
        ]
    );
    assert_eq!(
        asked("DROP NAMESPACE spark_catalog.n"),
        [
            "SELECT namespace `spark_catalog`.`n`",
            "DROP namespace `spark_catalog`.`n`"
        ]
    );
    assert_eq!(
        asked("SHOW TABLES"),
        ["SELECT namespace `spark_catalog`.`default`"]
    );
    assert_eq!(
        asked("CREATE GLOBAL TEMPORARY VIEW g AS SELECT 1"),
        [
            "SELECT namespace `spark_catalog`.`global_temp`",
            "CREATE namespace `spark_catalog`.`global_temp`"
        ]
    );
    // Naming a table answers whether it exists, so it needs the right to read it too.
    assert_eq!(
        asked("REFRESH TABLE t"),
        [
            "SELECT table `spark_catalog`.`default`.`t`",
            "MONITOR system `cluster`"
        ]
    );
    assert_eq!(asked("SET a = b"), Vec::<String>::new());
    // A computed name is evaluated first and checked as the name it gives (`ab`, which is absent).
    assert_eq!(
        class(block_on(s.sql("DROP TABLE IDENTIFIER('a' || 'b')"))),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn path_prefix_policy_keeps_paths_under_its_roots() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&root).unwrap();
    let setup = session(Arc::new(AllowAll));
    for dir in [&root, &outside] {
        block_on(
            setup
                .table("t")
                .write()
                .parquet(dir.join("data").to_str().unwrap()),
        )
        .unwrap();
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

    let policy = PathPrefixPolicy::new([&root], Arc::new(AllowAll)).unwrap();
    let s = session(Arc::new(policy.clone()));
    let read = |p: &str| class(rows(&s, &format!("SELECT * FROM parquet.`{p}`")));
    let r = root.display();
    assert_eq!(read(&format!("{r}/data")), "OK");
    assert_eq!(read(&format!("{r}/../outside/data")), DENIED);
    assert_eq!(read(&format!("{}/data", outside.display())), DENIED);
    #[cfg(unix)]
    assert_eq!(read(&format!("{r}/link/data")), DENIED);
    assert_eq!(read(&format!("{r}/%2e%2e/outside/data")), DENIED);
    assert_eq!(read(&format!("file://{r}/data")), "OK");

    // Writes: under the root only, nothing created outside it.
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    block_on(df.write().parquet(root.join("new").to_str().unwrap())).unwrap();
    let escaped = outside.join("new");
    assert_eq!(
        class(block_on(df.write().parquet(escaped.to_str().unwrap()))),
        DENIED
    );
    assert!(!escaped.exists());

    // Other schemes, hosts, relative and unresolved forms are denied.
    let principal = Principal::new("ana", PrincipalKind::User);
    for path in [
        "s3://bucket/key",
        "file://host/x",
        "file:relative",
        "/no/scheme",
        &format!("file:{r}/../outside"),
        &format!("file:{r}/a%2Fb"),
    ] {
        let securable = Securable::Path(OptionValue::from_user("path", path));
        let d = block_on(policy.check(&principal, &securable, Privilege::Select)).unwrap();
        assert!(matches!(d, AccessDecision::Deny { .. }), "{path}");
    }
    // Other securables go to the inner policy.
    let table = Securable::System("cluster".to_owned());
    let d = block_on(policy.check(&principal, &table, Privilege::Monitor)).unwrap();
    assert_eq!(d, AccessDecision::Allow);

    let e = PathPrefixPolicy::new([tmp.path().join("missing")], Arc::new(AllowAll)).unwrap_err();
    assert_eq!(e.error_class().name(), "PATH_NOT_FOUND");
}

/// Allows everything; on the first `Create` check of a path it runs
/// `swap`, standing in for another process changing the file system
/// between the check and the write.
struct SwapOnCheck {
    swapped: AtomicBool,
    swap: Box<dyn Fn() + Send + Sync>,
}

impl AccessPolicy for SwapOnCheck {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(0)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if privilege == Privilege::Create
            && matches!(securable, Securable::Path(_))
            && !self.swapped.swap(true, Ordering::SeqCst)
        {
            (self.swap)();
        }
        Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
    }
}

#[cfg(unix)]
#[test]
fn a_link_swapped_in_after_the_check_fails_the_write() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(root.join("out")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(root.join("out/old"), b"x").unwrap();
    std::fs::write(outside.join("keep"), b"y").unwrap();
    let before = tree(&outside);

    // The checked directory becomes a link out of the root: the overwrite
    // must neither delete nor write outside it.
    let (r, o) = (root.clone(), outside.clone());
    let s = session(Arc::new(SwapOnCheck {
        swapped: AtomicBool::new(false),
        swap: Box::new(move || {
            std::fs::remove_dir_all(r.join("out")).unwrap();
            std::os::unix::fs::symlink(&o, r.join("out")).unwrap();
        }),
    }));
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let out = root.join("out");
    assert_eq!(
        class(block_on(
            df.write().mode("overwrite").parquet(out.to_str().unwrap())
        )),
        DENIED
    );
    assert_eq!(tree(&outside), before, "nothing changed outside the root");

    // A missing parent that becomes a link.
    let (r, o) = (root.clone(), outside.clone());
    let s = session(Arc::new(SwapOnCheck {
        swapped: AtomicBool::new(false),
        swap: Box::new(move || std::os::unix::fs::symlink(&o, r.join("new")).unwrap()),
    }));
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let sub = root.join("new/sub");
    assert_eq!(
        class(block_on(df.write().parquet(sub.to_str().unwrap()))),
        DENIED
    );
    assert_eq!(tree(&outside), before, "nothing changed outside the root");
}

/// An overwrite replaces what the target directory holds; a link inside it
/// is removed itself and never followed (the listing used to follow it, so
/// the files it pointed at outside the root were deleted).
#[cfg(unix)]
#[test]
fn an_overwrite_never_deletes_through_a_link_in_the_target() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(root.join("out/sub")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(root.join("out/old"), b"x").unwrap();
    std::fs::write(outside.join("keep"), b"y").unwrap();
    std::fs::create_dir_all(outside.join("deep")).unwrap();
    std::fs::write(outside.join("deep/keep2"), b"z").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("out/link")).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("out/sub/link2")).unwrap();
    let before = tree(&outside);

    let policy = PathPrefixPolicy::new([&root], Arc::new(AllowAll)).unwrap();
    let s = session(Arc::new(policy));
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let out = root.join("out");
    block_on(df.write().mode("overwrite").parquet(out.to_str().unwrap())).unwrap();
    assert_eq!(
        tree(&outside),
        before,
        "nothing outside the root was deleted"
    );
    assert!(
        !root.join("out/old").exists(),
        "the overwrite replaced the old file"
    );
    assert!(
        std::fs::symlink_metadata(root.join("out/link")).is_err(),
        "the link itself was replaced"
    );
}

/// The target directory is renamed away and a link to a directory outside
/// the root put in its place while the query runs: nothing lands outside, the
/// write fails with the audited denial (not a task failure).
#[cfg(unix)]
#[test]
fn a_directory_swapped_during_the_write_is_denied_and_cleaned_up() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("keep"), b"y").unwrap();
    let before = tree(&outside);
    let out = root.join("out");

    let swapper = {
        let (out, outside) = (out.clone(), outside.clone());
        std::thread::spawn(move || {
            let start = std::time::Instant::now();
            // Wait for the first part file, then swap.
            while start.elapsed() < std::time::Duration::from_secs(60) {
                let has_part = std::fs::read_dir(&out).is_ok_and(|mut d| {
                    d.any(|e| e.is_ok_and(|e| e.file_name().to_string_lossy().starts_with("part-")))
                });
                if has_part {
                    let old = out.with_file_name("out-old");
                    std::fs::rename(&out, &old).unwrap();
                    std::os::unix::fs::symlink(&outside, &out).unwrap();
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            false
        })
    };
    let policy = PathPrefixPolicy::new([&root], Arc::new(AllowAll)).unwrap();
    let s = session(Arc::new(policy));
    let df = block_on(s.sql("SELECT id FROM range(0, 20000, 1, 1)")).unwrap();
    let result = block_on(
        df.write()
            .option("maxRecordsPerFile", "1")
            .parquet(out.to_str().unwrap()),
    );
    assert!(
        swapper.join().unwrap(),
        "the swap happened during the write"
    );
    assert_eq!(class(result), DENIED);
    assert_eq!(tree(&outside), before, "nothing changed outside the root");
    // Files written before the swap are in the renamed directory, inside
    // the root; the link now at `out` cannot be followed to remove them.
    assert!(root.join("out-old").is_dir());
}

/// Dropping the write's future mid-query cancels it; the writers still
/// abort through the store they hold, so no finished part file stays.
#[cfg(unix)]
#[test]
fn a_cancelled_write_removes_the_files_it_made() {
    use futures::FutureExt;
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let s = session(Arc::new(AllowAll));
    let df = block_on(s.sql("SELECT id FROM range(0, 200000, 1, 1)")).unwrap();
    let (tx, rx) = futures::channel::oneshot::channel::<()>();
    let watcher = {
        let out = out.clone();
        std::thread::spawn(move || {
            let start = std::time::Instant::now();
            while start.elapsed() < std::time::Duration::from_secs(60) {
                let parts = |d: &Path| {
                    std::fs::read_dir(d).map_or(0, |d| {
                        d.filter_map(Result::ok)
                            .filter(|e| e.file_name().to_string_lossy().starts_with("part-"))
                            .count()
                    })
                };
                if parts(&out) >= 3 {
                    tx.send(()).ok();
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })
    };
    let write = df
        .write()
        .option("maxRecordsPerFile", "1")
        .parquet(out.to_str().unwrap())
        .boxed();
    let cancelled = block_on(async {
        match futures::future::select(write, rx).await {
            futures::future::Either::Left(_) => false,
            futures::future::Either::Right(_) => true,
        }
    });
    watcher.join().unwrap();
    assert!(cancelled, "the write was still running when it was dropped");
    // The detached writers finish aborting on their own.
    let start = std::time::Instant::now();
    loop {
        let left = std::fs::read_dir(&out).map_or(0, |d| {
            d.filter_map(Result::ok)
                .filter(|e| e.file_name().to_string_lossy().starts_with("part-"))
                .count()
        });
        if left == 0 {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "{left} part files remain after the cancelled write"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// An existing file as the target: `overwrite` replaces it, `error` says it
/// exists, with the store named `file` (never the write's private id).
#[cfg(unix)]
#[test]
fn a_file_in_the_way_of_the_target_is_handled_as_before() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let target = root.join("t");
    std::fs::write(&target, b"in the way").unwrap();
    let s = session(Arc::new(AllowAll));
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let e = block_on(df.write().mode("error").parquet(target.to_str().unwrap())).unwrap_err();
    let text = e.to_string();
    assert_eq!(e.error_class().name(), "PATH_ALREADY_EXISTS", "{text}");
    assert!(
        text.contains(&format!(
            "file/{}",
            target.display().to_string().trim_start_matches('/')
        )),
        "{text}"
    );
    assert!(!text.contains("private"), "{text}");
    block_on(
        df.write()
            .mode("overwrite")
            .parquet(target.to_str().unwrap()),
    )
    .unwrap();
    assert!(target.is_dir());
}

/// A directory above the target whose name object paths would encode cannot
/// be addressed: refused up front (it used to write into a differently named
/// sibling directory).
#[cfg(unix)]
#[test]
fn a_special_character_in_an_ancestor_name_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let dir = root.join("jos\u{e9}");
    std::fs::create_dir_all(&dir).unwrap();
    let s = session(Arc::new(AllowAll));
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let out = dir.join("out");
    assert_eq!(
        class(block_on(df.write().parquet(out.to_str().unwrap()))),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert!(!out.exists());
    assert_eq!(
        std::fs::read_dir(&root).unwrap().count(),
        1,
        "no sibling was created"
    );
}

/// A file where a directory of the path should be fails as a storage error,
/// not an internal one.
#[cfg(unix)]
#[test]
fn a_file_on_the_path_is_a_storage_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    std::fs::write(root.join("f"), b"x").unwrap();
    let s = session(Arc::new(AllowAll));
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let out = root.join("f/sub/out");
    assert_eq!(
        class(block_on(df.write().parquet(out.to_str().unwrap()))),
        "KRISHIV_OBJECT_STORE_REQUEST_FAILED"
    );
    assert_eq!(std::fs::read(root.join("f")).unwrap(), b"x");
}

#[cfg(unix)]
#[test]
fn writes_go_to_the_directory_a_link_names_and_are_checked_there() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
    let policy = PathPrefixPolicy::new([&root], Arc::new(AllowAll)).unwrap();
    let s = session(Arc::new(policy));
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let e = block_on(df.write().parquet(root.join("link/o").to_str().unwrap())).unwrap_err();
    assert_eq!(e.error_class().name(), DENIED);
    // The denial names the directory the write would have gone to.
    assert!(
        e.to_string()
            .contains(&format!("{}/o", outside.canonicalize().unwrap().display())),
        "{e}"
    );
    assert!(!outside.join("o").exists());
}

fn no_create_on_paths(securable: &Securable, p: Privilege) -> AccessDecision {
    if p == Privilege::Create && matches!(securable, Securable::Path(_)) {
        deny()
    } else {
        AccessDecision::Allow
    }
}

#[test]
fn a_writer_without_create_learns_nothing_about_existence() {
    let s = session(Policy::new(no_create_on_paths));
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("p");
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    let write = || {
        block_on(df.write().mode("append").parquet(p.to_str().unwrap()))
            .unwrap_err()
            .to_string()
    };
    let missing = write();
    std::fs::create_dir(&p).unwrap();
    let existing = write();
    assert_eq!(missing, existing);
    assert!(
        missing.contains("[KRISHIV_INSUFFICIENT_PERMISSIONS]"),
        "{missing}"
    );
    assert_eq!(std::fs::read_dir(&p).unwrap().count(), 0, "nothing written");
}

#[test]
fn a_temporary_table_location_is_checked() {
    let s = session(Arc::new(DenyAll));
    let dir = tempfile::tempdir().unwrap();
    let loc = dir.path().join("tl");
    for sql in [
        format!(
            "CREATE TEMPORARY TABLE x USING parquet LOCATION '{}' AS SELECT 1 AS a",
            loc.display()
        ),
        format!(
            "CREATE TEMPORARY TABLE x USING parquet LOCATION '{}'",
            dir.path().display()
        ),
    ] {
        assert_eq!(class(block_on(s.sql(&sql))), DENIED, "{sql}");
    }
    assert!(!loc.exists());
}

fn no_reads_of_s(securable: &Securable, p: Privilege) -> AccessDecision {
    if p == Privilege::Select && matches!(securable, Securable::Table(t) if t.name == "s") {
        deny()
    } else {
        AccessDecision::Allow
    }
}

#[test]
fn merge_sources_are_never_read_unchecked() {
    let s = session(Policy::new(no_reads_of_s));
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    s.register_batches("s", schema, vec![]).unwrap();
    // A table source is a target of the statement: checked before dispatch.
    assert_eq!(
        class(block_on(s.sql(
            "MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN DELETE"
        ))),
        DENIED
    );
    // A query source is read only through analysis, which asks the policy;
    // the session does not run MERGE, so nothing is read at all: `t` is a
    // temporary table, which MERGE refuses as Spark does.
    assert_eq!(
        class(block_on(s.sql(
            "MERGE INTO t USING (SELECT * FROM s) x ON t.a = x.a WHEN MATCHED THEN DELETE"
        ))),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
    // Its reads, as analysis finds them, are denied.
    let e = block_on(s.statement_info(
        "MERGE INTO t USING (SELECT * FROM s) x ON t.a = x.a WHEN MATCHED THEN DELETE",
    ))
    .unwrap_err();
    assert_eq!(e.error_class().name(), DENIED);
}

/// A SQL function's body reads `s`: creating the function probes the body,
/// and calling it analyzes the body, both through the policy.
#[test]
fn sql_function_bodies_are_read_through_the_policy() {
    let policy = Policy::new(no_reads_of_s);
    let s = session(Arc::<Policy>::clone(&policy));
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    s.register_batches("s", schema, vec![]).unwrap();
    // Creating a function whose body reads `s` is denied.
    assert_eq!(
        class(block_on(s.sql(
            "CREATE TEMPORARY FUNCTION leak() RETURNS INT RETURN (SELECT max(a) FROM s)"
        ))),
        DENIED
    );
    // A function over a table the principal may read works ...
    assert_eq!(
        class(block_on(s.sql(
            "CREATE TEMPORARY FUNCTION ok_fn() RETURNS INT RETURN (SELECT max(a) FROM t)"
        ))),
        "OK"
    );
    assert_eq!(class(rows(&s, "SELECT ok_fn()")), "OK");
    // ... until the policy denies the reads: the call analyzes the body.
    policy.deny_everything();
    assert_eq!(class(rows(&s, "SELECT ok_fn()")), DENIED);
}

/// A stored function (`CREATE FUNCTION` without `TEMPORARY`) is read through
/// the policy as a temporary one is: creating it probes the body, and a
/// denied create stores nothing.
#[test]
fn stored_function_bodies_are_read_through_the_policy() {
    let policy = Policy::new(no_reads_of_s);
    let s = session(Arc::<Policy>::clone(&policy));
    block_on(s.sql("CREATE TABLE s (a INT) USING parquet")).unwrap();
    block_on(s.sql("CREATE TABLE p (a INT) USING parquet")).unwrap();
    assert_eq!(
        class(block_on(s.sql(
            "CREATE FUNCTION leak() RETURNS INT RETURN (SELECT max(a) FROM s)"
        ))),
        DENIED
    );
    // Nothing was stored by the denied statement.
    assert_eq!(class(rows(&s, "SELECT leak()")), "UNRESOLVED_ROUTINE");
    assert_eq!(
        class(block_on(s.sql(
            "CREATE FUNCTION ok_stored() RETURNS INT RETURN (SELECT max(a) FROM p)"
        ))),
        "OK"
    );
    assert_eq!(class(rows(&s, "SELECT ok_stored()")), "OK");
    policy.deny_everything();
    assert_eq!(class(rows(&s, "SELECT ok_stored()")), DENIED);
}

/// A function body that reads a permanent view is read through the policy
/// (the view and its securables) when the function is called, whatever the
/// function was allowed to be created over.
#[test]
fn a_function_body_reading_a_view_is_read_through_the_policy() {
    let policy = Policy::new(no_reads_of_s);
    let s = session(Arc::<Policy>::clone(&policy));
    block_on(s.sql("CREATE VIEW pv AS SELECT 1 AS a")).unwrap();
    block_on(s.sql("CREATE VIEW s AS SELECT 2 AS a")).unwrap();
    assert_eq!(
        class(block_on(s.sql(
            "CREATE FUNCTION via_s() RETURNS INT RETURN (SELECT max(a) FROM s)"
        ))),
        DENIED
    );
    assert_eq!(
        class(block_on(s.sql(
            "CREATE FUNCTION via_pv() RETURNS INT RETURN (SELECT max(a) FROM pv)"
        ))),
        "OK"
    );
    assert_eq!(class(rows(&s, "SELECT via_pv()")), "OK");
    policy.deny_everything();
    assert_eq!(class(rows(&s, "SELECT via_pv()")), DENIED);
}

fn only_function_privileges(securable: &Securable, p: Privilege) -> AccessDecision {
    if matches!(
        (securable, p),
        (Securable::Function(_), Privilege::Drop) | (Securable::Namespace(_), Privilege::Create)
    ) {
        deny()
    } else {
        AccessDecision::Allow
    }
}

/// Stored `CREATE FUNCTION` needs `CREATE` on the namespace (and `DROP` on
/// the function to replace it), stored `DROP FUNCTION` needs `DROP` on the
/// function; temporary functions are the session's own and need nothing.
#[test]
fn stored_function_statements_are_authorized() {
    // Denied everywhere: nothing is created or dropped.
    let policy = Policy::new(allow);
    let s = session(Arc::<Policy>::clone(&policy));
    assert_eq!(
        class(block_on(s.sql("CREATE FUNCTION p() RETURNS INT RETURN 1"))),
        "OK"
    );
    policy.deny_everything();
    assert_eq!(
        class(block_on(s.sql("CREATE FUNCTION q() RETURNS INT RETURN 1"))),
        DENIED
    );
    assert_eq!(class(block_on(s.sql("DROP FUNCTION p"))), DENIED);
    assert_eq!(class(block_on(s.sql("DROP FUNCTION IF EXISTS p"))), DENIED);
    assert_eq!(
        class(block_on(
            s.sql("CREATE OR REPLACE FUNCTION p() RETURNS INT RETURN 2")
        )),
        DENIED
    );
    // A temporary function is the session's own.
    assert_eq!(
        class(block_on(
            s.sql("CREATE TEMPORARY FUNCTION tmp() RETURNS INT RETURN 1")
        )),
        "OK"
    );
    assert_eq!(class(block_on(s.sql("DROP TEMPORARY FUNCTION tmp"))), "OK");
    // The inverse policy denies only creating in the namespace and dropping
    // a function: each statement is checked for its own privilege.
    let s = session(Policy::new(only_function_privileges));
    assert_eq!(
        class(block_on(s.sql("CREATE FUNCTION p() RETURNS INT RETURN 1"))),
        DENIED
    );
    assert_eq!(class(block_on(s.sql("DROP FUNCTION IF EXISTS p"))), DENIED);
    assert_eq!(class(rows(&s, "SELECT 1")), "OK");
}

/// A stored function cannot read a temporary table (Spark checks this
/// before any privilege): the temporary one `t` of the session stays
/// out of the catalog.
#[test]
fn a_stored_function_cannot_read_a_temporary_table() {
    let s = session(Policy::new(allow));
    assert_eq!(
        class(block_on(s.sql(
            "CREATE FUNCTION reads_t() RETURNS INT RETURN (SELECT max(a) FROM t)"
        ))),
        "INVALID_TEMP_OBJ_REFERENCE"
    );
}

static DENY_VIEWS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn deny_view_selects(securable: &Securable, p: Privilege) -> AccessDecision {
    if DENY_VIEWS.load(std::sync::atomic::Ordering::SeqCst)
        && p == Privilege::Select
        && matches!(securable, Securable::View(_))
    {
        deny()
    } else {
        AccessDecision::Allow
    }
}

static DENY_VIEW_READS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn deny_view_reads(securable: &Securable, p: Privilege) -> AccessDecision {
    if DENY_VIEW_READS.load(std::sync::atomic::Ordering::SeqCst)
        && p == Privilege::Select
        && matches!(securable, Securable::View(v) if v.name == "inner_view")
    {
        deny()
    } else {
        AccessDecision::Allow
    }
}

/// A policy that denies `SELECT` on a permanent view stops a query of it, even through another view,
/// although every table it reads is allowed.
#[test]
fn querying_a_permanent_view_checks_the_view_securable() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(Policy::new(deny_view_reads));
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    for sql in [
        "CREATE TABLE base (a INT) USING parquet",
        "CREATE VIEW inner_view AS SELECT a FROM base",
        "CREATE VIEW outer_view AS SELECT a FROM inner_view",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let read = |sql: &str| class(block_on(s.sql(sql)));
    assert_eq!(read("SELECT * FROM outer_view"), "OK");
    DENY_VIEW_READS.store(true, std::sync::atomic::Ordering::SeqCst);
    let results = [
        read("SELECT * FROM inner_view"),
        read("SELECT * FROM outer_view"),
        read("SELECT * FROM default.inner_view"),
        read("SELECT * FROM base"),
    ];
    DENY_VIEW_READS.store(false, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(results[0], DENIED);
    assert_eq!(results[1], DENIED, "a view read through another view");
    assert_eq!(results[2], DENIED, "a qualified name");
    assert_eq!(results[3], "OK", "the base table stays readable");
}

/// The three tests below share these flags: one at a time.
static FLAG_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

static DENY_ALL_NOW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static DENY_TABLE_V: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn flagged(securable: &Securable, p: Privilege) -> AccessDecision {
    use std::sync::atomic::Ordering::SeqCst;
    if DENY_ALL_NOW.load(SeqCst) && p == Privilege::Select {
        return deny();
    }
    if DENY_TABLE_V.load(SeqCst)
        && p == Privilege::Select
        && matches!(securable, Securable::Table(t) if t.name == "v")
    {
        return deny();
    }
    AccessDecision::Allow
}

/// A denied read of a permanent view cannot be told from a read of a missing name: the same securable
/// is named, the policy is asked about the name as written and folded, and a broken stored text is
/// never reported to a principal that may not read the view.
#[test]
fn permanent_view_reads_leak_nothing_to_a_denied_principal() {
    let _one_at_a_time = FLAG_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    use std::sync::atomic::Ordering::SeqCst;
    let dir = tempfile::tempdir().unwrap();
    let s = session(Policy::new(flagged));
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    for sql in [
        "CREATE TABLE base (a INT) USING parquet",
        "CREATE VIEW v AS SELECT a FROM base",
        "CREATE VIEW c1 AS SELECT a FROM base",
        "CREATE VIEW c2 AS SELECT a FROM c1",
        // Cycle through ALTER: c1 now reads c2, which reads c1.
        "ALTER VIEW c1 AS SELECT a FROM c2",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let message = |sql: &str| {
        block_on(s.sql(sql))
            .map(|_| "OK".to_owned())
            .unwrap_or_else(|e| format!("{} {e}", e.error_class().name()))
    };
    let allowed = message("SELECT * FROM V");
    DENY_ALL_NOW.store(true, SeqCst);
    let existing = message("SELECT * FROM v");
    let missing = message("SELECT * FROM nope");
    // The name as the plan spells it is what the denial shows, whichever it is.
    let upper_existing = message("SELECT * FROM default.V");
    let upper_missing = message("SELECT * FROM default.NOPE");
    let cyclic = message("SELECT * FROM c1");
    let order_existing = message("SELECT * FROM base, v");
    let order_missing = message("SELECT * FROM base, nope");
    DENY_ALL_NOW.store(false, SeqCst);
    assert_eq!(allowed, "OK");
    assert!(existing.starts_with(DENIED), "{existing}");
    assert_eq!(
        existing.replace("`v`", "`X`"),
        missing.replace("`nope`", "`X`")
    );
    assert_eq!(
        upper_existing.to_lowercase().replace("`v`", "`X`"),
        upper_missing.to_lowercase().replace("`nope`", "`X`")
    );
    assert!(cyclic.starts_with(DENIED), "{cyclic}");
    // The first denial is the first name in the plan, view or not.
    assert!(order_existing.contains("base"), "{order_existing}");
    assert!(order_missing.contains("base"), "{order_missing}");
    // With the grant back, the cycle is reported.
    assert!(message("SELECT * FROM c1").starts_with("INTERNAL_ERROR"));
}

/// A policy that denies the table-named securable of a view stops a query of the view, in any
/// spelling of its name.
#[test]
fn a_table_rule_on_a_view_name_applies_to_queries_of_it() {
    let _one_at_a_time = FLAG_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    use std::sync::atomic::Ordering::SeqCst;
    let dir = tempfile::tempdir().unwrap();
    let s = session(Policy::new(flagged));
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    for sql in [
        "CREATE TABLE base (a INT) USING parquet",
        "CREATE VIEW v AS SELECT a FROM base",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    DENY_TABLE_V.store(true, SeqCst);
    let results = [
        class(block_on(s.sql("SELECT * FROM v"))),
        class(block_on(s.sql("SELECT * FROM V"))),
        class(block_on(s.sql("SELECT * FROM spark_catalog.default.v"))),
    ];
    DENY_TABLE_V.store(false, SeqCst);
    assert_eq!(results, [DENIED, DENIED, DENIED]);
}

/// `statement_info` lists the view securable of a query of a permanent view.
#[test]
fn statement_info_lists_the_view_securable() {
    let _one_at_a_time = FLAG_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();
    let s = session(Policy::new(flagged));
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    for sql in [
        "CREATE TABLE base (a INT) USING parquet",
        "CREATE VIEW v AS SELECT a FROM base",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let info = block_on(s.statement_info("SELECT * FROM v")).unwrap();
    let ident = |name: &str| krishiv_plan::analyzer::TableIdent {
        catalog: "spark_catalog".to_owned(),
        namespace: vec!["default".to_owned()],
        name: name.to_owned(),
    };
    let found: Vec<(Securable, Privilege)> = info
        .securables
        .iter()
        .map(|u| (u.securable.clone(), u.privilege))
        .collect();
    assert!(
        found.contains(&(Securable::View(ident("v")), Privilege::Select)),
        "{found:?}"
    );
    assert!(
        found.contains(&(Securable::Table(ident("base")), Privilege::Select)),
        "{found:?}"
    );
}

/// A grant on the table-named securable does not show a permanent view's definition, columns or
/// properties: those statements need `SELECT` on the view, as a query of it does.
#[test]
fn describing_a_view_needs_the_view_securable() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(Policy::new(deny_view_selects));
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    block_on(async {
        s.sql("CREATE TABLE pt (a INT) USING parquet")
            .await?
            .collect()
            .await
    })
    .unwrap();
    block_on(async {
        s.sql("CREATE VIEW pv AS SELECT a FROM pt")
            .await?
            .collect()
            .await
    })
    .unwrap();
    DENY_VIEWS.store(true, std::sync::atomic::Ordering::SeqCst);
    let denied = [
        "SHOW CREATE TABLE pv",
        "SHOW TBLPROPERTIES pv",
        "DESCRIBE TABLE pv",
        "DESCRIBE TABLE EXTENDED pv",
        "SHOW COLUMNS IN pv",
    ];
    let results: Vec<_> = denied
        .iter()
        .map(|sql| class(block_on(s.sql(sql))))
        .collect();
    let table_ok = block_on(async { s.sql("SHOW CREATE TABLE pt").await?.collect().await }).is_ok();
    DENY_VIEWS.store(false, std::sync::atomic::Ordering::SeqCst);
    for (sql, c) in denied.iter().zip(&results) {
        assert_eq!(c, DENIED, "{sql}");
    }
    assert!(table_ok, "the table itself stays readable");
}

/// An access policy whose rules and version change while a session lives.
struct Flip {
    deny_views: std::sync::atomic::AtomicBool,
    version: std::sync::atomic::AtomicU64,
}

impl AccessPolicy for Flip {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(
            self.version.load(std::sync::atomic::Ordering::SeqCst),
        )
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let deny_now = self.deny_views.load(std::sync::atomic::Ordering::SeqCst)
            && privilege == Privilege::Select
            && matches!(securable, Securable::View(_));
        Box::pin(async move {
            Ok(if deny_now {
                deny()
            } else {
                AccessDecision::Allow
            })
        })
    }
}

/// A DataFrame analysed under one policy version is analysed again under the next: reads it was
/// allowed to make before are checked against the new rules.
#[test]
fn a_new_policy_version_invalidates_cached_analysis() {
    use std::sync::atomic::Ordering::SeqCst;
    let dir = tempfile::tempdir().unwrap();
    let policy = Arc::new(Flip {
        deny_views: std::sync::atomic::AtomicBool::new(false),
        version: std::sync::atomic::AtomicU64::new(1),
    });
    let s = session(Arc::clone(&policy) as Arc<dyn AccessPolicy>);
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    for sql in [
        "CREATE TABLE base (a INT) USING parquet",
        "CREATE VIEW v AS SELECT a FROM base",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let df = block_on(s.sql("SELECT * FROM v")).unwrap();
    block_on(df.collect()).unwrap();
    policy.deny_views.store(true, SeqCst);
    policy.version.store(2, SeqCst);
    assert_eq!(class(block_on(df.collect())), DENIED);
}

fn no_modifying(_: &Securable, p: Privilege) -> AccessDecision {
    if p == Privilege::Modify {
        deny()
    } else {
        AccessDecision::Allow
    }
}

/// A script is asked about everything any of its statements writes before the first one runs, also
/// in a handler body that may never run; nothing the earlier statements did is undone for a denial
/// later on.
#[test]
fn a_script_is_authorized_by_its_strongest_statement() {
    let s = session(Policy::new(no_modifying));
    for sql in [
        "BEGIN DECLARE x INT DEFAULT 1; INSERT INTO t VALUES (x); END",
        "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION INSERT INTO t VALUES (1); SELECT 1; END",
        "BEGIN IF TRUE THEN SELECT 1; ELSE INSERT INTO t VALUES (2); END IF; END",
        "BEGIN FOR r AS SELECT 1 DO INSERT INTO t VALUES (3); END FOR; END",
    ] {
        assert_eq!(class(block_on(s.sql(sql))), DENIED, "{sql}");
    }
    // Reads and session state alone are fine.
    for sql in [
        "BEGIN DECLARE x INT DEFAULT 1; SET x = 2; SELECT x; END",
        "BEGIN SELECT * FROM t; END",
    ] {
        assert_eq!(class(block_on(s.sql(sql))), "OK", "{sql}");
    }
}
