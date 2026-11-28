// SPDX-License-Identifier: Apache-2.0
//! One path form for a managed table's directory (T063 N1c, after #773): every statement that asks the
//! policy about a table's directories asks in the `file:/a/b` form `df.write` and `PathPrefixPolicy` use,
//! so one prefix rule gives the same answer for `INSERT`, `INSERT OVERWRITE`, `TRUNCATE` and `DROP TABLE`
//! of the same table (allow and deny), with the warehouse given with `file:` or without, and `CREATE
//! TABLE`, CTAS, `saveAsTable`, `ALTER TABLE .. RENAME` and partition DDL under a `PathPrefixPolicy` of
//! the warehouse run. `path_confinement.rs` covers the admin role matrix of the same directories.
// Unix only: the cases plant symbolic links and mode-000 directories and use Unix path forms.
#![cfg(unix)]
#![cfg(test)]

use std::path::Path;
use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, EngineRuntime, PathPrefixPolicy, Principal,
    PrincipalKind, Privilege, Securable, Session,
};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

/// One rule on path text: a path starting with `prefix` gets `inside`, any other path the opposite;
/// everything that is not a path is allowed.
struct PrefixRule {
    prefix: String,
    inside: bool,
}

impl AccessPolicy for PrefixRule {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = match securable {
            Securable::Path(p) => {
                let under = p.as_plain().is_some_and(|t| t.starts_with(&self.prefix));
                if under == self.inside {
                    AccessDecision::Allow
                } else {
                    AccessDecision::Deny {
                        reason: "path rule".to_owned(),
                    }
                }
            }
            _ => AccessDecision::Allow,
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

/// An unrestricted owner and a subject under `policy`, sharing one runtime (so one catalog and warehouse),
/// both with the warehouse `warehouse` as written.
fn sessions(warehouse: &str, policy: Arc<dyn AccessPolicy>) -> (Session, Session) {
    let rt = EngineRuntime::builder().build().unwrap();
    let owner = Session::builder()
        .runtime(Arc::clone(&rt))
        .principal(Principal::new("owner", PrincipalKind::User))
        .access_policy(Arc::new(AllowAll))
        .config("spark.sql.warehouse.dir", warehouse)
        .build()
        .unwrap();
    let subject = Session::builder()
        .runtime(rt)
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(policy)
        .config("spark.sql.warehouse.dir", warehouse)
        .build()
        .unwrap();
    (owner, subject)
}

fn run(s: &Session, sql: &str) -> Result<(), KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await.map(|_| ()) })
}

fn outcome(r: Result<(), KrishivError>) -> String {
    match r {
        Ok(()) => "ok".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    }
}

/// The subject's answer for each statement on the managed table `t`, which the owner creates (with a row)
/// before each one.
fn answers(warehouse: &str, policy: Arc<dyn AccessPolicy>) -> Vec<(&'static str, String)> {
    let (owner, subject) = sessions(warehouse, policy);
    let statements = [
        ("insert", "INSERT INTO t VALUES (2)"),
        ("insert overwrite", "INSERT OVERWRITE t VALUES (3)"),
        ("truncate", "TRUNCATE TABLE t"),
        ("drop", "DROP TABLE t"),
    ];
    statements
        .into_iter()
        .map(|(label, sql)| {
            run(&owner, "CREATE TABLE IF NOT EXISTS t (a INT) USING parquet").unwrap();
            run(&owner, "INSERT INTO t VALUES (1)").unwrap();
            (label, outcome(run(&subject, sql)))
        })
        .collect()
}

fn all(expected: &str) -> Vec<(&'static str, String)> {
    ["insert", "insert overwrite", "truncate", "drop"]
        .into_iter()
        .map(|l| (l, expected.to_owned()))
        .collect()
}

/// The warehouse, created, with its links resolved (the form the rule is written against).
fn warehouse(tmp: &Path) -> std::path::PathBuf {
    let wh = tmp.join("wh");
    std::fs::create_dir_all(&wh).unwrap();
    wh.canonicalize().unwrap()
}

/// The warehouse as written: plain or with `file:`.
fn written(wh: &Path, with_scheme: bool) -> String {
    if with_scheme {
        format!("file:{}", wh.display())
    } else {
        wh.display().to_string()
    }
}

#[test]
fn one_prefix_rule_allows_every_statement_on_the_table_directory() {
    for with_scheme in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let wh = warehouse(tmp.path());
        let rule = PrefixRule {
            prefix: format!("file:{}/t", wh.display()),
            inside: true,
        };
        assert_eq!(
            answers(&written(&wh, with_scheme), Arc::new(rule)),
            all("ok"),
            "warehouse with file: {with_scheme}"
        );
    }
}

#[test]
fn one_prefix_rule_denies_every_statement_on_the_table_directory() {
    for with_scheme in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let wh = warehouse(tmp.path());
        let rule = PrefixRule {
            prefix: format!("file:{}/t", wh.display()),
            inside: false,
        };
        assert_eq!(
            answers(&written(&wh, with_scheme), Arc::new(rule)),
            all(DENIED),
            "warehouse with file: {with_scheme}"
        );
    }
}

/// `PathPrefixPolicy` over the warehouse allows the statements on its tables; over another root it denies
/// all of them.
#[test]
fn a_path_prefix_policy_gives_one_answer_for_a_managed_table() {
    for with_scheme in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let wh = warehouse(tmp.path());
        let other = tmp.path().join("other");
        std::fs::create_dir(&other).unwrap();
        let inside = PathPrefixPolicy::new([&wh], Arc::new(AllowAll)).unwrap();
        let outside = PathPrefixPolicy::new([&other], Arc::new(AllowAll)).unwrap();
        let at = written(&wh, with_scheme);
        assert_eq!(answers(&at, Arc::new(inside)), all("ok"), "{at}");
        assert_eq!(answers(&at, Arc::new(outside)), all(DENIED), "{at}");
    }
}

/// The directory checks of the statements that create, move or add a managed directory use the same form:
/// under a `PathPrefixPolicy` of the warehouse they run.
#[test]
fn creates_renames_and_partitions_in_the_warehouse_pass_a_path_prefix_policy() {
    for with_scheme in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let wh = warehouse(tmp.path());
        let policy = PathPrefixPolicy::new([&wh], Arc::new(AllowAll)).unwrap();
        let (_, s) = sessions(&written(&wh, with_scheme), Arc::new(policy));
        for sql in [
            "CREATE TABLE c (a INT) USING parquet",
            "CREATE TABLE ctas USING parquet AS SELECT 1 AS a",
            "ALTER TABLE c RENAME TO c2",
            "CREATE TABLE p (a INT, k INT) USING parquet PARTITIONED BY (k)",
            "ALTER TABLE p ADD PARTITION (k = 1)",
            "INSERT INTO p PARTITION (k = 2) VALUES (5)",
            "ALTER TABLE p PARTITION (k = 1) RENAME TO PARTITION (k = 3)",
            "ALTER TABLE p DROP PARTITION (k = 2)",
        ] {
            assert_eq!(outcome(run(&s, sql)), "ok", "{sql} (file: {with_scheme})");
        }
        let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
        block_on(df.write().save_as_table("saved")).unwrap();
        assert!(wh.join("saved").is_dir());
        assert!(wh.join("c2").is_dir() && !wh.join("c").exists());
    }
}

/// Every statement that asks about a managed table's directory, each run by the subject in a fresh
/// runtime after the owner's setup (`""` for none).
const DIRECTORY_STATEMENTS: [(&str, &str, &str); 7] = [
    ("create", "", "CREATE TABLE c (a INT) USING parquet"),
    (
        "ctas",
        "",
        "CREATE TABLE ctas USING parquet AS SELECT 1 AS a",
    ),
    (
        "insert",
        "CREATE TABLE t (a INT) USING parquet",
        "INSERT INTO t VALUES (2)",
    ),
    (
        "truncate",
        "CREATE TABLE t (a INT) USING parquet",
        "TRUNCATE TABLE t",
    ),
    (
        "drop",
        "CREATE TABLE t (a INT) USING parquet",
        "DROP TABLE t",
    ),
    (
        "rename",
        "CREATE TABLE t (a INT) USING parquet",
        "ALTER TABLE t RENAME TO t2",
    ),
    (
        "add partition",
        "CREATE TABLE p (a INT, k INT) USING parquet PARTITIONED BY (k)",
        "ALTER TABLE p ADD PARTITION (k = 1)",
    ),
];

/// The subject's answer for each of [`DIRECTORY_STATEMENTS`]: `layout` makes a fresh warehouse under a new
/// temporary directory and returns the warehouse as written and the policy to ask.
fn directory_answers(
    layout: impl Fn(&Path) -> (String, Arc<dyn AccessPolicy>),
) -> Vec<(&'static str, String)> {
    DIRECTORY_STATEMENTS
        .into_iter()
        .map(|(label, setup, sql)| {
            let tmp = tempfile::tempdir().unwrap();
            let (at, policy) = layout(tmp.path());
            let (owner, subject) = sessions(&at, policy);
            if !setup.is_empty() {
                run(&owner, setup).unwrap();
                if label != "add partition" {
                    run(&owner, "INSERT INTO t VALUES (1)").unwrap();
                }
            }
            (label, outcome(run(&subject, sql)))
        })
        .collect()
}

fn every_directory_statement(expected: &str) -> Vec<(&'static str, String)> {
    DIRECTORY_STATEMENTS
        .into_iter()
        .map(|(l, _, _)| (l, expected.to_owned()))
        .collect()
}

/// `<tmp>/real/wh` (created) and the link `<tmp>/link` to `<tmp>/real`: the real warehouse (links
/// resolved) and the warehouse reached through the link.
fn linked_warehouse(tmp: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let real = tmp.join("real");
    std::fs::create_dir_all(real.join("wh")).unwrap();
    std::os::unix::fs::symlink(&real, tmp.join("link")).unwrap();
    (
        real.join("wh").canonicalize().unwrap(),
        tmp.join("link").join("wh"),
    )
}

/// Under a warehouse reached through a symbolic link, one prefix rule written on the real directory gives
/// one answer, allow or deny, for every statement on a managed table's directory: each asks the canonical
/// path (links and `..` resolved), the path the write goes to.
#[test]
fn a_linked_warehouse_gets_one_answer_from_a_rule_on_the_real_directory() {
    for inside in [true, false] {
        for with_scheme in [false, true] {
            let answers = directory_answers(|tmp| {
                let (real, link) = linked_warehouse(tmp);
                let rule = PrefixRule {
                    prefix: format!("file:{}/", real.display()),
                    inside,
                };
                (written(&link, with_scheme), Arc::new(rule))
            });
            let expected = if inside { "ok" } else { DENIED };
            assert_eq!(
                answers,
                every_directory_statement(expected),
                "inside {inside}, file: {with_scheme}"
            );
        }
    }
}

/// The same with a `PathPrefixPolicy` whose root is given as the link path: its root is made canonical
/// when it is built, so it matches the canonical paths the statements ask.
#[test]
fn a_path_prefix_root_given_through_the_link_matches_every_statement() {
    let allowed = directory_answers(|tmp| {
        let (_, link) = linked_warehouse(tmp);
        let policy = PathPrefixPolicy::new([&link], Arc::new(AllowAll)).unwrap();
        (written(&link, false), Arc::new(policy))
    });
    assert_eq!(allowed, every_directory_statement("ok"));
    let denied = directory_answers(|tmp| {
        let (_, link) = linked_warehouse(tmp);
        let other = tmp.join("other");
        std::fs::create_dir(&other).unwrap();
        std::os::unix::fs::symlink(&other, tmp.join("other-link")).unwrap();
        let policy = PathPrefixPolicy::new([tmp.join("other-link")], Arc::new(AllowAll)).unwrap();
        (written(&link, false), Arc::new(policy))
    });
    assert_eq!(denied, every_directory_statement(DENIED));
}

/// A `..` in the configured warehouse is resolved before any statement asks: a rule (and a
/// `PathPrefixPolicy`) on the real directory answers every statement alike.
#[test]
fn a_dot_dot_in_the_warehouse_is_resolved_for_every_statement() {
    let dotted = |tmp: &Path| {
        std::fs::create_dir_all(tmp.join("x")).unwrap();
        std::fs::create_dir_all(tmp.join("wh")).unwrap();
        (
            tmp.join("wh").canonicalize().unwrap(),
            tmp.join("x").join("..").join("wh"),
        )
    };
    let by_rule = directory_answers(|tmp| {
        let (real, at) = dotted(tmp);
        let rule = PrefixRule {
            prefix: format!("file:{}/", real.display()),
            inside: true,
        };
        (written(&at, true), Arc::new(rule))
    });
    assert_eq!(by_rule, every_directory_statement("ok"));
    let by_prefix = directory_answers(|tmp| {
        let (real, at) = dotted(tmp);
        let policy = PathPrefixPolicy::new([&real], Arc::new(AllowAll)).unwrap();
        (written(&at, false), Arc::new(policy))
    });
    assert_eq!(by_prefix, every_directory_statement("ok"));
}

/// Records every path the policy is asked about, and denies a path that is not a canonical `file:/` path.
struct PathLog(std::sync::Mutex<Vec<String>>);

impl AccessPolicy for PathLog {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = match securable {
            Securable::Path(p) => {
                let text = p.as_plain().unwrap_or("<not plain>").to_owned();
                let ok = text.starts_with("file:/");
                self.0.lock().unwrap().push(text);
                if ok {
                    AccessDecision::Allow
                } else {
                    AccessDecision::Deny {
                        reason: "not a canonical local path".to_owned(),
                    }
                }
            }
            _ => AccessDecision::Allow,
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

/// A relative warehouse is refused before the policy is asked: the path never reaches it as written.
#[test]
fn a_relative_warehouse_is_refused_not_passed_to_the_policy() {
    let name = format!("krishiv-relative-wh-{}", std::process::id());
    for sql in [
        "CREATE TABLE c (a INT) USING parquet",
        "CREATE TABLE ctas USING parquet AS SELECT 1 AS a",
    ] {
        let log = Arc::new(PathLog(std::sync::Mutex::new(Vec::new())));
        let (_, subject) = sessions(&name, Arc::clone(&log) as Arc<dyn AccessPolicy>);
        let e = run(&subject, sql).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{sql}: {e}"
        );
        assert_eq!(*log.0.lock().unwrap(), Vec::<String>::new(), "{sql}");
        assert!(!Path::new(&name).exists(), "{sql} made the directory");
    }
}

/// Allows the first `allowed` path questions and denies every later one; everything that is not a path
/// is allowed. With `allowed = 1` a `CREATE TABLE ... LOCATION` passes its statement pre-check (the
/// location as written) and is denied at the directory check, so what that check answers is visible.
struct FirstPaths {
    allowed: usize,
    asked: std::sync::atomic::AtomicUsize,
}

impl FirstPaths {
    fn new(allowed: usize) -> Arc<Self> {
        Arc::new(Self {
            allowed,
            asked: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

impl AccessPolicy for FirstPaths {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = match securable {
            Securable::Path(_)
                if self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= self.allowed =>
            {
                AccessDecision::Deny {
                    reason: "path".to_owned(),
                }
            }
            _ => AccessDecision::Allow,
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

/// The error text, with the class: what a caller sees.
fn seen(r: Result<(), KrishivError>) -> String {
    match r {
        Ok(()) => "ok".to_owned(),
        Err(e) => format!("{}: {e}", e.error_class().name()),
    }
}

/// `<dir>/d` made as each kind of entry a path cannot be resolved through, as a directory and as
/// nothing: a missing directory, an existing one, an unsearchable one (skipped when the test runs as root, which searches it anyway), a
/// dangling link and a link loop. `probe` gets `<written>/d/x` (`written` is `dir` or a link to it).
fn each_unresolvable_kind(
    dir: &Path,
    written: &Path,
    mut probe: impl FnMut(&str) -> String,
) -> Vec<(String, String)> {
    use std::os::unix::fs::PermissionsExt;
    let d = dir.join("d");
    let mut out = Vec::new();
    for kind in [
        "missing",
        "existing",
        "unsearchable",
        "dangling link",
        "link loop",
    ] {
        match kind {
            "existing" => std::fs::create_dir(&d).unwrap(),
            "unsearchable" => {
                std::fs::create_dir(&d).unwrap();
                std::fs::create_dir(d.join("inner")).unwrap();
                std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o000)).unwrap();
                if std::fs::read_dir(&d).is_ok() {
                    // Root: the directory is searchable after all.
                    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700)).unwrap();
                    std::fs::remove_dir_all(&d).unwrap();
                    continue;
                }
            }
            "dangling link" => std::os::unix::fs::symlink(dir.join("nowhere"), &d).unwrap(),
            "link loop" => std::os::unix::fs::symlink(&d, &d).unwrap(),
            _ => {}
        }
        let at = written.join("d").join("x");
        out.push((kind.to_owned(), probe(&at.display().to_string())));
        if kind == "unsearchable" {
            std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700)).unwrap();
            std::fs::remove_dir_all(&d).unwrap();
        } else if kind == "existing" {
            std::fs::remove_dir_all(&d).unwrap();
        } else if kind != "missing" {
            std::fs::remove_file(&d).unwrap();
        }
    }
    out
}

/// A path the session cannot resolve (unsearchable, a dangling link, a loop) is denied exactly as the
/// policy denies a missing one: the same 42501 text, so the answer never tells what is on disk. For a
/// write by path and for `CREATE TABLE ... LOCATION`.
#[test]
fn an_unresolvable_path_is_denied_as_a_missing_one_is() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().canonicalize().unwrap();
    let writes = each_unresolvable_kind(&dir, &dir, |path| {
        let s = Session::builder()
            .principal(Principal::new("ana", PrincipalKind::User))
            .access_policy(FirstPaths::new(0))
            .build()
            .unwrap();
        seen(block_on(async {
            s.sql("SELECT 1 AS a").await?.write().parquet(path).await
        }))
    });
    let creates = each_unresolvable_kind(&dir, &dir, |path| {
        let s = Session::builder()
            .principal(Principal::new("ana", PrincipalKind::User))
            .access_policy(FirstPaths::new(1))
            .build()
            .unwrap();
        seen(run(
            &s,
            &format!("CREATE TABLE e (a INT) USING parquet LOCATION '{path}'"),
        ))
    });
    for (what, answers) in [("write", writes), ("create", creates)] {
        let (_, missing) = &answers[0];
        assert!(missing.starts_with(DENIED), "{what}: {missing}");
        for (kind, answer) in &answers {
            assert_eq!(answer, missing, "{what}: {kind}");
        }
    }
}

/// `..` after a link is folded in the text first, as reads and writes fold it (`LocalPaths::locate`),
/// then links are resolved: with the warehouse `<tmp>/data/l/../wh` and `l -> <tmp>/mnt/x`, every
/// statement asks about (and CREATE makes) `<tmp>/data/wh/..`, never `<tmp>/mnt/wh/..`.
#[test]
fn a_dot_dot_after_a_link_is_folded_as_writes_fold_it() {
    let layout = |tmp: &Path, inside: bool| {
        let tmp = tmp.canonicalize().unwrap();
        for d in ["data/wh", "mnt/x", "mnt/wh"] {
            std::fs::create_dir_all(tmp.join(d)).unwrap();
        }
        std::os::unix::fs::symlink(tmp.join("mnt/x"), tmp.join("data/l")).unwrap();
        let rule = PrefixRule {
            prefix: format!("file:{}/data/wh/", tmp.display()),
            inside,
        };
        (
            format!("{}/data/l/../wh", tmp.display()),
            Arc::new(rule) as Arc<dyn AccessPolicy>,
        )
    };
    assert_eq!(
        directory_answers(|tmp| layout(tmp, true)),
        every_directory_statement("ok")
    );
    assert_eq!(
        directory_answers(|tmp| layout(tmp, false)),
        every_directory_statement(DENIED)
    );
    let tmp = tempfile::tempdir().unwrap();
    let (at, policy) = layout(tmp.path(), true);
    let (_, s) = sessions(&at, policy);
    run(&s, "CREATE TABLE made (a INT) USING parquet").unwrap();
    let root = tmp.path().canonicalize().unwrap();
    assert!(root.join("data/wh/made").is_dir());
    assert!(!root.join("mnt/wh/made").exists());
}

/// A location whose `..` climbs back out of a link (`<wh>/o -> <tmp>/outside`) gets one answer whether the
/// directory it names behind the link exists or not: the check never resolves `..` on the disk.
#[test]
fn a_location_climbing_out_of_a_link_does_not_tell_what_is_behind_it() {
    let answer = |behind_exists: bool| {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("wh")).unwrap();
        std::fs::create_dir_all(root.join("outside")).unwrap();
        if behind_exists {
            std::fs::create_dir(root.join("outside/maybe")).unwrap();
        }
        std::os::unix::fs::symlink(root.join("outside"), root.join("wh/o")).unwrap();
        let rule = PrefixRule {
            prefix: format!("file:{}/wh/", root.display()),
            inside: true,
        };
        let (_, s) = sessions(&format!("{}/wh", root.display()), Arc::new(rule));
        let location = format!("{}/wh/o/maybe/../../x", root.display());
        seen(run(
            &s,
            &format!("CREATE TABLE e (a INT) USING parquet LOCATION '{location}'"),
        ))
    };
    assert_eq!(answer(true), answer(false));
    assert_eq!(answer(false), "ok");
}

/// Swaps the warehouse `<tmp>/wh` for a link to `<tmp>/outside` the first time it is asked about a
/// path under it, and allows everything: the race between a directory check and the mkdir.
struct SwapOnCheck {
    root: std::path::PathBuf,
    done: std::sync::atomic::AtomicBool,
}

impl AccessPolicy for SwapOnCheck {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let wh = format!("file:{}/wh/", self.root.display());
        if let Securable::Path(p) = securable
            && p.as_plain().is_some_and(|t| t.starts_with(&wh))
            && !self.done.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            std::fs::rename(self.root.join("wh"), self.root.join("wh-old")).unwrap();
            std::os::unix::fs::symlink(self.root.join("outside"), self.root.join("wh")).unwrap();
        }
        Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
    }
}

/// A link swapped into the warehouse between the directory check and the mkdir: CREATE and CTAS make
/// nothing outside it and fail with 42501.
#[test]
fn a_link_swapped_in_after_the_check_creates_nothing_outside() {
    for sql in [
        "CREATE TABLE t (a INT) USING parquet",
        "CREATE TABLE t USING parquet AS SELECT 1 AS a",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("wh")).unwrap();
        std::fs::create_dir_all(root.join("outside")).unwrap();
        let policy = SwapOnCheck {
            root: root.clone(),
            done: std::sync::atomic::AtomicBool::new(false),
        };
        let (_, s) = sessions(&format!("{}/wh", root.display()), Arc::new(policy));
        assert_eq!(outcome(run(&s, sql)), DENIED, "{sql}");
        assert_eq!(
            std::fs::read_dir(root.join("outside")).unwrap().count(),
            0,
            "{sql} made a directory outside"
        );
    }
}

/// A warehouse that is a dangling link or a link loop: CREATE is denied as the policy denies (42501),
/// not refused with another class.
#[test]
fn a_dangling_or_looping_warehouse_is_denied() {
    for looping in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let wh = root.join("wh");
        if looping {
            std::os::unix::fs::symlink(&wh, &wh).unwrap();
        } else {
            std::os::unix::fs::symlink(root.join("nowhere"), &wh).unwrap();
        }
        let (_, s) = sessions(&wh.display().to_string(), Arc::new(AllowAll));
        for sql in [
            "CREATE TABLE c (a INT) USING parquet",
            "CREATE TABLE ctas USING parquet AS SELECT 1 AS a",
        ] {
            assert_eq!(outcome(run(&s, sql)), DENIED, "{sql} (loop: {looping})");
        }
    }
}

/// A link inside the warehouse that points out of it (`<wh>/ns.db -> <tmp>/outside`): under a
/// `PathPrefixPolicy` of the warehouse, a table created through it is denied and nothing is made there.
#[test]
fn a_link_escaping_the_warehouse_is_denied() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let wh = root.join("wh");
    std::fs::create_dir_all(&wh).unwrap();
    std::fs::create_dir_all(root.join("outside")).unwrap();
    let policy = PathPrefixPolicy::new([&wh], Arc::new(AllowAll)).unwrap();
    let (owner, s) = sessions(&wh.display().to_string(), Arc::new(policy));
    run(&owner, "CREATE DATABASE ns").unwrap();
    if wh.join("ns.db").is_dir() {
        std::fs::remove_dir(wh.join("ns.db")).unwrap();
    }
    std::os::unix::fs::symlink(root.join("outside"), wh.join("ns.db")).unwrap();
    for sql in [
        "CREATE TABLE ns.c (a INT) USING parquet",
        "CREATE TABLE ns.ctas USING parquet AS SELECT 1 AS a",
    ] {
        assert_eq!(outcome(run(&s, sql)), DENIED, "{sql}");
    }
    assert_eq!(std::fs::read_dir(root.join("outside")).unwrap().count(), 0);
}

/// A `file://host/...` location naming another host is refused as reads and writes refuse it, never
/// checked as the local path `/...`.
#[test]
fn a_file_url_with_a_host_is_refused() {
    let log = Arc::new(PathLog(std::sync::Mutex::new(Vec::new())));
    let tmp = tempfile::tempdir().unwrap();
    let (_, s) = sessions(
        &tmp.path().display().to_string(),
        Arc::clone(&log) as Arc<dyn AccessPolicy>,
    );
    let e = run(
        &s,
        "CREATE TABLE e (a INT) USING parquet LOCATION 'file://evil/a'",
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE", "{e}");
    assert!(
        !log.0.lock().unwrap().iter().any(|p| p == "file:/evil/a"),
        "{:?}",
        log.0.lock().unwrap()
    );
}

/// The `krishiv::audit` records of the current thread, as `field=value` lines.
#[derive(Clone, Default)]
struct AuditLines(Arc<std::sync::Mutex<Vec<String>>>);

struct Fields(String);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write as _;
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }
}

impl tracing::Subscriber for AuditLines {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        if event.metadata().target() == "krishiv::audit" {
            let mut fields = Fields(String::new());
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
    }

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

/// With a link on the path (`<tmp>/wh/o -> <tmp>/outside`, the entry `<tmp>/outside/d`), a denied write
/// to `<tmp>/wh/o/d/x` names the path as written (`..` folded from the text only), never the link's
/// target, and the same text whether `d` exists, is missing, unsearchable, a dangling link or a loop:
/// nothing read from the disk reaches the caller. The policy is asked, and the server-side audit record
/// names, the resolved path.
#[test]
fn a_denial_names_the_path_as_written_and_the_audit_the_resolved_one() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    std::fs::create_dir_all(root.join("wh")).unwrap();
    std::fs::create_dir_all(root.join("outside")).unwrap();
    std::os::unix::fs::symlink(root.join("outside"), root.join("wh/o")).unwrap();
    let audit = AuditLines::default();
    // Nothing else in this binary sets a global subscriber (nextest runs each test in its own process).
    let _ = tracing::subscriber::set_global_default(audit.clone());
    let mut records = Vec::new();
    let answers = each_unresolvable_kind(&root.join("outside"), &root.join("wh/o"), |path| {
        let s = Session::builder()
            .principal(Principal::new("audited", PrincipalKind::User))
            .access_policy(FirstPaths::new(0))
            .build()
            .unwrap();
        let answer = seen(block_on(async {
            s.sql("SELECT 1 AS a").await?.write().parquet(path).await
        }));
        // The checks may run on the runtime's threads: the subscriber is process-wide, and only this
        // test's principal is kept.
        let mine: Vec<String> = std::mem::take(&mut *audit.0.lock().unwrap())
            .into_iter()
            .filter(|r| r.contains("principal=audited"))
            .collect();
        assert!(!mine.is_empty(), "no audit record for {path}");
        records.push(mine.join("\n"));
        answer
    });
    let (_, missing) = &answers[0];
    let written = format!("'file:{}/wh/o/d/x'", root.display());
    assert!(missing.starts_with(DENIED), "{missing}");
    assert!(missing.contains(&written), "{missing}");
    for (kind, answer) in &answers {
        assert_eq!(answer, missing, "{kind}");
        assert!(!answer.contains("outside"), "{kind}: {answer}");
    }
    let (kind, _) = &answers[1];
    assert_eq!(kind, "existing");
    let resolved = format!("'file:{}/outside/d/x'", root.display());
    assert!(records[1].contains(&resolved), "{}", records[1]);
}

/// Swaps `<table>/a=1` for a link to `<tmp>/outside` (which holds a planted `b=2`) the first time the
/// rename's target is asked about: the race between RENAME PARTITION's checks and its move.
struct SwapPartitionParent {
    table: std::path::PathBuf,
    outside: std::path::PathBuf,
    done: std::sync::atomic::AtomicBool,
}

impl AccessPolicy for SwapPartitionParent {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if let Securable::Path(p) = securable
            && p.as_plain().is_some_and(|t| t.ends_with("/a=2/b=2"))
            && !self.done.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            std::fs::rename(self.table.join("a=1"), self.table.with_extension("a1-old")).unwrap();
            std::os::unix::fs::symlink(&self.outside, self.table.join("a=1")).unwrap();
        }
        Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
    }
}

/// A link swapped above the source between RENAME PARTITION's checks and its move: the files behind the
/// link are not moved into the table (they are put back), and the statement fails with 42501.
#[test]
fn a_link_swapped_under_a_partition_rename_moves_nothing_into_the_table() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let wh = root.join("wh");
    let outside = root.join("outside");
    std::fs::create_dir_all(&wh).unwrap();
    std::fs::create_dir_all(outside.join("b=2")).unwrap();
    std::fs::write(outside.join("b=2/planted"), b"x").unwrap();
    let policy = Arc::new(SwapPartitionParent {
        table: wh.join("p"),
        outside: outside.clone(),
        done: std::sync::atomic::AtomicBool::new(false),
    });
    let (owner, s) = sessions(&wh.display().to_string(), policy);
    run(
        &owner,
        "CREATE TABLE p (v INT, a INT, b INT) USING parquet PARTITIONED BY (a, b)",
    )
    .unwrap();
    run(&owner, "ALTER TABLE p ADD PARTITION (a = 1, b = 2)").unwrap();
    let r = run(
        &s,
        "ALTER TABLE p PARTITION (a = 1, b = 2) RENAME TO PARTITION (a = 2, b = 2)",
    );
    assert_eq!(outcome(r), DENIED);
    assert!(
        outside.join("b=2/planted").is_file(),
        "the files behind the link left it"
    );
    assert!(
        !wh.join("p/a=2/b=2/planted").exists(),
        "moved into the table"
    );
}

/// DROP TABLE and RENAME of a managed table under a warehouse reached through a link
/// (`<tmp>/wh/o -> <tmp>/outside`): the denial names the directory as written, never the link's target,
/// and the same for both statements.
#[test]
fn drop_and_rename_denials_name_the_directory_as_written() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    std::fs::create_dir_all(root.join("wh")).unwrap();
    std::fs::create_dir_all(root.join("outside")).unwrap();
    std::os::unix::fs::symlink(root.join("outside"), root.join("wh/o")).unwrap();
    let (owner, s) = sessions(&format!("{}/wh/o", root.display()), FirstPaths::new(0));
    run(&owner, "CREATE TABLE t (a INT) USING parquet").unwrap();
    run(&owner, "INSERT INTO t VALUES (1)").unwrap();
    let written = format!("'file:{}/wh/o/t'", root.display());
    let mut texts = Vec::new();
    for sql in ["DROP TABLE t", "ALTER TABLE t RENAME TO t2"] {
        let text = seen(run(&s, sql));
        assert!(text.starts_with(DENIED), "{sql}: {text}");
        assert!(text.contains(&written), "{sql}: {text}");
        assert!(!text.contains("outside"), "{sql}: {text}");
        texts.push(text);
    }
    assert_eq!(texts[0], texts[1]);
}
