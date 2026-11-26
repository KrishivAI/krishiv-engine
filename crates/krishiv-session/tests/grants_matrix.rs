// SPDX-License-Identifier: Apache-2.0
//! P3-T062 unit G4: the no-bypass matrix (design `P3-T062-access-policy-providers.md` §6) under the
//! `grants` provider. A principal with no grant on the table `canary` runs one statement per row
//! (SQL, DataFrame, catalog API, analyze JSON): every row is refused with
//! `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501) and leaves exactly one `krishiv::audit` record. A
//! second table, `vault`, is never written in any statement the principal runs and sits only
//! behind objects it holds no grant on: neither its name nor either table's data shows in any
//! error, plan, listing, JSON or `system.queries` row. Listing rows (run by a principal that may
//! list the namespace) succeed without naming either; a lookup of one hidden object by name
//! answers as for a missing one (review B3). The read rows run again authorised (granted the
//! canary) and revoked; the revoke-after-cache rows, for the reader's and the definer's grants,
//! are in `grants_views.rs`. One test: the audit subscriber is process-wide.
//!
//! Not covered here: the `catalog` and `external` providers (G5, G6), Connect `Relation` kinds the
//! session API does not reach (`cached_remote_relation` is the session relation cache, covered by
//! `grants_views.rs`), and a generator over the IR node kinds (`PlanNode` is `#[non_exhaustive]`
//! outside `krishiv-plan`).
#![cfg(test)]
mod literal_path;
use literal_path::sql_path;

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session};
use tracing::field::{Field as TraceField, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";
/// What no row may show: the never-written table and both tables' data.
const LEAKS: [&str; 3] = ["vault", "424242", "987654"];

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &TraceField, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if event.metadata().target() == "krishiv::audit" {
            let mut fields = Fields(String::new());
            event.record(&mut fields);
            RECORDS.lock().unwrap().push(fields.0);
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

fn take() -> Vec<String> {
    std::mem::take(&mut *RECORDS.lock().unwrap())
}

type Run<'a> = Box<dyn Fn(&Session) -> Result<String, KrishivError> + 'a>;

type Row<'a> = (&'a str, Run<'a>);

fn sql(text: &str) -> Run<'_> {
    Box::new(move |s: &Session| {
        block_on(async {
            let batches = s.sql(text).await?.collect().await?;
            Ok(format!("{batches:?}"))
        })
    })
}

fn collected(df: &krishiv_session::DataFrame) -> Result<String, KrishivError> {
    block_on(df.collect()).map(|b| format!("{b:?}"))
}

fn leaks(text: &str) -> Option<&'static str> {
    LEAKS.into_iter().find(|l| text.contains(l))
}

#[test]
#[expect(clippy::too_many_lines, reason = "one row per matrix cell")]
fn no_row_reads_the_canary_and_none_shows_the_vault() {
    tracing::subscriber::set_global_default(Capture).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let wh = dir.path().join("wh");
    let rt = EngineRuntime::builder()
        .config("spark.sql.warehouse.dir", wh.to_str().unwrap())
        .build()
        .unwrap();
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let session = |name: &str, admin: bool, system: Arc<dyn AccessPolicy>| {
        let policy = Arc::new(GrantsPolicy::new(Arc::clone(&grants), system, admin));
        Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(Principal::new(name, PrincipalKind::User))
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .grants(policy)
            .build()
            .unwrap()
    };
    let viewer = || -> Arc<dyn AccessPolicy> {
        Arc::new(RolePolicy::new(
            Some(Role::Viewer),
            Role::Admin,
            PolicyVersion(1),
        ))
    };
    let root = session("root", true, Arc::new(AllowAll));
    let ana = session("ana", false, viewer());
    let lister = session("lister", false, viewer());
    for statement in [
        "CREATE TABLE canary (a INT, s STRING) USING parquet",
        "INSERT INTO canary VALUES (1, 'canary-987654')",
        "CREATE TABLE vault (a INT, s STRING) USING parquet",
        "INSERT INTO vault VALUES (2, 'vault-424242')",
        "CREATE TABLE pub (a INT) USING parquet",
        "INSERT INTO pub VALUES (1)",
        "GRANT SELECT, MODIFY ON TABLE pub TO user:ana",
        // An invoker view over the canary that ana may read (its text names only the canary).
        "CREATE VIEW cv AS SELECT a FROM canary",
        "GRANT SELECT ON VIEW cv TO user:ana",
        // A definer view over the vault that ana holds nothing on.
        "CREATE VIEW dvc SQL SECURITY DEFINER AS SELECT * FROM vault",
        // A function ana may not call, and one it may call whose body reads the canary.
        "CREATE FUNCTION canary_fn() RETURNS INT RETURN (SELECT max(a) FROM canary)",
        "CREATE FUNCTION ufn() RETURNS INT RETURN (SELECT max(a) FROM canary)",
        "GRANT SELECT ON FUNCTION ufn TO user:ana",
        "GRANT USAGE ON SCHEMA default TO user:lister",
        "GRANT USAGE ON CATALOG spark_catalog TO user:lister",
        // A namespace lister holds nothing on (review B3).
        "CREATE NAMESPACE vaultns",
        "CREATE TABLE vaultns.vt (a INT) USING parquet",
        "CREATE FUNCTION vaultns.f() RETURNS INT RETURN 1",
        // A definer view ana may read over `pub` (review B1).
        "CREATE VIEW dpub SQL SECURITY DEFINER AS SELECT a FROM pub",
        "GRANT SELECT ON VIEW dpub TO user:ana",
        // Definers that lose the canary under views ana may read (review B2): dana's view over
        // the canary, and dora's view over dana's.
        "GRANT CREATE ON SCHEMA default TO user:dana",
        "GRANT CREATE ON SCHEMA default TO user:dora",
        "GRANT SELECT ON TABLE canary TO user:dana",
    ] {
        block_on(async { root.sql(statement).await?.collect().await })
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }
    let dana = session("dana", false, Arc::new(AllowAll));
    let dora = session("dora", false, Arc::new(AllowAll));
    for (s, statement) in [
        (
            &dana,
            "CREATE VIEW dcan SQL SECURITY DEFINER AS SELECT a FROM canary WHERE a > 0",
        ),
        (&dana, "GRANT SELECT ON VIEW dcan TO user:ana"),
        (&dana, "GRANT SELECT ON VIEW dcan TO user:dora"),
        (
            &dora,
            "CREATE VIEW dnest SQL SECURITY DEFINER AS SELECT a FROM dcan",
        ),
        (&dora, "GRANT SELECT ON VIEW dnest TO user:ana"),
        (&root, "REVOKE SELECT ON TABLE canary FROM user:dana"),
    ] {
        block_on(async { s.sql(statement).await?.collect().await })
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }
    let canary_dir = wh.join("canary");
    let canary_path = canary_dir.to_str().unwrap().to_owned();
    let path_read = format!("SELECT * FROM parquet.`{canary_path}`");
    let directory = format!(
        "INSERT OVERWRITE DIRECTORY '{}' USING parquet SELECT * FROM canary",
        sql_path(&dir.path().join("out"))
    );
    // design §6: one row per cell (the unauthorised case of the `grants` column).
    let mut refused: Vec<Row<'_>> = vec![
        ("SELECT", sql("SELECT * FROM canary")),
        ("SELECT a column", sql("SELECT a FROM canary WHERE a = 1")),
        ("SELECT count", sql("SELECT count(*) FROM canary")),
        ("INSERT", sql("INSERT INTO canary VALUES (2, 'x')")),
        ("INSERT SELECT", sql("INSERT INTO pub SELECT a FROM canary")),
        (
            "MERGE",
            sql("MERGE INTO pub USING canary ON pub.a = canary.a WHEN MATCHED THEN DELETE"),
        ),
        ("UPDATE", sql("UPDATE canary SET a = 1")),
        ("DELETE", sql("DELETE FROM canary WHERE a = 1")),
        ("TRUNCATE", sql("TRUNCATE TABLE canary")),
        ("CREATE AS", sql("CREATE TABLE c2 AS SELECT * FROM canary")),
        ("CREATE LIKE", sql("CREATE TABLE c3 LIKE canary")),
        ("CREATE VIEW", sql("CREATE VIEW v2 AS SELECT * FROM canary")),
        ("DROP", sql("DROP TABLE canary")),
        ("ALTER", sql("ALTER TABLE canary ADD COLUMNS (z INT)")),
        ("ALTER RENAME", sql("ALTER TABLE canary RENAME TO c4")),
        ("COPY (directory)", sql(&directory)),
        ("CACHE", sql("CACHE TABLE canary")),
        ("CACHE AS", sql("CACHE TABLE cc AS SELECT * FROM canary")),
        ("UNCACHE", sql("UNCACHE TABLE canary")),
        ("ANALYZE", sql("ANALYZE TABLE canary COMPUTE STATISTICS")),
        ("DESCRIBE", sql("DESCRIBE canary")),
        ("DESCRIBE EXTENDED", sql("DESCRIBE EXTENDED canary")),
        ("SHOW COLUMNS", sql("SHOW COLUMNS IN canary")),
        ("SHOW CREATE TABLE", sql("SHOW CREATE TABLE canary")),
        ("SHOW TBLPROPERTIES", sql("SHOW TBLPROPERTIES canary")),
        (
            "information_schema",
            sql("SELECT * FROM information_schema.tables"),
        ),
        (
            "SET a server-scoped store key",
            sql("SET fs.s3a.endpoint = http://127.0.0.1:1"),
        ),
        (
            "EXECUTE IMMEDIATE",
            sql("EXECUTE IMMEDIATE 'SELECT * FROM canary'"),
        ),
        (
            "EXECUTE IMMEDIATE USING",
            sql("EXECUTE IMMEDIATE 'SELECT * FROM canary WHERE a = ?' USING 1"),
        ),
        ("scripting", sql("BEGIN SELECT * FROM canary; END")),
        ("via invoker view", sql("SELECT * FROM cv")),
        ("via definer view", sql("SELECT * FROM dvc")),
        ("via SQL UDF (no grant)", sql("SELECT canary_fn()")),
        ("via SQL UDF body", sql("SELECT ufn()")),
        (
            "via CTE",
            sql("WITH c AS (SELECT * FROM canary) SELECT * FROM c"),
        ),
        (
            "via IN",
            sql("SELECT * FROM pub WHERE a IN (SELECT a FROM canary)"),
        ),
        (
            "via EXISTS",
            sql("SELECT * FROM pub WHERE EXISTS (SELECT 1 FROM canary WHERE canary.a = pub.a)"),
        ),
        (
            "via scalar subquery",
            sql("SELECT (SELECT max(a) FROM canary) FROM pub"),
        ),
        (
            "via set operation",
            sql("SELECT a FROM pub UNION SELECT a FROM canary"),
        ),
        ("via join", sql("SELECT * FROM pub JOIN canary USING (a)")),
        (
            "via time travel",
            sql("SELECT * FROM canary VERSION AS OF 1"),
        ),
        (
            "via time travel on a hidden view",
            sql("SELECT * FROM dvc VERSION AS OF 1"),
        ),
        (
            "via time travel on a hidden view, suffix",
            sql("SELECT * FROM dvc@v1"),
        ),
        ("via path", sql(&path_read)),
        (
            "via temporary view",
            sql("CREATE TEMPORARY VIEW tv AS SELECT * FROM canary"),
        ),
        (
            "DataFrame table",
            Box::new(|s: &Session| collected(&s.table("canary"))),
        ),
        (
            "DataFrame alias",
            Box::new(|s: &Session| collected(&s.table("canary").alias("x").select(["a"]))),
        ),
        (
            "DataFrame read",
            Box::new(|s: &Session| collected(&s.read().parquet([canary_path.as_str()]))),
        ),
        (
            "DataFrame schema",
            Box::new(|s: &Session| {
                block_on(s.table("canary").schema()).map(|schema| format!("{schema:?}"))
            }),
        ),
        (
            "DataFrame cache",
            Box::new(|s: &Session| collected(&s.table("canary").cache())),
        ),
        (
            "catalog get_table",
            Box::new(|s: &Session| {
                block_on(s.catalog().get_table("canary")).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "catalog list_columns",
            Box::new(|s: &Session| {
                block_on(s.catalog().list_columns("canary")).map(|c| format!("{c:?}"))
            }),
        ),
        (
            "catalog cache_table",
            Box::new(|s: &Session| {
                block_on(s.catalog().cache_table("canary")).map(|()| String::new())
            }),
        ),
        (
            "analyze JSON",
            Box::new(|s: &Session| {
                block_on(s.analyze("SELECT * FROM pub WHERE a IN (SELECT a FROM canary)"))
                    .map(|a| a.to_json())
            }),
        ),
        (
            "analyze JSON via definer view",
            Box::new(|s: &Session| block_on(s.analyze("SELECT * FROM dvc")).map(|a| a.to_json())),
        ),
        (
            "statement_info",
            Box::new(|s: &Session| {
                block_on(s.statement_info("SELECT * FROM cv")).map(|i| format!("{i:?}"))
            }),
        ),
    ];
    // Review B2: a definer that lost the canary, read through a view ana may read (alone and
    // nested): neither definer, view nor table behind it shows.
    refused.push(("B2 definer lost the canary", sql("SELECT * FROM dcan")));
    refused.push((
        "B2 nested definer lost the canary",
        sql("SELECT * FROM dnest"),
    ));
    take();
    let mut bad = Vec::new();
    for (row, run) in &refused {
        let result = run(&ana);
        let records = take();
        match &result {
            Err(e) if e.error_class().name() == DENIED => {
                if e.sql_state() != Some("42501") {
                    bad.push(format!("{row}: SQLSTATE {:?}", e.sql_state()));
                }
                if let Some(l) = leaks(&e.to_string()) {
                    bad.push(format!("{row}: the error shows {l}: {e}"));
                }
                if row.starts_with("B2")
                    && let Some(l) = ["dana", "dora", "dcan", "canary", "> 0"]
                        .into_iter()
                        .find(|l| e.to_string().contains(l))
                {
                    bad.push(format!("{row}: the error shows {l}: {e}"));
                }
            }
            Err(e) => bad.push(format!("{row}: not a denial: {e}")),
            Ok(text) => bad.push(format!("{row}: allowed: {text}")),
        }
        if records.len() != 1 {
            bad.push(format!(
                "{row}: {} audit records: {records:?}",
                records.len()
            ));
        }
        if let Some(l) = records.iter().find_map(|r| leaks(r)) {
            bad.push(format!("{row}: the audit record shows {l}"));
        }
    }
    // `EXPLAIN` describes a refused analysis in its text, as Spark describes a failed one: the
    // text is the refusal, with no plan and no data.
    for statement in [
        "EXPLAIN SELECT * FROM canary",
        "EXPLAIN EXTENDED SELECT * FROM canary",
        "EXPLAIN FORMATTED SELECT * FROM pub WHERE a IN (SELECT a FROM canary)",
        "EXPLAIN SELECT * FROM dvc",
        "EXPLAIN EXTENDED SELECT * FROM cv",
        "EXPLAIN COST SELECT * FROM canary",
    ] {
        let result = sql(statement)(&ana);
        let records = take();
        match &result {
            // A metadata explain (P3 PR-14d) that cannot show a withheld plan prints the
            // planning failure with its message redacted.
            Ok(text)
                if !text.contains("Scan")
                    && (text.contains(DENIED)
                        || text.contains("Error occurred during query planning")) => {}
            Err(e) if e.error_class().name() == DENIED => {}
            other => bad.push(format!("{statement}: not refused: {other:?}")),
        }
        let text = match &result {
            Ok(t) => t.clone(),
            Err(e) => e.to_string(),
        };
        if let Some(l) = leaks(&text) {
            bad.push(format!("{statement}: shows {l}: {text}"));
        }
        if records.len() > 1 {
            bad.push(format!("{statement}: {} audit records", records.len()));
        }
    }
    // Listings and the principal's own history: allowed, and they name neither table nor their
    // data (`lister` may list `default`, holding no grant on anything in it).
    let listings: Vec<Row<'_>> = vec![
        ("SHOW TABLES", sql("SHOW TABLES")),
        ("SHOW VIEWS", sql("SHOW VIEWS")),
        ("SHOW TABLES LIKE", sql("SHOW TABLES LIKE 'c*'")),
        ("SHOW NAMESPACES", sql("SHOW NAMESPACES")),
        (
            "catalog list_databases",
            Box::new(|s: &Session| {
                block_on(s.catalog().list_databases(None)).map(|d| format!("{d:?}"))
            }),
        ),
        (
            "catalog list_tables",
            Box::new(|s: &Session| {
                block_on(s.catalog().list_tables(None, None)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "catalog list_functions",
            Box::new(|s: &Session| {
                block_on(s.catalog().list_functions(None)).map(|f| format!("{f:?}"))
            }),
        ),
        (
            "system.queries",
            sql("SELECT query_text FROM system.queries"),
        ),
    ];
    for (row, run) in &listings {
        let result = run(&lister);
        let records = take();
        match &result {
            Ok(text) => {
                let shown = leaks(text).or_else(|| {
                    ["canary", "dvc", "canary_fn", "ufn", "dcan", "dnest", "dpub"]
                        .into_iter()
                        .find(|n| text.contains(n))
                });
                if let Some(l) = shown {
                    bad.push(format!("{row}: lists {l}: {text}"));
                }
                if !records.is_empty() {
                    bad.push(format!("{row}: {} audit records", records.len()));
                }
            }
            Err(e) => bad.push(format!("{row}: {e}")),
        }
    }
    // Review B1: Spark Connect's `Relation.sql` (embedded text, numbered by Bind from 0 in plan
    // order) beside a definer view ana may read never borrows its definer's rights, whichever
    // number the canary's text gets (a fresh session numbers its view texts from 1): each plan is
    // refused, with one audit record.
    #[cfg(feature = "unstable-ir")]
    for n in 0..8 {
        use krishiv_plan::{LogicalPlan, Plan, PlanNode, SqlArgs};
        let sql_node = |query: &str| {
            Arc::new(LogicalPlan::new(PlanNode::Sql {
                query: query.to_owned(),
                args: SqlArgs::None,
            }))
        };
        let mut inputs = vec![Arc::new(LogicalPlan::table(["dpub"]))];
        inputs.extend((0..n).map(|_| sql_node("SELECT 1 AS a")));
        inputs.push(sql_node("SELECT a FROM canary"));
        let plan = Plan::try_new(Arc::new(LogicalPlan::new(PlanNode::Union {
            inputs,
            by_name: false,
            allow_missing_columns: false,
        })))
        .unwrap();
        let fresh = session("ana", false, viewer());
        let result = collected(&fresh.from_plan(&plan));
        let records = take();
        match &result {
            Err(e) if e.error_class().name() == DENIED && records.len() == 1 => {}
            other => bad.push(format!(
                "B1 Connect sql {n}: {other:?}, {} audit records",
                records.len()
            )),
        }
    }
    // Review B3: one object looked up by name reads for `lister`, who may not list it, exactly as
    // a missing one (class, text with the name swapped, audit records).
    type Look<'a> = Box<dyn Fn(&Session, &str) -> Result<String, KrishivError> + 'a>;
    type Lookup<'a> = (&'a str, &'a str, Look<'a>);
    let statement = |template: &'static str| -> Look<'static> {
        Box::new(move |s: &Session, name: &str| sql(&template.replace("{}", name))(s))
    };
    let lookups: Vec<Lookup<'_>> = vec![
        ("canary", "nosuch", statement("DESCRIBE {}")),
        ("canary", "nosuch", statement("DESCRIBE EXTENDED {}")),
        ("canary", "nosuch", statement("SHOW COLUMNS IN {}")),
        ("canary", "nosuch", statement("SHOW CREATE TABLE {}")),
        ("canary", "nosuch", statement("SHOW TBLPROPERTIES {}")),
        ("dvc", "nosuch", statement("DESCRIBE {}")),
        ("dvc", "nosuch", statement("SHOW CREATE TABLE {}")),
        (
            "canary",
            "nosuch",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().get_table(n)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "canary",
            "nosuch",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().table_exists(n)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "canary",
            "nosuch",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().list_columns(n)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "default.canary_fn",
            "default.nosuch_fn",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().get_function(n)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "canary_fn",
            "nosuch_fn",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().function_exists(n)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().get_database(n)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().database_exists(n)).map(|t| format!("{t:?}"))
            }),
        ),
        // Review r2 R4: the SQL namespace lookups, by `database_exists`'s rule.
        ("vaultns", "nosuchns", statement("DESCRIBE NAMESPACE {}")),
        ("vaultns", "nosuchns", statement("DESCRIBE DATABASE {}")),
        (
            "vaultns",
            "nosuchns",
            statement("DESCRIBE SCHEMA EXTENDED {}"),
        ),
        ("vaultns", "nosuchns", statement("SHOW TABLES IN {}")),
        (
            "vaultns",
            "nosuchns",
            statement("SHOW TABLES FROM {} LIKE 'v*'"),
        ),
        ("vaultns", "nosuchns", statement("SHOW VIEWS IN {}")),
        ("vaultns", "nosuchns", statement("SHOW FUNCTIONS IN {}")),
        (
            "vaultns",
            "nosuchns",
            statement("SHOW USER FUNCTIONS IN {}"),
        ),
        (
            "vaultns",
            "nosuchns",
            statement("SHOW TABLE EXTENDED IN {} LIKE '*'"),
        ),
        ("vaultns", "nosuchns", statement("SHOW CREATE TABLE {}.vt")),
        ("vaultns", "nosuchns", statement("DESCRIBE {}.vt")),
        ("vaultns", "nosuchns", statement("SHOW COLUMNS IN vt IN {}")),
        ("vaultns", "nosuchns", statement("DESCRIBE FUNCTION {}.f")),
        // Review r3 X1: the rest of the by-namespace lookups, SQL and `spark.catalog`.
        ("vaultns", "nosuchns", statement("DESCRIBE EXTENDED {}.vt")),
        ("vaultns", "nosuchns", statement("SHOW TBLPROPERTIES {}.vt")),
        ("vaultns", "nosuchns", statement("SHOW COLUMNS IN {}.vt")),
        (
            "vaultns",
            "nosuchns",
            statement("DESCRIBE FUNCTION EXTENDED {}.f"),
        ),
        (
            "vaultns",
            "nosuchns",
            statement("SHOW TABLES IN spark_catalog.{}"),
        ),
        (
            "vaultns",
            "nosuchns",
            statement("SHOW VIEWS FROM {} LIKE '*'"),
        ),
        (
            "vaultns",
            "nosuchns",
            statement("SHOW FUNCTIONS FROM {} LIKE '*'"),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().list_tables(Some(n), None)).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(
                    s.catalog()
                        .list_tables(Some(&format!("spark_catalog.{n}")), Some("v*")),
                )
                .map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().get_table(&format!("{n}.vt"))).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().table_exists(&format!("{n}.vt"))).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().list_columns(&format!("{n}.vt"))).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().get_function(&format!("{n}.f"))).map(|t| format!("{t:?}"))
            }),
        ),
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().function_exists(&format!("{n}.f"))).map(|t| format!("{t:?}"))
            }),
        ),
        // Last: a `USE` that succeeds moves the session.
        (
            "vaultns",
            "nosuchns",
            Box::new(|s: &Session, n: &str| {
                block_on(s.catalog().set_current_database(n)).map(|t| format!("{t:?}"))
            }),
        ),
        ("vaultns", "nosuchns", statement("USE NAMESPACE {}")),
        ("vaultns", "nosuchns", statement("USE {}")),
    ];
    // The name's last part, wherever the text quotes it.
    let answer = |r: Result<String, KrishivError>, name: &str| {
        let last = name.rsplit('.').next().unwrap_or(name);
        match r {
            Ok(text) => format!("ok {}", text.replace(last, "<name>")),
            Err(e) => format!(
                "{} {}",
                e.error_class().name(),
                e.to_string().replace(last, "<name>")
            ),
        }
    };
    take();
    for (hidden, missing, look) in &lookups {
        let shown = answer(look(&lister, hidden), hidden);
        let shown_records = take().len();
        let absent = answer(look(&lister, missing), missing);
        let absent_records = take().len();
        if shown != absent || shown_records != absent_records {
            bad.push(format!(
                "lookup of {hidden}: {shown} ({shown_records} records) vs {missing}: {absent} \
                 ({absent_records} records)"
            ));
        }
    }
    sql("USE default")(&lister).unwrap();
    take();
    // Authorised, then revoked (design §6: each cell in all three states): granted the canary, ana
    // reads it through every read shape; revoked, each is refused again, however much of the
    // first answer a cache kept.
    let reads = [
        "SELECT",
        "SELECT a column",
        "SELECT count",
        "DESCRIBE",
        "SHOW COLUMNS",
        "via CTE",
        "via IN",
        "via scalar subquery",
        "via set operation",
        "via join",
        "DataFrame table",
        "DataFrame alias",
        "DataFrame schema",
        "catalog get_table",
        "catalog list_columns",
        "analyze JSON",
    ];
    block_on(async {
        root.sql("GRANT SELECT ON TABLE canary TO user:ana")
            .await?
            .collect()
            .await
    })
    .unwrap();
    for (row, run) in refused.iter().filter(|(r, _)| reads.contains(r)) {
        if let Err(e) = run(&ana) {
            bad.push(format!("{row}: refused once granted: {e}"));
        }
    }
    block_on(async {
        root.sql("REVOKE SELECT ON TABLE canary FROM user:ana")
            .await?
            .collect()
            .await
    })
    .unwrap();
    take();
    for (row, run) in refused.iter().filter(|(r, _)| reads.contains(r)) {
        match run(&ana) {
            Err(e) if e.error_class().name() == DENIED => {}
            other => bad.push(format!("{row}: after the revoke: {other:?}")),
        }
    }
    take();
    // ana's own history holds only ana's statements: root's (which name the vault) are not shown.
    if let Ok(text) = sql("SELECT query_text FROM system.queries")(&ana)
        && let Some(l) = leaks(&text)
    {
        bad.push(format!("ana's system.queries shows {l}"));
    }
    take();
    assert!(bad.is_empty(), "{bad:#?}");
}
