// SPDX-License-Identifier: Apache-2.0
//! The analyze API never executes (design §3.5, M2) and reads DataFrame plans.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, PoisonError};

use arrow_array::{ArrayRef, Int32Array};
use futures::executor::block_on;
use krishiv_types::DataType;
use krishiv_udf_sdk::{
    ArgField, InvokeContext, ReturnField, ScalarUdf, Signature, UdfArg, UdfError, UdfInputType,
};

use crate::{AnalysisWarning, Session, StatementClass};

/// `bump()` / `bump(x)`: counts its calls.
#[derive(Debug)]
struct Bump {
    calls: Arc<AtomicUsize>,
    args: usize,
    name: &'static str,
}

impl ScalarUdf for Bump {
    fn name(&self) -> &str {
        self.name
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        static NONE: OnceLock<Signature> = OnceLock::new();
        static ONE: OnceLock<Signature> = OnceLock::new();
        if self.args == 0 {
            NONE.get_or_init(|| Signature::Exact(Vec::new()))
        } else {
            ONE.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)]))
        }
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: false,
        })
    }

    fn invoke(
        &self,
        _: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(Int32Array::from(vec![1; rows])))
    }
}

fn session(calls: &Arc<AtomicUsize>) -> Session {
    let s = Session::new().unwrap();
    for (name, args) in [("bump", 0), ("bump_of", 1)] {
        s.register_udf(Bump {
            calls: Arc::clone(calls),
            args,
            name,
        })
        .unwrap();
    }
    block_on(s.sql("CREATE TABLE t (a INT, b INT)")).unwrap();
    s
}

/// A recursive CTE, a scalar subquery and scalar UDFs (constant and column arguments): analysis runs none
/// of them, opens no spill directory and runs no query.
#[test]
fn analyze_never_executes() {
    let calls = Arc::new(AtomicUsize::new(0));
    let s = session(&calls);
    let runs_before = s.state.runtime.history.snapshot(None).len();
    for sql in [
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 1000000) \
         SELECT max(n) FROM r",
        "SELECT a, (SELECT max(b) FROM t) AS m FROM t",
        "SELECT bump_of(a) AS x FROM t",
        "SELECT bump() AS y, bump_of(3) AS z",
        "INSERT INTO t SELECT bump_of(a), bump() FROM t",
        "CREATE TABLE u AS SELECT bump_of(a) AS c FROM t",
    ] {
        let a = block_on(s.analyze(sql)).unwrap();
        assert!(!a.partial, "{sql}");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "{sql}");
    }
    let a = block_on(s.analyze("SELECT bump_of(a) AS x FROM t")).unwrap();
    assert!(a.warnings.contains(&AnalysisWarning::LineageOpaqueUdf));
    assert!(
        s.state
            .runtime
            .spill
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_none(),
        "a spill directory was opened"
    );
    assert_eq!(
        s.state.runtime.history.snapshot(None).len(),
        runs_before,
        "a query ran"
    );
    // Nothing was written: `t` is still empty, `u` does not exist.
    let rows = block_on(block_on(s.sql("SELECT count(*) FROM t")).unwrap().collect()).unwrap();
    let count = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 0);
    assert!(block_on(s.sql("SELECT * FROM u")).is_err());
    // A run does call the UDF (the counter works).
    block_on(block_on(s.sql("SELECT bump_of(1)")).unwrap().collect()).unwrap();
    assert!(calls.load(Ordering::SeqCst) > 0);
}

/// `analyze_plan` of a DataFrame is the analysis of the same query in SQL.
#[test]
fn analyze_plan_reads_a_dataframe() {
    let calls = Arc::new(AtomicUsize::new(0));
    let s = session(&calls);
    let sql = "SELECT a, b + 1 AS c FROM t WHERE a > 0";
    let df = block_on(s.sql(sql)).unwrap();
    let a = block_on(s.analyze_plan(&df.plan)).unwrap();
    assert_eq!(a.info.class, StatementClass::Read);
    assert_eq!(a.lineage.len(), 2);
    assert_eq!(a.to_json(), block_on(s.analyze(sql)).unwrap().to_json());
}

/// The JSON's fixed parts: upper-case class names, `null`s, ordinals, and string escaping.
#[test]
fn json_shape() {
    let calls = Arc::new(AtomicUsize::new(0));
    let s = session(&calls);
    let a = block_on(s.analyze("SELECT a, a, 'x\"\\\n' AS `q\"` FROM t ORDER BY b")).unwrap();
    let json = a.to_json();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["statement_class"], "READ");
    assert_eq!(v["explain"], serde_json::Value::Null);
    assert_eq!(v["lineage"][1]["ordinal"], 1);
    assert_eq!(v["lineage"][1]["output"], "a");
    assert_eq!(v["lineage"][2]["transform"], "literal");
    assert_eq!(v["output_schema"][2]["name"], "q\"");
    assert_eq!(v["output_schema"][0]["type_json"], "\"integer\"");
    assert_eq!(v["filters"][0]["kind"], "sort");
    let e = block_on(s.analyze("EXPLAIN ANALYZE SELECT a FROM t")).unwrap();
    assert!(e.to_json().contains("\"explain\":\"ANALYZE\""));
    let e = block_on(s.analyze("EXPLAIN SELECT a FROM t")).unwrap();
    assert!(e.to_json().contains("\"explain\":\"PLAN\""));
    assert!(e.output_schema.is_empty() && e.lineage.is_empty());
    assert_eq!(super::string("\t\r\u{1}"), "\"\\t\\r\\u0001\"");
}

/// Denies everything.
struct DenyAll;

impl krishiv_plan::analyzer::AccessPolicy for DenyAll {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a krishiv_plan::analyzer::Principal,
        _: &'a krishiv_plan::analyzer::Securable,
        _: krishiv_plan::analyzer::Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<
        'a,
        Result<krishiv_plan::analyzer::AccessDecision, krishiv_common::error::KrishivError>,
    > {
        Box::pin(std::future::ready(Ok(
            krishiv_plan::analyzer::AccessDecision::Deny {
                reason: "private".to_owned(),
            },
        )))
    }
}

/// The JSON's `display` of a credentialed URL and of secret references is byte-equal to the securable in
/// the 42501 denial text, and no credential reaches either (ADR 0038, reused: review m8).
#[test]
fn redaction_matches_the_denial_text() {
    use krishiv_common::secret::OptionValue;
    use krishiv_plan::analyzer::{Privilege, Securable, SecurableUse};
    let principal = krishiv_common::Principal::new("ana", krishiv_common::PrincipalKind::User);
    for securable in [
        Securable::Path(OptionValue::from_user(
            "path",
            "s3://AKIDEXAMPLE:wJalrXUtnFEMIK7MDENG@bucket/data/x?X-Amz-Signature=abc123",
        )),
        Securable::Path(OptionValue::reference("s", "k")),
        Securable::Secret {
            scope: "s".to_owned(),
        },
    ] {
        let denial = block_on(krishiv_plan::analyzer::authorize(
            &DenyAll,
            &principal,
            &securable,
            Privilege::Usage,
        ))
        .unwrap_err();
        let KrishivErrorText { securable: denied } = KrishivErrorText::of(&denial);
        let json = super::securable_json(&SecurableUse {
            securable,
            privilege: Privilege::Usage,
            columns: std::collections::BTreeSet::new(),
        });
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["display"].as_str().unwrap(), denied, "{json}");
        assert!(
            !json.contains("wJalrXUtnFEMIK7MDENG") && !json.contains("abc123"),
            "{json}"
        );
        assert!(!denial.to_string().contains("wJalrXUtnFEMIK7MDENG"));
    }
}

/// The `securable` parameter of a 42501 error.
struct KrishivErrorText {
    securable: String,
}

impl KrishivErrorText {
    fn of(e: &krishiv_common::error::KrishivError) -> Self {
        let securable = e
            .message_parameters()
            .iter()
            .find(|(k, _)| *k == "securable")
            .map(|(_, v)| v.clone())
            .unwrap();
        Self { securable }
    }
}
