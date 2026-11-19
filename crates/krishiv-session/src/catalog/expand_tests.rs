// SPDX-License-Identifier: Apache-2.0
//! The view side table: every view inlined is recorded once per expansion and located in the analyzed plan
//! by its (alias, origin).

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::analyzer::views::ViewSecurity;

/// A session with `base (a, b)`, a permanent view `pv` and a temporary view `tv` over it.
fn session() -> crate::Session {
    let s = crate::Session::new().unwrap();
    for sql in [
        "CREATE TABLE base (a INT, b INT)",
        "CREATE VIEW pv AS SELECT a + 1 AS k FROM base WHERE b > 0",
        "CREATE TEMPORARY VIEW tv AS SELECT b FROM base",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    s
}

fn analyzed(df: &crate::DataFrame) -> Arc<AnalyzedPlan> {
    Arc::clone(&block_on(df.analysis()).unwrap().analyzed)
}

/// The side table as (name, alias, security, located).
fn table(a: &AnalyzedPlan) -> Vec<(String, String, ViewSecurity, bool)> {
    let mut v: Vec<_> = a
        .views
        .iter()
        .map(|v| {
            (
                v.name.join("."),
                v.alias.clone(),
                v.security,
                v.output.is_some(),
            )
        })
        .collect();
    v.sort();
    v
}

#[test]
fn views_read_by_a_query_are_recorded_and_located() {
    let s = session();
    let a = analyzed(&block_on(s.sql("SELECT k, b FROM pv, tv")).unwrap());
    assert_eq!(
        table(&a),
        [
            (
                "spark_catalog.default.pv".to_owned(),
                "pv".to_owned(),
                ViewSecurity::Invoker,
                true
            ),
            (
                "tv".to_owned(),
                "tv".to_owned(),
                ViewSecurity::Temporary,
                true
            ),
        ]
    );
    // Both are expanded through: their base tables were checked against the caller.
    assert!(
        a.views
            .iter()
            .all(krishiv_plan::analyzer::views::ViewExpansion::expands)
    );
}

#[test]
fn a_query_without_views_has_an_empty_side_table() {
    let s = session();
    let a = analyzed(&block_on(s.sql("SELECT a FROM base")).unwrap());
    assert!(a.views.is_empty());
}

/// One view read twice: each expansion has its own entry. Two reads are two nodes, each entry located at its
/// own. One read shared by two parents (`df.union(df)`) is inlined twice, with one (alias, origin); the
/// analyzer resolves the two equal subtrees to one node (the second side under a `Shift`), so the key's first
/// entry is located and a consumer matches the node by (alias, origin).
#[test]
fn a_view_read_twice_has_an_entry_per_expansion() {
    let s = session();
    let a = analyzed(&s.table("pv").union(&s.table("pv")));
    assert_eq!(a.views.len(), 2, "{:?}", a.views);
    assert_ne!(a.views[0].origin, a.views[1].origin);
    assert!(a.views.iter().all(|v| v.output.is_some()), "{:?}", a.views);
    assert_ne!(a.views[0].output, a.views[1].output);
    for df in [
        {
            let df = s.table("pv");
            df.union(&df)
        },
        {
            let df = s.table("pv");
            df.cross_join(&df)
        },
    ] {
        let a = analyzed(&df);
        assert_eq!(a.views.len(), 2, "{:?}", a.views);
        assert!(a.views[0].is_at(&a.views[1].alias, a.views[1].origin));
        assert!(a.views[0].output.is_some(), "{:?}", a.views);
    }
}

#[test]
fn a_view_read_in_a_sql_function_body_is_recorded() {
    let s = session();
    block_on(s.sql("CREATE TEMPORARY FUNCTION tf() RETURNS INT RETURN SELECT max(k) FROM pv"))
        .unwrap();
    let a = analyzed(&block_on(s.sql("SELECT tf() AS x")).unwrap());
    assert_eq!(
        table(&a),
        [(
            "spark_catalog.default.pv".to_owned(),
            "pv".to_owned(),
            ViewSecurity::Invoker,
            true
        )]
    );
    // A body called twice is expanded once per query: one entry, located.
    let a = analyzed(&block_on(s.sql("SELECT tf() AS x, tf() + 1 AS y")).unwrap());
    assert_eq!(a.views.len(), 1, "{:?}", a.views);
    assert!(a.views[0].output.is_some());
}

/// A view whose query repeats a column name is inlined with an inner rename `SubqueryAlias` of the same
/// (alias, origin) under the one recorded. One read locates its entry at the outer node; a shared read
/// (`df.union(df)`) locates only the first entry, never the second at the inner rename node.
#[test]
fn a_view_that_repeats_a_column_name_is_located_at_its_outer_node() {
    let s = session();
    block_on(s.sql("CREATE VIEW rv (x, y) AS SELECT a, a FROM base")).unwrap();
    let a = analyzed(&s.table("rv"));
    assert_eq!(a.views.len(), 1, "{:?}", a.views);
    let out: Vec<_> = a.output.iter().map(|c| c.id).collect();
    assert_eq!(a.views[0].output.as_deref(), Some(out.as_slice()));
    let df = s.table("rv");
    let a = analyzed(&df.union(&df));
    assert_eq!(a.views.len(), 2, "{:?}", a.views);
    assert!(a.views[0].output.is_some(), "{:?}", a.views);
    assert_eq!(a.views[1].output, None, "{:?}", a.views);
}
