// SPDX-License-Identifier: Apache-2.0
//! Column lineage (`krishiv_plan::analyzer::lineage`, P3-T063 unit N1a) of plans the SQL front end
//! lowers and the analyzer resolves: one test per row of the design's §3.1 table, and a property over
//! generated queries.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::lineage::{FilterKind, Leaf, Lineage, LineageWarning, lineage};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::views::{ViewExpansion, ViewSecurity, locate_views};
use krishiv_plan::analyzer::{
    AnalysisContext, AnalyzedPlan, Analyzer as _, BoxFuture, BuiltinFunctions, DefaultAnalyzer,
    FileSchemaResolver, FunctionLookup, FunctionRegistry, MemoryCatalog, ResolutionConf,
    SourceKind, SparkProfile, TableColumn,
};
use krishiv_plan::{
    Expr, FunctionId, JoinCondition, JoinKind, LogicalPlan, Plan, PlanNode, SourceId,
};
use krishiv_sql::{ParserOptions, parse_and_lower};
use proptest::prelude::*;

fn col(name: &str, t: &str) -> TableColumn {
    TableColumn {
        name: name.to_owned(),
        data_type: krishiv_types::parse_data_type(t).unwrap(),
        nullable: true,
        char_varchar: None,
    }
}

/// `t(a, b, c, arr)` and `u(a, x)` in `spark_catalog.default`.
fn catalog() -> MemoryCatalog {
    MemoryCatalog::new("spark_catalog")
        .with_table(
            &["default"],
            "t",
            vec![
                col("a", "int"),
                col("b", "int"),
                col("c", "string"),
                col("arr", "array<int>"),
            ],
            None,
        )
        .with_table(
            &["default"],
            "u",
            vec![col("a", "int"), col("x", "int")],
            None,
        )
}

/// The built-ins and `my_udf`, a session UDF typed as `abs`.
struct Udfs(BuiltinFunctions);

impl FunctionRegistry for Udfs {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        if let [n] = name
            && n.eq_ignore_ascii_case("my_udf")
            && let FunctionLookup::Found(mut e) = self.0.lookup(&["abs".to_owned()], profile)
        {
            e.id = FunctionId::Udf {
                name: "my_udf".to_owned(),
                version: 1,
            };
            return FunctionLookup::Found(e);
        }
        self.0.lookup(name, profile)
    }

    fn version(&self) -> u64 {
        1
    }
}

/// Parquet files: every path has `p INT, q STRING`.
struct Files;

impl FileSchemaResolver for Files {
    fn source_kind(&self, name: &str) -> SourceKind {
        if name.eq_ignore_ascii_case("parquet") {
            SourceKind::File
        } else {
            SourceKind::Unknown
        }
    }

    fn infer<'a>(
        &'a self,
        _: &'a str,
        _: &'a [OptionValue],
        _: &'a BTreeMap<String, OptionValue>,
    ) -> BoxFuture<'a, Result<Vec<TableColumn>, KrishivError>> {
        Box::pin(std::future::ready(Ok(vec![
            col("p", "int"),
            col("q", "string"),
        ])))
    }
}

fn ctx() -> AnalysisContext {
    let mut ctx = context(catalog(), ResolutionConf::default());
    ctx.functions = Arc::new(Udfs(BuiltinFunctions::new()));
    ctx.files = Some(Arc::new(Files));
    ctx
}

fn analyzed(sql: &str) -> AnalyzedPlan {
    let plan = parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
        .map_err(|e| e.error)
        .unwrap();
    analyzed_plan(&plan)
}

fn analyzed_plan(plan: &Plan) -> AnalyzedPlan {
    block_on(DefaultAnalyzer.analyze(plan, &ctx())).unwrap()
}

fn of(sql: &str) -> Lineage {
    lineage(&analyzed(sql)).unwrap()
}

/// A leaf as `t.a`, `path:<url>.p` or `view:v.a` (the via view as `@v`).
fn leaf(l: &Leaf) -> String {
    match l {
        Leaf::Table { name, column } => format!("{}.{column}", name.last().unwrap()),
        Leaf::Path { url, column } => format!("path:{url}.{column}"),
        Leaf::View { name, column } => format!("view:{}.{column}", name.join(".")),
        _ => unreachable!("a new leaf kind"),
    }
}

/// Per output: `name: sources -> transform`.
fn columns(l: &Lineage) -> Vec<String> {
    l.columns
        .iter()
        .map(|c| {
            let sources: Vec<String> = c
                .sources
                .iter()
                .map(|s| match &s.via_view {
                    Some(v) => format!("{}@{}", leaf(&s.leaf), v.join(".")),
                    None => leaf(&s.leaf),
                })
                .collect();
            format!("{}: [{}] -> {}", c.name, sources.join(", "), c.transform)
        })
        .collect()
}

/// The filters as `kind t.a`.
fn filters(l: &Lineage) -> Vec<String> {
    l.filters
        .iter()
        .map(|(k, l)| format!("{k} {}", leaf(l)))
        .collect()
}

#[test]
fn project_filter_sort_and_case() {
    let l = of(
        "SELECT a, b + 1 AS b1, CASE WHEN c = 'x' THEN a ELSE b END AS k, 1 AS one \
         FROM t WHERE b > 0 ORDER BY a",
    );
    assert_eq!(
        columns(&l),
        [
            "a: [t.a] -> identity",
            "b1: [t.b] -> expression",
            "k: [t.a, t.b, t.c] -> expression",
            "one: [] -> literal",
        ]
    );
    assert_eq!(filters(&l), ["filter t.b", "sort t.a", "conditional t.c"]);
    assert!(l.warnings.is_empty());
}

/// A condition decides the value: it is a source of the output (not only a statement-level filter).
#[test]
fn case_and_if_conditions_are_sources() {
    let l = of(
        "SELECT CASE WHEN b > 100 THEN 'high' ELSE 'low' END AS band,          IF(c IS NULL, 'missing', 'present') AS has_c, if(a > 0, b, 0) AS pick FROM t",
    );
    assert_eq!(
        columns(&l),
        [
            "band: [t.b] -> expression",
            "has_c: [t.c] -> expression",
            "pick: [t.a, t.b] -> expression",
        ]
    );
    assert_eq!(
        filters(&l),
        ["conditional t.a", "conditional t.b", "conditional t.c"]
    );
}

#[test]
fn duplicate_output_names_have_an_entry_each() {
    let l = of("SELECT a, a FROM t");
    assert_eq!(
        columns(&l),
        ["a: [t.a] -> identity", "a: [t.a] -> identity"]
    );
    assert_eq!(
        l.columns.iter().map(|c| c.ordinal).collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn aggregate_keys_stay_identity() {
    let l = of("SELECT a, a + 1 AS a1, sum(b) AS s FROM t GROUP BY a HAVING count(*) > 1");
    assert_eq!(
        columns(&l),
        [
            "a: [t.a] -> identity",
            "a1: [t.a] -> expression",
            "s: [t.b] -> aggregate",
        ]
    );
    assert_eq!(filters(&l), ["group_by t.a"]);
}

#[test]
fn distinct_is_identity() {
    assert_eq!(
        columns(&of("SELECT DISTINCT a FROM t")),
        ["a: [t.a] -> identity"]
    );
}

/// One rule for every set operation: an output column reads both sides' column at its position, `identity`.
#[test]
fn set_operations_read_both_sides() {
    for sql in [
        "SELECT a FROM t INTERSECT SELECT a FROM u",
        "SELECT a FROM t EXCEPT SELECT a FROM u",
        "SELECT a FROM t INTERSECT ALL SELECT a FROM u",
        "SELECT a FROM t EXCEPT ALL SELECT a FROM u",
        "SELECT a FROM t UNION SELECT a FROM u",
    ] {
        let l = of(sql);
        assert_eq!(columns(&l), ["a: [t.a, u.a] -> identity"], "{sql}");
    }
    // Positional over several columns, and the `DISTINCT` rewrite's `join` filters stay; its distinct
    // aggregate groups the merged columns, so both sides show under `group_by` too.
    let l = of("SELECT a, b FROM t INTERSECT SELECT x, a FROM u");
    assert_eq!(
        columns(&l),
        ["a: [t.a, u.x] -> identity", "b: [t.b, u.a] -> identity"]
    );
    assert_eq!(
        filters(&l),
        [
            "join t.a",
            "join t.b",
            "join u.a",
            "join u.x",
            "group_by t.a",
            "group_by t.b",
            "group_by u.a",
            "group_by u.x"
        ]
    );
    // A user's semi join is no set operation: the right side only decides the rows.
    let l = of("SELECT a FROM t LEFT SEMI JOIN u ON t.a = u.a");
    assert_eq!(columns(&l), ["a: [t.a] -> identity"]);
}

#[test]
fn union_is_positional() {
    let l = of("SELECT a, b FROM t UNION ALL SELECT x, 1 FROM u");
    assert_eq!(
        columns(&l),
        ["a: [t.a, u.x] -> identity", "b: [t.b] -> identity"]
    );
}

#[test]
fn rollup_cube_and_values_go_through_expand() {
    let l = of("SELECT a, b, sum(b) AS s, grouping_id() AS g FROM t GROUP BY ROLLUP(a, b)");
    assert_eq!(
        columns(&l),
        [
            "a: [t.a] -> identity",
            "b: [t.b] -> identity",
            "s: [t.b] -> aggregate",
            "g: [] -> literal",
        ]
    );
    assert_eq!(filters(&l), ["group_by t.a", "group_by t.b"]);
    let l = of("SELECT a, count(*) AS n FROM t GROUP BY CUBE(a)");
    assert_eq!(columns(&l), ["a: [t.a] -> identity", "n: [] -> aggregate"]);
    let l = of("SELECT * FROM VALUES (1, 2), (3, 4) AS v(x, y)");
    assert_eq!(columns(&l), ["x: [] -> literal", "y: [] -> literal"]);
}

#[test]
fn generate_explode() {
    let l = of("SELECT a, e FROM t LATERAL VIEW explode(arr) AS e");
    assert_eq!(
        columns(&l),
        ["a: [t.a] -> identity", "e: [t.arr] -> generated"]
    );
    let l = of("SELECT pos, e + b AS v FROM t LATERAL VIEW posexplode(arr) AS pos, e");
    assert_eq!(
        columns(&l),
        ["pos: [t.arr] -> generated", "v: [t.arr, t.b] -> generated"]
    );
}

#[test]
fn range_is_generated() {
    let l = of("SELECT id, id + 1 AS n FROM range(3)");
    assert_eq!(columns(&l), ["id: [] -> generated", "n: [] -> generated"]);
}

#[test]
fn window_outputs_read_args_partition_and_order() {
    let l = of("SELECT a, rank() OVER (PARTITION BY b ORDER BY c) AS r, \
         sum(a) OVER (PARTITION BY b) AS s FROM t");
    assert_eq!(
        columns(&l),
        [
            "a: [t.a] -> identity",
            "r: [t.b, t.c] -> window",
            "s: [t.a, t.b] -> window",
        ]
    );
    assert_eq!(filters(&l), ["window t.b", "window t.c"]);
}

#[test]
fn cte_referenced_twice() {
    let l = of(
        "WITH q AS (SELECT a, b + 1 AS c FROM t) SELECT q.c, q2.a FROM q JOIN q q2 ON q.a = q2.a",
    );
    assert_eq!(
        columns(&l),
        ["c: [t.b] -> expression", "a: [t.a] -> identity"]
    );
    assert_eq!(filters(&l), ["join t.a"]);
}

#[test]
fn recursive_cte_reaches_a_fixed_point() {
    let l = of(
        "WITH RECURSIVE r(n, m) AS (SELECT a, b FROM t UNION ALL SELECT n + 1, n FROM r WHERE n < 5) \
         SELECT n, m FROM r",
    );
    // `m` reads `n` from the second round on, and `n` is `n + 1` from then: only the fixed point sees
    // `t.a` (and the expression) under `m`.
    assert_eq!(
        columns(&l),
        ["n: [t.a] -> expression", "m: [t.a, t.b] -> expression"]
    );
    assert_eq!(filters(&l), ["filter t.a"]);
}

#[test]
fn correlated_in_and_exists_are_join_filters() {
    let l = of(
        "SELECT a FROM t WHERE a IN (SELECT a FROM u WHERE u.x = t.b) \
         AND EXISTS (SELECT 1 FROM u WHERE u.a = t.c)",
    );
    assert_eq!(columns(&l), ["a: [t.a] -> identity"]);
    assert_eq!(
        filters(&l),
        ["join t.a", "join t.b", "join t.c", "join u.a", "join u.x"]
    );
}

#[test]
fn uncorrelated_exists_is_no_constant() {
    let l = of("SELECT a, EXISTS (SELECT 1 FROM u WHERE u.x > 0) AS e FROM t");
    assert_eq!(columns(&l), ["a: [t.a] -> identity", "e: [] -> expression"]);
    assert_eq!(filters(&l), ["filter u.x"]);
}

/// An analyzed plan keeps what the optimizer would fold: lineage reads the analyzed plan, so a filter
/// column the optimizer drops still shows.
#[test]
fn a_folded_condition_still_lists_its_columns() {
    let l = of("SELECT a FROM t WHERE 1 = 1 OR b > 0");
    assert_eq!(filters(&l), ["filter t.b"]);
}

#[test]
fn uncorrelated_subqueries() {
    let l = of("SELECT a, (SELECT max(x) FROM u) AS m FROM t WHERE b IN (SELECT a FROM u)");
    assert_eq!(
        columns(&l),
        ["a: [t.a] -> identity", "m: [u.x] -> aggregate"]
    );
    assert!(
        filters(&l).contains(&"join u.a".to_owned()),
        "{:?}",
        filters(&l)
    );
    assert!(
        filters(&l).contains(&"join t.b".to_owned()),
        "{:?}",
        filters(&l)
    );
}

#[test]
fn correlated_scalar_subquery() {
    let l = of("SELECT a, (SELECT max(x) FROM u WHERE u.a = t.a) AS m FROM t");
    assert_eq!(
        columns(&l),
        ["a: [t.a] -> identity", "m: [u.x] -> aggregate"]
    );
    assert_eq!(filters(&l), ["join t.a", "join u.a", "group_by u.a"]);
}

#[test]
fn lateral_join_reads_outer_columns() {
    let l = of("SELECT t.a, l.y FROM t, LATERAL (SELECT t.b + u.x AS y FROM u WHERE u.a = t.a) l");
    assert_eq!(
        columns(&l),
        ["a: [t.a] -> identity", "y: [t.b, u.x] -> expression"]
    );
    assert!(
        filters(&l).contains(&"join u.a".to_owned()),
        "{:?}",
        filters(&l)
    );
}

#[test]
fn full_join_using_key_reads_both_sides() {
    let l = of("SELECT a, x FROM t FULL JOIN u USING (a)");
    assert_eq!(
        columns(&l),
        ["a: [t.a, u.a] -> expression", "x: [u.x] -> identity"]
    );
    assert_eq!(filters(&l), ["join t.a", "join u.a"]);
}

#[test]
fn higher_order_function_lambda_reads_the_array() {
    let l = of("SELECT transform(arr, v -> v + b) AS r, filter(arr, v -> v > 0) AS f FROM t");
    assert_eq!(
        columns(&l),
        ["r: [t.arr, t.b] -> expression", "f: [t.arr] -> expression"]
    );
}

#[test]
fn file_source_with_metadata() {
    let l = of("SELECT p, _metadata.file_name AS f FROM parquet.`/data/x`");
    assert_eq!(
        columns(&l),
        [
            "p: [path:/data/x.p] -> identity",
            "f: [path:/data/x._metadata.file_name] -> expression",
        ]
    );
}

#[test]
fn file_source_path_is_redacted() {
    let mut plan = LogicalPlan::new(PlanNode::Read(krishiv_plan::ReadSource::Files {
        format: "parquet".to_owned(),
        paths: vec![OptionValue::from_user(
            "path",
            "s3://bucket/x?X-Amz-Signature=sekrit",
        )],
        options: BTreeMap::new(),
    }));
    plan = plan.project(vec![Expr::column(["p"])]);
    let l = lineage(&analyzed_plan(&Plan::try_new(plan).unwrap())).unwrap();
    let text = format!("{l:?}");
    assert!(!text.contains("sekrit"), "{text}");
    assert!(text.contains("redacted"), "{text}");
}

#[test]
fn lineage_opaque_udf_not_dropped() {
    let l = of("SELECT my_udf(a + b) AS u FROM t");
    assert_eq!(columns(&l), ["u: [t.a, t.b] -> opaque"]);
    // An aggregate of a UDF stays opaque: the UDF's step hides the derivation.
    let s = of("SELECT sum(my_udf(a)) AS s FROM t WHERE my_udf(b) > 0");
    assert_eq!(columns(&s), ["s: [t.a] -> opaque"]);
    assert_eq!(filters(&s), ["filter t.b"]);
    assert_eq!(
        l.warnings.iter().copied().collect::<Vec<_>>(),
        [LineageWarning::LineageOpaqueUdf]
    );
    assert_eq!(
        LineageWarning::LineageOpaqueUdf.code(),
        "LINEAGE_OPAQUE_UDF"
    );
}

/// A DataFrame self-join: both sides are one shared subtree, the analyzer gives the right one a `Shift`.
#[test]
fn dataframe_self_join_keeps_its_sides_apart() {
    let df = Arc::new(LogicalPlan::table(["u"]).project(vec![
        Expr::column(["a"]),
        Expr::function("+", vec![Expr::column(["x"]), Expr::column(["x"])]).alias("x2"),
    ]));
    // The right side reads the shared subtree through a filter, whose column is the shifted instance's.
    let right = LogicalPlan::new(PlanNode::Filter {
        input: Arc::clone(&df),
        predicate: Arc::new(Expr::function(
            ">",
            vec![
                Expr::column(["x2"]),
                Expr::literal(krishiv_plan::Literal::Integer(0)),
            ],
        )),
    });
    let join = LogicalPlan::new(PlanNode::Join {
        left: df,
        right: Arc::new(right),
        kind: JoinKind::Cross,
        condition: JoinCondition::None,
    });
    let a = analyzed_plan(&Plan::try_new(join).unwrap());
    let l = lineage(&a).unwrap();
    assert_eq!(
        columns(&l),
        [
            "a: [u.a] -> identity",
            "x2: [u.x] -> expression",
            "a: [u.a] -> identity",
            "x2: [u.x] -> expression",
        ]
    );
    assert_eq!(filters(&l), ["filter u.x"]);
}

/// The resolved `SubqueryAlias v`'s origin (the alias the inlining of a view builds).
fn alias_origin(p: &LogicalPlan, alias: &str) -> Option<krishiv_plan::Origin> {
    if let PlanNode::SubqueryAlias { alias: a, .. } = &p.node
        && a == alias
    {
        return Some(p.origin);
    }
    let mut found = None;
    p.for_each_child(&mut |c| {
        if found.is_none() {
            found = alias_origin(c, alias);
        }
    });
    found
}

fn with_view(sql: &str, security: ViewSecurity) -> AnalyzedPlan {
    let mut a = analyzed(sql);
    let origin = alias_origin(a.plan.root(), "v").unwrap();
    a.views = locate_views(a.plan.root(), vec![view_entry(origin, security)]);
    a
}

/// A side-table entry for `spark_catalog.default.v`, read as `SubqueryAlias v` at `origin`.
fn view_entry(origin: krishiv_plan::Origin, security: ViewSecurity) -> ViewExpansion {
    ViewExpansion::new(
        vec![
            "spark_catalog".to_owned(),
            "default".to_owned(),
            "v".to_owned(),
        ],
        "v".to_owned(),
        origin,
        security,
        true,
    )
}

const VIEW_QUERY: &str =
    "SELECT v.k + 1 AS k1, u.x FROM (SELECT a AS k FROM t WHERE b > 0) v JOIN u ON v.k = u.a";

#[test]
fn lineage_view_expanded() {
    for security in [ViewSecurity::Invoker, ViewSecurity::Temporary] {
        let l = lineage(&with_view(VIEW_QUERY, security)).unwrap();
        assert_eq!(
            columns(&l),
            [
                "k1: [t.a@spark_catalog.default.v] -> expression",
                "x: [u.x] -> identity",
            ]
        );
        assert_eq!(filters(&l), ["filter t.b", "join t.a", "join u.a"]);
        assert!(l.warnings.is_empty());
    }
}

#[test]
fn lineage_definer_view_is_leaf() {
    let a = with_view(VIEW_QUERY, ViewSecurity::Definer);
    let l = lineage(&a).unwrap();
    assert_eq!(
        columns(&l),
        [
            "k1: [view:spark_catalog.default.v.k] -> view",
            "x: [u.x] -> identity",
        ]
    );
    // Nothing under the view shows: its base table is in no source and no filter.
    assert_eq!(
        filters(&l),
        ["join u.a", "join view:spark_catalog.default.v.k"]
    );
    assert!(!format!("{l:?}").contains("\"t\""));
    assert_eq!(
        l.warnings.iter().copied().collect::<Vec<_>>(),
        [LineageWarning::ViewNotExpanded]
    );
    // A view whose bases were not authorized for the caller is a leaf too, whatever its mode.
    let mut b = with_view(VIEW_QUERY, ViewSecurity::Invoker);
    b.views[0].bases_authorized = false;
    assert_eq!(lineage(&b).unwrap().columns, l.columns);
}

#[test]
fn a_view_not_expanded_that_is_not_found_fails_closed() {
    // A `DEFINER` entry whose (alias, origin) no node in the plan carries fails closed, even with stale
    // located ids.
    let mut a = with_view(VIEW_QUERY, ViewSecurity::Definer);
    a.views[0].origin = krishiv_plan::Origin::plan(krishiv_plan::PlanId(999));
    let e = lineage(&a).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    a.views[0].output = None;
    assert_eq!(
        lineage(&a).unwrap_err().error_class(),
        ErrorClass::InternalError
    );
    // An expanded view that is not found only loses its `via_view`.
    a.views[0].security = ViewSecurity::Invoker;
    assert!(lineage(&a).is_ok());
    // A node that is found hides the view whether or not the entry was located.
    let mut b = with_view(VIEW_QUERY, ViewSecurity::Definer);
    let want = lineage(&b).unwrap();
    b.views[0].output = None;
    assert_eq!(lineage(&b).unwrap(), want);
}

/// The node is found by (alias, origin), never by ids: an entry with other ids (a renumbered copy) still
/// hides the view.
#[test]
fn a_definer_view_is_matched_by_alias_and_origin_not_ids() {
    let mut a = with_view(VIEW_QUERY, ViewSecurity::Definer);
    let want = lineage(&a).unwrap();
    a.views[0].output = Some(vec![krishiv_plan::AttrId(123_456)]);
    assert_eq!(lineage(&a).unwrap(), want);
}

/// `df.union(df)` over one view: the shared read is inlined twice, two subtrees with one (alias, origin)
/// and two entries. Both reads are found: tagged `via_view` when expanded, hidden when `DEFINER` (no `t`
/// anywhere).
#[test]
fn a_view_read_twice_through_a_shared_dataframe() {
    let origin = krishiv_plan::Origin::plan(krishiv_plan::PlanId(7));
    let read = || {
        Arc::new(
            LogicalPlan::table(["t"])
                .filter(Expr::function(
                    ">",
                    vec![
                        Expr::column(["b"]),
                        Expr::literal(krishiv_plan::Literal::Integer(0)),
                    ],
                ))
                .project(vec![Expr::column(["a"]).alias("k")])
                .alias("v")
                .with_origin(origin),
        )
    };
    let union = LogicalPlan::new(PlanNode::Union {
        inputs: vec![read(), read()],
        by_name: false,
        allow_missing_columns: false,
    });
    let base = analyzed_plan(&Plan::try_new(union).unwrap());
    for (security, want) in [
        (
            ViewSecurity::Invoker,
            "k: [t.a@spark_catalog.default.v] -> identity",
        ),
        (
            ViewSecurity::Definer,
            "k: [view:spark_catalog.default.v.k] -> view",
        ),
    ] {
        let mut a = base.clone();
        a.views = locate_views(
            a.plan.root(),
            vec![view_entry(origin, security), view_entry(origin, security)],
        );
        // The key's first entry is located (the analyzer may keep the two equal subtrees as one node).
        assert!(a.views[0].output.is_some(), "{:?}", a.views);
        let l = lineage(&a).unwrap();
        assert_eq!(columns(&l), [want], "{security:?}");
        if security == ViewSecurity::Definer {
            assert!(l.filters.is_empty(), "{:?}", filters(&l));
            assert!(!format!("{l:?}").contains("\"t\""));
        } else {
            assert_eq!(filters(&l), ["filter t.b"]);
        }
    }
    // Entries of one key that disagree: the node is hidden.
    let mut a = base;
    a.views = locate_views(
        a.plan.root(),
        vec![
            view_entry(origin, ViewSecurity::Invoker),
            view_entry(origin, ViewSecurity::Definer),
        ],
    );
    assert_eq!(
        columns(&lineage(&a).unwrap()),
        ["k: [view:spark_catalog.default.v.k] -> view"]
    );
}

#[test]
fn vocabulary_names_are_stable() {
    use krishiv_plan::analyzer::lineage::Transform;
    let names: Vec<&str> = [
        Transform::Literal,
        Transform::Identity,
        Transform::Expression,
        Transform::Generated,
        Transform::Window,
        Transform::Aggregate,
        Transform::Opaque,
        Transform::Masked,
        Transform::View,
    ]
    .iter()
    .map(|t| t.name())
    .collect();
    assert_eq!(
        names,
        [
            "literal",
            "identity",
            "expression",
            "generated",
            "window",
            "aggregate",
            "opaque",
            "masked",
            "view"
        ]
    );
    let kinds: Vec<String> = [
        FilterKind::Filter,
        FilterKind::Join,
        FilterKind::GroupBy,
        FilterKind::Sort,
        FilterKind::Window,
        FilterKind::Conditional,
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    assert_eq!(
        kinds,
        [
            "filter",
            "join",
            "group_by",
            "sort",
            "window",
            "conditional"
        ]
    );
    assert_eq!(
        LineageWarning::ViewNotExpanded.to_string(),
        "VIEW_NOT_EXPANDED"
    );
}

/// A generated query over `t` and `u`.
fn query() -> impl Strategy<Value = String> {
    let column = prop::sample::select(vec!["a", "b", "c", "x", "1", "arr"]);
    let item = (
        column.clone(),
        prop::sample::select(vec!["", " + 1", " IS NULL"]),
    )
        .prop_map(|(c, op)| {
            if c == "arr" || c == "c" {
                c.to_owned()
            } else {
                format!("{c}{op}")
            }
        });
    let shape = prop::sample::select(vec![
        "SELECT {items} FROM t, u",
        "SELECT {items} FROM t JOIN u ON t.a = u.a WHERE b > 0",
        "SELECT {items} FROM t JOIN u USING (a) ORDER BY 1",
        "SELECT {items} FROM (SELECT * FROM t, u) s UNION ALL SELECT {items} FROM t, u",
        "WITH q AS (SELECT * FROM t, u) SELECT {items} FROM q",
        "SELECT {items} FROM t, u WHERE EXISTS (SELECT 1 FROM u u2 WHERE u2.a = t.b)",
    ]);
    (shape, prop::collection::vec(item, 1..5)).prop_map(|(shape, items)| {
        let items = items
            .iter()
            .map(|i| {
                i.replace('a', "t.a")
                    .replace("t.arr", "arr")
                    .replace("t.at.a", "t.a")
            })
            .collect::<Vec<_>>()
            .join(", ");
        shape.replace("{items}", &items)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every output ordinal has an entry, and every leaf names a real column (no attribute id leaks).
    #[test]
    fn every_ordinal_has_an_entry_and_no_attr_id_leaks(sql in query()) {
        let plan = parse_and_lower(&sql, &ParserOptions::default(), SourceId(0)).map_err(|e| e.error).unwrap();
        let Ok(a) = block_on(DefaultAnalyzer.analyze(&plan, &ctx())) else {
            return Ok(());
        };
        let l = lineage(&a).unwrap();
        prop_assert_eq!(l.columns.len(), a.output.len());
        for (i, c) in l.columns.iter().enumerate() {
            prop_assert_eq!(c.ordinal, i);
            prop_assert_eq!(&c.name, &a.output[i].name);
        }
        let real = ["t.a", "t.b", "t.c", "t.arr", "u.a", "u.x"];
        let leaves = l
            .columns
            .iter()
            .flat_map(|c| c.sources.iter().map(|s| &s.leaf))
            .chain(l.filters.iter().map(|(_, l)| l));
        for lf in leaves {
            let name = leaf(lf);
            prop_assert!(real.contains(&name.as_str()), "{} in {}", name, sql);
        }
        let text = format!("{l:?}");
        prop_assert!(!text.contains('#'), "{}", text);
    }
}
