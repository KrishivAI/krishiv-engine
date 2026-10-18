// SPDX-License-Identifier: Apache-2.0
//! Reads by path through the `FileSchemaResolver` hook (P2-T154): the
//! DataFrame `read.parquet(path)` form and SQL's `` parquet.`path` ``,
//! checked against Spark 4.2.0 runs (column nullability, qualifiers, error
//! classes and which name wins).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use krishiv_common::error::{ErrorClass, KrishivError, PathNotFound, UnableToInferSchema};
use krishiv_common::secret::OptionValue;
use krishiv_types::DataType;

use super::dsl::c;
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AccessDecision, AccessPolicy, AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _,
    BoxFuture, DefaultAnalyzer, FileSchemaResolver, MemoryCatalog, PolicyVersion, Principal,
    Privilege, ResolutionConf, Securable, SourceKind, TableColumn,
};
use crate::{Expr, LogicalPlan, Origin, Plan, PlanId, PlanNode, ReadSource};

/// Files under `/data`: `/data/one` has `a INT NOT NULL, B STRING`;
/// anything else is missing. Counts its inferences.
#[derive(Default)]
struct Files {
    calls: Mutex<Vec<String>>,
}

impl FileSchemaResolver for Files {
    /// Relative paths live under `/data`.
    fn qualify(&self, path: &OptionValue) -> Result<OptionValue, KrishivError> {
        let text = path.to_string();
        if text == "bad" {
            return Err(UnableToInferSchema {
                format: "bad".to_owned(),
            }
            .into());
        }
        Ok(if text.starts_with('/') {
            path.clone()
        } else {
            OptionValue::from_user("path", format!("/data/{text}"))
        })
    }

    fn source_kind(&self, name: &str) -> SourceKind {
        match name.to_ascii_lowercase().as_str() {
            "parquet" | "json" => SourceKind::File,
            "jdbc" => SourceKind::Other,
            _ => SourceKind::Unknown,
        }
    }

    fn infer<'a>(
        &'a self,
        format: &'a str,
        paths: &'a [OptionValue],
        _: &'a BTreeMap<String, OptionValue>,
    ) -> BoxFuture<'a, Result<Vec<TableColumn>, KrishivError>> {
        let path = paths.first().map(ToString::to_string).unwrap_or_default();
        self.calls.lock().unwrap().push(format!("{format}:{path}"));
        let result = if paths.is_empty() {
            Err(UnableToInferSchema {
                format: "Parquet".to_owned(),
            }
            .into())
        } else if path == "/data/one" {
            Ok(vec![
                TableColumn {
                    name: "a".to_owned(),
                    data_type: DataType::Integer,
                    nullable: false,
                },
                TableColumn {
                    name: "B".to_owned(),
                    data_type: DataType::String,
                    nullable: true,
                },
            ])
        } else {
            Err(PathNotFound {
                path: format!("file:{path}"),
            }
            .into())
        };
        Box::pin(std::future::ready(result))
    }
}

fn catalog() -> MemoryCatalog {
    let col = |n: &str| TableColumn {
        name: n.to_owned(),
        data_type: DataType::Long,
        nullable: false,
    };
    MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "t", vec![col("x")], None)
        .with_table(&["parquet"], "tt", vec![col("q")], None)
}

fn ctx_with(files: Option<Arc<Files>>, conf: ResolutionConf) -> AnalysisContext {
    let mut ctx = context(catalog(), conf);
    ctx.files = files.map(|f| f as Arc<dyn FileSchemaResolver>);
    ctx
}

fn ctx(files: &Arc<Files>) -> AnalysisContext {
    ctx_with(Some(Arc::clone(files)), ResolutionConf::default())
}

fn read(format: &str, paths: &[&str]) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: format.to_owned(),
        paths: paths
            .iter()
            .map(|p| OptionValue::from_user("path", *p))
            .collect(),
        options: BTreeMap::new(),
    }))
}

fn run(plan: LogicalPlan, ctx: &AnalysisContext) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, ctx))
}

fn class(r: &Result<AnalyzedPlan, AnalysisError>) -> ErrorClass {
    r.as_ref().unwrap_err().error.error_class()
}

/// Output names, types and nullability.
fn output(r: &Result<AnalyzedPlan, AnalysisError>) -> Vec<(String, DataType, bool)> {
    let a = r.as_ref().unwrap();
    a.output
        .iter()
        .map(|c| (c.name.clone(), c.data_type.clone(), c.nullable))
        .collect()
}

fn nullable_one() -> Vec<(String, DataType, bool)> {
    vec![
        ("a".to_owned(), DataType::Integer, true),
        ("B".to_owned(), DataType::String, true),
    ]
}

#[test]
fn reads_by_path_resolve_to_nullable_file_scans() {
    let files = Arc::new(Files::default());
    let r = run(read("parquet", &["/data/one"]), &ctx(&files));
    // Spark: struct<a:int,B:string>, every column nullable.
    assert_eq!(output(&r), nullable_one());
    let a = r.unwrap();
    assert!(a.reads_files);
    assert!(a.context.files);
    let PlanNode::Scan(scan) = &a.plan.root().node else {
        panic!("not a scan: {}", a.plan.root());
    };
    assert!(matches!(&scan.source, ReadSource::Files { format, .. } if format == "parquet"));
    assert_eq!(scan.version, None);
    // A path is not a qualifier; an alias is.
    let qualified = |q: &str| Expr::column([q, "a"]);
    let r = run(
        read("parquet", &["/data/one"]).project(vec![qualified("/data/one")]),
        &ctx(&files),
    );
    assert_eq!(class(&r), ErrorClass::UnresolvedColumnWithSuggestion);
    let r = run(
        read("parquet", &["/data/one"])
            .alias("t")
            .project(vec![qualified("t")]),
        &ctx(&files),
    );
    assert_eq!(output(&r)[0].0, "a");
}

#[test]
fn sql_on_files_falls_back_from_the_catalog() {
    let files = Arc::new(Files::default());
    let t = |parts: &[&str]| LogicalPlan::table(parts.iter().copied());
    // Upper-case format names read files too; the format is kept lower-case.
    let r = run(t(&["PARQUET", "/data/one"]), &ctx(&files));
    assert_eq!(output(&r), nullable_one());
    let a = r.unwrap();
    let PlanNode::Scan(scan) = &a.plan.root().node else {
        panic!("not a scan")
    };
    assert!(matches!(&scan.source, ReadSource::Files { format, .. } if format == "parquet"));
    // The format name qualifies nothing either.
    let r = run(
        t(&["parquet", "/data/one"]).project(vec![c("parquet.a")]),
        &ctx(&files),
    );
    assert_eq!(class(&r), ErrorClass::UnresolvedColumnWithSuggestion);
    // A table of a namespace named like a format wins over the path.
    let r = run(t(&["parquet", "tt"]), &ctx(&files));
    assert_eq!(output(&r)[0].0, "q");
    // A missing path of a file format: PATH_NOT_FOUND (also for JSON).
    let r = run(t(&["json", "/data/nope"]), &ctx(&files));
    assert_eq!(class(&r), ErrorClass::PathNotFound);
    // Not a data source: the table is missing, with its full name.
    let r = run(t(&["foo", "/data/one"]), &ctx(&files));
    assert_eq!(class(&r), ErrorClass::TableOrViewNotFound);
    assert_eq!(
        r.unwrap_err().error.message_parameters()[0].1,
        "`foo`.`/data/one`"
    );
    // A data source that reads no files.
    let r = run(t(&["jdbc", "/data/one"]), &ctx(&files));
    assert_eq!(class(&r), ErrorClass::UnsupportedDatasourceForDirectQuery);
    assert_eq!(
        r.unwrap_err().error.message_parameters(),
        [("dataSourceType", "jdbc".to_owned())]
    );
    // Three parts are always a table name.
    let r = run(t(&["spark_catalog", "parquet", "/data/one"]), &ctx(&files));
    assert_eq!(class(&r), ErrorClass::TableOrViewNotFound);
    // `spark.sql.runSQLOnFiles` off: a table name only.
    let off = ResolutionConf {
        run_sql_on_files: false,
        ..ResolutionConf::default()
    };
    let r = run(
        t(&["parquet", "/data/one"]),
        &ctx_with(Some(Arc::clone(&files)), off),
    );
    assert_eq!(class(&r), ErrorClass::TableOrViewNotFound);
    assert!(!r.is_ok_and(|a| a.reads_files));
}

#[test]
fn resolver_errors_carry_the_read() {
    let files = Arc::new(Files::default());
    let origin = Origin::plan(PlanId(4));
    let r = run(
        read("parquet", &["/data/nope"]).with_origin(origin),
        &ctx(&files),
    );
    let e = r.unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::PathNotFound);
    assert_eq!(e.error.message_parameters()[0].1, "file:/data/nope");
    assert_eq!(e.origin, origin);
    let r = run(read("parquet", &[]), &ctx(&files));
    assert_eq!(class(&r), ErrorClass::UnableToInferSchema);
}

#[test]
fn without_a_resolver_files_are_unsupported() {
    let none = ctx_with(None, ResolutionConf::default());
    let r = run(read("parquet", &["/data/one"]), &none);
    assert_eq!(class(&r), ErrorClass::KrishivUnsupportedFeature);
    let r = run(LogicalPlan::table(["parquet", "/data/one"]), &none);
    assert_eq!(class(&r), ErrorClass::TableOrViewNotFound);
    let a = run(LogicalPlan::table(["t"]), &none).unwrap();
    assert!(!a.reads_files && !a.context.files);
}

#[test]
fn each_read_is_inferred_once() {
    let files = Arc::new(Files::default());
    let one = read("parquet", &["/data/one"]);
    let plan = one.clone().alias("l").join(
        one.alias("r"),
        crate::JoinKind::Inner,
        crate::JoinCondition::None,
    );
    let r = run(plan, &ctx(&files));
    assert_eq!(output(&r).len(), 4);
    assert_eq!(*files.calls.lock().unwrap(), ["parquet:/data/one"]);
}

/// Denies one path.
struct DenyPath;

impl AccessPolicy for DenyPath {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        s: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = if matches!(s, Securable::Path(_)) {
            AccessDecision::Deny {
                reason: String::new(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

#[test]
fn denied_paths_are_never_listed() {
    let files = Arc::new(Files::default());
    let mut denied = ctx(&files);
    denied.policy = Arc::new(DenyPath);
    for plan in [
        read("parquet", &["/data/nope"]),
        LogicalPlan::table(["parquet", "/data/nope"]),
    ] {
        let r = run(plan, &denied);
        assert_eq!(class(&r), ErrorClass::KrishivInsufficientPermissions);
    }
    assert!(files.calls.lock().unwrap().is_empty());
}

#[test]
fn resolved_file_scans_are_checked_against_the_files() {
    let files = Arc::new(Files::default());
    let a = run(read("parquet", &["/data/one"]), &ctx(&files)).unwrap();
    // Analyzing a resolved plan again keeps it.
    let again = run((**a.plan.root()).clone(), &ctx(&files)).unwrap();
    assert_eq!(again.output, a.output);
    assert!(again.reads_files);
    // A scan claiming a column the files lack is refused.
    let PlanNode::Scan(scan) = &a.plan.root().node else {
        panic!("not a scan")
    };
    let mut wrong = scan.clone();
    wrong.output[0].data_type = DataType::Long;
    let r = run(LogicalPlan::new(PlanNode::Scan(wrong)), &ctx(&files));
    assert_eq!(class(&r), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn sql_reads_by_path_follow_resolve_sql_on_files() {
    let files = Arc::new(Files::default());
    // Lowering keeps `format.`a/path`` typed as a read by path, with the
    // format as written.
    let sql = |format: &str| {
        read(format, &["/data/one"])
            .with_origin(Origin::sql(crate::SourceId(0), crate::Span::new(14, 30)))
    };
    let r = run(sql("PARQUET"), &ctx(&files));
    assert_eq!(output(&r), nullable_one());
    let a = r.unwrap();
    let PlanNode::Scan(scan) = &a.plan.root().node else {
        panic!("not a scan")
    };
    assert!(matches!(&scan.source, ReadSource::Files { format, .. } if format == "parquet"));
    let r = run(sql("Foo"), &ctx(&files));
    let e = r.unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(e.error.message_parameters()[0].1, "`Foo`.`/data/one`");
    let r = run(sql("jdbc"), &ctx(&files));
    assert_eq!(class(&r), ErrorClass::UnsupportedDatasourceForDirectQuery);
    let off = ResolutionConf {
        run_sql_on_files: false,
        ..ResolutionConf::default()
    };
    for ctx in [ctx_with(Some(Arc::clone(&files)), off), ctx_with(None, off)] {
        assert_eq!(
            class(&run(sql("parquet"), &ctx)),
            ErrorClass::TableOrViewNotFound
        );
    }
    // Without a resolver, with the config on: not supported.
    let r = run(sql("parquet"), &ctx_with(None, ResolutionConf::default()));
    assert_eq!(class(&r), ErrorClass::KrishivUnsupportedFeature);
    // A DataFrame read names its format as it likes.
    let r = run(read("Parquet", &["/data/one"]), &ctx(&files));
    assert_eq!(output(&r), nullable_one());
    // Only the successful reads reached the resolver.
    assert_eq!(files.calls.lock().unwrap().len(), 2);
}

/// No current catalog: a two-part name is still tried as a read by path.
struct NoCatalogs;

impl crate::analyzer::CatalogResolver for NoCatalogs {
    fn current_catalog(&self) -> &str {
        "missing"
    }

    fn current_namespace(&self) -> &[String] {
        &[]
    }

    fn catalog(&self, _: &str) -> Option<Arc<dyn crate::analyzer::Catalog>> {
        None
    }

    fn epoch(&self) -> u64 {
        0
    }
}

#[test]
fn reads_by_path_need_no_catalog_and_mark_the_plan() {
    let files = Arc::new(Files::default());
    let mut no_catalogs = ctx(&files);
    no_catalogs.catalogs = Arc::new(NoCatalogs);
    let r = run(LogicalPlan::table(["parquet", "/data/one"]), &no_catalogs);
    assert_eq!(output(&r), nullable_one());
    // Equal analyses, unless one reads files and the other does not.
    let a = r.unwrap();
    let mut b = a.clone();
    assert_eq!(a, b);
    b.reads_files = false;
    assert_ne!(a, b);
}

/// Records the paths it is asked about and denies them all.
#[derive(Default)]
struct AuditPaths(Mutex<Vec<String>>);

impl AccessPolicy for AuditPaths {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(2)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        s: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = if let Securable::Path(p) = s {
            self.0.lock().unwrap().push(p.to_string());
            AccessDecision::Deny {
                reason: String::new(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

#[test]
fn the_policy_sees_the_path_the_scan_reads() {
    let files = Arc::new(Files::default());
    // Allowed: the scan reads the qualified path.
    let a = run(read("parquet", &["one"]), &ctx(&files)).unwrap();
    let PlanNode::Scan(scan) = &a.plan.root().node else {
        panic!("not a scan")
    };
    let ReadSource::Files { paths, .. } = &scan.source else {
        panic!("not files")
    };
    assert_eq!(paths[0].to_string(), "/data/one");
    // Denied: the policy was asked about the qualified path.
    let audit = Arc::new(AuditPaths::default());
    let mut denied = ctx(&files);
    denied.policy = Arc::clone(&audit) as Arc<dyn AccessPolicy>;
    let r = run(LogicalPlan::table(["parquet", "one"]), &denied);
    assert_eq!(class(&r), ErrorClass::KrishivInsufficientPermissions);
    assert_eq!(*audit.0.lock().unwrap(), ["/data/one"]);
    // A crafted scan of files is asked about like a read.
    let PlanNode::Scan(resolved) = &a.plan.root().node else {
        panic!("not a scan")
    };
    let r = run(LogicalPlan::new(PlanNode::Scan(resolved.clone())), &denied);
    assert_eq!(class(&r), ErrorClass::KrishivInsufficientPermissions);
    audit.0.lock().unwrap().pop();
    // A path read twice is asked about once.
    let twice = read("parquet", &["one"]).join(
        read("parquet", &["one"]),
        crate::JoinKind::Cross,
        crate::JoinCondition::None,
    );
    let _ = run(twice, &denied);
    assert_eq!(audit.0.lock().unwrap().len(), 2);
    // SQL names that read no files never reach the policy.
    for format in ["foo", "jdbc"] {
        let sql = read(format, &["/data/one"])
            .with_origin(Origin::sql(crate::SourceId(0), crate::Span::new(0, 1)));
        let r = run(sql, &denied);
        assert_ne!(
            class(&r),
            ErrorClass::KrishivInsufficientPermissions,
            "{format}"
        );
    }
    assert_eq!(audit.0.lock().unwrap().len(), 2);
    // A path the resolver cannot place fails before the policy.
    let r = run(read("parquet", &["bad"]), &denied);
    assert_eq!(class(&r), ErrorClass::UnableToInferSchema);
    assert_eq!(audit.0.lock().unwrap().len(), 2);
}
