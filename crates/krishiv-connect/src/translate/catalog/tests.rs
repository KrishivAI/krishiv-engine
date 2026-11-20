// SPDX-License-Identifier: Apache-2.0
//! `Relation.catalog`: each of the 36 operations read into its call, the overload choice, the
//! `createTable` argument rules, and the translator's request for the host.

use krishiv_common::error::ErrorClass;
use krishiv_plan::{PlanNode, ReadSource};

use super::super::testing::{fails, relation};
use super::super::{LocalRelations, LocalRequest, TranslateConf, translate_relation};
use super::*;
use crate::proto::v4_3::data_type::{Integer, Kind, Struct, StructField};

fn msg(kind: CatType) -> pb::Catalog {
    pb::Catalog {
        cat_type: Some(kind),
    }
}

fn some(s: &str) -> Option<String> {
    Some(s.to_owned())
}

fn s(v: &str) -> String {
    v.to_owned()
}

fn struct_of(names: &[&str]) -> pb::DataType {
    pb::DataType {
        kind: Some(Kind::Struct(Struct {
            fields: names
                .iter()
                .map(|n| StructField {
                    name: (*n).to_owned(),
                    data_type: Some(pb::DataType {
                        kind: Some(Kind::Integer(Integer::default())),
                    }),
                    nullable: true,
                    metadata: None,
                })
                .collect(),
            type_variation_reference: 0,
        })),
    }
}

#[test]
fn the_name_only_arms_carry_their_name() {
    let cases = [
        (
            CatType::SetCurrentDatabase(pb::SetCurrentDatabase { db_name: s("d") }),
            CatalogCall::SetCurrentDatabase(s("d")),
        ),
        (
            CatType::GetDatabase(pb::GetDatabase { db_name: s("d") }),
            CatalogCall::GetDatabase(s("d")),
        ),
        (
            CatType::DatabaseExists(pb::DatabaseExists { db_name: s("d") }),
            CatalogCall::DatabaseExists(s("d")),
        ),
        (
            CatType::DropTempView(pb::DropTempView { view_name: s("v") }),
            CatalogCall::DropTempView(s("v")),
        ),
        (
            CatType::DropGlobalTempView(pb::DropGlobalTempView { view_name: s("v") }),
            CatalogCall::DropGlobalTempView(s("v")),
        ),
        (
            CatType::RecoverPartitions(pb::RecoverPartitions { table_name: s("t") }),
            CatalogCall::RecoverPartitions(s("t")),
        ),
        (
            CatType::IsCached(pb::IsCached { table_name: s("t") }),
            CatalogCall::IsCached(s("t")),
        ),
        (
            CatType::UncacheTable(pb::UncacheTable { table_name: s("t") }),
            CatalogCall::UncacheTable(s("t")),
        ),
        (
            CatType::RefreshTable(pb::RefreshTable { table_name: s("t") }),
            CatalogCall::RefreshTable(s("t")),
        ),
        (
            CatType::RefreshByPath(pb::RefreshByPath { path: s("/p") }),
            CatalogCall::RefreshByPath(s("/p")),
        ),
        (
            CatType::SetCurrentCatalog(pb::SetCurrentCatalog {
                catalog_name: s("c"),
            }),
            CatalogCall::SetCurrentCatalog(s("c")),
        ),
        (
            CatType::ListPartitions(pb::ListPartitions { table_name: s("t") }),
            CatalogCall::ListPartitions(s("t")),
        ),
        (
            CatType::GetTableProperties(pb::GetTableProperties { table_name: s("t") }),
            CatalogCall::GetTableProperties(s("t")),
        ),
        (
            CatType::TruncateTable(pb::TruncateTable { table_name: s("t") }),
            CatalogCall::TruncateTable(s("t")),
        ),
    ];
    for (kind, want) in cases {
        assert_eq!(catalog_call(&msg(kind)).expect("reads"), want);
    }
}

#[test]
fn the_arms_without_arguments_read_as_themselves() {
    let cases = [
        (
            CatType::CurrentDatabase(pb::CurrentDatabase {}),
            CatalogCall::CurrentDatabase,
        ),
        (
            CatType::CurrentCatalog(pb::CurrentCatalog {}),
            CatalogCall::CurrentCatalog,
        ),
        (
            CatType::ClearCache(pb::ClearCache {}),
            CatalogCall::ClearCache,
        ),
    ];
    for (kind, want) in cases {
        assert_eq!(catalog_call(&msg(kind)).expect("reads"), want);
    }
}

#[test]
fn the_listings_keep_an_absent_database_and_pattern_apart() {
    let tables = |db: Option<&str>, pattern: Option<&str>| {
        catalog_call(&msg(CatType::ListTables(pb::ListTables {
            db_name: db.map(str::to_owned),
            pattern: pattern.map(str::to_owned),
        })))
        .expect("reads")
    };
    assert_eq!(tables(None, None), CatalogCall::ListTables(None, None));
    assert_eq!(
        tables(Some("d"), Some("p*")),
        CatalogCall::ListTables(some("d"), some("p*"))
    );
    assert_eq!(
        catalog_call(&msg(CatType::ListFunctions(pb::ListFunctions {
            db_name: None,
            pattern: some("f"),
        })))
        .expect("reads"),
        CatalogCall::ListFunctions(None, some("f"))
    );
    assert_eq!(
        catalog_call(&msg(CatType::ListViews(pb::ListViews {
            db_name: some("d"),
            pattern: None,
        })))
        .expect("reads"),
        CatalogCall::ListViews(some("d"), None)
    );
    assert_eq!(
        catalog_call(&msg(CatType::ListDatabases(pb::ListDatabases {
            pattern: some("d*"),
        })))
        .expect("reads"),
        CatalogCall::ListDatabases(some("d*"))
    );
    assert_eq!(
        catalog_call(&msg(CatType::ListCatalogs(pb::ListCatalogs {
            pattern: None,
        })))
        .expect("reads"),
        CatalogCall::ListCatalogs(None)
    );
}

#[test]
fn the_lookups_carry_an_optional_database() {
    assert_eq!(
        catalog_call(&msg(CatType::ListColumns(pb::ListColumns {
            table_name: s("t"),
            db_name: some("d"),
        })))
        .expect("reads"),
        CatalogCall::ListColumns {
            db: some("d"),
            table: s("t")
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::GetTable(pb::GetTable {
            table_name: s("t"),
            db_name: None,
        })))
        .expect("reads"),
        CatalogCall::GetTable {
            db: None,
            table: s("t")
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::GetFunction(pb::GetFunction {
            function_name: s("f"),
            db_name: some("d"),
        })))
        .expect("reads"),
        CatalogCall::GetFunction {
            db: some("d"),
            function: s("f")
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::TableExists(pb::TableExists {
            table_name: s("t"),
            db_name: some("d"),
        })))
        .expect("reads"),
        CatalogCall::TableExists {
            db: some("d"),
            table: s("t")
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::FunctionExists(pb::FunctionExists {
            function_name: s("f"),
            db_name: None,
        })))
        .expect("reads"),
        CatalogCall::FunctionExists {
            db: None,
            function: s("f")
        }
    );
}

#[test]
fn the_flags_of_the_drops_and_maintenance_calls_are_kept() {
    assert_eq!(
        catalog_call(&msg(CatType::DropTable(pb::DropTable {
            table_name: s("t"),
            if_exists: true,
            purge: true,
        })))
        .expect("reads"),
        CatalogCall::DropTable {
            table: s("t"),
            if_exists: true,
            purge: true
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::DropView(pb::DropView {
            view_name: s("v"),
            if_exists: true,
        })))
        .expect("reads"),
        CatalogCall::DropView {
            view: s("v"),
            if_exists: true
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::DropDatabase(pb::DropDatabase {
            db_name: s("d"),
            if_exists: false,
            cascade: true,
        })))
        .expect("reads"),
        CatalogCall::DropDatabase {
            db: s("d"),
            if_exists: false,
            cascade: true
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::GetCreateTableString(
            pb::GetCreateTableString {
                table_name: s("t"),
                as_serde: true,
            }
        )))
        .expect("reads"),
        CatalogCall::GetCreateTableString {
            table: s("t"),
            as_serde: true
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::AnalyzeTable(pb::AnalyzeTable {
            table_name: s("t"),
            no_scan: true,
        })))
        .expect("reads"),
        CatalogCall::AnalyzeTable {
            table: s("t"),
            no_scan: true
        }
    );
    let created = catalog_call(&msg(CatType::CreateDatabase(pb::CreateDatabase {
        db_name: s("d"),
        if_not_exists: true,
        properties: [(s("k"), s("v"))].into(),
    })))
    .expect("reads");
    assert_eq!(
        created,
        CatalogCall::CreateDatabase {
            db: s("d"),
            if_not_exists: true,
            properties: [(s("k"), s("v"))].into()
        }
    );
}

#[test]
fn cache_table_keeps_the_storage_level_it_was_given() {
    let level = pb::StorageLevel {
        use_disk: true,
        use_memory: false,
        use_off_heap: false,
        deserialized: false,
        replication: 2,
    };
    assert_eq!(
        catalog_call(&msg(CatType::CacheTable(pb::CacheTable {
            table_name: s("t"),
            storage_level: Some(level),
        })))
        .expect("reads"),
        CatalogCall::CacheTable {
            table: s("t"),
            level: Some(level)
        }
    );
    assert_eq!(
        catalog_call(&msg(CatType::CacheTable(pb::CacheTable {
            table_name: s("t"),
            storage_level: None,
        })))
        .expect("reads"),
        CatalogCall::CacheTable {
            table: s("t"),
            level: None
        }
    );
}

#[test]
fn create_table_turns_the_path_into_the_last_option_and_the_struct_into_ddl() {
    let CatalogCall::CreateTable(t) = catalog_call(&msg(CatType::CreateTable(pb::CreateTable {
        table_name: s("t"),
        path: some("/data/t"),
        source: some("parquet"),
        description: some("d"),
        schema: Some(struct_of(&["a", "we`ird"])),
        options: [(s("path"), s("/ignored")), (s("k"), s("v"))].into(),
    })))
    .expect("reads") else {
        panic!("create table");
    };
    assert_eq!(t.table, "t");
    assert_eq!(t.source.as_deref(), Some("parquet"));
    assert_eq!(t.schema.as_deref(), Some("`a` INT, `we``ird` INT"));
    assert_eq!(t.description, "d");
    // The `path` message field is appended last, so it wins over a `path` option.
    assert_eq!(t.options.last(), Some(&(s("path"), s("/data/t"))));
    assert_eq!(t.options.len(), 3);
}

#[test]
fn a_field_comment_becomes_the_columns_comment_clause() {
    let mut schema = struct_of(&["a", "b"]);
    let Some(Kind::Struct(st)) = &mut schema.kind else {
        panic!("struct");
    };
    st.fields[0].metadata = Some(r#"{"comment":"it's a \\ test"}"#.to_owned());
    let CatalogCall::CreateTable(t) = catalog_call(&msg(CatType::CreateTable(pb::CreateTable {
        table_name: s("t"),
        schema: Some(schema),
        ..pb::CreateTable::default()
    })))
    .expect("reads") else {
        panic!("create table");
    };
    assert_eq!(
        t.schema.as_deref(),
        Some(r"`a` INT COMMENT 'it\'s a \\ test', `b` INT")
    );
    assert!(t.lossy_in_escaped_mode);
}

#[test]
fn a_non_nullable_field_keeps_its_not_null() {
    let mut schema = struct_of(&["a", "b"]);
    let Some(Kind::Struct(st)) = &mut schema.kind else {
        panic!("struct");
    };
    st.fields[0].nullable = false;
    let CatalogCall::CreateTable(t) = catalog_call(&msg(CatType::CreateTable(pb::CreateTable {
        table_name: s("t"),
        schema: Some(schema),
        ..pb::CreateTable::default()
    })))
    .expect("reads") else {
        panic!("create table");
    };
    assert_eq!(t.schema.as_deref(), Some("`a` INT NOT NULL, `b` INT"));
    assert!(!t.lossy_in_escaped_mode);
}

#[test]
fn create_table_without_description_or_schema_reads_as_spark_does() {
    let CatalogCall::CreateTable(t) = catalog_call(&msg(CatType::CreateTable(pb::CreateTable {
        table_name: s("t"),
        ..pb::CreateTable::default()
    })))
    .expect("reads") else {
        panic!("create table");
    };
    assert_eq!(t.description, "");
    assert_eq!(t.schema, None);
    assert_eq!(t.source, None);
    assert!(t.options.is_empty());
    // An empty struct is Spark's `new StructType`: no schema.
    let CatalogCall::CreateTable(t) = catalog_call(&msg(CatType::CreateExternalTable(
        pb::CreateExternalTable {
            table_name: s("e"),
            schema: Some(struct_of(&[])),
            path: some("/p"),
            ..pb::CreateExternalTable::default()
        },
    )))
    .expect("reads") else {
        panic!("create external table");
    };
    assert_eq!(t.schema, None);
    assert_eq!(t.description, "");
    assert_eq!(t.options, [(s("path"), s("/p"))]);
}

#[test]
fn a_schema_that_is_not_a_struct_is_an_assertion_failure() {
    let e = catalog_call(&msg(CatType::CreateTable(pb::CreateTable {
        table_name: s("t"),
        schema: Some(pb::DataType {
            kind: Some(Kind::Integer(Integer::default())),
        }),
        ..pb::CreateTable::default()
    })))
    .expect_err("not a struct");
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
}

#[test]
fn an_unset_operation_is_not_set() {
    let e = catalog_call(&pb::Catalog { cat_type: None }).expect_err("unset");
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

fn catalog_relation(kind: CatType) -> pb::Relation {
    relation(crate::proto::v4_3::relation::RelType::Catalog(msg(kind)))
}

#[test]
fn the_translator_records_the_call_for_the_host_and_reads_its_relation() {
    let locals = LocalRelations::collecting();
    let conf = TranslateConf {
        local_relations: locals.clone(),
        ..TranslateConf::default()
    };
    let plan = translate_relation(
        &catalog_relation(CatType::ListTables(pb::ListTables {
            db_name: None,
            pattern: some("t*"),
        })),
        &conf,
    )
    .expect("translates");
    assert!(matches!(
        &plan.root().node,
        PlanNode::Read(ReadSource::Local(_))
    ));
    assert_eq!(
        locals.take_requests(),
        [LocalRequest::Catalog(CatalogCall::ListTables(
            None,
            some("t*")
        ))]
    );
}

#[test]
fn without_a_host_a_catalog_relation_is_refused() {
    let e = fails(translate_relation(
        &catalog_relation(CatType::ClearCache(pb::ClearCache {})),
        &TranslateConf::default(),
    ));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn a_bad_message_fails_before_the_host_is_asked() {
    let locals = LocalRelations::collecting();
    let conf = TranslateConf {
        local_relations: locals.clone(),
        ..TranslateConf::default()
    };
    let e = translate_relation(
        &relation(crate::proto::v4_3::relation::RelType::Catalog(
            pb::Catalog { cat_type: None },
        )),
        &conf,
    )
    .expect_err("unset");
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    assert!(locals.take_requests().is_empty());
}

#[test]
fn a_backslash_in_a_nested_field_comment_is_refused_and_a_quote_is_lossy_when_escaped() {
    let nested = |comment: &str| {
        let mut schema = struct_of(&["a"]);
        let Some(Kind::Struct(st)) = &mut schema.kind else {
            panic!("struct");
        };
        let mut inner = struct_of(&["x"]);
        let Some(Kind::Struct(inner_st)) = &mut inner.kind else {
            panic!("struct");
        };
        inner_st.fields[0].metadata = Some(format!(r#"{{"comment":"{comment}"}}"#));
        st.fields[0].data_type = Some(inner);
        catalog_call(&msg(CatType::CreateTable(pb::CreateTable {
            table_name: s("t"),
            schema: Some(schema),
            ..pb::CreateTable::default()
        })))
    };
    assert!(nested(r"a\\b").is_err());
    let CatalogCall::CreateTable(t) = nested("it's").expect("reads") else {
        panic!("create table");
    };
    assert!(t.lossy_in_escaped_mode);
    let CatalogCall::CreateTable(t) = nested("plain").expect("reads") else {
        panic!("create table");
    };
    assert!(!t.lossy_in_escaped_mode);
}
