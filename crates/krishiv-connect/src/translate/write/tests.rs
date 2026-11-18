// SPDX-License-Identifier: Apache-2.0
//! The write commands' writer state.

use std::collections::BTreeMap;

use krishiv_common::error::ErrorClass;

use super::*;
use crate::proto::v4_3::write_operation::{BucketBy, SaveTable};

fn attribute(name: &str) -> pb::Expression {
    pb::Expression {
        expr_type: Some(pb::expression::ExprType::UnresolvedAttribute(
            pb::expression::UnresolvedAttribute {
                unparsed_identifier: name.to_owned(),
                ..Default::default()
            },
        )),
        ..Default::default()
    }
}

#[test]
fn a_path_write_carries_the_writer_state() {
    let w = pb::WriteOperation {
        source: Some("csv".to_owned()),
        save_type: Some(SaveType::Path("/tmp/out".to_owned())),
        mode: SaveMode::Overwrite as i32,
        sort_column_names: vec!["s".to_owned()],
        partitioning_columns: vec!["p".to_owned(), "q".to_owned()],
        bucket_by: Some(BucketBy {
            bucket_column_names: vec!["b".to_owned()],
            num_buckets: 4,
        }),
        options: BTreeMap::from([
            ("sep".to_owned(), ";".to_owned()),
            ("header".to_owned(), "true".to_owned()),
        ]),
        ..Default::default()
    };
    let spec = write_spec(&w).expect("a write");
    assert_eq!(
        spec,
        WriteSpec {
            source: Some("csv".to_owned()),
            mode: Some("overwrite"),
            options: vec![
                ("header".to_owned(), "true".to_owned()),
                ("sep".to_owned(), ";".to_owned()),
            ],
            sort_by: Some(vec!["s".to_owned()]),
            bucket_by: Some((4, vec!["b".to_owned()])),
            partition_by: vec!["p".to_owned(), "q".to_owned()],
            target: WriteTarget::Path(Some("/tmp/out".to_owned())),
        }
    );
}

#[test]
fn every_save_mode_has_its_spark_name() {
    for (mode, name) in [
        (SaveMode::Unspecified, None),
        (SaveMode::Append, Some("append")),
        (SaveMode::Overwrite, Some("overwrite")),
        (SaveMode::ErrorIfExists, Some("errorifexists")),
        (SaveMode::Ignore, Some("ignore")),
    ] {
        let w = pb::WriteOperation {
            mode: mode as i32,
            ..Default::default()
        };
        let spec = write_spec(&w).expect("a write");
        assert_eq!(spec.mode, name, "{mode:?}");
        assert_eq!(spec.target, WriteTarget::Path(None));
    }
    let w = pb::WriteOperation {
        mode: 99,
        ..Default::default()
    };
    let e = write_spec(&w).expect_err("an unknown mode");
    assert_eq!(e.error_class(), ErrorClass::ConnectInvalidPlanInvalidEnum);
}

#[test]
fn a_table_write_names_its_save_method() {
    let table = |method: TableSaveMethod| pb::WriteOperation {
        save_type: Some(SaveType::Table(SaveTable {
            table_name: "db.t".to_owned(),
            save_method: method as i32,
        })),
        ..Default::default()
    };
    assert_eq!(
        write_spec(&table(TableSaveMethod::SaveAsTable))
            .expect("saveAsTable")
            .target,
        WriteTarget::SaveAsTable("db.t".to_owned())
    );
    assert_eq!(
        write_spec(&table(TableSaveMethod::InsertInto))
            .expect("insertInto")
            .target,
        WriteTarget::InsertInto("db.t".to_owned())
    );
    let e = write_spec(&table(TableSaveMethod::Unspecified)).expect_err("unspecified");
    assert_eq!(e.error_class(), ErrorClass::ConnectInvalidPlanInvalidEnum);
    assert!(
        e.to_string().contains("TABLE_SAVE_METHOD_UNSPECIFIED"),
        "{e}"
    );
}

#[test]
fn cluster_by_and_schema_evolution_are_refused_by_name() {
    let clustered = pb::WriteOperation {
        clustering_columns: vec!["c".to_owned()],
        ..Default::default()
    };
    let e = write_spec(&clustered).expect_err("clusterBy");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.to_string().contains("WriteOperation.clustering_columns"),
        "{e}"
    );
    let evolving = pb::WriteOperation {
        with_schema_evolution: true,
        ..Default::default()
    };
    let e = write_spec(&evolving).expect_err("schema evolution");
    assert!(e.to_string().contains("with_schema_evolution"), "{e}");
    let evolving = pb::WriteOperationV2 {
        mode: Mode::Append as i32,
        with_schema_evolution: true,
        ..Default::default()
    };
    let e = write_v2_spec(&evolving).expect_err("schema evolution");
    assert!(
        e.to_string()
            .contains("WriteOperationV2.with_schema_evolution"),
        "{e}"
    );
}

#[test]
fn a_v2_write_carries_its_mode_and_reads_the_provider_only_where_spark_does() {
    for (mode, name, provider) in [
        (Mode::Create, "create", true),
        (Mode::Overwrite, "overwrite", false),
        (Mode::OverwritePartitions, "overwritePartitions", false),
        (Mode::Append, "append", false),
        (Mode::Replace, "replace", true),
        (Mode::CreateOrReplace, "createOrReplace", true),
    ] {
        let w = pb::WriteOperationV2 {
            table_name: "t".to_owned(),
            provider: Some("parquet".to_owned()),
            partitioning_columns: vec![attribute("p")],
            options: BTreeMap::from([("k".to_owned(), "v".to_owned())]),
            table_properties: BTreeMap::from([("tp".to_owned(), "x".to_owned())]),
            clustering_columns: vec!["c".to_owned()],
            mode: mode as i32,
            ..Default::default()
        };
        let spec = write_v2_spec(&w).expect("a write");
        assert_eq!(spec.mode, name);
        assert_eq!(spec.provider.is_some(), provider, "{name}");
        assert_eq!(spec.table, "t");
        assert_eq!(spec.partitioned_by, ["p"]);
        assert_eq!(spec.options, [("k".to_owned(), "v".to_owned())]);
        assert_eq!(spec.properties, [("tp".to_owned(), "x".to_owned())]);
        assert_eq!(spec.cluster_by, ["c"]);
    }
    let e = write_v2_spec(&pb::WriteOperationV2::default()).expect_err("unspecified");
    assert_eq!(e.error_class(), ErrorClass::ConnectInvalidPlanInvalidEnum);
    assert!(e.to_string().contains("MODE_UNSPECIFIED"), "{e}");
}

#[test]
fn a_partition_transform_is_refused() {
    let transform = pb::Expression {
        expr_type: Some(pb::expression::ExprType::UnresolvedFunction(
            pb::expression::UnresolvedFunction {
                function_name: "years".to_owned(),
                arguments: vec![attribute("d")],
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    let w = pb::WriteOperationV2 {
        mode: Mode::Create as i32,
        partitioning_columns: vec![transform],
        ..Default::default()
    };
    let e = write_v2_spec(&w).expect_err("a transform");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("partitioning_columns"), "{e}");
}

#[test]
fn debug_shows_no_path_option_or_table_name() {
    let w = pb::WriteOperation {
        save_type: Some(SaveType::Path("/secret/path".to_owned())),
        options: BTreeMap::from([("password".to_owned(), "hunter2".to_owned())]),
        ..Default::default()
    };
    let text = format!("{:?}", write_spec(&w).expect("a write"));
    assert!(
        !text.contains("secret") && !text.contains("hunter2") && !text.contains("password"),
        "{text}"
    );
    let w = pb::WriteOperationV2 {
        table_name: "secret_table".to_owned(),
        mode: Mode::Append as i32,
        options: BTreeMap::from([("token".to_owned(), "hunter2".to_owned())]),
        ..Default::default()
    };
    let text = format!("{:?}", write_v2_spec(&w).expect("a write"));
    assert!(
        !text.contains("secret") && !text.contains("hunter2"),
        "{text}"
    );
}
