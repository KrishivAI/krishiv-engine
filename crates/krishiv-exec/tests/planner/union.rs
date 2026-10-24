// SPDX-License-Identifier: Apache-2.0
//! `UNION ALL` branches that carry one logical type in different Arrow types.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::error::ErrorClass;
use krishiv_exec::planner::{DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner as _};
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{MemoryCatalog, ResolutionConf, TableColumn};
use krishiv_plan::{LogicalPlan, Plan, PlanNode};
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

use super::{Env, env, table};

/// `t` of [`env`] (`s` is `Utf8View`) and `u(a INT, s STRING)` with `s`
/// stored as `Utf8`: the same logical type in two Arrow carriers.
fn carriers() -> Env {
    let (t_schema, t_batches) = table(10);
    let u_schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("s", ArrowType::Utf8, true),
    ]));
    let u = RecordBatch::try_new(
        Arc::clone(&u_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("u1"), None])),
        ],
    )
    .unwrap();
    let columns = |schema: &Schema| -> Vec<TableColumn> {
        schema
            .fields()
            .iter()
            .map(|f| TableColumn {
                name: f.name().clone(),
                data_type: DataType::from_arrow(f.data_type()).unwrap(),
                nullable: f.is_nullable(),
            })
            .collect()
    };
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "t", columns(&t_schema), None)
        .with_table(&["default"], "u", columns(&u_schema), None);
    let tables = MemoryTables::new()
        .with_table(
            &["spark_catalog", "default", "t"],
            Arc::new(MemoryTable::new(t_schema, t_batches)),
        )
        .with_table(
            &["spark_catalog", "default", "u"],
            Arc::new(MemoryTable::new(u_schema, vec![Batch::new(u)])),
        );
    Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    }
}

/// A `Utf8` string column and a NULL (or literal) string beside a
/// `Utf8View` one.
#[test]
fn string_branches_in_different_carriers_union() {
    let env = carriers();
    let mixed = env
        .plan("SELECT a, s FROM u UNION ALL SELECT a, s FROM t")
        .unwrap();
    assert!(format!("{:?}", mixed.root()).contains("Cast"), "{mixed}");
    for (sql, expected) in [
        (
            "SELECT a, s FROM u UNION ALL SELECT a, s FROM t WHERE a = 1",
            vec!["1|s1", "1|u1", "2|NULL"],
        ),
        (
            "SELECT a, s FROM t WHERE a = 1 UNION ALL SELECT a, s FROM u",
            vec!["1|s1", "1|u1", "2|NULL"],
        ),
        (
            "SELECT a, NULL AS s FROM u WHERE a = 1 UNION ALL SELECT a, s FROM u",
            vec!["1|NULL", "1|u1", "2|NULL"],
        ),
        (
            "SELECT a, s FROM u WHERE a = 2 UNION ALL SELECT a, 'x' AS s FROM u WHERE a = 2",
            vec!["2|NULL", "2|x"],
        ),
    ] {
        let (schema, mut rows) = env.rows(sql);
        assert_eq!(schema.field(1).data_type(), &ArrowType::Utf8View, "{sql}");
        rows.sort();
        assert_eq!(rows, expected, "{sql}");
    }
}

/// Branches that agree keep their carrier: no cast, no copy.
#[test]
fn branches_in_one_carrier_keep_it() {
    let env = carriers();
    let plan = env
        .plan("SELECT a, s FROM u UNION ALL SELECT a, s FROM u")
        .unwrap();
    assert_eq!(plan.schema().field(1).data_type(), &ArrowType::Utf8);
    assert!(!format!("{:?}", plan.root()).contains("Cast"), "{plan}");
}

/// Branches of different logical types are still an engine error: the
/// analyzer widens them, so a plan that skips it is a bug upstream.
#[test]
fn branches_of_different_logical_types_are_rejected() {
    let env = env();
    let branch = |sql: &str| Arc::new(env.analyze(sql).root().as_ref().clone());
    let root = LogicalPlan::new(PlanNode::Union {
        inputs: vec![branch("SELECT a FROM t"), branch("SELECT b FROM t")],
        by_name: false,
        allow_missing_columns: false,
    });
    let e = DefaultPlanner
        .plan(&Plan::try_new(root).unwrap(), &env.cx())
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("different physical types"), "{e}");
}
