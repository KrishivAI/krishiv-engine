// SPDX-License-Identifier: Apache-2.0
//! `ReadSource::Local` (P3-T012a part 2a): the carrier's equality, `Debug`
//! and canonical encoding, its analysis to a `Scan`, and the prefix cache.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use krishiv_memory::RelationStore;
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

use super::dsl::{add, int};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    Analyzer as _, CachingAnalyzer, DefaultAnalyzer, MemoryCatalog, ResolutionConf,
};
use crate::{
    ColumnRef, Expr, ExprNode, LocalRelation, LogicalPlan, Origin, Plan, PlanId, PlanNode,
    ReadSource, normalized_plan_hash,
};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, false),
        Field::new("b", ArrowType::Utf8, true),
    ]))
}

fn relation(store: &Arc<RelationStore>, ids: &[i32]) -> LocalRelation {
    let s = schema();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(ids.to_vec())),
        Arc::new(StringArray::from(vec![Some("secret-value"); ids.len()])),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&s), cols).unwrap();
    let lease = store.insert(&s, &[Batch::new(batch)]).unwrap();
    LocalRelation::from_lease(s, &lease, ids.len() as u64, 0)
}

fn read(l: &LocalRelation) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Read(ReadSource::Local(l.clone())))
}

fn plan(p: LogicalPlan) -> Plan {
    Plan::try_new(p).unwrap()
}

#[test]
fn debug_prints_the_shape_and_never_values() {
    let store = RelationStore::new();
    let l = relation(&store, &[1, 2, 3]);
    let text = format!("{l:?}");
    assert!(text.contains("rows: 3"), "{text}");
    assert!(!text.contains("secret-value"), "{text}");
    let in_plan = format!("{:?}", read(&l));
    assert!(!in_plan.contains("secret-value"), "{in_plan}");
}

#[test]
fn equal_content_is_structurally_equal_and_different_content_is_not() {
    let store = RelationStore::new();
    let (a, b) = (relation(&store, &[1, 2]), relation(&store, &[1, 2]));
    let other = relation(&store, &[1, 2, 3]);
    assert_eq!(a, b);
    assert_ne!(a, other);
    let (pa, pb, po) = (plan(read(&a)), plan(read(&b)), plan(read(&other)));
    assert!(pa.structurally_eq(&pb));
    assert_eq!(pa.canonical_encoding(), pb.canonical_encoding());
    assert_eq!(pa.structural_hash(), pb.structural_hash());
    assert_eq!(normalized_plan_hash(&pa), normalized_plan_hash(&pb));
    assert!(!pa.structurally_eq(&po));
    assert_ne!(pa.canonical_encoding(), po.canonical_encoding());
    assert_ne!(normalized_plan_hash(&pa), normalized_plan_hash(&po));
    // A decoded carrier (no lease) equals the leased one.
    let decoded = LocalRelation::from_decoded(
        Arc::clone(a.schema()),
        *a.relation().hash.as_bytes(),
        a.batches(),
        a.rows(),
        a.bytes(),
    );
    assert_eq!(decoded, a);
    assert!(plan(read(&decoded)).structurally_eq(&pa));
}

#[test]
fn a_read_resolves_to_a_scan_with_the_arrow_columns() {
    let store = RelationStore::new();
    let l = relation(&store, &[1]);
    let c = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    let analyzed = block_on(DefaultAnalyzer.analyze(&plan(read(&l)), &c)).unwrap();
    assert_eq!(analyzed.output.len(), 2);
    assert_eq!(analyzed.output[0].name, "a");
    assert_eq!(analyzed.output[0].data_type, DataType::Integer);
    assert!(!analyzed.output[0].nullable);
    assert_eq!(analyzed.output[1].data_type, DataType::String);
    assert!(analyzed.output[1].nullable);
    assert!(analyzed.securables.is_empty());
    let PlanNode::Scan(scan) = &analyzed.plan.root().node else {
        panic!("not a scan: {:?}", analyzed.plan.root().node);
    };
    assert!(matches!(scan.source, ReadSource::Local(_)));
    // Re-analysing the resolved plan keeps the source.
    let again = block_on(DefaultAnalyzer.analyze(&analyzed.plan, &c)).unwrap();
    assert_eq!(again.output, analyzed.output);
}

#[test]
fn an_arrow_type_without_a_spark_type_is_unsupported() {
    let store = RelationStore::new();
    let s: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "t",
        ArrowType::Time64(arrow_schema::TimeUnit::Nanosecond),
        true,
    )]));
    let batch = RecordBatch::new_empty(Arc::clone(&s));
    let lease = store.insert(&s, &[Batch::new(batch)]).unwrap();
    let l = LocalRelation::from_lease(s, &lease, 0, 0);
    let c = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    let err = block_on(DefaultAnalyzer.analyze(&plan(read(&l)), &c)).unwrap_err();
    assert_eq!(
        err.error.error_class(),
        krishiv_common::error::ErrorClass::UnsupportedTimeType
    );
}

#[test]
fn the_prefix_cache_reuses_a_local_relation() {
    let store = RelationStore::new();
    let l = relation(&store, &[1, 2]);
    let c = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    let col = |name: &str, id: i64| {
        Expr::new(ExprNode::Column(ColumnRef {
            plan_id: Some(PlanId(id)),
            ..ColumnRef::new(vec![name.to_owned()])
        }))
    };
    let chain = |steps: i64| {
        let mut p = read(&l).with_origin(Origin::plan(PlanId(0)));
        for i in 1..=steps {
            let prev = if i == 1 {
                "a".to_owned()
            } else {
                format!("c{}", i - 1)
            };
            p = LogicalPlan::new(PlanNode::WithColumns {
                input: Arc::new(p),
                columns: vec![(format!("c{i}"), Arc::new(add(col(&prev, i - 1), int(1))))],
            })
            .with_origin(Origin::plan(PlanId(i)));
        }
        plan(p)
    };
    let cache = CachingAnalyzer::new();
    block_on(cache.analyze(&chain(2), &c)).unwrap();
    for depth in 3..=5 {
        let before = cache.warm_starts();
        let warm = block_on(cache.analyze(&chain(depth), &c)).unwrap();
        assert!(cache.warm_starts() > before, "depth {depth} starts warm");
        assert_eq!(
            warm,
            block_on(DefaultAnalyzer.analyze(&chain(depth), &c)).unwrap()
        );
    }
}

#[test]
fn display_names_the_columns_and_rows() {
    let store = RelationStore::new();
    let l = relation(&store, &[1, 2]);
    assert_eq!(read(&l).to_string(), "LocalRelation [a, b] (2 rows)\n");
    let c = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    let analyzed = block_on(DefaultAnalyzer.analyze(&plan(read(&l)), &c)).unwrap();
    // Spark-format text is golden-pending; the row count is Krishiv's own.
    insta::assert_snapshot!(analyzed.plan.root().to_string(), @"LocalRelation [a#0, b#1] (2 rows)");
}
