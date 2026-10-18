// SPDX-License-Identifier: Apache-2.0
//! Tests of the entry points, the limits and the stack safety of the
//! translation.

use krishiv_common::error::ErrorClass;
use prost::Message;

use super::testing::{attr, boxed, conf, expr, fails, func, lit, read_table, relation};
use super::*;
use crate::proto::v4_2::expression::{ExprType, literal::LiteralType};
use crate::proto::v4_2::relation::RelType;
use krishiv_plan::{Grouping, PlanNode};

fn filter_over(input: pb::Relation) -> pb::Relation {
    relation(RelType::Filter(Box::new(pb::Filter {
        input: boxed(input),
        condition: Some(attr("a")),
    })))
}

/// `filters` filters over a table read, built without recursion.
fn filter_chain(filters: usize) -> pb::Relation {
    let mut r = read_table("t");
    for _ in 0..filters {
        r = filter_over(r);
    }
    r
}

/// Runs `f` on a thread with `stack` bytes. Deep protos are built, translated
/// and dropped inside it, so the test thread's 2 MiB does not matter.
fn on_stack<T: Send + 'static>(stack: usize, f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn the_default_conf_is_sparks() {
    assert_eq!(TranslateConf::default().default_data_source, "parquet");
}

#[test]
fn plans_translate_from_their_root() {
    let plan = pb::Plan {
        op_type: Some(OpType::Root(Box::new(filter_chain(1)))),
    };
    let ir = translate_plan(&plan, &conf()).unwrap();
    assert_eq!(ir.to_string(), "Filter a\n+- Read t\n");
    assert_eq!(ir.ir_version(), krishiv_plan::IR_VERSION);
    // The same plan from its bytes.
    assert_eq!(plan_from_bytes(&plan.encode_to_vec(), &conf()).unwrap(), ir);
}

#[test]
fn commands_compressed_plans_and_empty_plans_are_not_relations() {
    let command = pb::Plan {
        op_type: Some(OpType::Command(Box::default())),
    };
    let e = fails(translate_plan(&command, &conf()));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.message().contains("Plan.command"), "{}", e.message());
    let compressed = pb::Plan {
        op_type: Some(OpType::CompressedOperation(Box::default())),
    };
    let e = fails(translate_plan(&compressed, &conf()));
    assert!(
        e.message().contains("Plan.compressed_operation"),
        "{}",
        e.message()
    );
    let e = fails(translate_plan(&pb::Plan::default(), &conf()));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    assert!(e.message().contains("OP_TYPE_NOT_SET"));
}

#[test]
fn relations_and_expressions_translate_from_bytes() {
    let r = filter_chain(2);
    let plan = relation_from_bytes(&r.encode_to_vec(), &conf()).unwrap();
    assert_eq!(plan.to_string(), "Filter a\n+- Filter a\n   +- Read t\n");
    let e = func("+", vec![attr("a"), lit(LiteralType::Integer(1))]);
    let ir = expression_from_bytes(&e.encode_to_vec(), &conf()).unwrap();
    assert_eq!(ir.to_string(), "(a + 1)");
    // Bytes that are not a message are a decode error, not a panic.
    let e = fails(plan_from_bytes(&[0xff; 12], &conf()));
    assert_eq!(e.error_class(), ErrorClass::ConnectInvalidPlanCannotParse);
    let e = fails(relation_from_bytes(&[0x0b], &conf()));
    assert_eq!(e.error_class(), ErrorClass::ConnectInvalidPlanCannotParse);
    let e = fails(expression_from_bytes(&[0x0e], &conf()));
    assert_eq!(e.error_class(), ErrorClass::ConnectInvalidPlanCannotParse);
}

#[test]
fn the_translator_counts_nodes_against_the_budget() {
    let c = conf();
    let mut t = Translator::new(&c);
    t.nodes = wire::MAX_MESSAGES - 2;
    // A function and its two arguments are three nodes: the third is over.
    let e = fails(t.expression(&func("f", vec![attr("a"), attr("b")])));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax
    );
    // Within the budget, it translates.
    let mut t = Translator::new(&c);
    t.nodes = wire::MAX_MESSAGES - 3;
    t.expression(&func("f", vec![attr("a"), attr("b")]))
        .unwrap();
    assert_eq!(t.nodes, wire::MAX_MESSAGES);
}

#[test]
fn relations_deeper_than_the_ir_are_too_complex() {
    // `k` filters over a read are k + 1 levels deep (the predicate of the
    // lowest filter is a level of its own, but a sibling of the read).
    for (filters, ok) in [(MAX_DEPTH - 1, true), (MAX_DEPTH, false)] {
        let result = on_stack(64 << 20, move || {
            let r = filter_chain(filters);
            let out = translate_relation(&r, &conf()).map(|p| p.root().origin);
            drop(r);
            out
        });
        match (ok, result) {
            (true, Ok(_)) => {}
            (false, Err(e)) => {
                assert_eq!(
                    e.error_class(),
                    ErrorClass::FailedToParseTooComplex,
                    "{filters} filters"
                );
            }
            (ok, r) => panic!("{filters} filters: expected ok={ok}, got {r:?}"),
        }
    }
}

#[test]
fn deep_relations_translate_on_a_small_stack() {
    // 900 nested filters, translated on a 1 MiB stack: after a few levels the
    // translator continues on a thread of its own. The input is built and
    // dropped on a big stack; only the translation runs on the small one.
    on_stack(64 << 20, || {
        let r = filter_chain(900);
        let plan = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(1 << 20)
                .spawn_scoped(scope, || translate_relation(&r, &conf()))
                .unwrap()
                .join()
                .unwrap()
        })
        .unwrap();
        assert!(plan.to_string().starts_with("Filter a\n+- Filter a\n"));
        drop(plan);
        drop(r);
    });
}

#[test]
fn a_too_deep_plan_from_bytes_never_reaches_the_translator() {
    // 600 filters are 1,200 messages deep: over the wire cap of 1,024, so the
    // walk refuses the bytes before anything recurses.
    let bytes = on_stack(64 << 20, || {
        let r = filter_chain(600);
        let bytes = r.encode_to_vec();
        drop(r);
        bytes
    });
    let e = fails(relation_from_bytes(&bytes, &conf()));
    assert_eq!(e.error_class(), ErrorClass::FailedToParseTooComplex);
    // 400 filters (800 deep) are accepted.
    let ok = on_stack(64 << 20, || filter_chain(400).encode_to_vec());
    assert!(relation_from_bytes(&ok, &conf()).is_ok());
}

#[test]
fn an_expression_in_a_plan_is_translated_once_per_use() {
    // The aggregate's keys are shared between its grouping and its output.
    let key = attr("a");
    let aggregate = relation(RelType::Aggregate(Box::new(pb::Aggregate {
        input: boxed(read_table("t")),
        group_type: 1,
        grouping_expressions: vec![key],
        aggregate_expressions: vec![expr(ExprType::UnresolvedStar(Default::default()))],
        ..pb::Aggregate::default()
    })));
    let plan = translate_relation(&aggregate, &conf()).unwrap();
    let PlanNode::Aggregate {
        grouping: Grouping::GroupBy(keys),
        aggregates,
        ..
    } = &plan.root().node
    else {
        panic!("not an aggregate")
    };
    assert!(Arc::ptr_eq(&keys[0], &aggregates[0]));
}

#[test]
fn threads_for_deep_subtrees_are_capped() {
    let c = conf();
    // A leaf at the inline depth needs no thread, whatever the count.
    let mut t = Translator::new(&c);
    t.depth = INLINE_LEVELS - 1;
    t.hops = MAX_HOPS;
    t.expression(&attr("a")).unwrap();
    // A composite one does, and the cap refuses the one after the last.
    let call = func("not", vec![attr("a")]);
    let mut t = Translator::new(&c);
    t.depth = INLINE_LEVELS - 1;
    t.hops = MAX_HOPS - 1;
    t.expression(&call).unwrap();
    assert_eq!(t.hops, MAX_HOPS);
    t.depth = INLINE_LEVELS - 1;
    let e = fails(t.expression(&call));
    assert_eq!(e.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(
        t.depth,
        INLINE_LEVELS - 1,
        "the depth is restored on the error"
    );
}

#[test]
fn leaves_count_as_a_level_of_the_ir() {
    // The expression at the cap is a leaf one level below it: refused.
    let c = conf();
    let mut t = Translator::new(&c);
    t.depth = MAX_DEPTH;
    let e = fails(t.expression(&lit(LiteralType::Integer(1))));
    assert_eq!(e.error_class(), ErrorClass::FailedToParseTooComplex);
    t.depth = MAX_DEPTH - 1;
    t.expression(&lit(LiteralType::Integer(1))).unwrap();
}

#[test]
fn debug_of_a_write_hides_a_path_held_by_a_oneof() {
    use crate::proto::v4_2::{WriteOperation, write_operation::SaveType};
    let write = WriteOperation {
        save_type: Some(SaveType::Path(
            "https://h/p?X-Amz-Signature=Canary-4d1".into(),
        )),
        ..WriteOperation::default()
    };
    let text = format!("{write:?}");
    assert!(!text.contains("Canary-4d1"), "{text}");
    assert!(text.contains("Path(<redacted: 38 bytes>)"), "{text}");
}
