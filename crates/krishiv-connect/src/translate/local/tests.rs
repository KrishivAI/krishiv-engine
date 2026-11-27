// SPDX-License-Identifier: Apache-2.0
//! Tests of the local relation arms: the host protocol (requests, ordinals) and the refusals.

use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use krishiv_plan::{LocalRelation, PlanNode, ReadSource};
use krishiv_sql::ParserOptions;

use super::super::testing::{boxed, fails, relation};
use super::super::{TranslateConf, translate_relation};
use super::*;
use crate::proto::v4_3::{Join, Relation, join::JoinType, relation::RelType};

fn local(data: Option<&[u8]>, schema: Option<&str>) -> Relation {
    relation(RelType::LocalRelation(pb::LocalRelation {
        data: data.map(<[u8]>::to_vec),
        schema: schema.map(str::to_owned),
    }))
}

fn cached(hash: &str) -> Relation {
    relation(RelType::CachedLocalRelation(pb::CachedLocalRelation {
        hash: hash.to_owned(),
    }))
}

fn conf_with(local_relations: LocalRelations) -> TranslateConf {
    TranslateConf {
        local_relations,
        ..TranslateConf::default()
    }
}

fn built(rows: u64) -> LocalRelation {
    LocalRelation::from_decoded(Arc::new(arrow_schema::Schema::empty()), [7; 32], 1, rows, 0)
}

#[test]
fn the_first_pass_records_each_request_in_order() {
    let locals = LocalRelations::collecting();
    let join = relation(RelType::Join(Box::new(Join {
        left: boxed(local(Some(b"abc"), Some("a INT"))),
        right: boxed(cached("h1")),
        join_type: JoinType::Cross as i32,
        ..Join::default()
    })));
    translate_relation(&join, &conf_with(locals.clone())).expect("translates");
    let requests = locals.take_requests();
    assert_eq!(requests.len(), 2);
    let LocalRequest::Inline { data, schema } = &requests[0] else {
        panic!("{:?}", requests[0])
    };
    assert_eq!(data.as_deref(), Some(&b"abc"[..]));
    assert_eq!(schema.as_ref().map(|s| s.fields().len()), Some(1));
    assert_eq!(requests[1], LocalRequest::Cached { hash: "h1".into() });
}

#[test]
fn the_second_pass_hands_out_the_relation_of_each_request() {
    let join = relation(RelType::Join(Box::new(Join {
        left: boxed(local(Some(b"abc"), None)),
        right: boxed(local(None, Some("b LONG"))),
        join_type: JoinType::Cross as i32,
        ..Join::default()
    })));
    let conf = conf_with(LocalRelations::resolved(vec![built(1), built(2)]));
    let plan = translate_relation(&join, &conf).expect("translates");
    let PlanNode::Join { left, right, .. } = &plan.root().node else {
        panic!("not a join")
    };
    let rows = |p: &Arc<krishiv_plan::LogicalPlan>| match &p.node {
        PlanNode::Read(ReadSource::Local(l)) => l.rows(),
        other => panic!("{other:?}"),
    };
    assert_eq!((rows(left), rows(right)), (1, 2));
    // A host that built too few is an internal error, never a silent empty relation.
    let few = conf_with(LocalRelations::resolved(vec![built(1)]));
    assert_eq!(
        fails(translate_relation(&join, &few)).error_class(),
        ErrorClass::InternalError
    );
}

#[test]
fn no_data_and_no_schema_is_an_invalid_relation() {
    let e = fails(translate_relation(
        &local(None, None),
        &conf_with(LocalRelations::collecting()),
    ));
    assert_eq!(
        e.error_class().name(),
        "CONNECT_INVALID_PLAN.SCHEMA_REQUIRED_FOR_LOCAL_RELATION"
    );
    // A schema alone is an empty relation of that schema.
    let locals = LocalRelations::collecting();
    translate_relation(&local(None, Some("a INT")), &conf_with(locals.clone())).expect("ok");
    assert_eq!(locals.take_requests().len(), 1);
}

#[test]
fn a_bad_schema_text_fails_at_translation_and_is_not_echoed() {
    let e = fails(translate_relation(
        &local(Some(b"x"), Some("CANARY nonsense nonsense")),
        &conf_with(LocalRelations::collecting()),
    ));
    assert!(!e.message().contains("CANARY"), "{}", e.message());
    // The schema parser's own error, not the refusal of local relations.
    assert_ne!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature,
        "{}",
        e.message()
    );
}

#[test]
fn a_host_that_builds_none_refuses_local_relations() {
    for r in [local(Some(b"x"), None), cached("h")] {
        let e = fails(translate_relation(&r, &TranslateConf::default()));
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    }
}

#[test]
fn the_chunked_cached_relation_stays_refused() {
    let r = relation(RelType::ChunkedCachedLocalRelation(
        pb::ChunkedCachedLocalRelation::default(),
    ));
    let e = fails(translate_relation(
        &r,
        &conf_with(LocalRelations::collecting()),
    ));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSupported
    );
}

#[test]
fn local_relations_compare_by_mode() {
    assert_eq!(LocalRelations::default(), LocalRelations::default());
    let c = LocalRelations::collecting();
    assert_eq!(c, c.clone());
    assert_ne!(c, LocalRelations::collecting());
    assert_eq!(
        LocalRelations::resolved(vec![built(1)]),
        LocalRelations::resolved(vec![built(1)])
    );
    assert_ne!(c, LocalRelations::default());
    assert!(format!("{c:?}").contains("Collect"));
}

#[test]
fn a_cached_relation_is_read_as_the_inline_one_it_holds() {
    use prost::Message as _;
    let bytes = pb::LocalRelation {
        data: Some(b"ipc".to_vec()),
        schema: Some("a INT".to_owned()),
    }
    .encode_to_vec();
    let (data, schema) = cached_request(&bytes, &ParserOptions::default()).expect("decodes");
    assert_eq!(data.as_deref(), Some(&b"ipc"[..]));
    assert_eq!(schema.expect("schema").fields().len(), 1);
}

#[test]
fn a_cached_relation_that_does_not_decode_is_refused() {
    // A field key cut off mid-varint, then a valid tag of an unknown wire type (a group).
    for bytes in [&b"\xff\xff"[..], &b"\x0b"[..]] {
        assert_eq!(
            fails(cached_request(bytes, &ParserOptions::default())).error_class(),
            ErrorClass::ConnectInvalidPlanCannotParse
        );
    }
    // Neither data nor schema: as an inline relation.
    assert_eq!(
        fails(cached_request(&[], &ParserOptions::default())).error_class(),
        ErrorClass::ConnectInvalidPlanSchemaRequiredForLocalRelation
    );
}
