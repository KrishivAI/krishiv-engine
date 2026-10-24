// SPDX-License-Identifier: Apache-2.0
//! The feature header (ADR 0040): what the encoder declares, how a reader
//! checks it before reading a node, and how corrupt headers are refused.
#![expect(clippy::unwrap_used, reason = "test code")]

#[path = "common/inventory.rs"]
mod inventory;

use krishiv_exec::physical::PhysNode;
use krishiv_plan_codec::features::{Construct, feature_of};
use krishiv_plan_codec::proto as pb;
use krishiv_plan_codec::{
    CodecError, EPOCH, Feature, FeatureKind, Limits, PlanFeatures, ReaderFeatures, decode,
    decode_with, encode, features_of, plan_features, to_message,
};
use prost::Message as _;

use inventory::{cases, plan_of};

fn message(name: &str) -> pb::PhysicalPlan {
    let (_, plan) = cases().into_iter().find(|(n, _)| n == name).unwrap();
    to_message(&plan).unwrap()
}

fn corrupt(m: &pb::PhysicalPlan, expected: &str) {
    let err = decode(&m.encode_to_vec()).unwrap_err();
    assert_eq!(err, CodecError::Corrupt(expected.to_owned()));
}

#[test]
fn the_header_lists_exactly_the_features_in_use() {
    let m = message("window_ranking");
    assert_eq!(m.features, ["window"]);
    assert_eq!(m.epoch, EPOCH);
    assert_eq!(message("limit_offset").features, ["limit_offset"]);
    assert!(message("limit").features.is_empty());
    assert_eq!(
        message("window_frames").features,
        ["window", "window_frame_offsets"]
    );
    assert_eq!(
        message("window_offset_values").features,
        ["window", "window_frame_offsets", "window_offset_functions"]
    );
    // An offset of zero is the baseline `Limit`.
    let zero = plan_of(PhysNode::Limit {
        input: Box::new(PhysNode::OneRow),
        offset: 0,
        limit: 1,
    });
    assert!(plan_features(&zero).is_empty());
    // Every registered feature occurs in some inventory case.
    for f in Feature::ALL {
        assert!(
            cases().iter().any(|(_, p)| plan_features(p).contains(f)),
            "{f}: no inventory case uses it"
        );
    }
}

#[test]
fn a_feature_in_a_subquery_plan_counts() {
    let (_, window) = cases()
        .into_iter()
        .find(|(n, _)| n == "window_ranking")
        .unwrap();
    let plan = plan_of(PhysNode::Filter {
        input: Box::new(PhysNode::OneRow),
        layout: Vec::new(),
        predicate: inventory::predicate(),
        subqueries: vec![krishiv_exec::physical::SubqueryPlan {
            plan: window.root().clone(),
            ..inventory::subquery()
        }],
    });
    assert_eq!(
        plan_features(&plan).into_iter().collect::<Vec<_>>(),
        [Feature::Window]
    );
    assert_eq!(decode(&encode(&plan).unwrap()).unwrap(), plan);
}

#[test]
fn constructs_map_to_one_feature_or_the_baseline() {
    let (_, window) = cases()
        .into_iter()
        .find(|(n, _)| n == "window_ranking")
        .unwrap();
    assert_eq!(
        feature_of(Construct::Node(window.root())),
        Some(Feature::Window)
    );
    assert_eq!(feature_of(Construct::Node(&PhysNode::OneRow)), None);
    let null_of = krishiv_exec::physical::ProjectItemPlan::NullOf {
        index: 0,
        name: "x".to_owned(),
    };
    assert_eq!(feature_of(Construct::Item(&null_of)), Some(Feature::Expand));
    assert_eq!(
        feature_of(Construct::Join(krishiv_exec::JoinType::LeftSemi)),
        None
    );
    assert_eq!(
        feature_of(Construct::Partitioning(
            &krishiv_exec::physical::Partitioning::Single
        )),
        None
    );
    assert_eq!(
        feature_of(Construct::Window(
            &krishiv_exec::window::WindowFunction::RowNumber
        )),
        Some(Feature::Window)
    );
    assert_eq!(
        feature_of(Construct::Aggregate(
            krishiv_exec::aggregate::AggregateFunction::Sum
        )),
        None
    );
    // A frame needs a feature only for an offset bound: the `Window`
    // node's format carries the others, `CURRENT ROW` at both ends included.
    use krishiv_exec::window::{FrameBound, FrameUnits, WindowFrame};
    let frame = |lower, upper| WindowFrame {
        units: FrameUnits::Rows,
        lower,
        upper,
    };
    for (lower, upper) in [
        (
            FrameBound::UnboundedPreceding,
            FrameBound::UnboundedFollowing,
        ),
        (FrameBound::CurrentRow, FrameBound::CurrentRow),
        (FrameBound::CurrentRow, FrameBound::UnboundedFollowing),
    ] {
        assert_eq!(feature_of(Construct::Frame(&frame(lower, upper))), None);
    }
    for (lower, upper) in [
        (FrameBound::Offset(-1), FrameBound::CurrentRow),
        (FrameBound::UnboundedPreceding, FrameBound::Offset(2)),
    ] {
        assert_eq!(
            feature_of(Construct::Frame(&frame(lower, upper))),
            Some(Feature::WindowFrameOffsets)
        );
    }
    assert_eq!(
        feature_of(Construct::Window(
            &krishiv_exec::window::WindowFunction::NthValue {
                input: 0,
                n: 1,
                ignore_nulls: false,
                data_type: arrow_schema::DataType::Int64,
            }
        )),
        Some(Feature::WindowOffsetFunctions)
    );
    assert_eq!(Feature::Window.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::WindowFrameOffsets.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::WindowOffsetFunctions.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::LimitOffset.to_string(), "limit_offset");
    assert_eq!(Feature::from_id("no_such_feature"), None);
}

#[test]
fn an_unknown_feature_is_named_before_any_node_is_read() {
    let mut m = message("scan");
    m.features = vec!["zeta_future".to_owned()];
    // A node the reader would refuse: the feature check comes first.
    m.nodes[0].kind = None;
    let err = decode(&m.encode_to_vec()).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["zeta_future".to_owned()]
        }
    );
    assert_eq!(
        err.to_string(),
        "the plan uses features [zeta_future] that this reader does not implement"
    );
}

#[test]
fn a_reader_without_a_feature_refuses_the_plan_naming_it() {
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "window_ranking")
            .unwrap()
            .1,
    )
    .unwrap();
    let reader = ReaderFeatures::only([Feature::LimitOffset]);
    assert!(!reader.contains(Feature::Window));
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["window".to_owned()]
        }
    );
    assert_eq!(ReaderFeatures::default(), ReaderFeatures::all());
}

#[test]
fn an_unknown_epoch_is_refused() {
    let mut m = message("scan");
    m.epoch = EPOCH + 1;
    let err = decode(&m.encode_to_vec()).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedEpoch {
            epoch: EPOCH + 1,
            supported: EPOCH
        }
    );
    assert_eq!(
        err.to_string(),
        "the plan is of epoch 2, this reader implements epoch 1"
    );
}

#[test]
fn malformed_headers_are_corrupt() {
    for bad in ["", "Window", "1window", "win-dow", &"x".repeat(65)] {
        let mut m = message("window_ranking");
        m.features = vec![bad.to_owned()];
        corrupt(&m, "a malformed feature id");
    }
    let mut m = message("window_ranking");
    m.features = (0..257).map(|i| format!("f{i:04}")).collect();
    corrupt(&m, "more feature ids than any plan uses");
    let mut m = message("window_ranking");
    m.features = vec!["window".to_owned(), "window".to_owned()];
    corrupt(&m, "feature ids that are not sorted or repeat");
    let mut m = message("window_ranking");
    m.features = vec!["window".to_owned(), "limit_offset".to_owned()];
    corrupt(&m, "feature ids that are not sorted or repeat");
    // The legacy reader version must follow from the features.
    let mut m = message("window_ranking");
    m.min_reader_version = 2;
    corrupt(&m, "a reader version that does not match the features");
    let mut m = message("scan");
    m.format_version = 0;
    corrupt(&m, "no format version");
}

#[test]
fn features_of_reads_the_header_only() {
    let mut m = message("window_ranking");
    let header = features_of(&m.encode_to_vec()).unwrap();
    assert_eq!(
        header,
        PlanFeatures {
            epoch: EPOCH,
            features: vec!["window".to_owned()]
        }
    );
    assert_eq!(
        header.missing(&ReaderFeatures::only([])),
        ["window".to_owned()]
    );
    assert!(header.missing(&ReaderFeatures::all()).is_empty());
    // The tables are not decoded: a broken node does not matter.
    m.nodes[0].kind = None;
    assert_eq!(features_of(&m.encode_to_vec()).unwrap(), header);
    // An unknown id is reported, not refused.
    m.features = vec!["zeta_future".to_owned()];
    let header = features_of(&m.encode_to_vec()).unwrap();
    assert_eq!(header.features, ["zeta_future"]);
    assert_eq!(
        header.missing(&ReaderFeatures::all()),
        ["zeta_future".to_owned()]
    );
}

#[test]
fn features_of_a_legacy_plan_follow_from_its_content() {
    let mut m = message("limit_offset");
    m.epoch = 0;
    m.features.clear();
    let header = features_of(&m.encode_to_vec()).unwrap();
    assert_eq!(header.features, ["limit_offset"]);
    assert_eq!(header.epoch, EPOCH);
}

#[test]
fn features_of_refuses_what_it_cannot_interpret() {
    let mut m = message("scan");
    m.epoch = 9;
    assert!(matches!(
        features_of(&m.encode_to_vec()).unwrap_err(),
        CodecError::UnsupportedEpoch { epoch: 9, .. }
    ));
    assert_eq!(
        features_of(&[0xff, 0xff, 0xff, 0xff]).unwrap_err(),
        CodecError::Corrupt("malformed protobuf".to_owned())
    );
    let mut m = message("scan");
    m.features = vec!["b".to_owned(), "a".to_owned()];
    assert_eq!(
        features_of(&m.encode_to_vec()).unwrap_err(),
        CodecError::Corrupt("feature ids that are not sorted or repeat".to_owned())
    );
}

#[test]
fn nested_types_need_their_feature() {
    let m = message("nested_types");
    assert_eq!(m.features, ["nested_types"]);
    let bytes = m.encode_to_vec();
    // A reader without the feature refuses the plan before reading a node,
    // naming it; one with it reads the plan back.
    let older = ReaderFeatures::only(
        Feature::ALL
            .iter()
            .copied()
            .filter(|f| *f != Feature::NestedTypes),
    );
    assert_eq!(
        decode_with(&bytes, &Limits::default(), &older).unwrap_err(),
        CodecError::UnsupportedFeatures {
            missing: vec!["nested_types".to_owned()]
        }
    );
    let (_, plan) = cases()
        .into_iter()
        .find(|(n, _)| n == "nested_types")
        .unwrap();
    assert_eq!(decode(&bytes).unwrap(), plan);
    assert_eq!(Feature::NestedTypes.kind(), FeatureKind::Semantic);
    // The header cannot hide it.
    let mut hidden = m;
    hidden.features.clear();
    hidden.min_reader_version = 2;
    corrupt(
        &hidden,
        "the feature header does not match the plan's constructs",
    );
}

#[test]
fn a_malformed_nested_type_is_corrupt() {
    let m = message("nested_types");
    let nested = pb::data_type::Kind::Nested as i32;
    let with_type = |edit: &dyn Fn(&mut pb::DataType)| {
        let mut m = m.clone();
        let mut done = false;
        for e in &mut m.exprs {
            if let Some(pb::expr::Kind::Cast(c)) = &mut e.kind
                && let Some(t) = c.to.as_mut().filter(|t| t.kind == nested)
            {
                edit(t);
                done = true;
            }
        }
        assert!(done, "no nested cast in the case");
        m
    };
    for json in ["", "not json", r#""integer""#, &"[".repeat(10_000)] {
        corrupt(
            &with_type(&|t| t.json = json.to_owned()),
            "an invalid nested type",
        );
    }
    let deep = format!(
        "{}\"integer\"{}",
        r#"{"type":"array","elementType":"#.repeat(40),
        r#","containsNull":true}"#.repeat(40)
    );
    corrupt(
        &with_type(&|t| t.json = deep.clone()),
        "an invalid nested type",
    );
    corrupt(
        &with_type(&|t| {
            t.kind = pb::data_type::Kind::Integer as i32;
        }),
        "a JSON form on a type that is not nested",
    );
}

#[test]
fn a_type_too_deep_for_readers_does_not_encode() {
    use krishiv_types::{ArrayType, DataType};
    let mut t = DataType::Integer;
    for _ in 0..=DataType::MAX_NESTING_DEPTH {
        t = ArrayType::new(t, true).into();
    }
    let plan = plan_of(PhysNode::Project {
        input: Box::new(PhysNode::OneRow),
        layout: Vec::new(),
        items: vec![krishiv_exec::physical::ProjectItemPlan::Expr {
            name: "x".to_owned(),
            expr: inventory::lit(krishiv_plan::Literal::Null(Some(t))),
        }],
        nullable: None,
        subqueries: Vec::new(),
    });
    assert!(matches!(encode(&plan), Err(CodecError::Unsupported(_))));
}
