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
    for (scheme, feature) in [
        (krishiv_exec::physical::HashScheme::Unpinned, None),
        (
            krishiv_exec::physical::HashScheme::KeyGroupV1,
            Some(Feature::KeyGroupHash),
        ),
    ] {
        assert_eq!(
            feature_of(Construct::Partitioning(
                &krishiv_exec::physical::Partitioning::Hash {
                    keys: vec![0],
                    scheme
                }
            )),
            feature
        );
    }
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
    assert_eq!(Feature::Window.kind(), FeatureKind::Semantic);
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
