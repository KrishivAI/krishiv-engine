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

use inventory::{cases, plan_of, plan_of_env};

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
    // A lambda over arrays needs both features.
    assert_eq!(
        message("lambda_functions").features,
        ["lambda_functions", "nested_types"]
    );
    assert_eq!(
        message("window_frames").features,
        ["window", "window_frame_offsets"]
    );
    assert_eq!(
        message("window_offset_values").features,
        ["window", "window_frame_offsets", "window_offset_functions"]
    );
    assert_eq!(
        message("window_range_values").features,
        ["window", "window_range_bounds"]
    );
    assert_eq!(
        message("aggregate_mode_regr_bit_listagg").features,
        ["mode_regr_bit_listagg_aggs"]
    );
    // An offset of zero is the baseline `Limit`.
    let zero = plan_of(PhysNode::Limit {
        input: Box::new(PhysNode::OneRow),
        offset: 0,
        limit: 1,
    });
    assert!(plan_features(&zero).is_empty());
    assert_eq!(message("aggregate_filter").features, ["aggregate_filter"]);
    assert_eq!(message("session_zone_name").features, ["session_zone_name"]);
    assert_eq!(
        message("legacy_charset_switches").features,
        ["legacy_charset_switches"]
    );
    // Every registered feature occurs in some inventory case, except
    // `legacy_statistical_aggregate`: it fires only around a moment
    // aggregate, and the encoder refuses every moment aggregate (pinned by
    // the TPC-DS ratchet), so no encodable case can use it. The test below
    // pins its `plan_features` rule on plans that are never encoded.
    for f in Feature::ALL {
        if *f == Feature::LegacyStatisticalAggregate {
            continue;
        }
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

/// `legacy_statistical_aggregate` fires only for a plan that sets the flag
/// around a moment aggregate. The flag alone changes nothing the encoder
/// can write (it refuses every moment aggregate), so such plans list no
/// feature: a reader without it keeps reading them, and the flag itself
/// still round-trips.
#[test]
fn legacy_statistical_aggregate_needs_a_moment_aggregate() {
    use krishiv_exec::aggregate::{AggregateExpr, AggregateFunction, AggregateOutput};
    use krishiv_exec::physical::{AggregatePlan, SubqueryPlan};
    use krishiv_exec::window::{
        FrameUnits, WindowFrame, WindowFunction, WindowFunctionPlan, WindowPlan,
    };

    let moment = || {
        AggregateExpr::new(
            AggregateFunction::StddevSamp,
            0,
            arrow_schema::DataType::Float64,
            "s",
        )
    };
    let aggregate_node = || PhysNode::Aggregate {
        input: Box::new(inventory::scan()),
        aggregate: AggregatePlan {
            group_by: Vec::new(),
            aggregates: vec![moment()],
            distinct: vec![false],
            output: AggregateOutput::Final,
        },
    };
    let legacy = || {
        let mut env = inventory::env();
        env.legacy_statistical_aggregate = true;
        env
    };

    // The flag alone lists no feature.
    let (_, flag_only) = cases()
        .into_iter()
        .find(|(n, _)| n == "one_row_legacy_statistical_aggregate")
        .unwrap();
    assert!(plan_features(&flag_only).is_empty());
    assert!(to_message(&flag_only).unwrap().features.is_empty());
    // A reader without the feature still reads such a plan, flag included.
    let bytes = encode(&flag_only).unwrap();
    let without = ReaderFeatures::only([]);
    assert_eq!(
        decode_with(&bytes, &Limits::default(), &without).unwrap(),
        flag_only
    );
    // A moment aggregate without the flag lists none either.
    assert!(plan_features(&plan_of(aggregate_node())).is_empty());
    // Both together list the feature: in an aggregate node, ...
    let with_both = plan_of_env(aggregate_node(), legacy());
    assert_eq!(
        plan_features(&with_both).into_iter().collect::<Vec<_>>(),
        [Feature::LegacyStatisticalAggregate]
    );
    // ... in a window function, ...
    let window = plan_of_env(
        PhysNode::Window {
            input: Box::new(inventory::scan()),
            window: WindowPlan {
                partition_by: Vec::new(),
                order_by: Vec::new(),
                functions: vec![WindowFunctionPlan {
                    function: WindowFunction::Aggregate(moment()),
                    frame: WindowFrame::whole(FrameUnits::Rows),
                    filter: None,
                    name: "w".to_owned(),
                }],
            },
        },
        legacy(),
    );
    assert_eq!(
        plan_features(&window).into_iter().collect::<Vec<_>>(),
        [Feature::LegacyStatisticalAggregate, Feature::Window]
    );
    // ... and in a subquery plan ...
    let subquery = plan_of_env(
        PhysNode::Filter {
            input: Box::new(PhysNode::OneRow),
            layout: Vec::new(),
            predicate: inventory::predicate(),
            subqueries: vec![SubqueryPlan {
                plan: aggregate_node(),
                ..inventory::subquery()
            }],
        },
        legacy(),
    );
    assert_eq!(
        plan_features(&subquery).into_iter().collect::<Vec<_>>(),
        [Feature::LegacyStatisticalAggregate]
    );
    // ... but never for another aggregate under the flag.
    let plain = plan_of_env(
        PhysNode::Aggregate {
            input: Box::new(inventory::scan()),
            aggregate: AggregatePlan {
                group_by: Vec::new(),
                aggregates: vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    0,
                    arrow_schema::DataType::Float64,
                    "s",
                )],
                distinct: vec![false],
                output: AggregateOutput::Final,
            },
        },
        legacy(),
    );
    assert!(plan_features(&plain).is_empty());
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
    assert_eq!(
        feature_of(Construct::Lambda),
        Some(Feature::LambdaFunctions)
    );
    assert_eq!(Feature::LambdaFunctions.kind(), FeatureKind::Semantic);
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
        feature_of(Construct::Join(krishiv_exec::JoinType::Existence)),
        Some(Feature::ExistenceJoin)
    );
    // The single-match flag is a property of the hash join node: a plain
    // left outer join is the baseline, a single-match one needs the
    // feature. `distinct_build` needs none: a reader that ignores it builds
    // every row and returns the same result.
    let join = |single_match, distinct_build, join_type| {
        let (_, plan) = cases()
            .into_iter()
            .find(|(n, _)| n == "hash_join_single_match")
            .unwrap();
        match plan.root() {
            PhysNode::HashJoin(join) => feature_of(Construct::Node(&PhysNode::HashJoin(
                krishiv_exec::physical::HashJoinPlan {
                    single_match,
                    distinct_build,
                    join_type,
                    ..join.clone()
                },
            ))),
            _ => None,
        }
    };
    let (outer, semi) = (
        krishiv_exec::JoinType::LeftOuter,
        krishiv_exec::JoinType::LeftSemi,
    );
    assert_eq!(join(false, false, outer), None);
    assert_eq!(join(true, false, outer), Some(Feature::JoinSingleMatch));
    assert_eq!(join(false, true, semi), None);
    assert_eq!(Feature::ExistenceJoin.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::JoinSingleMatch.kind(), FeatureKind::Semantic);
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
            &krishiv_exec::aggregate::AggregateExpr::new(
                krishiv_exec::aggregate::AggregateFunction::Sum,
                0,
                arrow_schema::DataType::Int64,
                "s",
            )
        )),
        None
    );
    assert_eq!(
        feature_of(Construct::AggregateFunction(
            &krishiv_exec::aggregate::AggregateFunction::Sum
        )),
        None
    );
    assert_eq!(
        feature_of(Construct::Aggregate(
            &krishiv_exec::aggregate::AggregateExpr {
                mask: Some(1),
                ..krishiv_exec::aggregate::AggregateExpr::new(
                    krishiv_exec::aggregate::AggregateFunction::Sum,
                    0,
                    arrow_schema::DataType::Int64,
                    "s",
                )
            }
        )),
        Some(Feature::AggregateFilter)
    );
    assert_eq!(
        feature_of(Construct::AggregateFunction(
            &krishiv_exec::aggregate::AggregateFunction::CollectSet
        )),
        Some(Feature::CollectAggregates)
    );
    assert_eq!(
        feature_of(Construct::AggregateFunction(
            &krishiv_exec::aggregate::AggregateFunction::VectorAvg
        )),
        Some(Feature::CollectAggregates)
    );
    assert_eq!(
        feature_of(Construct::AggregateFunction(
            &krishiv_exec::aggregate::AggregateFunction::RegrR2
        )),
        Some(Feature::ModeRegrBitListaggAggs)
    );
    assert_eq!(
        feature_of(Construct::AggregateFunction(
            &krishiv_exec::aggregate::AggregateFunction::ListAgg {
                delimiter: Vec::new(),
                orders: Vec::new(),
            }
        )),
        Some(Feature::ModeRegrBitListaggAggs)
    );
    assert_eq!(
        feature_of(Construct::AggregateFunction(
            &krishiv_exec::aggregate::AggregateFunction::ApproxCountDistinct { p: 14 }
        )),
        Some(Feature::PercentileAggregates)
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
    for bound in [
        FrameBound::RangeInt(3),
        FrameBound::RangeFloat(1.0_f64.to_bits()),
    ] {
        assert_eq!(
            feature_of(Construct::Frame(&frame(
                FrameBound::UnboundedPreceding,
                bound
            ))),
            Some(Feature::WindowRangeBounds)
        );
    }
    assert_eq!(Feature::WindowRangeBounds.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::Window.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::WindowFrameOffsets.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::WindowOffsetFunctions.kind(), FeatureKind::Semantic);
    // A plain write plan is the `write_plan` feature; a layout (partition
    // columns) is the `partitioned_writes` feature.
    use krishiv_exec::physical::WritePlan;
    let base = inventory::write_plan();
    let node_of = |write: WritePlan| PhysNode::WriteFiles {
        input: Box::new(PhysNode::OneRow),
        write: Some(write),
    };
    assert_eq!(
        feature_of(Construct::Node(&node_of(base.clone()))),
        Some(Feature::WritePlan)
    );
    let write = WritePlan {
        partition_by: vec![0],
        ..base
    };
    let node = node_of(write);
    assert_eq!(
        feature_of(Construct::Node(&node)),
        Some(Feature::PartitionedWrites),
        "{node:?}"
    );
    assert_eq!(Feature::PartitionedWrites.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::PartitionedWrites.to_string(), "partitioned_writes");
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
    // A reader without `partial_aggregate` refuses states computed below a
    // join.
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "partial_aggregate")
            .unwrap()
            .1,
    )
    .unwrap();
    let err = decode_with(&bytes, &Limits::default(), &ReaderFeatures::only([])).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["partial_aggregate".to_owned()]
        }
    );
    // A reader without `shared_subplan` refuses a plan with shared plans.
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "shared_subplan")
            .unwrap()
            .1,
    )
    .unwrap();
    let err = decode_with(&bytes, &Limits::default(), &ReaderFeatures::only([])).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["shared_subplan".to_owned()]
        }
    );
    // A reader without `lambda_functions` refuses the lambda plan naming
    // the feature, even though it knows `nested_types`.
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "lambda_functions")
            .unwrap()
            .1,
    )
    .unwrap();
    let reader = ReaderFeatures::only([Feature::NestedTypes]);
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["lambda_functions".to_owned()]
        }
    );
    // A reader without `collect_aggregates` refuses the collect plan
    // naming the feature.
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "aggregate_collects")
            .unwrap()
            .1,
    )
    .unwrap();
    let reader = ReaderFeatures::only([Feature::NestedTypes]);
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["collect_aggregates".to_owned()]
        }
    );
    // A reader without `mode_regr_bit_listagg_aggs` refuses the plan of
    // the new aggregates naming it.
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "aggregate_mode_regr_bit_listagg")
            .unwrap()
            .1,
    )
    .unwrap();
    let reader = ReaderFeatures::only([Feature::NestedTypes]);
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["mode_regr_bit_listagg_aggs".to_owned()]
        }
    );
}

#[test]
fn a_reader_without_the_filter_feature_refuses_a_masked_aggregate() {
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "aggregate_filter")
            .unwrap()
            .1,
    )
    .unwrap();
    let reader = ReaderFeatures::only([Feature::Window]);
    assert!(!reader.contains(Feature::AggregateFilter));
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["aggregate_filter".to_owned()]
        }
    );
}

#[test]
fn a_reader_without_partitioned_writes_refuses_the_plan_naming_it() {
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "write_files_partitioned")
            .unwrap()
            .1,
    )
    .unwrap();
    // The unpartitioned write needs no such feature.
    let plain = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "write_files_with_plan")
            .unwrap()
            .1,
    )
    .unwrap();
    let reader = ReaderFeatures::only([Feature::WritePlan]);
    assert!(!reader.contains(Feature::PartitionedWrites));
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["partitioned_writes".to_owned()]
        }
    );
    assert_eq!(
        decode_with(&plain, &Limits::default(), &reader).unwrap(),
        decode(&plain).unwrap()
    );
    // A reader without `percentile_aggregates` refuses the percentile plan
    // naming the feature.
    let bytes = encode(
        &cases()
            .into_iter()
            .find(|(n, _)| n == "aggregate_percentiles")
            .unwrap()
            .1,
    )
    .unwrap();
    let reader = ReaderFeatures::only([Feature::NestedTypes]);
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["percentile_aggregates".to_owned()]
        }
    );
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
fn only_udf_plans_need_rust_udf_and_every_expression_position_counts() {
    // Negative: no inventory case but `rust_udf` uses it, and the
    // `expressions` case (a registry call) does not.
    for (name, plan) in cases() {
        if name == "rust_udf" {
            assert!(plan_features(&plan).contains(&Feature::RustUdf), "{name}");
        } else {
            assert!(
                !plan_features(&plan).contains(&Feature::RustUdf),
                "{name} must not need rust_udf"
            );
        }
    }
    // The UDF expression of the inventory case, reused in every position
    // holding an expression.
    let udf_expr = match inventory::udf_call() {
        PhysNode::Project { items, .. } => match &items[..] {
            [krishiv_exec::physical::ProjectItemPlan::Expr { expr, .. }] => {
                std::sync::Arc::clone(expr)
            }
            _ => panic!("inventory udf_call holds one expression"),
        },
        _ => panic!("inventory udf_call is a project"),
    };
    let builtin_expr = inventory::predicate();
    // A UDF in a filter predicate needs the feature; a built-in does not.
    for (predicate, wants) in [(&udf_expr, true), (&builtin_expr, false)] {
        let plan = plan_of(PhysNode::Filter {
            input: Box::new(PhysNode::OneRow),
            layout: Vec::new(),
            predicate: std::sync::Arc::clone(predicate),
            subqueries: Vec::new(),
        });
        assert_eq!(plan_features(&plan).contains(&Feature::RustUdf), wants);
    }
    // A UDF in a scan's pushed filter needs the feature; a built-in does not.
    for (predicate, wants) in [(&udf_expr, true), (&builtin_expr, false)] {
        let mut scan = inventory::table_scan();
        scan.filter = Some(krishiv_exec::physical::ScanFilter {
            layout: vec![krishiv_plan::AttrId(1)],
            predicate: std::sync::Arc::clone(predicate),
            exact: true,
        });
        let plan = plan_of(PhysNode::Scan(scan));
        assert_eq!(plan_features(&plan).contains(&Feature::RustUdf), wants);
    }
    // A UDF in a join residual needs the feature; a built-in does not.
    for (conjunct, wants) in [(&udf_expr, true), (&builtin_expr, false)] {
        let residual = krishiv_exec::physical::Residual {
            layout: vec![krishiv_plan::AttrId(1)],
            conjuncts: vec![std::sync::Arc::clone(conjunct)],
        };
        let plan = plan_of(PhysNode::HashJoin(krishiv_exec::physical::HashJoinPlan {
            probe: Box::new(PhysNode::OneRow),
            build: Box::new(PhysNode::OneRow),
            join_type: krishiv_exec::JoinType::Inner,
            probe_keys: Vec::new(),
            build_keys: Vec::new(),
            null_safe: Vec::new(),
            residual: Some(residual),
            single_match: false,
            distinct_build: false,
            runtime_filters: Vec::new(),
            distribution: krishiv_exec::physical::JoinDistribution::Broadcast,
        }));
        assert_eq!(plan_features(&plan).contains(&Feature::RustUdf), wants);
    }
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
fn time_and_spatial_types_need_their_features() {
    use krishiv_types::{DataType, Srid};
    assert_eq!(message("time_project").features, ["time_type"]);
    assert_eq!(message("geo_scan").features, ["geo_types"]);
    assert_eq!(message("geo_join_residual").features, ["geo_types"]);
    let geography = DataType::Geography { srid: Srid::Any };
    assert_eq!(
        feature_of(Construct::Type(&geography)),
        Some(Feature::GeoTypes)
    );
    assert_eq!(
        feature_of(Construct::Type(&DataType::Time { precision: 6 })),
        Some(Feature::TimeType)
    );
    assert_eq!(feature_of(Construct::Type(&DataType::TimestampNtz)), None);
    for f in [Feature::TimeType, Feature::GeoTypes] {
        assert_eq!(f.kind(), FeatureKind::Semantic);
        assert_eq!(f.legacy_level(), 6);
    }
    // A reader without the feature refuses the plan before reading a node.
    let bytes = message("time_project").encode_to_vec();
    let older = ReaderFeatures::only(
        Feature::ALL
            .iter()
            .copied()
            .filter(|f| *f != Feature::TimeType),
    );
    assert_eq!(
        decode_with(&bytes, &Limits::default(), &older).unwrap_err(),
        CodecError::UnsupportedFeatures {
            missing: vec!["time_type".to_owned()]
        }
    );
    // A header that hides the type is corrupt.
    let mut m = message("geo_scan");
    m.features.clear();
    m.min_reader_version = 2;
    corrupt(
        &m,
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
    // Only the canonical form (the encoder's) decodes.
    corrupt(
        &with_type(&|t| t.json = t.json.replace(':', ": ")),
        "an invalid nested type",
    );
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

/// An Arrow type holding an array, a map or a struct.
fn arrow_list() -> arrow_schema::DataType {
    use arrow_schema::{DataType as A, Field};
    A::List(std::sync::Arc::new(Field::new("item", A::Int32, true)))
}

#[test]
fn a_nested_type_only_in_an_arrow_field_needs_the_feature() {
    use arrow_schema::{DataType as A, Field, Fields};
    use krishiv_exec::physical::{AggregatePlan, SubqueryPlan};
    let aggregate = {
        let mut a = inventory::aggregate_exprs().remove(1);
        a.arg_type = arrow_list();
        a
    };
    let struct_type = A::Struct(Fields::from(vec![Field::new("a", A::Int32, true)]));
    let with_schema = |t: A| {
        let plan = plan_of(PhysNode::OneRow);
        krishiv_exec::physical::PhysicalPlan::new(
            plan.root().clone(),
            std::sync::Arc::new(arrow_schema::Schema::new(vec![Field::new("c", t, true)])),
            plan.explain().clone(),
            plan.env().clone(),
        )
    };
    let plans = [
        // A subquery's value type.
        plan_of(PhysNode::Filter {
            input: Box::new(PhysNode::OneRow),
            layout: Vec::new(),
            predicate: inventory::predicate(),
            subqueries: vec![SubqueryPlan {
                data_type: arrow_list(),
                ..inventory::subquery()
            }],
        }),
        // An aggregate's argument type.
        plan_of(PhysNode::Aggregate {
            input: Box::new(PhysNode::OneRow),
            aggregate: AggregatePlan {
                group_by: Vec::new(),
                distinct: vec![false],
                aggregates: vec![aggregate],
                output: krishiv_exec::aggregate::AggregateOutput::Partial,
            },
        }),
        // The plan's output schema.
        with_schema(struct_type),
    ];
    let older = ReaderFeatures::only(
        Feature::ALL
            .iter()
            .copied()
            .filter(|f| *f != Feature::NestedTypes),
    );
    for plan in plans {
        assert!(plan_features(&plan).contains(&Feature::NestedTypes));
        let bytes = encode(&plan).unwrap();
        assert_eq!(
            decode_with(&bytes, &Limits::default(), &older).unwrap_err(),
            CodecError::UnsupportedFeatures {
                missing: vec!["nested_types".to_owned()]
            }
        );
        assert_eq!(decode(&bytes).unwrap(), plan);
    }
}

/// A hash exchange naming a routing function this reader does not know is
/// corrupt: it is never routed by some other function.
#[test]
fn an_unknown_hash_scheme_is_corrupt() {
    let mut m = message("exchange_hash_key_group");
    let mut hashed = 0;
    for n in &mut m.nodes {
        if let Some(pb::node::Kind::Exchange(e)) = &mut n.kind
            && let Some(pb::partitioning::Kind::Hash(h)) =
                e.partitioning.as_mut().and_then(|p| p.kind.as_mut())
        {
            h.scheme = 99;
            hashed += 1;
        }
    }
    assert_eq!(hashed, 1);
    corrupt(&m, "an unknown hash scheme");
}

/// A `Subquery` expression's `IN` values are entered when scanning for UDF
/// calls, so a UDF hiding there still needs the feature.
#[test]
fn subquery_in_values_with_a_udf_need_rust_udf() {
    use krishiv_plan::{Expr, ExprNode, SubqueryKind};
    let udf_expr = match inventory::udf_call() {
        krishiv_exec::physical::PhysNode::Project { items, .. } => match &items[..] {
            [krishiv_exec::physical::ProjectItemPlan::Expr { expr, .. }] => {
                std::sync::Arc::clone(expr)
            }
            _ => panic!("inventory udf_call holds one expression"),
        },
        _ => panic!("inventory udf_call is a project"),
    };
    let subquery = std::sync::Arc::new(Expr::new(ExprNode::Subquery {
        kind: SubqueryKind::In(vec![std::sync::Arc::clone(&udf_expr)]),
        plan: std::sync::Arc::new(krishiv_plan::LogicalPlan::new(
            krishiv_plan::PlanNode::OneRowRelation,
        )),
    }));
    let plan = plan_of(krishiv_exec::physical::PhysNode::Filter {
        input: Box::new(krishiv_exec::physical::PhysNode::OneRow),
        layout: Vec::new(),
        predicate: subquery,
        subqueries: Vec::new(),
    });
    assert!(plan_features(&plan).contains(&Feature::RustUdf));
}

#[test]
fn type_parameters_are_checked_on_decode() {
    use pb::data_type::Kind;
    // `geo_scan`'s first output column is `geometry(any)`.
    let edit = |f: &dyn Fn(&mut pb::DataType)| {
        let mut m = message("geo_scan");
        let Some(pb::node::Kind::Scan(scan)) = m.nodes[0].kind.as_mut() else {
            panic!("not a scan");
        };
        f(scan.output[0].data_type.as_mut().unwrap());
        m
    };
    let srid = |kind: Kind, srid: i32| {
        edit(&move |t: &mut pb::DataType| {
            t.kind = kind as i32;
            t.srid = srid;
        })
    };
    assert!(decode(&srid(Kind::Geometry, 3857).encode_to_vec()).is_ok());
    corrupt(&srid(Kind::Geometry, 9999), "an invalid geometry type");
    corrupt(&srid(Kind::Geography, 3857), "an invalid geography type");
    let time = |precision: u32, scale: u32| {
        let mut m = edit(&move |t: &mut pb::DataType| {
            t.kind = Kind::Time as i32;
            t.srid = 0;
            t.precision = precision;
            t.scale = scale;
        });
        m.features = vec!["geo_types".to_owned(), "time_type".to_owned()];
        m
    };
    corrupt(&time(7, 0), "an invalid time type");
    corrupt(&time(3, 1), "a type parameter on a type that has none");
    let srid_on_int = edit(&|t: &mut pb::DataType| {
        t.kind = Kind::Integer as i32;
        t.srid = 4326;
    });
    corrupt(&srid_on_int, "a type parameter on a type that has none");
}

#[test]
fn interval_types_and_literals_need_their_feature() {
    use krishiv_plan::{Expr, ExprNode, Literal, YearMonthField};
    use krishiv_types::DataType;
    assert_eq!(message("interval_literals").features, ["interval_types"]);
    assert_eq!(message("interval_scan").features, ["interval_types"]);
    assert_eq!(
        message("interval_nested").features,
        ["interval_types", "nested_types"]
    );
    for t in [
        DataType::DEFAULT_YEAR_MONTH_INTERVAL,
        DataType::DEFAULT_DAY_TIME_INTERVAL,
        DataType::CalendarInterval,
    ] {
        assert_eq!(
            feature_of(Construct::Type(&t)),
            Some(Feature::IntervalTypes)
        );
        let arrow = t.to_arrow();
        assert_eq!(
            feature_of(Construct::ArrowType(&arrow)),
            Some(Feature::IntervalTypes)
        );
    }
    // The bare literal predates the feature (`base/expressions` holds one);
    // as an argument it needs it, since nothing else then names the type.
    let literal = std::sync::Arc::new(Expr::new(ExprNode::Literal(Literal::YearMonthInterval {
        months: 1,
        start: YearMonthField::Month,
        end: YearMonthField::Month,
    })));
    assert_eq!(feature_of(Construct::Expr(&literal)), None);
    let cast = Expr::new(ExprNode::Cast {
        expr: std::sync::Arc::clone(&literal),
        to: DataType::String,
        mode: krishiv_plan::CastMode::Ansi,
    });
    assert_eq!(
        feature_of(Construct::Expr(&cast)),
        Some(Feature::IntervalTypes)
    );
    assert_eq!(Feature::IntervalTypes.kind(), FeatureKind::Semantic);
    assert_eq!(Feature::IntervalTypes.legacy_level(), 6);
    let older = ReaderFeatures::only(
        Feature::ALL
            .iter()
            .copied()
            .filter(|f| *f != Feature::IntervalTypes),
    );
    for case in ["interval_literals", "interval_scan", "interval_nested"] {
        let bytes = message(case).encode_to_vec();
        assert_eq!(
            decode_with(&bytes, &Limits::default(), &older).unwrap_err(),
            CodecError::UnsupportedFeatures {
                missing: vec!["interval_types".to_owned()]
            },
            "{case}"
        );
        assert!(decode(&bytes).is_ok(), "{case}");
    }
}

/// An interval type's fields are external input: a missing field, fields
/// that do not increase and fields on another type are corrupt.
#[test]
fn a_decoded_interval_type_is_checked() {
    use pb::data_type::Kind;
    let edit = |change: &dyn Fn(&mut pb::DataType)| {
        let mut m = message("interval_scan");
        let Some(pb::node::Kind::Scan(scan)) = m.nodes[0].kind.as_mut() else {
            panic!("not a scan");
        };
        change(scan.output[0].data_type.as_mut().unwrap());
        m
    };
    let ym = |start: pb::YearMonthField, end: pb::YearMonthField| {
        edit(&move |t: &mut pb::DataType| {
            t.kind = Kind::YearMonthInterval as i32;
            t.ym_start = start as i32;
            t.ym_end = end as i32;
        })
    };
    assert!(
        decode(&ym(pb::YearMonthField::Year, pb::YearMonthField::Month).encode_to_vec()).is_ok()
    );
    corrupt(
        &ym(pb::YearMonthField::Month, pb::YearMonthField::Year),
        "an invalid interval type",
    );
    let dt = edit(&|t: &mut pb::DataType| {
        t.kind = Kind::DayTimeInterval as i32;
        t.ym_start = 0;
        t.ym_end = 0;
        t.dt_start = pb::DayTimeField::Second as i32;
        t.dt_end = pb::DayTimeField::Hour as i32;
    });
    corrupt(&dt, "an invalid interval type");
    let on_calendar = edit(&|t: &mut pb::DataType| {
        t.kind = Kind::CalendarInterval as i32;
        t.dt_start = pb::DayTimeField::Hour as i32;
    });
    corrupt(&on_calendar, "a type parameter on a type that has none");
    let on_int = edit(&|t: &mut pb::DataType| {
        t.kind = Kind::Integer as i32;
        t.ym_start = pb::YearMonthField::Month as i32;
    });
    corrupt(&on_int, "a type parameter on a type that has none");
    let missing = edit(&|t: &mut pb::DataType| {
        t.kind = Kind::YearMonthInterval as i32;
        t.ym_start = 0;
        t.ym_end = pb::YearMonthField::Month as i32;
    });
    corrupt(&missing, "an unknown interval field");
}

/// A literal's fields are external input too, for both field families.
#[test]
fn a_decoded_interval_literal_is_checked() {
    let edit = |change: &dyn Fn(&mut pb::literal::Kind)| {
        let mut m = message("interval_literals");
        for e in &mut m.exprs {
            if let Some(pb::expr::Kind::Literal(pb::Literal { kind: Some(k) })) = &mut e.kind {
                change(k);
            }
        }
        m
    };
    let year_month = edit(&|k: &mut pb::literal::Kind| {
        if let pb::literal::Kind::YearMonth(v) = k {
            v.start = pb::YearMonthField::Month as i32;
            v.end = pb::YearMonthField::Year as i32;
        }
    });
    corrupt(&year_month, "an invalid interval literal");
    let day_time = edit(&|k: &mut pb::literal::Kind| {
        if let pb::literal::Kind::DayTime(v) = k {
            v.start = pb::DayTimeField::Second as i32;
            v.end = pb::DayTimeField::Hour as i32;
        }
    });
    corrupt(&day_time, "an invalid interval literal");
}

/// A time inside an array is a time plan: the header names `time_type`
/// (beside `nested_types`), so a reader without it refuses the plan as
/// unsupported rather than misreading the nested type as corrupt.
#[test]
fn a_time_inside_a_nested_type_needs_time_type() {
    use krishiv_plan::{Expr, ExprNode, Literal};
    use krishiv_types::{ArrayType, DataType, MapType, StructField, StructType};
    let time = DataType::Time { precision: 3 };
    for nested in [
        DataType::Array(ArrayType::new(time.clone(), true).into()),
        DataType::Map(MapType::new(DataType::String, time.clone(), true).into()),
        DataType::Struct(StructType::new([StructField::new("t", time, true)])),
    ] {
        let plan = plan_of(PhysNode::Project {
            input: Box::new(PhysNode::OneRow),
            layout: Vec::new(),
            items: vec![krishiv_exec::physical::ProjectItemPlan::Expr {
                name: "x".to_owned(),
                expr: std::sync::Arc::new(Expr::new(ExprNode::Literal(Literal::Null(Some(
                    nested.clone(),
                ))))),
            }],
            nullable: None,
            subqueries: Vec::new(),
        });
        let features = plan_features(&plan);
        assert!(features.contains(&Feature::TimeType), "{nested:?}");
        let bytes = encode(&plan).unwrap();
        let older = ReaderFeatures::only(
            Feature::ALL
                .iter()
                .copied()
                .filter(|f| *f != Feature::TimeType),
        );
        assert_eq!(
            decode_with(&bytes, &Limits::default(), &older).unwrap_err(),
            CodecError::UnsupportedFeatures {
                missing: vec!["time_type".to_owned()]
            },
            "{nested:?}"
        );
        assert_eq!(decode(&bytes).unwrap(), plan);
    }
}

/// A `Range` node of a decoded plan is external input: a zero step (which
/// would divide by zero counting rows) and a zero slice count are corrupt.
#[test]
fn range_nodes_with_a_zero_step_or_zero_slices_are_corrupt() {
    let edit = |f: &dyn Fn(&mut pb::RangeNode)| {
        let mut m = message("range");
        for node in &mut m.nodes {
            if let Some(pb::node::Kind::Range(r)) = node.kind.as_mut() {
                f(r);
            }
        }
        m
    };
    assert!(decode(&edit(&|_| {}).encode_to_vec()).is_ok());
    corrupt(&edit(&|r| r.step = 0), "a range with step 0");
    corrupt(&edit(&|r| r.num_slices = Some(0)), "a range with 0 slices");
}

/// `partial_aggregate` is a partial aggregate below a join, on either side
/// and through other nodes; one above any join stays the older, plain form.
#[test]
fn a_partial_aggregate_is_a_feature_only_below_a_join() {
    use inventory::{join_plan, partial_aggregate};
    use krishiv_exec::JoinType;
    use krishiv_exec::physical::HashJoinPlan;
    use krishiv_plan_codec::features::plan_features;

    let has = |root: PhysNode| plan_features(&plan_of(root)).contains(&Feature::PartialAggregate);
    assert!(!has(partial_aggregate()));
    assert!(!has(PhysNode::HashJoin(join_plan(JoinType::Inner))));
    assert!(has(PhysNode::HashJoin(HashJoinPlan {
        probe: Box::new(partial_aggregate()),
        ..join_plan(JoinType::Inner)
    })));
    assert!(has(PhysNode::HashJoin(HashJoinPlan {
        build: Box::new(PhysNode::Limit {
            input: Box::new(partial_aggregate()),
            offset: 0,
            limit: 1,
        }),
        ..join_plan(JoinType::Inner)
    })));
    // In a shared plan too: every tree of the plan counts.
    let shared = plan_of(PhysNode::OneRow)
        .with_shared(vec![krishiv_exec::physical::SharedPlan {
            id: krishiv_exec::physical::SharedId(0),
            plan: PhysNode::HashJoin(HashJoinPlan {
                probe: Box::new(partial_aggregate()),
                ..join_plan(JoinType::Inner)
            }),
            explain: krishiv_exec::planner::ExplainNode::leaf("HashJoin", ""),
            deterministic: true,
            ordered: false,
        }])
        .unwrap();
    assert!(plan_features(&shared).contains(&Feature::PartialAggregate));
}

/// The header covers what a shared plan holds, and a listed plan with no
/// reader still needs the feature.
#[test]
fn a_shared_plans_constructs_are_in_the_header() {
    use krishiv_exec::physical::{SharedId, SharedPlan};
    use krishiv_exec::planner::ExplainNode;
    use krishiv_plan_codec::features::plan_features;

    let (_, windowed) = cases()
        .into_iter()
        .find(|(n, _)| n == "window_ranking")
        .unwrap();
    let plan = plan_of(PhysNode::OneRow)
        .with_shared(vec![SharedPlan {
            id: SharedId(0),
            plan: windowed.root().clone(),
            explain: ExplainNode::leaf("Window", ""),
            deterministic: true,
            ordered: false,
        }])
        .unwrap();
    let features = plan_features(&plan);
    assert!(features.contains(&Feature::Window), "{features:?}");
    assert!(features.contains(&Feature::SharedSubplan), "{features:?}");
    assert_eq!(decode(&encode(&plan).unwrap()).unwrap(), plan);
}
