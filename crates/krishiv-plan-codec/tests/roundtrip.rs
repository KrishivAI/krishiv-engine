// SPDX-License-Identifier: Apache-2.0
//! The round-trip inventory (every `PhysNode` variant and every kind of
//! value a plan holds), the secrets rule (ADR 0038) and the decoder's
//! defences against corrupt bytes.
#![expect(clippy::unwrap_used, reason = "test code")]

#[path = "common/inventory.rs"]
mod inventory;

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::secret::OptionValue;
use krishiv_exec::physical::{PhysNode, PhysicalPlan};
use krishiv_plan::{Expr, ReadSource};
use krishiv_plan_codec::proto as pb;
use krishiv_plan_codec::{
    CodecError, FORMAT_VERSION, Feature, FeatureKind, Limits, MIN_READER_VERSION,
    POST_MIGRATION_READER_VERSION, ReaderFeatures, decode, decode_with, encode, plan_features,
    to_message,
};
use prost::Message as _;

use inventory::{cases, plan_of};

/// A test-local name for each `PhysNode` variant. `variant_of` matches
/// without a wildcard, so a new `PhysNode` variant does not compile here
/// until it is listed; `variants!` makes the list `ALL`, and the inventory
/// test fails until a case of the new variant exists.
macro_rules! variants {
    ($($v:ident),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        enum Variant { $($v),* }
        const ALL: &[Variant] = &[$(Variant::$v),*];
    };
}

variants!(
    Scan,
    OneRow,
    Range,
    Filter,
    Project,
    Limit,
    ReplicateRows,
    Expand,
    Sort,
    TopN,
    Window,
    WindowGroupLimit,
    Aggregate,
    HashJoin,
    NestedLoopJoin,
    Union,
    WriteFiles,
    Exchange,
);

fn variant_of(node: &PhysNode) -> Variant {
    match node {
        PhysNode::Scan(_) => Variant::Scan,
        PhysNode::OneRow => Variant::OneRow,
        PhysNode::Range(_) => Variant::Range,
        PhysNode::Filter { .. } => Variant::Filter,
        PhysNode::Project { .. } => Variant::Project,
        PhysNode::Limit { .. } => Variant::Limit,
        PhysNode::ReplicateRows { .. } => Variant::ReplicateRows,
        PhysNode::Expand { .. } => Variant::Expand,
        PhysNode::Sort { .. } => Variant::Sort,
        PhysNode::TopN { .. } => Variant::TopN,
        PhysNode::Window { .. } => Variant::Window,
        PhysNode::WindowGroupLimit { .. } => Variant::WindowGroupLimit,
        PhysNode::Aggregate { .. } => Variant::Aggregate,
        PhysNode::HashJoin(_) => Variant::HashJoin,
        PhysNode::NestedLoopJoin(_) => Variant::NestedLoopJoin,
        PhysNode::Union { .. } => Variant::Union,
        PhysNode::WriteFiles { .. } => Variant::WriteFiles,
        PhysNode::Exchange(_) => Variant::Exchange,
    }
}

/// Every variant that occurs in `node`'s tree, subquery plans included.
fn variants_in(node: &PhysNode, out: &mut Vec<Variant>) {
    out.push(variant_of(node));
    for child in node.children() {
        variants_in(child, out);
    }
}

fn round_trip(plan: &PhysicalPlan) -> PhysicalPlan {
    decode(&encode(plan).unwrap()).unwrap()
}

#[test]
fn every_variant_has_a_case_and_round_trips() {
    let mut seen: Vec<Variant> = Vec::new();
    for (name, plan) in cases() {
        variants_in(plan.root(), &mut seen);
        let bytes = encode(&plan).unwrap();
        assert_eq!(decode(&bytes).unwrap(), plan, "{name}");
        assert_eq!(
            encode(&plan).unwrap(),
            bytes,
            "{name}: encoding is deterministic"
        );
        // Encoding the decoded plan gives the same bytes.
        assert_eq!(encode(&decode(&bytes).unwrap()).unwrap(), bytes, "{name}");
    }
    seen.sort();
    seen.dedup();
    let missing: Vec<_> = ALL.iter().filter(|v| !seen.contains(v)).collect();
    assert!(missing.is_empty(), "no round-trip case for {missing:?}");
}

#[test]
fn a_shared_subexpression_is_stored_once() {
    // `a > 10 AND a IS NOT NULL`: the column, the literal and three calls.
    let predicate = inventory::predicate();
    let plan = plan_of(PhysNode::Filter {
        input: Box::new(PhysNode::OneRow),
        layout: Vec::new(),
        predicate: Arc::clone(&predicate),
        subqueries: Vec::new(),
    });
    assert_eq!(to_message(&plan).unwrap().exprs.len(), 5);
    assert_eq!(round_trip(&plan), plan);
}

#[test]
fn a_deep_expression_is_not_encoded_recursively() {
    let mut e = inventory::col(1, "a", krishiv_types::DataType::Boolean);
    for _ in 0..900 {
        e = inventory::call(
            krishiv_plan::Builtin::Not,
            vec![e],
            krishiv_types::DataType::Boolean,
        );
    }
    let plan = plan_of(PhysNode::Filter {
        input: Box::new(PhysNode::OneRow),
        layout: Vec::new(),
        predicate: e,
        subqueries: Vec::new(),
    });
    assert_eq!(round_trip(&plan), plan);
}

fn scan_with(options: BTreeMap<String, OptionValue>) -> PhysicalPlan {
    let mut scan = inventory::table_scan();
    scan.scan.source = ReadSource::Table {
        name: vec!["t".to_owned()],
        options,
    };
    plan_of(PhysNode::Scan(scan))
}

fn write_with(options: BTreeMap<String, OptionValue>) -> PhysicalPlan {
    let mut write = inventory::write_plan();
    write.options = options;
    plan_of(PhysNode::WriteFiles {
        input: Box::new(inventory::scan()),
        write: Some(write),
    })
}

#[test]
fn a_write_plan_with_a_secret_reference_round_trips_without_the_secret() {
    let canary = "canary-write-4d2a-secret";
    let plan = write_with(BTreeMap::from([
        (
            "compression".to_owned(),
            OptionValue::Plain("zstd".to_owned()),
        ),
        (
            "token".to_owned(),
            OptionValue::reference("prod-scope", "warehouse-key"),
        ),
    ]));
    let bytes = encode(&plan).unwrap();
    for needle in [&b"prod-scope"[..], b"warehouse-key"] {
        assert!(bytes.windows(needle.len()).any(|w| w == needle));
    }
    assert!(!bytes.windows(canary.len()).any(|w| w == canary.as_bytes()));
    assert_eq!(decode(&bytes).unwrap(), plan);
}

#[test]
fn a_write_plan_with_an_inline_secret_is_refused_without_naming_it() {
    let canary = "canary-write-9f3a-hunter2";
    let plan = write_with(BTreeMap::from([(
        "password".to_owned(),
        OptionValue::from_user("password", canary),
    )]));
    let err = encode(&plan).unwrap_err();
    assert!(matches!(err, CodecError::Unsupported(_)), "{err:?}");
    let text = format!("{err} {err:?}");
    assert!(text.contains("`password` is an inline secret"), "{text}");
    assert!(!text.contains(canary), "{text}");
}

#[test]
fn a_write_plan_with_a_url_with_credentials_is_refused_without_naming_it() {
    let canary = "canary-write-7c21-sig";
    let url = format!("s3://bucket/data?X-Amz-Signature={canary}");
    let option = OptionValue::from_user("path", url);
    assert!(matches!(option, OptionValue::Url(_)), "not a URL option");
    let plan = write_with(BTreeMap::from([("path".to_owned(), option)]));
    let err = encode(&plan).unwrap_err();
    let text = format!("{err} {err:?}");
    assert!(text.contains("`path` is a URL with credentials"), "{text}");
    assert!(!text.contains(canary), "{text}");
}

#[test]
fn a_reader_without_write_plan_refuses_the_plan_by_name() {
    let (_, plan) = cases()
        .into_iter()
        .find(|(n, _)| n == "write_files_with_plan")
        .unwrap();
    assert!(plan_features(&plan).contains(&Feature::WritePlan));
    let bytes = encode(&plan).unwrap();
    let reader = ReaderFeatures::only([Feature::LimitOffset]);
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["write_plan".to_owned()]
        }
    );
    assert_eq!(decode(&bytes).unwrap(), plan);
}

#[test]
fn a_write_plan_without_a_format_job_id_or_store_is_corrupt() {
    let m = message("write_files_with_plan");
    let edit = |f: &dyn Fn(&mut pb::WritePlan)| {
        let mut m = m.clone();
        for node in &mut m.nodes {
            if let Some(pb::node::Kind::WriteFiles(w)) = &mut node.kind
                && let Some(write) = w.write.as_mut()
            {
                f(write);
            }
        }
        m
    };
    corrupt(
        &edit(&|w| w.format.clear()),
        "a write plan without a format",
    );
    corrupt(
        &edit(&|w| w.job_uuid.clear()),
        "a write plan without a job id",
    );
    corrupt(&edit(&|w| w.store.clear()), "a write plan without a store");
    // An absent `write` is a plan written before the `write_plan`
    // feature, not corruption: it decodes to the legacy plan.
    let mut missing = message("write_files_with_plan");
    for node in &mut missing.nodes {
        if let Some(pb::node::Kind::WriteFiles(w)) = &mut node.kind {
            w.write = None;
        }
    }
    missing.features.retain(|f| f != "write_plan");
    missing.min_reader_version = MIN_READER_VERSION;
    let (_, legacy) = cases()
        .into_iter()
        .find(|(n, _)| n == "write_files")
        .unwrap();
    assert_eq!(decode(&missing.encode_to_vec()).unwrap(), legacy);
}

#[test]
fn a_write_without_a_write_plan_is_the_baseline() {
    let (_, plan) = cases()
        .into_iter()
        .find(|(n, _)| n == "write_files")
        .unwrap();
    assert!(plan_features(&plan).is_empty());
    let m = to_message(&plan).unwrap();
    assert!(m.features.is_empty());
    assert_eq!(m.min_reader_version, MIN_READER_VERSION);
    for node in &m.nodes {
        if let Some(pb::node::Kind::WriteFiles(w)) = &node.kind {
            assert!(w.write.is_none());
        }
    }
    assert_eq!(round_trip(&plan), plan);
}

#[test]
fn an_inline_secret_is_refused_without_naming_it() {
    let canary = "canary-9f3a-hunter2";
    let plan = scan_with(BTreeMap::from([(
        "password".to_owned(),
        OptionValue::from_user("password", canary),
    )]));
    let err = encode(&plan).unwrap_err();
    assert!(matches!(err, CodecError::Unsupported(_)), "{err:?}");
    let text = format!("{err} {err:?}");
    assert!(text.contains("`password` is an inline secret"), "{text}");
    assert!(!text.contains(canary), "{text}");
}

#[test]
fn a_url_with_credentials_is_refused_without_naming_it() {
    let canary = "canary-7c21-sig";
    let url = format!("s3://bucket/data?X-Amz-Signature={canary}");
    let option = OptionValue::from_user("path", url);
    // The message must not format the option: it holds the credential.
    assert!(matches!(option, OptionValue::Url(_)), "not a URL option");
    let plan = scan_with(BTreeMap::from([("path".to_owned(), option)]));
    let err = encode(&plan).unwrap_err();
    let text = format!("{err} {err:?}");
    assert!(text.contains("`path` is a URL with credentials"), "{text}");
    assert!(!text.contains(canary), "{text}");
}

#[test]
fn a_secret_reference_travels_as_a_reference() {
    let plan = scan_with(BTreeMap::from([(
        "password".to_owned(),
        OptionValue::reference("prod-scope", "db-password-key"),
    )]));
    let bytes = encode(&plan).unwrap();
    for needle in [&b"prod-scope"[..], b"db-password-key"] {
        assert!(bytes.windows(needle.len()).any(|w| w == needle));
    }
    assert_eq!(decode(&bytes).unwrap(), plan);
}

#[test]
fn unresolved_expressions_are_refused() {
    let unresolved = [
        Expr::column(["a"]),
        Expr::function("upper", vec![Expr::column(["a"])]),
        Expr::column(["a"]).alias("b"),
    ];
    for e in unresolved {
        let plan = plan_of(PhysNode::Filter {
            input: Box::new(PhysNode::OneRow),
            layout: Vec::new(),
            predicate: Arc::new(e),
            subqueries: Vec::new(),
        });
        let err = encode(&plan).unwrap_err();
        assert!(
            matches!(&err, CodecError::Unsupported(m) if m.contains("unresolved")),
            "{err:?}"
        );
    }
}

/// The message of the `name` case.
fn message(name: &str) -> pb::PhysicalPlan {
    let (_, plan) = cases().into_iter().find(|(n, _)| n == name).unwrap();
    to_message(&plan).unwrap()
}

fn corrupt(m: &pb::PhysicalPlan, fragment: &str) {
    let err = decode(&m.encode_to_vec()).unwrap_err();
    assert!(
        matches!(&err, CodecError::Corrupt(msg) if msg.contains(fragment)),
        "expected corrupt `{fragment}`, got {err:?}"
    );
}

/// The decimal literal of `m`'s expression table.
fn decimal_literal(m: &mut pb::PhysicalPlan) -> &mut pb::DecimalLiteral {
    m.exprs
        .iter_mut()
        .find_map(|e| match e.kind.as_mut() {
            Some(pb::expr::Kind::Literal(pb::Literal {
                kind: Some(pb::literal::Kind::Decimal(d)),
            })) => Some(d),
            _ => None,
        })
        .unwrap()
}

fn node_mut(m: &mut pb::PhysicalPlan, at: usize) -> &mut pb::node::Kind {
    m.nodes[at].kind.as_mut().unwrap()
}

#[test]
fn bytes_that_are_not_a_plan_are_corrupt() {
    let err = decode(&[]).unwrap_err();
    assert_eq!(err, CodecError::Corrupt("no format version".to_owned()));
    let err = decode(&[0xff, 0xff, 0xff, 0xff]).unwrap_err();
    assert_eq!(err, CodecError::Corrupt("malformed protobuf".to_owned()));
    let bytes = encode(&cases().remove(0).1).unwrap();
    for cut in [1, bytes.len() / 2, bytes.len() - 1] {
        let err = decode(&bytes[..cut]).unwrap_err();
        assert!(matches!(err, CodecError::Corrupt(_)), "cut {cut}: {err:?}");
    }
}

/// `m` as a writer that predates the feature header wrote it.
fn legacy(mut m: pb::PhysicalPlan) -> pb::PhysicalPlan {
    m.epoch = 0;
    m.features.clear();
    m
}

#[test]
fn a_legacy_plan_that_needs_a_newer_reader_is_refused() {
    let mut m = legacy(message("scan"));
    m.min_reader_version = FORMAT_VERSION + 1;
    let err = decode(&m.encode_to_vec()).unwrap_err();
    assert_eq!(
        err,
        CodecError::TooNew {
            required: FORMAT_VERSION + 1,
            supported: FORMAT_VERSION
        }
    );
}

#[test]
fn readers_that_predate_features_still_refuse_what_they_cannot_read() {
    // The frozen format version and the legacy `min_reader_version` gate:
    // an offset `Limit` needs a v3 reader (an older one would ignore the
    // skipped rows), a `Window` a v4 reader, a plan without features v2.
    assert_eq!(FORMAT_VERSION, 5);
    for (name, plan) in cases() {
        let m = to_message(&plan).unwrap();
        assert_eq!(m.format_version, FORMAT_VERSION, "{name}");
        let expected = match name.as_str() {
            "limit_offset" => 3,
            "window_ranking" | "window_aggregates" => 4,
            n if n.starts_with("expand") => 5,
            // A feature added after the freeze: above every frozen version.
            "window_group_limit"
            | "rust_udf"
            | "nested_types"
            | "struct_field"
            | "lambda_functions"
            | "map_key_dedup_policy"
            | "aggregate_filter"
            | "range"
            | "aggregate_collects"
            | "aggregate_vectors"
            | "aggregate_mode_regr_bit_listagg"
            | "aggregate_listagg" => 6,
            // Features added after the version was frozen: above every
            // sequential format version.
            n if n.starts_with("time_") || n.starts_with("geo_") => 6,
            n if n.starts_with("hash_join_right") => POST_MIGRATION_READER_VERSION,
            "hash_join_existence"
            | "hash_join_single_match"
            | "nested_loop_existence"
            | "exchange_hash_key_group"
            | "window_frames"
            | "window_offset_values"
            | "write_files_with_plan" => POST_MIGRATION_READER_VERSION,
            _ => MIN_READER_VERSION,
        };
        assert_eq!(m.min_reader_version, expected, "{name}");
    }
}

#[test]
fn a_legacy_plan_decodes_by_its_content() {
    // Written before the header: no epoch and no feature list.
    for name in ["scan", "limit_offset", "window_ranking"] {
        let (_, plan) = cases().into_iter().find(|(n, _)| n == name).unwrap();
        let bytes = legacy(to_message(&plan).unwrap()).encode_to_vec();
        assert_eq!(decode(&bytes).unwrap(), plan, "{name}");
    }
    // A legacy plan whose content is newer than its reader version.
    let mut m = legacy(message("window_ranking"));
    m.min_reader_version = 3;
    corrupt(&m, "a construct newer than the plan's reader version");
    // A feature list needs an epoch.
    let mut m = message("window_ranking");
    m.epoch = 0;
    corrupt(&m, "a feature list without an epoch");
}

#[test]
fn a_window_plan_survives_the_wire_with_every_field() {
    for name in [
        "window_ranking",
        "window_aggregates",
        "window_frames",
        "window_offset_values",
    ] {
        let (_, plan) = cases().into_iter().find(|(n, _)| n == name).unwrap();
        assert_eq!(round_trip(&plan), plan, "{name}");
    }
}

#[test]
fn a_corrupt_window_is_refused() {
    use pb::node::Kind;
    let edit = |f: &dyn Fn(&mut pb::WindowNode)| {
        let mut m = message("window_ranking");
        for node in &mut m.nodes {
            if let Some(Kind::Window(w)) = &mut node.kind {
                f(w);
            }
        }
        m
    };
    corrupt(
        &edit(&|w| w.functions[0].function = None),
        "a window function is missing",
    );
    corrupt(
        &edit(&|w| w.functions[0].frame = None),
        "a window frame is missing",
    );
    corrupt(
        &edit(&|w| w.functions[0].frame.as_mut().unwrap().units = 0),
        "unknown window frame unit",
    );
    corrupt(
        &edit(&|w| w.functions[0].frame.as_mut().unwrap().lower = 9),
        "unknown window frame bound",
    );
    corrupt(
        &edit(&|w| {
            w.functions[5].function = Some(pb::window_function_plan::Function::Ntile(0));
        }),
        "ntile with a bucket count below one",
    );
}

#[test]
fn a_corrupt_window_group_limit_is_refused() {
    use pb::node::Kind;
    let edit = |f: &dyn Fn(&mut pb::WindowGroupLimitNode)| {
        let mut m = message("window_group_limit");
        for node in &mut m.nodes {
            if let Some(Kind::WindowGroupLimit(g)) = &mut node.kind {
                f(g);
            }
        }
        m
    };
    corrupt(&edit(&|g| g.limit = 0), "window group limit of 0 rows");
    corrupt(&edit(&|g| g.function = 0), "unknown rank-like function");
    corrupt(&edit(&|g| g.function = 9), "unknown rank-like function");
}

#[test]
fn an_expand_plan_survives_the_wire_with_every_field() {
    for name in ["expand_rollup", "expand_single", "expand_subquery"] {
        let (_, plan) = cases().into_iter().find(|(n, _)| n == name).unwrap();
        assert_eq!(round_trip(&plan), plan, "{name}");
    }
}

#[test]
fn a_loose_null_of_item_still_needs_the_expand_feature() {
    let plan = plan_of(PhysNode::Project {
        input: Box::new(PhysNode::OneRow),
        layout: Vec::new(),
        items: vec![krishiv_exec::physical::ProjectItemPlan::NullOf {
            index: 0,
            name: "x".to_owned(),
        }],
        nullable: None,
        subqueries: Vec::new(),
    });
    let m = to_message(&plan).unwrap();
    assert_eq!(m.min_reader_version, 5);
    assert_eq!(m.features, ["expand"]);
    assert_eq!(round_trip(&plan), plan);
}

#[test]
fn a_corrupt_expand_is_refused() {
    use pb::node::Kind;
    let edit = |f: &dyn Fn(&mut pb::ExpandNode)| {
        let mut m = message("expand_rollup");
        for node in &mut m.nodes {
            if let Some(Kind::Expand(e)) = &mut node.kind {
                f(e);
            }
        }
        m
    };
    corrupt(
        &edit(&|e| e.projections.clear()),
        "an expand without a projection",
    );
    corrupt(
        &edit(&|e| {
            e.nullable.pop();
        }),
        "do not match the nullability",
    );
    corrupt(
        &edit(&|e| e.projections[0].items[0].kind = None),
        "a project item kind",
    );
}

#[test]
fn joins_that_build_their_preserved_side_are_a_feature() {
    // The three join types are the `join_build_preserved` feature, which
    // the planner can avoid (it builds the other side instead). A reader
    // that predates features refuses the plan as too new rather than as an
    // unknown join type.
    assert_eq!(
        Feature::JoinBuildPreserved.kind(),
        FeatureKind::Optimization
    );
    for name in ["hash_join", "hash_join_anti", "nested_loop_join"] {
        let m = message(name);
        assert!(m.features.is_empty(), "{name}");
        assert_eq!(m.min_reader_version, MIN_READER_VERSION, "{name}");
    }
    for name in [
        "hash_join_right_semi",
        "hash_join_right_anti",
        "hash_join_right_outer",
    ] {
        let m = message(name);
        assert_eq!(m.features, ["join_build_preserved"], "{name}");
        assert_eq!(
            m.min_reader_version, POST_MIGRATION_READER_VERSION,
            "{name}"
        );
    }
}

/// The existence join type and the single-match flag are two semantic
/// features (no equivalent plan exists without either), each refused by
/// name by a reader without it; `distinct_build` needs none.
#[test]
fn existence_joins_and_single_match_are_features() {
    for (name, feature) in [
        ("hash_join_existence", Feature::ExistenceJoin),
        ("nested_loop_existence", Feature::ExistenceJoin),
        ("hash_join_single_match", Feature::JoinSingleMatch),
    ] {
        assert_eq!(feature.kind(), FeatureKind::Semantic);
        let m = message(name);
        assert_eq!(m.features, [feature.id()], "{name}");
        assert_eq!(
            m.min_reader_version, POST_MIGRATION_READER_VERSION,
            "{name}"
        );
        // A reader without the feature refuses the plan, naming it.
        let (_, plan) = cases().into_iter().find(|(n, _)| n == name).unwrap();
        let bytes = encode(&plan).unwrap();
        let reader = ReaderFeatures::only([Feature::LimitOffset]);
        let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
        assert_eq!(
            err,
            CodecError::UnsupportedFeatures {
                missing: vec![feature.id().to_owned()]
            },
            "{name}"
        );
        // A reader with every feature reads it back exactly.
        assert_eq!(round_trip(&plan), plan, "{name}");
    }
    assert_eq!(Feature::ExistenceJoin.to_string(), "existence_join");
    assert_eq!(Feature::JoinSingleMatch.to_string(), "join_single_match");
}

/// `distinct_build` is only legal where the build's rows are never output
/// nor read beyond their keys; anywhere else the plan is corrupt.
#[test]
fn a_misplaced_distinct_build_flag_is_corrupt() {
    use pb::node::Kind;
    // `hash_join` is an inner join with a residual condition.
    let mut m = message("hash_join");
    let last = m.nodes.len() - 1;
    let Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.distinct_build = true;
    corrupt(
        &m,
        "a distinct build on a join that outputs or reads build rows",
    );
    // An existence join with a residual condition reads build columns.
    let mut m = message("hash_join");
    let Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.join_type = pb::JoinType::Existence as i32;
    j.distinct_build = true;
    m.features = vec!["existence_join".to_owned()];
    corrupt(
        &m,
        "a distinct build on a join that outputs or reads build rows",
    );
}

/// A single match on a join that is not left outer is corrupt.
#[test]
fn a_misplaced_single_match_flag_is_corrupt() {
    use pb::node::Kind;
    let mut m = message("hash_join");
    let last = m.nodes.len() - 1;
    let Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.single_match = true;
    corrupt(&m, "single match");
}

#[test]
fn an_unspecified_join_type_is_corrupt() {
    let mut m = message("hash_join");
    for node in &mut m.nodes {
        if let Some(pb::node::Kind::HashJoin(j)) = &mut node.kind {
            j.join_type = pb::JoinType::Unspecified as i32;
        }
    }
    corrupt(&m, "join type");
}

#[test]
fn a_newer_writer_that_stays_compatible_is_read() {
    // A writer that only added ignorable fields: unknown fields (here
    // number 999, a varint) are skipped.
    let (_, plan) = cases().remove(0);
    let m = to_message(&plan).unwrap();
    let mut bytes = m.encode_to_vec();
    bytes.extend_from_slice(&[0xB8, 0x3E, 0x07]);
    assert_eq!(decode(&bytes).unwrap(), plan);
}

#[test]
fn references_must_point_to_earlier_entries() {
    let mut m = message("filter");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::Filter(f) = node_mut(&mut m, last) else {
        panic!("not a filter")
    };
    f.input = u32::try_from(last).unwrap();
    corrupt(&m, "later or missing entry");

    let mut m = message("filter");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::Filter(f) = node_mut(&mut m, last) else {
        panic!("not a filter")
    };
    f.predicate = u32::MAX;
    corrupt(&m, "later or missing entry");

    let mut m = message("filter");
    m.root = 999;
    corrupt(&m, "later or missing entry");
}

#[test]
fn a_node_with_two_parents_or_none_is_corrupt() {
    let mut m = message("union");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::Union(u) = node_mut(&mut m, last) else {
        panic!("not a union")
    };
    u.inputs[1] = u.inputs[0];
    corrupt(&m, "used by two parents");

    let mut m = message("one_row");
    m.nodes.push(pb::Node {
        kind: Some(pb::node::Kind::OneRow(pb::Empty {})),
    });
    m.root = 0;
    corrupt(&m, "no other node or the root uses");
}

#[test]
fn counts_and_enums_are_validated() {
    let mut m = message("hash_join");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.null_safe.pop();
    corrupt(&m, "key lists differ in length");

    let mut m = message("hash_join");
    let pb::node::Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.join_type = 0;
    corrupt(&m, "unknown join type");
    let mut m = message("hash_join");
    let pb::node::Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.join_type = 99;
    corrupt(&m, "unknown join type");

    let mut m = message("aggregate");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::Aggregate(a) = node_mut(&mut m, last) else {
        panic!("not an aggregate")
    };
    a.distinct.pop();
    corrupt(&m, "distinct flags");
}

#[test]
fn values_are_validated() {
    let mut m = message("expressions");
    let decimal = m
        .exprs
        .iter_mut()
        .find_map(|e| match e.kind.as_mut() {
            Some(pb::expr::Kind::Literal(pb::Literal {
                kind: Some(pb::literal::Kind::Decimal(d)),
            })) => Some(d),
            _ => None,
        })
        .unwrap();
    decimal.unscaled.pop();
    corrupt(&m, "not 16 bytes");

    let mut m = message("expressions");
    let decimal = m
        .exprs
        .iter_mut()
        .find_map(|e| match e.kind.as_mut() {
            Some(pb::expr::Kind::Literal(pb::Literal {
                kind: Some(pb::literal::Kind::Decimal(d)),
            })) => Some(d),
            _ => None,
        })
        .unwrap();
    decimal.data_type.as_mut().unwrap().precision = 39;
    corrupt(&m, "invalid decimal type");

    // The unscaled value must fit the precision: 10^5 is one too many for
    // DECIMAL(5,2), 10^5 - 1 is the largest.
    for (unscaled, fits) in [(100_000_i128, false), (99_999, true), (-99_999, true)] {
        let mut m = message("expressions");
        let decimal = decimal_literal(&mut m);
        decimal.unscaled = unscaled.to_le_bytes().to_vec();
        let ty = decimal.data_type.as_mut().unwrap();
        (ty.precision, ty.scale) = (5, 2);
        if fits {
            // The decoded literal keeps its value and type: read it back
            // from the re-encoded message.
            let decoded = decode(&m.encode_to_vec()).unwrap();
            let mut again = to_message(&decoded).unwrap();
            let back = decimal_literal(&mut again);
            let bytes: [u8; 16] = back.unscaled.as_slice().try_into().unwrap();
            assert_eq!(i128::from_le_bytes(bytes), unscaled);
            let ty = back.data_type.as_ref().unwrap();
            assert_eq!((ty.precision, ty.scale), (5, 2));
        } else {
            corrupt(&m, "beyond its precision");
        }
    }

    let mut m = message("expressions");
    let byte = m
        .exprs
        .iter_mut()
        .find_map(|e| match e.kind.as_mut() {
            Some(pb::expr::Kind::Literal(pb::Literal {
                kind: Some(pb::literal::Kind::Byte(v)),
            })) => Some(v),
            _ => None,
        })
        .unwrap();
    *byte = 300;
    corrupt(&m, "out of range for its type");

    let mut m = message("scan");
    m.schema = vec![1, 2, 3];
    corrupt(&m, "invalid Arrow schema");

    let mut m = message("scan");
    m.env = None;
    corrupt(&m, "no expression environment");

    let mut m = message("scan");
    m.explain_root = 77;
    corrupt(&m, "later or missing entry");
}

#[test]
fn limits_bound_size_and_depth() {
    let (_, plan) = cases().remove(0);
    let bytes = encode(&plan).unwrap();
    let small = Limits {
        max_bytes: bytes.len() - 1,
        ..Limits::default()
    };
    assert!(matches!(
        decode_with(&bytes, &small, &ReaderFeatures::all()).unwrap_err(),
        CodecError::Limit(m) if m.contains("bytes")
    ));
    let few = Limits {
        max_exprs: 0,
        ..Limits::default()
    };
    assert!(matches!(
        decode_with(&bytes, &few, &ReaderFeatures::all()).unwrap_err(),
        CodecError::Limit(m) if m.contains("expressions")
    ));

    let mut node = PhysNode::OneRow;
    for _ in 0..1030 {
        node = PhysNode::Limit {
            input: Box::new(node),
            offset: 0,
            limit: 1,
        };
    }
    // The plan itself is deeper than the default bound; its explain tree
    // is shallow.
    let deep = plan_of(node);
    let bytes = encode(&deep).unwrap();
    let err = decode(&bytes).unwrap_err();
    assert!(
        matches!(&err, CodecError::Limit(m) if m.contains("deeper than 1024")),
        "{err:?}"
    );
    let roomy = Limits {
        max_depth: 2000,
        ..Limits::default()
    };
    assert_eq!(
        decode_with(&bytes, &roomy, &ReaderFeatures::all()).unwrap(),
        deep
    );
}

#[test]
fn limits_bound_expression_depth() {
    let mut e = inventory::col(1, "a", krishiv_types::DataType::Boolean);
    for _ in 0..50 {
        e = inventory::call(
            krishiv_plan::Builtin::Not,
            vec![e],
            krishiv_types::DataType::Boolean,
        );
    }
    let plan = plan_of(PhysNode::Filter {
        input: Box::new(PhysNode::OneRow),
        layout: Vec::new(),
        predicate: e,
        subqueries: Vec::new(),
    });
    let bytes = encode(&plan).unwrap();
    // 50 calls over a column: 51 deep.
    let exact = Limits {
        max_depth: 51,
        ..Limits::default()
    };
    assert_eq!(
        decode_with(&bytes, &exact, &ReaderFeatures::all()).unwrap(),
        plan
    );
    let tight = Limits {
        max_depth: 50,
        ..Limits::default()
    };
    let err = decode_with(&bytes, &tight, &ReaderFeatures::all()).unwrap_err();
    assert!(
        matches!(&err, CodecError::Limit(m) if m.contains("expression deeper than 50")),
        "{err:?}"
    );
}

#[test]
fn v1_hash_join_without_a_distribution_decodes_as_broadcast() {
    // v1 bytes carry no `distribution` (field 9): the partitioned join did
    // not exist yet, and every hash join broadcast its build.
    let mut m = message("hash_join");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.distribution = pb::JoinDistribution::Unspecified as i32;
    let plan = decode(&m.encode_to_vec()).unwrap();
    let PhysNode::HashJoin(join) = plan.root() else {
        panic!("not a join")
    };
    assert_eq!(
        join.distribution,
        krishiv_exec::physical::JoinDistribution::Broadcast
    );
}

#[test]
fn an_unknown_join_distribution_is_corrupt() {
    let mut m = message("hash_join");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::HashJoin(j) = node_mut(&mut m, last) else {
        panic!("not a join")
    };
    j.distribution = 99;
    corrupt(&m, "unknown join distribution");
}

#[test]
fn an_exchange_without_a_partitioning_is_corrupt() {
    let mut m = message("exchange_hash");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::Exchange(e) = node_mut(&mut m, last) else {
        panic!("not an exchange")
    };
    e.partitioning = None;
    corrupt(&m, "partitioning is missing");
}

#[test]
fn an_exchange_without_a_partitioning_kind_is_corrupt() {
    let mut m = message("exchange_hash");
    let last = m.nodes.len() - 1;
    let pb::node::Kind::Exchange(e) = node_mut(&mut m, last) else {
        panic!("not an exchange")
    };
    e.partitioning = Some(pb::Partitioning { kind: None });
    corrupt(&m, "partitioning kind is missing");
}

/// `any_value`, `max_by` and `min_by` have no ids in the format yet: a plan
/// holding one is refused with the typed error (not encoded as another
/// function), in an aggregate node and as a window function.
#[test]
fn aggregates_the_format_does_not_know_are_refused() {
    use krishiv_exec::aggregate::{AggregateExpr, AggregateFunction, AggregateOutput};
    use krishiv_exec::physical::AggregatePlan;
    use krishiv_exec::window::{WindowFrame, WindowFunction, WindowFunctionPlan, WindowPlan};
    let exprs = [
        AggregateExpr::new(
            AggregateFunction::AnyValue { ignore_nulls: true },
            0,
            arrow_schema::DataType::Int32,
            "a",
        ),
        AggregateExpr::by(
            AggregateFunction::MaxBy,
            (0, 1),
            (
                arrow_schema::DataType::Utf8View,
                arrow_schema::DataType::Int64,
            ),
            "b",
        ),
        AggregateExpr::by(
            AggregateFunction::MinBy,
            (0, 1),
            (
                arrow_schema::DataType::Int32,
                arrow_schema::DataType::Float64,
            ),
            "c",
        ),
    ];
    for e in exprs {
        let refused = |plan: &PhysicalPlan| {
            let err = encode(plan).unwrap_err();
            assert!(
                matches!(&err, CodecError::Unsupported(m) if m.contains("aggregate function")),
                "{err:?}"
            );
        };
        refused(&plan_of(PhysNode::Aggregate {
            input: Box::new(inventory::scan()),
            aggregate: AggregatePlan {
                group_by: vec![0],
                aggregates: vec![e.clone()],
                distinct: vec![false],
                output: AggregateOutput::Final,
            },
        }));
        refused(&plan_of(PhysNode::Window {
            input: Box::new(inventory::scan()),
            window: WindowPlan {
                partition_by: Vec::new(),
                order_by: Vec::new(),
                functions: vec![WindowFunctionPlan {
                    function: WindowFunction::Aggregate(e),
                    frame: WindowFrame::whole(krishiv_exec::window::FrameUnits::Rows),
                    filter: None,
                    name: "w".to_owned(),
                }],
            },
        }));
    }
}
