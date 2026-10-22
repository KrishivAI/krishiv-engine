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
    CodecError, FORMAT_VERSION, Limits, decode, decode_with, encode, to_message,
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
    Filter,
    Project,
    Limit,
    ReplicateRows,
    Sort,
    TopN,
    Aggregate,
    HashJoin,
    NestedLoopJoin,
    Union,
    WriteFiles,
);

fn variant_of(node: &PhysNode) -> Variant {
    match node {
        PhysNode::Scan(_) => Variant::Scan,
        PhysNode::OneRow => Variant::OneRow,
        PhysNode::Filter { .. } => Variant::Filter,
        PhysNode::Project { .. } => Variant::Project,
        PhysNode::Limit { .. } => Variant::Limit,
        PhysNode::ReplicateRows { .. } => Variant::ReplicateRows,
        PhysNode::Sort { .. } => Variant::Sort,
        PhysNode::TopN { .. } => Variant::TopN,
        PhysNode::Aggregate { .. } => Variant::Aggregate,
        PhysNode::HashJoin(_) => Variant::HashJoin,
        PhysNode::NestedLoopJoin(_) => Variant::NestedLoopJoin,
        PhysNode::Union { .. } => Variant::Union,
        PhysNode::WriteFiles { .. } => Variant::WriteFiles,
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

#[test]
fn a_plan_that_needs_a_newer_reader_is_refused() {
    let mut m = message("scan");
    m.format_version = FORMAT_VERSION + 1;
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
fn a_newer_writer_that_stays_compatible_is_read() {
    // N reads N+1 when the writer only added fields: unknown fields (here
    // number 999, a varint) are skipped, and the version is higher.
    let (_, plan) = cases().remove(0);
    let mut m = to_message(&plan).unwrap();
    m.format_version = FORMAT_VERSION + 1;
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
        decode_with(&bytes, &small).unwrap_err(),
        CodecError::Limit(m) if m.contains("bytes")
    ));
    let few = Limits {
        max_exprs: 0,
        ..Limits::default()
    };
    assert!(matches!(
        decode_with(&bytes, &few).unwrap_err(),
        CodecError::Limit(m) if m.contains("expressions")
    ));

    let mut node = PhysNode::OneRow;
    for _ in 0..1030 {
        node = PhysNode::Limit {
            input: Box::new(node),
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
    assert_eq!(decode_with(&bytes, &roomy).unwrap(), deep);
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
    assert_eq!(decode_with(&bytes, &exact).unwrap(), plan);
    let tight = Limits {
        max_depth: 50,
        ..Limits::default()
    };
    let err = decode_with(&bytes, &tight).unwrap_err();
    assert!(
        matches!(&err, CodecError::Limit(m) if m.contains("expression deeper than 50")),
        "{err:?}"
    );
}
