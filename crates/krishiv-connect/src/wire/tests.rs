// SPDX-License-Identifier: Apache-2.0
//! Tests of the wire limits (`wire`): the nesting cap with its limit+1 case,
//! the size and message caps, and malformed input.

use krishiv_common::error::{ErrorClass, KrishivError};

use super::*;
use crate::proto::layout::{EXPRESSION, MESSAGES, PLAN, RELATION};
use crate::proto::v4_2::{expression, relation};

// Protobuf wire builders: raw bytes, so a hostile depth needs no recursion.

fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(u8::try_from(v & 0x7f).expect("7 bits") | 0x80);
        v >>= 7;
    }
    out.push(u8::try_from(v).expect("7 bits"));
}

/// A length-delimited field `number` holding `body`.
fn field(number: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    varint(u64::from(number) << 3 | 2, &mut out);
    varint(body.len() as u64, &mut out);
    out.extend_from_slice(body);
    out
}

/// `Relation { sql { query: "x" } }`: the SQL message is one level below the
/// relation.
fn sql_leaf() -> Vec<u8> {
    field(10, &field(1, b"x"))
}

/// `Relation { read { named_table { unparsed_identifier: "t" } } }`: the
/// table is two levels below the relation.
fn table_leaf() -> Vec<u8> {
    field(2, &field(1, &field(1, b"t")))
}

/// The bytes of a varint.
fn varint_len(v: usize) -> usize {
    let mut out = Vec::new();
    varint(v as u64, &mut out);
    out.len()
}

/// `filters` filters stacked on `leaf`. Each filter adds two levels, the
/// `Filter` and the `Relation` that holds it. Built in linear time: the
/// sizes are computed inside out, the bytes written outside in.
fn filter_chain(filters: usize, leaf: &[u8]) -> Vec<u8> {
    // sizes[i] is the size of the relation under i filters.
    let mut sizes = vec![leaf.len()];
    for _ in 0..filters {
        let inner = sizes[sizes.len() - 1];
        let filter = 1 + varint_len(inner) + inner; // Filter { input }
        sizes.push(1 + varint_len(filter) + filter); // Relation { filter }
    }
    let mut out = Vec::with_capacity(sizes[filters]);
    for i in (1..=filters).rev() {
        let inner = sizes[i - 1];
        let filter = 1 + varint_len(inner) + inner;
        out.push(4 << 3 | 2);
        varint(filter as u64, &mut out);
        out.push(1 << 3 | 2);
        varint(inner as u64, &mut out);
    }
    out.extend_from_slice(leaf);
    out
}

fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

#[test]
fn nesting_limit_accepts_the_limit_and_rejects_one_more() {
    // 511 filters on a leaf 2 deep: the deepest message is at depth 1024.
    let at_limit = filter_chain(MAX_NESTING / 2 - 1, &table_leaf());
    let rel = decode_relation(&at_limit).expect("depth 1024 decodes");
    assert!(matches!(rel.rel_type, Some(relation::RelType::Filter(_))));
    // 512 filters on a leaf 1 deep: depth 1025.
    let over = filter_chain(MAX_NESTING / 2, &sql_leaf());
    let err = decode_relation(&over).expect_err("depth 1025 is rejected");
    assert_eq!(class(&err), ErrorClass::FailedToParseTooComplex);
    assert_eq!(err.sql_state(), Some("54001"));
    // The same shape one level shallower (depth 1023) is fine.
    decode_relation(&filter_chain(MAX_NESTING / 2 - 1, &sql_leaf())).expect("depth 1023");
}

#[test]
fn nesting_limit_counts_from_the_decoded_message() {
    // Under a Plan, the relation is one level deeper.
    let plan = |filters, leaf: &[u8]| field(1, &filter_chain(filters, leaf));
    decode_plan(&plan(MAX_NESTING / 2 - 1, &sql_leaf())).expect("depth 1024 under a plan");
    let err = decode_plan(&plan(MAX_NESTING / 2 - 1, &table_leaf())).expect_err("depth 1025");
    assert_eq!(class(&err), ErrorClass::FailedToParseTooComplex);
}

#[test]
fn nested_expressions_are_bounded_too() {
    // Expression { unresolved_function { arguments: [Expression] } } costs
    // two levels per call; the literal is one more.
    let leaf = field(1, &[0x18, 0x01]); // literal { boolean: true }
    let chain = |calls: usize| {
        let mut e = leaf.clone();
        for _ in 0..calls {
            e = field(3, &field(2, &e));
        }
        e
    };
    decode_expression(&chain(MAX_NESTING / 2 - 1)).expect("depth 1023");
    let err = decode_expression(&chain(MAX_NESTING / 2)).expect_err("depth 1025");
    assert_eq!(class(&err), ErrorClass::FailedToParseTooComplex);
}

#[test]
fn deepest_allowed_plan_decodes_and_drops_on_a_small_stack() {
    // A test thread has 2 MiB: decoding at the cap and dropping the result
    // must not overflow it.
    let rel = decode_relation(&filter_chain(MAX_NESTING / 2 - 1, &table_leaf())).expect("decodes");
    drop(rel);
}

#[test]
fn hostile_depth_is_rejected_without_recursing() {
    // A million levels: only an iterative walk survives this.
    let err = decode_relation(&filter_chain(1_000_000, &sql_leaf())).expect_err("too deep");
    assert_eq!(class(&err), ErrorClass::FailedToParseTooComplex);
}

/// A table with one message (id 0) of 100 bytes that holds itself in field
/// 1, a string in field 2, a packed `int64` list in field 3 and a fixed
/// scalar in field 4.
const SELF_NESTING: [Layout; 1] = [Layout {
    name: "Self",
    size: 100,
    fields: &[
        (1, Field::Message(0)),
        (2, Field::Bytes),
        (3, Field::Varint(8)),
        (4, Field::Fixed),
    ],
}];

const fn limits(bytes: usize, messages: usize, nesting: usize) -> Limits {
    Limits {
        bytes,
        messages,
        decoded: usize::MAX,
        nesting,
    }
}

const fn decoded(decoded: usize) -> Limits {
    Limits {
        bytes: usize::MAX,
        messages: usize::MAX,
        decoded,
        nesting: usize::MAX,
    }
}

#[test]
fn size_message_and_nesting_caps() {
    let two_deep = field(1, &field(1, &[]));
    assert_eq!(two_deep.len(), 4);
    let walk = |l: &Limits| scan(&SELF_NESTING, 0, &two_deep, l);
    assert_eq!(walk(&limits(4, 3, 2)), Ok(2));
    assert_eq!(walk(&limits(3, 3, 2)), Err(Reject::TooBig(4)));
    // The root and two nested messages are three.
    assert_eq!(walk(&limits(4, 2, 2)), Err(Reject::TooManyMessages));
    assert_eq!(walk(&limits(4, 3, 1)), Err(Reject::TooDeep));
}

#[test]
fn decoded_size_counts_types_not_wire_bytes() {
    let walk = |bytes: &[u8], budget| scan(&SELF_NESTING, 0, bytes, &decoded(budget));
    // Three messages of 100 bytes from four wire bytes.
    let two_deep = field(1, &field(1, &[]));
    assert_eq!(walk(&two_deep, 300), Ok(2));
    assert_eq!(walk(&two_deep, 299), Err(Reject::TooLarge));
    // A string: its header and its bytes.
    let text = field(2, b"abc");
    assert_eq!(walk(&text, 100 + LEAF_SIZE + 3), Ok(0));
    assert_eq!(walk(&text, 100 + LEAF_SIZE + 2), Err(Reject::TooLarge));
    // A packed int64 list: each one-byte varint decodes to eight bytes.
    let packed = field(3, &[1, 2, 3, 4]);
    assert_eq!(walk(&packed, 100 + LEAF_SIZE + 32), Ok(0));
    assert_eq!(walk(&packed, 100 + LEAF_SIZE + 31), Err(Reject::TooLarge));
    // Scalars sent one by one (unpacked or fixed) count eight bytes each.
    let scalars = [0x18, 0x01, 0x18, 0x02, 0x25, 1, 2, 3, 4];
    assert_eq!(walk(&scalars, 100 + 3 * SCALAR_SIZE), Ok(0));
    assert_eq!(
        walk(&scalars, 100 + 3 * SCALAR_SIZE - 1),
        Err(Reject::TooLarge)
    );
}

#[test]
fn fields_the_version_does_not_define_are_rejected() {
    // Relation field 5000 is not in Spark Connect 4.2: prost would skip it,
    // and the plan would run without it.
    let unknown = [sql_leaf(), field(5000, b"x")].concat();
    let err = decode_relation(&unknown).expect_err("unknown field");
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        err.message()
            .contains("field 5000 of spark.connect.Relation"),
        "{}",
        err.message()
    );
    // Deeper down too, and for every wire type: Filter.condition's
    // expression with a varint field 900.
    let filter = field(
        4,
        &[field(1, &sql_leaf()), field(2, &[0xa0, 0x38, 0x01])].concat(),
    );
    let err = decode_relation(&filter).expect_err("unknown nested field");
    assert!(
        err.message()
            .contains("field 900 of spark.connect.Expression"),
        "{}",
        err.message()
    );
    // Literal fields 27 and 28 are reserved in 4.2 (geometry, geography).
    let geometry = field(1, &[0xda, 0x01, 0x00]);
    let err = decode_expression(&geometry).expect_err("reserved field");
    assert!(
        err.message()
            .contains("field 27 of spark.connect.Expression.Literal")
    );
}

#[test]
fn flat_hostile_strings_hit_the_decoded_size_cap() {
    // Join.using_columns (field 5) filled with empty strings: two wire bytes
    // each, a 24-byte `String` each decoded.
    let n = MAX_DECODED_BYTES / LEAF_SIZE + 1;
    let mut join = Vec::with_capacity(2 * n);
    for _ in 0..n {
        join.extend_from_slice(&[5 << 3 | 2, 0]);
    }
    assert!(join.len() < MAX_PLAN_BYTES);
    let err = decode_relation(&field(5, &join)).expect_err("too large decoded");
    assert_eq!(
        class(&err),
        ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax
    );
    assert!(err.message().contains("decoded"), "{}", err.message());
}

#[test]
fn rejects_map_to_spark_error_classes() {
    let big = KrishivError::from(Reject::TooBig(MAX_PLAN_BYTES + 1));
    assert_eq!(
        class(&big),
        ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax
    );
    assert!(big.message().contains("134217729"), "{}", big.message());
    let many = KrishivError::from(Reject::TooManyMessages);
    assert_eq!(
        class(&many),
        ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax
    );
    let bad = KrishivError::from(Reject::Malformed("x"));
    assert_eq!(class(&bad), ErrorClass::ConnectInvalidPlanCannotParse);
}

#[test]
fn malformed_input_is_rejected_by_the_walk() {
    let walk = |bytes: &[u8]| scan(&MESSAGES, RELATION, bytes, &limits(1 << 20, 1 << 20, 64));
    let malformed = |bytes: &[u8]| matches!(walk(bytes), Err(Reject::Malformed(_)));
    assert!(malformed(&[0x80]), "truncated key");
    assert!(malformed(&[0x08, 0x80]), "truncated varint value");
    assert!(
        malformed(&[
            0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01
        ]),
        "11-byte varint"
    );
    assert!(
        malformed(&[
            0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02
        ]),
        "overflows 64 bits"
    );
    assert!(malformed(&[0x00, 0x01]), "field number 0");
    assert!(malformed(&[0x0a, 0x05, 0x01]), "length past the end");
    assert!(
        malformed(&[
            0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01
        ]),
        "length of 2^64 - 1"
    );
    assert!(malformed(&[0x0b]), "start group");
    assert!(malformed(&[0x0c]), "end group");
    assert!(malformed(&[0x0e]), "wire type 6");
    assert!(malformed(&[0x09, 1, 2, 3]), "fixed64 cut short");
    assert!(malformed(&[0x0d, 1, 2]), "fixed32 cut short");
    assert!(
        malformed(&[0x22, 0x02, 0x0a, 0x05]),
        "a nested field that overruns its parent"
    );
    assert!(
        malformed(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x01]),
        "field number over 32 bits"
    );
    assert!(malformed(&[0x0f]), "wire type 7");
    // Well-formed fields of every wire type are walked: a literal's integer
    // (varint), double (fixed64), float (fixed32) and string.
    let literal = |body: &[u8]| field(1, body);
    let walk_expr =
        |bytes: &[u8]| scan(&MESSAGES, EXPRESSION, bytes, &limits(1 << 20, 1 << 20, 64));
    assert_eq!(walk_expr(&literal(&[0x30, 0x96, 0x01])), Ok(1));
    assert_eq!(walk_expr(&literal(&[0x59, 1, 2, 3, 4, 5, 6, 7, 8])), Ok(1));
    assert_eq!(walk_expr(&literal(&[0x55, 1, 2, 3, 4])), Ok(1));
    assert_eq!(walk_expr(&literal(&field(13, b"text"))), Ok(1));
    assert_eq!(walk(&[]), Ok(0));
}

#[test]
fn decode_reports_what_prost_rejects_after_the_walk() {
    // The walk accepts it (a query is bytes to it); prost rejects the invalid
    // UTF-8, and the error is a Spark class that does not echo the bytes.
    let err = decode_relation(&field(10, &field(1, &[0xff, 0xfe]))).expect_err("invalid utf-8");
    assert_eq!(class(&err), ErrorClass::ConnectInvalidPlanCannotParse);
    assert!(!err.message().contains('\u{fffd}'));
}

#[test]
fn oversized_input_is_rejected_before_any_decoding() {
    let err = decode_plan(&vec![0_u8; MAX_PLAN_BYTES + 1]).expect_err("too big");
    assert_eq!(
        class(&err),
        ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax
    );
}

#[test]
fn flat_hostile_input_hits_the_message_cap() {
    // Two million empty grouping sets in an Aggregate (Relation.aggregate is
    // field 9, Aggregate.grouping_sets field 6): each is a message of its
    // own, small enough decoded that the message cap is reached before the
    // decoded-size cap.
    let one = field(6, &[]);
    let mut aggregate = Vec::with_capacity(one.len() * (MAX_MESSAGES + 1));
    for _ in 0..=MAX_MESSAGES {
        aggregate.extend_from_slice(&one);
    }
    let err = decode_relation(&field(9, &aggregate)).expect_err("too many messages");
    assert_eq!(
        class(&err),
        ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax
    );
    assert!(err.message().contains("messages"), "{}", err.message());
}

#[test]
fn flat_hostile_messages_hit_the_decoded_size_cap() {
    // Two million empty expressions in a Project: two wire bytes each, about
    // 300 decoded, so the decoded-size cap is reached first.
    let one = field(3, &[]);
    let n = MAX_DECODED_BYTES / size_of::<crate::proto::v4_2::Expression>() + 1;
    assert!(n < MAX_MESSAGES);
    let mut project = Vec::with_capacity(one.len() * n);
    for _ in 0..n {
        project.extend_from_slice(&one);
    }
    let err = decode_relation(&field(3, &project)).expect_err("too large decoded");
    assert_eq!(
        class(&err),
        ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax
    );
    assert!(err.message().contains("decoded"), "{}", err.message());
}

#[test]
fn decodes_the_three_roots() {
    let plan = decode_plan(&field(1, &sql_leaf())).expect("plan");
    assert!(plan.op_type.is_some());
    let relation = decode_relation(&sql_leaf()).expect("relation");
    assert!(matches!(relation.rel_type, Some(relation::RelType::Sql(_))));
    let expr = decode_expression(&field(1, &[0x18, 0x01])).expect("expression");
    assert!(matches!(
        expr.expr_type,
        Some(expression::ExprType::Literal(_))
    ));
}

#[test]
fn layout_table_matches_the_generated_types() {
    // Plan.root (1), Relation.filter (4) and Expression.literal (1) hold
    // messages; an unknown field is absent and `Any` is a leaf.
    let kind = |message: u16, number: u32| {
        let fields = MESSAGES[usize::from(message)].fields;
        fields
            .binary_search_by_key(&number, |&(n, _)| n)
            .ok()
            .map(|i| fields[i].1)
    };
    assert!(matches!(kind(PLAN, 1), Some(Field::Message(RELATION))));
    assert!(matches!(kind(RELATION, 4), Some(Field::Message(_))));
    assert!(matches!(kind(EXPRESSION, 1), Some(Field::Message(_))));
    assert_eq!(kind(RELATION, 5000), None);
    assert_eq!(kind(EXPRESSION, 999), Some(Field::Bytes), "an Any");
    assert_eq!(MESSAGES[usize::from(PLAN)].name, "Plan");
    assert_eq!(MESSAGES[usize::from(RELATION)].name, "Relation");
    assert_eq!(MESSAGES[usize::from(EXPRESSION)].name, "Expression");
    assert_eq!(
        MESSAGES[usize::from(RELATION)].size,
        size_of::<crate::proto::v4_2::Relation>()
    );
    for (id, m) in MESSAGES.iter().enumerate() {
        // The walk binary-searches each row, and descends only to ids it has.
        assert!(m.fields.windows(2).all(|w| w[0].0 < w[1].0), "{}", m.name);
        for &(_, f) in m.fields {
            if let Field::Message(child) = f {
                assert!(usize::from(child) < MESSAGES.len(), "{id}");
            }
        }
    }
}
