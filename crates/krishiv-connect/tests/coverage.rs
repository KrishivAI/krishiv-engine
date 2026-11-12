// SPDX-License-Identifier: Apache-2.0
//! The coverage table is tied to the translator (P3-T012a, owner decision 7): for every arm of
//! `Relation`, `Expression`, `Command`, `Expression.Literal` and `DataType`, an empty message of that
//! arm answers `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` exactly when its row says the arm is not translated;
//! every field name and number matches the vendored descriptor; the table holds one row per arm.
//! (`Catalog.cat_type` is reached only through `Relation.catalog`, a missing arm.)
#![forbid(unsafe_code)]

use krishiv_common::error::ErrorClass;
use krishiv_connect::coverage::{
    COMMAND_ROWS, Coverage, DATA_KIND_ROWS, EXPRESSION_ROWS, LITERAL_ROWS, RELATION_ROWS, Status,
    all,
};
use krishiv_connect::translate::{
    TranslateConf, expression_from_bytes, relation_from_bytes, translate_plan,
};
use prost_reflect::DescriptorPool;

/// A varint.
fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// A message whose only field is `number`, holding `payload`.
fn field(number: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    varint(u64::from(number) << 3 | 2, &mut out);
    varint(payload.len() as u64, &mut out);
    out.extend_from_slice(payload);
    out
}

fn refused(answer: &Result<(), krishiv_common::error::KrishivError>, row: &Coverage) {
    let got = answer.as_ref().err();
    let is_56k00 = got.is_some_and(|e| {
        e.error_class() == ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSupported
    });
    let expect = matches!(row.status, Status::Missing | Status::OutOfScope);
    assert_eq!(
        is_56k00,
        expect,
        "{} {} ({}) is `{}`: {:?}",
        row.oneof,
        row.field,
        row.number,
        row.status.word(),
        got.map(ToString::to_string)
    );
    if is_56k00 {
        let text = got.map(ToString::to_string).unwrap_or_default();
        assert!(
            text.contains(&format!(
                "{}({})",
                row.field.to_ascii_uppercase(),
                row.number
            )),
            "{}: {text}",
            row.field
        );
    }
}

#[test]
fn coverage_matches_translator() {
    let conf = TranslateConf::default();
    for row in RELATION_ROWS {
        refused(
            &relation_from_bytes(&field(row.number, &[]), &conf).map(|_| ()),
            row,
        );
    }
    for row in EXPRESSION_ROWS {
        refused(
            &expression_from_bytes(&field(row.number, &[]), &conf).map(|_| ()),
            row,
        );
    }
    // Plan.command is field 2; a Command holds the arm.
    for row in COMMAND_ROWS {
        let plan = field(2, &field(row.number, &[]));
        let decoded = krishiv_connect::wire::decode_plan(&plan).expect("a plan");
        refused(&translate_plan(&decoded, &conf).map(|_| ()), row);
    }
    // Expression.literal is field 1.
    for row in LITERAL_ROWS {
        refused(
            &expression_from_bytes(&field(1, &field(row.number, &[])), &conf).map(|_| ()),
            row,
        );
    }
    // Expression.cast is field 7: Cast.type is field 2 (read before the expression, field 1).
    for row in DATA_KIND_ROWS {
        let cast = field(2, &field(row.number, &[]));
        refused(
            &expression_from_bytes(&field(7, &cast), &conf).map(|_| ()),
            row,
        );
    }
}

#[test]
fn every_row_matches_the_descriptor() {
    let pool = DescriptorPool::decode(
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/spark-connect-4.2.fds"),
        )
        .expect("descriptor set")
        .as_slice(),
    )
    .expect("a valid descriptor set");
    for (oneof, message, group) in [
        ("Relation.rel_type", "spark.connect.Relation", "rel_type"),
        (
            "Expression.expr_type",
            "spark.connect.Expression",
            "expr_type",
        ),
        (
            "Command.command_type",
            "spark.connect.Command",
            "command_type",
        ),
        ("Catalog.cat_type", "spark.connect.Catalog", "cat_type"),
        (
            "Expression.Literal.literal_type",
            "spark.connect.Expression.Literal",
            "literal_type",
        ),
        ("DataType.kind", "spark.connect.DataType", "kind"),
    ] {
        let desc = pool.get_message_by_name(message).expect(message);
        let oneof_desc = desc
            .oneofs()
            .find(|o| o.name() == group)
            .expect("the oneof");
        let mut fields: Vec<(String, u32)> = oneof_desc
            .fields()
            .map(|f| (f.name().to_owned(), f.number()))
            .collect();
        fields.sort_by_key(|f| f.1);
        let rows: Vec<(String, u32)> = all()
            .into_iter()
            .filter(|r| r.oneof == oneof)
            .map(|r| (r.field.to_owned(), r.number))
            .collect();
        assert_eq!(rows, fields, "{oneof}");
    }
}

#[test]
fn every_arm_has_one_row_and_a_reason() {
    let rows = all();
    for (i, a) in rows.iter().enumerate() {
        for b in &rows[i + 1..] {
            assert!(
                !(a.oneof == b.oneof && a.number == b.number),
                "{} {} twice",
                a.oneof,
                a.number
            );
        }
        match a.status {
            Status::Missing => assert!(
                !a.task.is_empty(),
                "{} {} has no owning task",
                a.oneof,
                a.field
            ),
            Status::Partial => {
                assert!(
                    !a.refused.is_empty(),
                    "{} {} names no refused field",
                    a.oneof,
                    a.field
                );
            }
            Status::Done | Status::OutOfScope => {}
        }
    }
    // The six oneofs of the design (s7.1).
    let mut names: Vec<&str> = rows.iter().map(|r| r.oneof).collect();
    names.dedup();
    assert_eq!(names.len(), 6, "{names:?}");
}
