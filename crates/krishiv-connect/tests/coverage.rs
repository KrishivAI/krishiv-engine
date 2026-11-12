// SPDX-License-Identifier: Apache-2.0
//! The coverage table is tied to the translator (P3-T012a, owner decision 7): for every `Relation`
//! arm, an empty message of that arm answers `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` exactly when its
//! row says the arm is not translated, and the table holds one row per arm.
#![forbid(unsafe_code)]

use krishiv_common::error::ErrorClass;
use krishiv_connect::coverage::{RELATION_ROWS, Status, all};
use krishiv_connect::translate::{TranslateConf, relation_from_bytes};

/// A `Relation` whose only field is `number`, an empty message.
fn relation_bytes(number: u32) -> Vec<u8> {
    let mut tag = u64::from(number) << 3 | 2;
    let mut out = Vec::new();
    loop {
        let byte = (tag & 0x7f) as u8;
        tag >>= 7;
        if tag == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
    out.push(0);
    out
}

#[test]
fn coverage_matches_translator() {
    for row in RELATION_ROWS {
        let answer = relation_from_bytes(&relation_bytes(row.number), &TranslateConf::default());
        let refused = answer.as_ref().err().is_some_and(|e| {
            e.error_class() == ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSupported
        });
        let expect_refused = matches!(row.status, Status::Missing | Status::OutOfScope);
        assert_eq!(
            refused,
            expect_refused,
            "{} ({}) is `{}`: {:?}",
            row.field,
            row.number,
            row.status.word(),
            answer.err().map(|e| e.to_string())
        );
        if refused {
            let text = answer.err().map(|e| e.to_string()).unwrap_or_default();
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
                    !a.refused.is_empty() || a.field == "sql" || a.field == "sort_order",
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
