// SPDX-License-Identifier: Apache-2.0
//! The generated `Debug` and `Scrub` of the protocol types (ADR 0038): a
//! canary put in every kind of text a plan carries never prints, and the
//! credentials of a decoded plan are wiped before it is dropped.

use std::collections::BTreeMap;

use prost::Message;

use super::Scrub;
use super::v4_2::{
    Expression, Filter, Join, Plan, Project, Read, Relation, Sql, expression, join, plan,
    read::{DataSource, NamedTable, ReadType},
    relation::RelType,
};

const CANARY: &str = "Canary-91c4e";

fn relation(rel_type: RelType) -> Relation {
    Relation {
        common: None,
        rel_type: Some(rel_type),
    }
}

fn literal(value: expression::literal::LiteralType) -> Expression {
    Expression {
        common: None,
        expr_type: Some(expression::ExprType::Literal(expression::Literal {
            data_type: None,
            literal_type: Some(value),
        })),
    }
}

fn any() -> prost_types::Any {
    prost_types::Any {
        type_url: "type.googleapis.com/krishiv.v1.Example".to_owned(),
        value: CANARY.as_bytes().to_vec(),
    }
}

fn read(read_type: ReadType) -> Relation {
    relation(RelType::Read(Read {
        is_streaming: false,
        read_type: Some(read_type),
    }))
}

/// A plan with the canary in SQL text, string and binary literals, a column
/// name, read options, paths, a table name and an `Any` extension.
fn canary_plan() -> Plan {
    use expression::literal::LiteralType;
    let options = BTreeMap::from([("password".to_owned(), CANARY.to_owned())]);
    let files = read(ReadType::DataSource(DataSource {
        format: Some("parquet".to_owned()),
        schema: None,
        options: options.clone(),
        paths: vec![format!("s3://b/k?X-Amz-Signature={CANARY}")],
        predicates: vec![format!("x = '{CANARY}'")],
        source_name: None,
    }));
    let table = read(ReadType::NamedTable(NamedTable {
        unparsed_identifier: CANARY.to_owned(),
        options,
    }));
    let sql = relation(RelType::Sql(Sql {
        query: format!("SELECT '{CANARY}'"),
        named_arguments: BTreeMap::from([(
            "p".to_owned(),
            literal(LiteralType::String(CANARY.to_owned())),
        )]),
        ..Sql::default()
    }));
    let project = relation(RelType::Project(Box::new(Project {
        input: Some(Box::new(sql)),
        expressions: vec![
            literal(LiteralType::String(CANARY.to_owned())),
            literal(LiteralType::Binary(CANARY.as_bytes().to_vec())),
            Expression {
                common: None,
                expr_type: Some(expression::ExprType::UnresolvedAttribute(
                    expression::UnresolvedAttribute {
                        unparsed_identifier: CANARY.to_owned(),
                        ..Default::default()
                    },
                )),
            },
            Expression {
                common: None,
                expr_type: Some(expression::ExprType::Extension(any())),
            },
        ],
    })));
    let join = relation(RelType::Join(Box::new(Join {
        left: Some(Box::new(files)),
        right: Some(Box::new(table)),
        join_type: join::JoinType::Inner.into(),
        ..Join::default()
    })));
    let filter = relation(RelType::Filter(Box::new(Filter {
        input: Some(Box::new(join)),
        condition: Some(literal(LiteralType::Boolean(true))),
    })));
    let union = relation(RelType::Extension(any()));
    Plan {
        op_type: Some(plan::OpType::Root(Box::new(relation(
            RelType::WithRelations(Box::new(super::v4_2::WithRelations {
                root: Some(Box::new(filter)),
                references: vec![project, union],
            })),
        )))),
    }
}

#[test]
fn debug_prints_the_structure_and_no_text() {
    let plan = canary_plan();
    let decoded = crate::wire::decode_plan(&plan.encode_to_vec()).expect("decodes");
    for text in [
        format!("{plan:?}"),
        format!("{plan:#?}"),
        format!("{:?}", *decoded),
    ] {
        assert!(!text.contains(CANARY), "Debug leaks: {text}");
        // The structure, enum names and sizes still show.
        assert!(
            text.contains("Project") && text.contains("Filter"),
            "{text}"
        );
        assert!(text.contains("JOIN_TYPE_INNER"), "{text}");
        assert!(text.contains("<redacted: 1 entries>"), "{text}");
        assert!(text.contains("<redacted: 21 bytes>"), "{text}");
        assert!(
            text.contains("type.googleapis.com/krishiv.v1.Example"),
            "{text}"
        );
    }
}

#[test]
fn an_unknown_enum_value_prints_its_number() {
    let join = Join {
        join_type: 99,
        ..Join::default()
    };
    assert!(format!("{join:?}").contains("join_type: 99"));
}

#[test]
fn scrub_wipes_option_values_paths_and_extensions_everywhere() {
    let credentials = [
        format!("X-Amz-Signature={CANARY}"),
        format!("password\x12\x0c{CANARY}"),
        format!("Example\x12\x0c{CANARY}"),
    ];
    let found = |plan: &Plan, needle: &[u8]| {
        let bytes = plan.encode_to_vec();
        bytes.windows(needle.len()).any(|w| w == needle)
    };
    let mut plan = canary_plan();
    for c in &credentials {
        assert!(found(&plan, c.as_bytes()), "{c}");
    }
    plan.scrub();
    // Option values, paths and Any payloads are gone, in every nested
    // relation and expression.
    for c in &credentials {
        assert!(!found(&plan, c.as_bytes()), "{c}");
    }
    // Keys and what is not a credential stay: SQL text, a literal (the
    // translated plan holds those as plain text too).
    assert!(found(&plan, b"password"));
    assert!(found(&plan, format!("SELECT '{CANARY}'").as_bytes()));
}

#[test]
fn debug_of_the_deepest_plan_fits_a_small_stack() {
    // A test thread has 2 MiB; `Debug` recurses once per level, like the
    // decoder, on a plan at the nesting cap (two levels per filter).
    let mut rel = relation(RelType::Sql(Sql::default()));
    for _ in 0..crate::wire::MAX_NESTING / 2 - 1 {
        rel = relation(RelType::Filter(Box::new(Filter {
            input: Some(Box::new(rel)),
            condition: None,
        })));
    }
    let decoded = crate::wire::decode_relation(&rel.encode_to_vec()).expect("at the cap");
    crate::stack::on_deep_stack(move || drop(rel)).expect("dropped");
    assert!(format!("{:?}", *decoded).matches("Filter").count() >= 511);
}
