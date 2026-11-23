// SPDX-License-Identifier: Apache-2.0
use super::*;
use prost::Message as _;

fn any(name: &str, value: Vec<u8>) -> Any {
    Any {
        type_url: format!("{TYPE_URL_PREFIX}{name}"),
        value,
    }
}

fn echo(text: &str) -> Any {
    any(
        "Echo",
        Echo {
            message: text.to_owned(),
        }
        .encode_to_vec(),
    )
}

#[test]
fn a_registered_message_resolves_in_its_slots() {
    for slot in [Slot::Relation, Slot::Expression] {
        assert_eq!(
            resolve(&echo("hi"), slot).unwrap(),
            Extension::Echo(Echo {
                message: "hi".to_owned()
            })
        );
    }
}

#[test]
fn a_command_slot_is_refused_for_echo() {
    let text = resolve(&echo("hi"), Slot::Command).unwrap_err().to_string();
    assert!(
        text.contains("krishiv.v1 extension `Echo` in a command"),
        "{text}"
    );
}

#[test]
fn an_unknown_message_names_its_capability_not_a_decode_failure() {
    let err = resolve(&any("Vacuum", vec![0xFF, 0xFF, 0xFF]), Slot::Relation).unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
    let text = err.to_string();
    assert!(text.contains("capability `vacuum`"), "{text}");
    assert!(text.contains("system.capabilities"), "{text}");
}

#[test]
fn another_package_is_refused_with_a_bounded_printable_name() {
    let foreign = Any {
        type_url: format!("type.googleapis.com/evil.\u{7}{}", "x".repeat(500)),
        value: Vec::new(),
    };
    let text = resolve(&foreign, Slot::Expression).unwrap_err().to_string();
    assert!(text.contains("only `krishiv.v1` extensions"), "{text}");
    assert!(text.len() < 300, "{}", text.len());
    assert!(!text.contains('\u{7}'));
    let empty = resolve(&Any::default(), Slot::Command)
        .unwrap_err()
        .to_string();
    assert!(empty.contains("only `krishiv.v1` extensions"), "{empty}");
}

#[test]
fn a_registered_message_with_bad_bytes_is_refused_with_its_own_class() {
    use krishiv_common::error::ErrorClass;
    // A length that runs past the end of the message: the wire check.
    let overrun = resolve(&any("Echo", vec![0x0A, 0x7F]), Slot::Relation).unwrap_err();
    assert_eq!(
        overrun.error_class(),
        ErrorClass::ConnectInvalidPlanCannotParse
    );
    assert!(
        overrun.to_string().contains("runs past the end"),
        "{overrun}"
    );
    // A valid length with bytes that are not UTF-8: passes the wire check, prost refuses it.
    let utf8 = resolve(&any("Echo", vec![0x0A, 0x01, 0xFF]), Slot::Relation).unwrap_err();
    assert_eq!(
        utf8.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    assert!(utf8.to_string().contains("does not decode"), "{utf8}");
}

/// prost recurses once per level of an unknown group and this crate builds it without a recursion
/// limit: a payload of group-start bytes would overflow the stack. It is refused before prost sees it.
#[test]
fn nested_groups_in_an_extension_are_refused_without_recursing() {
    use krishiv_common::error::ErrorClass;
    for slot in [Slot::Relation, Slot::Expression] {
        let hostile = any("Echo", vec![0x13; 4 * 1024 * 1024]);
        let err = resolve(&hostile, slot).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::ConnectInvalidPlanCannotParse);
        assert!(err.to_string().contains("group"), "{err}");
    }
}

#[test]
fn echo_debug_shows_the_length_only() {
    let shown = format!(
        "{:?}",
        Echo {
            message: "secret".to_owned()
        }
    );
    assert!(!shown.contains("secret"), "{shown}");
    assert_eq!(shown, "Echo { message_len: 6 }");
}

#[test]
fn capabilities_are_sorted_and_joined() {
    assert_eq!(capabilities(), vec!["echo", "persistpythonfunction"]);
    assert_eq!(capabilities_value(), "echo,persistpythonfunction");
}

/// The `.proto` snapshot lists exactly the messages the registry carries, and the pinned fields
/// below only grow (additive within v1).
#[test]
fn the_proto_snapshot_matches_the_messages_and_only_grows() {
    let proto = include_str!("../../../../proto/krishiv/v1/krishiv.proto");
    assert!(proto.contains("package krishiv.v1;"));
    // (message, field, number, type): append only.
    let pinned: &[(&str, &str, u32, &str)] = &[
        ("Echo", "message", 1, "string"),
        ("PersistPythonFunction", "name", 1, "string"),
        ("PersistPythonFunction", "drop", 2, "bool"),
    ];
    for (message, field, number, ty) in pinned {
        let start = proto
            .find(&format!("message {message} {{"))
            .unwrap_or_else(|| panic!("message {message} is gone"));
        let body = &proto[start..start + proto[start..].find('}').expect("closed")];
        assert!(
            body.contains(&format!("{ty} {field} = {number};")),
            "{message}.{field} changed or was removed"
        );
    }
    let mut declared: Vec<&str> = proto
        .lines()
        .filter_map(|l| l.strip_prefix("message "))
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    declared.sort_unstable();
    let mut registered: Vec<&str> = REGISTRY.iter().map(|r| r.message).collect();
    registered.sort_unstable();
    assert_eq!(declared, registered);
}

fn persist(name: &str, drop: bool) -> Any {
    any(
        "PersistPythonFunction",
        PersistPythonFunction {
            name: name.to_owned(),
            drop,
        }
        .encode_to_vec(),
    )
}

#[test]
fn persist_python_function_resolves_in_a_command_only() {
    assert_eq!(
        resolve(&persist("f", true), Slot::Command).unwrap(),
        Extension::PersistPythonFunction(PersistPythonFunction {
            name: "f".to_owned(),
            drop: true
        })
    );
    for slot in [Slot::Relation, Slot::Expression] {
        let e = resolve(&persist("f", false), slot).unwrap_err();
        assert!(e.to_string().contains("persistpythonfunction"), "{e}");
    }
}

#[test]
fn persist_python_function_refuses_a_group_and_an_overrun() {
    use krishiv_common::error::ErrorClass;
    // As Echo's bytes (above): the wire check, then prost's own refusal; never "unsupported extension".
    let group = resolve(
        &any("PersistPythonFunction", vec![0x0B, 0x0C]),
        Slot::Command,
    )
    .unwrap_err();
    assert_eq!(
        group.error_class(),
        ErrorClass::ConnectInvalidPlanCannotParse
    );
    let overrun = resolve(
        &any("PersistPythonFunction", vec![0x0A, 0x7F]),
        Slot::Command,
    )
    .unwrap_err();
    assert_eq!(
        overrun.error_class(),
        ErrorClass::ConnectInvalidPlanCannotParse
    );
    assert!(
        overrun.to_string().contains("runs past the end"),
        "{overrun}"
    );
    let utf8 = resolve(
        &any("PersistPythonFunction", vec![0x0A, 0x01, 0xFF]),
        Slot::Command,
    )
    .unwrap_err();
    assert_eq!(
        utf8.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    assert!(utf8.to_string().contains("does not decode"), "{utf8}");
}
