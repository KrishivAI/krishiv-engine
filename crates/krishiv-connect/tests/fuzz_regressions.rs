// SPDX-License-Identifier: Apache-2.0
//! Inputs the `connect_decode` fuzz target reported as out-of-memory
//! (first byte: 0 plan, 1 relation, 2 expression; the rest is the message).
//! Each is a few dozen bytes. These tests assert the decode ends in a typed
//! error, not a panic; they do not measure allocation. The memory bound is
//! checked where the finding came from: `fuzz.yml` replays
//! `fuzz/regressions/connect_decode` under the fuzz target's memory limit.

use krishiv_connect::translate::{
    TranslateConf, plan_from_bytes, relation_from_bytes, translate_expression,
};
use krishiv_connect::wire::{decode_expression, decode_plan, decode_relation};

/// What the `connect_decode` target does with `data`; whether it decoded.
fn run(data: &[u8]) -> bool {
    let Some((&which, bytes)) = data.split_first() else {
        return false;
    };
    let conf = TranslateConf::default();
    match which % 3 {
        0 => {
            let decoded = decode_plan(bytes);
            if let Ok(plan) = &decoded {
                drop(format!("{:?}", **plan));
            }
            drop(plan_from_bytes(bytes, &conf));
            decoded.is_ok()
        }
        1 => {
            let decoded = decode_relation(bytes);
            if let Ok(relation) = &decoded {
                drop(format!("{:?}", **relation));
            }
            drop(relation_from_bytes(bytes, &conf));
            decoded.is_ok()
        }
        _ => {
            let decoded = decode_expression(bytes);
            if let Ok(expression) = &decoded {
                drop(format!("{:?}", **expression));
                drop(translate_expression(expression, &conf));
            }
            decoded.is_ok()
        }
    }
}

/// A field number Spark Connect 4.2 does not define, after nested messages.
#[test]
fn unknown_plan_field_after_nested_messages() {
    let input = [
        0x00, 0x0a, 0x04, 0x22, 0x02, 0x12, 0x00, 0x28, 0x05, 0x0a, 0x02, 0x0a, 0x00, 0x0a, 0x00,
        0x52, 0x0a, 0x00, 0x52, 0x06, 0x1a, 0x02, 0x0a, 0x00, 0x0a, 0x02, 0x0a, 0x00, 0x0a, 0x00,
        0x52, 0x06, 0x1a, 0x02, 0xea, 0xfd, 0x28, 0x00, 0x06, 0x1a, 0x02, 0xea, 0xfd, 0x28, 0x00,
        0x0a, 0x00,
    ];
    assert!(!run(&input));
}

/// A group wire type.
#[test]
fn group_wire_type_in_a_plan() {
    let input = [
        0xe7, 0x2c, 0x12, 0x0a, 0x0a, 0x88, 0x0a, 0x00, 0x22, 0x04, 0x0a, 0x0a, 0x12, 0x00,
    ];
    assert!(!run(&input));
}

/// A length that runs past the end of its message.
#[test]
fn length_past_the_end_of_a_relation() {
    let input = [
        0xa9, 0x12, 0x16, 0x0a, 0x04, 0x0a, 0x78, 0x78, 0x78, 0x78, 0x78, 0x10, 0xba, 0x02, 0x00,
        0xfa, 0x02, 0x00, 0xba,
    ];
    assert!(!run(&input));
}
