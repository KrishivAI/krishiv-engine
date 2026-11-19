// SPDX-License-Identifier: Apache-2.0
//! `system.catalogs` (design P3-T067 §7): one row per declared catalog the session's principal
//! sees (§2.4: a catalog hidden from its tenant has no row). No URI and no credential, ever: a URI
//! can name an internal host or carry userinfo, and a token is a token.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray};

use crate::system::{Kind, Sources, text};

/// The rows.
pub(crate) fn columns(sources: &Sources) -> Vec<ArrayRef> {
    let rows = sources.catalogs.rows();
    vec![
        text(rows.iter().map(|r| Some(r.name.as_str()))),
        text(rows.iter().map(|r| Some(r.kind))),
        text(rows.iter().map(|r| Some(r.auth))),
        text(rows.iter().map(|r| Some(r.storage))),
        text(rows.iter().map(|r| r.delegation)),
        Arc::new(BooleanArray::from_iter(
            rows.iter().map(|r| Some(r.default)),
        )),
    ]
}

pub(crate) fn layout() -> Vec<(&'static str, Kind, bool)> {
    use Kind::{Flag, Text};
    vec![
        ("name", Text, false),
        ("type", Text, false),
        ("auth", Text, false),
        ("storage", Text, false),
        ("delegation", Text, true),
        ("default", Flag, false),
    ]
}
