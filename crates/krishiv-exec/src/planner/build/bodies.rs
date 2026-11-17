// SPDX-License-Identifier: Apache-2.0
//! The deciding planning pass's bodies: each deterministic shared definition planned once.

use std::collections::HashMap;

use krishiv_plan::AttrId;

use super::Planned;

/// The deciding pass's bodies. Planning every reference of a deterministic definition anew is exponential in a
/// chain of definitions that each read the previous one twice, so this pass plans each body once, at its first
/// reference, and gives every later reference the body's planned shape (schema, layout, runtime-filter sites,
/// placement, order) over a stand-in node. The sites name the first copy's scans, so a runtime filter that
/// reaches any reference reaches that copy, as it would have reached the reference's own copy.
#[derive(Default)]
pub(super) struct Bodies {
    /// Each planned body's shape, by definition.
    pub(super) shapes: HashMap<Vec<AttrId>, Planned>,
    /// The enclosing definition of each reference (the innermost body being planned; `None` at the top or in a
    /// shared plan, which is planned once), by definition.
    pub(super) references: HashMap<Vec<AttrId>, Vec<Option<Vec<AttrId>>>>,
    /// The bodies being planned, innermost last; `None` for a shared plan's.
    pub(super) enclosing: Vec<Option<Vec<AttrId>>>,
    /// Whether a reference reused a body.
    pub(super) reused: bool,
}

impl Bodies {
    /// The number of copies of each definition that planning every reference anew makes: a reference inside a
    /// body counts once per copy of that body (saturating).
    pub(super) fn copies(&self) -> HashMap<Vec<AttrId>, u64> {
        fn count(
            key: &Vec<AttrId>,
            references: &HashMap<Vec<AttrId>, Vec<Option<Vec<AttrId>>>>,
            counts: &mut HashMap<Vec<AttrId>, u64>,
        ) -> u64 {
            if let Some(&n) = counts.get(key) {
                return n;
            }
            // A body never references its own definition, so this ends.
            let n = references.get(key).map_or(0, |enclosing| {
                enclosing.iter().fold(0u64, |n, e| {
                    n.saturating_add(e.as_ref().map_or(1, |e| count(e, references, counts)))
                })
            });
            counts.insert(key.clone(), n);
            n
        }
        let mut counts = HashMap::new();
        for key in self.references.keys() {
            count(key, &self.references, &mut counts);
        }
        counts
    }
}
