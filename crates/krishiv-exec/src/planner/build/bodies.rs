// SPDX-License-Identifier: Apache-2.0
//! The deciding planning pass's bodies: each deterministic shared definition planned once.

use std::collections::HashMap;
use std::ops::Range;

use krishiv_plan::AttrId;

use super::{Planned, ScanId};

/// The deciding pass's bodies. Planning every reference of a deterministic definition anew is exponential in a
/// chain of definitions that each read the previous one twice, so this pass plans each body once, at its first
/// reference, and gives every later reference the body's planned shape (schema, layout, runtime-filter sites,
/// placement, order) over a stand-in node. Each site of a reused shape names a new scan id standing for the first
/// copy's scan (`aliases`), so a runtime filter that reaches the reference counts against the bodies planned
/// around the reference and against the definitions the scan belongs to, as it would for the reference's own
/// copy. The pass then makes the decisions of planning every reference anew in time linear in the plan.
///
/// This holds because no planning step that creates a runtime filter reads a reused reference's node: the
/// filter-over-scan site check and the top-N bound look only at a `Scan` (or `Filter` over one) directly below,
/// and a shared definition's body always holds a pipeline breaker, so a reference's node is never one.
///
/// When the pass finds nothing to share, the second pass still plans every reference anew; a chain of
/// definitions that runtime filters reach plans in time exponential in its length, as before.
#[derive(Default)]
pub(super) struct Bodies {
    /// Each planned body's shape, by definition and whether it was planned under a `LIMIT 0` (which plans no
    /// observed node, so its sites may differ).
    pub(super) shapes: HashMap<(Vec<AttrId>, bool), Planned>,
    /// The first copy's scan id each reused reference's scan id stands for, with the reused definition: the
    /// scan is in the copies of that definition and of those planned within its first copy only.
    pub(super) aliases: HashMap<u64, (u64, Vec<AttrId>)>,
    /// The definitions each definition's first copy was planned within, by definition.
    pub(super) within: HashMap<Vec<AttrId>, Vec<Vec<AttrId>>>,
    /// The enclosing definition of each reference (the innermost body being planned; `None` at the top or in a
    /// shared plan, which is planned once), by definition.
    pub(super) references: HashMap<Vec<AttrId>, Vec<Option<Vec<AttrId>>>>,
    /// The bodies being planned, innermost last; `None` for a shared plan's.
    pub(super) enclosing: Vec<Option<Vec<AttrId>>>,
    /// Whether a reference reused a body.
    pub(super) reused: bool,
}

impl Bodies {
    /// A later reference of definition `def`: its body's `shape`, each scan a site names under a new id (from
    /// `next_scan`) standing for this reference's copy of it, so a filter that reaches the reference reaches
    /// the bodies around it, not those around the first copy.
    pub(super) fn reuse(
        &mut self,
        mut shape: Planned,
        def: &[AttrId],
        next_scan: &mut u64,
    ) -> Planned {
        let mut fresh = HashMap::new();
        for site in &mut shape.sites {
            let id = *fresh.entry(site.scan.0).or_insert_with(|| {
                let id = *next_scan;
                *next_scan += 1;
                id
            });
            self.aliases.insert(id, (site.scan.0, def.to_vec()));
            site.scan = ScanId(id);
        }
        shape
    }

    /// Whether `scan`, a reused reference's (`aliases`), is in a copy of the definition `key` whose first copy
    /// has the scans `first`: the reference copied `key` or a definition within whose first copy `key`'s first
    /// copy was planned, at some step of the walk to the scans it stands for (ids only fall, so it ends).
    pub(super) fn covers(&self, key: &Vec<AttrId>, first: &Range<u64>, mut scan: u64) -> bool {
        while let Some((original, def)) = self.aliases.get(&scan) {
            let nested = || self.within.get(key).is_some_and(|w| w.contains(def));
            if def == key || (first.contains(original) && nested()) {
                return true;
            }
            scan = *original;
        }
        false
    }

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
