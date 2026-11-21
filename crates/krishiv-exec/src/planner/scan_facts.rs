// SPDX-License-Identifier: Apache-2.0
//! What a scan's planner facts record (P3-T073 missing statistics): its estimate and, when the provider
//! reported no statistics, the table's name.

use super::build::util::source_name;
use super::explain::NodeFacts;

/// The facts of a scan of `scan` whose provider reported statistics when `has_stats`, estimated at `estimated_rows` rows.
pub(super) fn of(scan: &krishiv_plan::Scan, has_stats: bool, estimated_rows: f64) -> NodeFacts {
    NodeFacts {
        estimated_rows: Some(estimated_rows),
        tables_without_stats: if has_stats {
            Vec::new()
        } else {
            vec![source_name(scan)]
        },
        ..NodeFacts::default()
    }
}
