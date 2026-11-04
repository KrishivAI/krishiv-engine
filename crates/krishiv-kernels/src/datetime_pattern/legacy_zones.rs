// SPDX-License-Identifier: Apache-2.0
//! What `SimpleDateFormat` (`Locale.US`) reads under LEGACY, from the JVM of Spark 4.2.0
//! (Java 21.0.11): its zone strings and the decimal digits of numeric fields. The tables are
//! generated into `vendored/legacy_zones.rs` by `tests/scripts/gen_legacy_zone_table.py`.

/// A row of `DateFormatSymbols.getZoneStrings()`.
pub(super) struct ZoneStrings {
    /// The zone id.
    pub id: &'static str,
    /// Long standard, short standard, long daylight, short daylight name.
    pub names: [&'static str; 4],
    /// `TimeZone.getRawOffset()` of the id, in seconds.
    pub raw: i32,
    /// `TimeZone.getDSTSavings()` of the id, in seconds.
    pub dst: i32,
}

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendored/legacy_zones.rs"
));

/// The row `DateFormatSymbols.getZoneIndex(id)` finds: the one whose id is
/// `id` exactly, with that id's own offsets.
pub(super) fn zone_row(id: &str) -> Option<ZoneStrings> {
    if let Some(row) = ZONE_STRINGS.iter().find(|r| r.id == id) {
        return Some(ZoneStrings { ..*row });
    }
    let &(id, row, raw, dst) = ZONE_ALIASES.iter().find(|a| a.0 == id)?;
    Some(ZoneStrings {
        id,
        names: ZONE_STRINGS.get(usize::from(row))?.names,
        raw,
        dst,
    })
}
