// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2's registry of spatial reference systems: which SRIDs `GEOMETRY(srid)` and
//! `GEOGRAPHY(srid)` accept, the name each has in the JSON form of a type, and which are
//! geographic.
//!
//! The data is Spark's own `srs_registry.csv` (apache/spark v4.2.0,
//! `sql/api/src/main/resources/org/apache/spark/sql/`, generated from PROJ 9.8.1:
//! 9,307 projected and 1,406 geographic systems), vendored unchanged in
//! `vendored/srs_registry.csv`. On top of it Spark's `SpatialReferenceSystemCache` adds SRID 0
//! (`SRID:0`, Cartesian) and gives three systems an OGC name, keeping the EPSG name as an alias:
//! 4326 `OGC:CRS84`, 4267 `OGC:CRS27`, 4269 `OGC:CRS83`.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The registry as shipped.
const REGISTRY: &str = include_str!("../vendored/srs_registry.csv");

/// The OGC names that replace the EPSG name of a system: `(srid, ogc name)`.
const OGC_NAMES: [(i32, &str); 3] = [
    (4326, "OGC:CRS84"),
    (4267, "OGC:CRS27"),
    (4269, "OGC:CRS83"),
];

/// One reference system.
#[derive(Clone, Copy)]
struct Entry {
    name: &'static str,
    geographic: bool,
}

struct Registry {
    by_srid: HashMap<i32, Entry>,
    /// Every name (primary and alias) to its SRID.
    by_name: HashMap<&'static str, i32>,
}

fn registry() -> &'static Registry {
    static REGISTRY_CELL: OnceLock<Registry> = OnceLock::new();
    REGISTRY_CELL.get_or_init(load)
}

fn load() -> Registry {
    let mut by_srid = HashMap::new();
    let mut by_name = HashMap::new();
    for line in REGISTRY.lines() {
        // Comment lines and the header are skipped, as Spark does.
        if line.starts_with('#') || line.starts_with("srid,") {
            continue;
        }
        let mut parts = line.splitn(3, ',');
        let (Some(srid), Some(name), Some(geographic)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let Ok(srid) = srid.trim().parse::<i32>() else {
            continue;
        };
        let (name, geographic) = (name.trim(), geographic.trim() == "true");
        by_srid.insert(srid, Entry { name, geographic });
        by_name.insert(name, srid);
    }
    by_srid.insert(
        0,
        Entry {
            name: "SRID:0",
            geographic: false,
        },
    );
    by_name.insert("SRID:0", 0);
    for (srid, ogc) in OGC_NAMES {
        if let Some(entry) = by_srid.get_mut(&srid) {
            entry.name = ogc;
            by_name.insert(ogc, srid);
        }
    }
    Registry { by_srid, by_name }
}

/// The name Spark gives `srid` in the JSON form of a type, when the registry has it and, with
/// `geographic`, when it is a geographic system.
pub(crate) fn name_of(srid: i32, geographic: bool) -> Option<&'static str> {
    registry()
        .by_srid
        .get(&srid)
        .filter(|e| e.geographic || !geographic)
        .map(|e| e.name)
}

/// The SRID a name (primary or alias, `EPSG:4326` as well as `OGC:CRS84`) stands for.
pub(crate) fn srid_of(name: &str) -> Option<i32> {
    registry().by_name.get(name).copied()
}

/// Whether `srid` is a geographic system in the registry.
pub(crate) fn is_geographic(srid: i32) -> bool {
    registry().by_srid.get(&srid).is_some_and(|e| e.geographic)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_has_spark_4_2_counts() {
        let r = registry();
        let geographic = r.by_srid.values().filter(|e| e.geographic).count();
        assert_eq!(geographic, 1406);
        // 10,713 rows of the file plus SRID 0.
        assert_eq!(r.by_srid.len(), 9307 + 1406 + 1);
    }

    #[test]
    fn spark_specific_entries_and_ogc_names() {
        assert_eq!(name_of(0, false), Some("SRID:0"));
        assert_eq!(name_of(0, true), None);
        assert_eq!(name_of(3857, false), Some("EPSG:3857"));
        assert_eq!(name_of(3857, true), None);
        assert_eq!(name_of(4326, true), Some("OGC:CRS84"));
        assert_eq!(name_of(4267, true), Some("OGC:CRS27"));
        assert_eq!(name_of(4269, true), Some("OGC:CRS83"));
        assert_eq!(name_of(4258, true), Some("EPSG:4258"));
        assert_eq!(name_of(4979, true), Some("EPSG:4979"));
        assert_eq!(name_of(-1, false), None);
        assert_eq!(name_of(1, false), None);
    }

    #[test]
    fn a_name_or_its_epsg_alias_finds_the_srid() {
        assert_eq!(srid_of("OGC:CRS84"), Some(4326));
        assert_eq!(srid_of("EPSG:4326"), Some(4326));
        assert_eq!(srid_of("EPSG:4269"), Some(4269));
        assert_eq!(srid_of("OGC:CRS83"), Some(4269));
        assert_eq!(srid_of("SRID:0"), Some(0));
        assert_eq!(srid_of("EPSG:3857"), Some(3857));
        assert_eq!(srid_of("nonsense"), None);
        assert!(is_geographic(4326) && !is_geographic(3857) && !is_geographic(0));
    }
}
