// SPDX-License-Identifier: Apache-2.0
//! Storage levels by name, and the session default (`spark.sql.defaultCacheStorageLevel`).

use krishiv_common::config::{CACHE_DEFAULT_STORAGE_LEVEL, Config};

use crate::StorageLevel;

impl StorageLevel {
    /// The level Spark's `StorageLevel.fromString` names (ASCII case-insensitive), `None` for an
    /// unknown name. The `_SER` levels keep the rows serialized (`deserialized` false).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let upper = name.trim().to_ascii_uppercase();
        let (base, ser) = match upper.strip_suffix("_SER_2") {
            Some(b) => (format!("{b}_2"), true),
            None => match upper.strip_suffix("_SER") {
                Some(b) => (b.to_owned(), true),
                None => (upper, false),
            },
        };
        let level = match base.as_str() {
            "NONE" => Self::NONE,
            "DISK_ONLY" => Self::DISK_ONLY,
            "DISK_ONLY_2" => Self::DISK_ONLY_2,
            "DISK_ONLY_3" => Self {
                replication: 3,
                ..Self::DISK_ONLY
            },
            "MEMORY_ONLY" => Self::MEMORY_ONLY,
            "MEMORY_ONLY_2" => Self::MEMORY_ONLY_2,
            "MEMORY_AND_DISK" => Self::MEMORY_AND_DISK,
            "MEMORY_AND_DISK_2" => Self::MEMORY_AND_DISK_2,
            "OFF_HEAP" if !ser => Self::OFF_HEAP,
            _ => return None,
        };
        Some(if ser && level.use_memory {
            Self {
                deserialized: false,
                ..level
            }
        } else {
            level
        })
    }

    /// The level `cache()`, `catalog.cacheTable` and `CACHE TABLE` use under `config`
    /// (`MEMORY_AND_DISK` unless `spark.sql.defaultCacheStorageLevel` says otherwise).
    pub(crate) fn default_of(config: &Config) -> Self {
        config
            .get(&CACHE_DEFAULT_STORAGE_LEVEL)
            .and_then(|v| v.as_str())
            .and_then(Self::from_name)
            .unwrap_or(Self::MEMORY_AND_DISK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_to_levels() {
        assert_eq!(
            StorageLevel::from_name("disk_only"),
            Some(StorageLevel::DISK_ONLY)
        );
        assert_eq!(
            StorageLevel::from_name("DISK_ONLY_3").map(|l| l.replication),
            Some(3)
        );
        let ser = StorageLevel::from_name("MEMORY_AND_DISK_SER_2").unwrap();
        assert!(ser.use_disk && ser.use_memory && !ser.deserialized && ser.replication == 2);
        assert_eq!(StorageLevel::from_name("bogus"), None);
    }
}
