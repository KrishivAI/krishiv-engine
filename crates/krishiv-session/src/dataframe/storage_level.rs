// SPDX-License-Identifier: Apache-2.0
//! [`StorageLevel`]: the levels of Spark's `StorageLevel` object (Spark 4.2.0
//! `common/utils/.../storage/StorageLevel.scala`) and its names
//! (`StorageLevelMapper`).

use krishiv_common::config::{Config, ConfigValue, DEFAULT_CACHE_STORAGE_LEVEL};

/// A storage level for [`DataFrame::persist`](crate::DataFrame::persist), as
/// Spark's `StorageLevel`. The relation cache holds every level's snapshot as
/// Arrow batches in memory under the `relation_cache` pool for now; the level
/// is kept and reported as registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLevel {
    /// Keep entries on disk (accepted; stored in memory for now).
    pub use_disk: bool,
    /// Keep entries in memory.
    pub use_memory: bool,
    /// Keep entries off-heap (accepted; stored in memory for now).
    pub use_off_heap: bool,
    /// Keep entries deserialized.
    pub deserialized: bool,
    /// How many copies to keep (accepted; one copy is kept for now).
    pub replication: i32,
}

impl StorageLevel {
    /// `NONE`: not cached (an uncached frame's `storageLevel`; `persist`
    /// with it does nothing).
    pub const NONE: Self = Self {
        use_disk: false,
        use_memory: false,
        use_off_heap: false,
        deserialized: false,
        replication: 1,
    };
    /// `MEMORY_ONLY`: one in-memory copy.
    pub const MEMORY_ONLY: Self = Self {
        use_disk: false,
        use_memory: true,
        use_off_heap: false,
        deserialized: true,
        replication: 1,
    };
    /// `MEMORY_ONLY_2`: accepted as `MEMORY_ONLY`.
    pub const MEMORY_ONLY_2: Self = Self {
        replication: 2,
        ..Self::MEMORY_ONLY
    };
    /// `MEMORY_AND_DISK`: one in-memory copy (`cache`'s default level).
    pub const MEMORY_AND_DISK: Self = Self {
        use_disk: true,
        ..Self::MEMORY_ONLY
    };
    /// `MEMORY_AND_DISK_2`: accepted as `MEMORY_AND_DISK`.
    pub const MEMORY_AND_DISK_2: Self = Self {
        use_disk: true,
        ..Self::MEMORY_ONLY_2
    };
    /// `DISK_ONLY`: accepted as `MEMORY_ONLY`.
    pub const DISK_ONLY: Self = Self {
        use_disk: true,
        use_memory: false,
        deserialized: false,
        ..Self::MEMORY_ONLY
    };
    /// `DISK_ONLY_2`: accepted as `MEMORY_ONLY`.
    pub const DISK_ONLY_2: Self = Self {
        replication: 2,
        ..Self::DISK_ONLY
    };
    /// `DISK_ONLY_3`: accepted as `MEMORY_ONLY`.
    pub const DISK_ONLY_3: Self = Self {
        replication: 3,
        ..Self::DISK_ONLY
    };
    /// `OFF_HEAP`: disk, memory and off-heap, serialized (accepted as
    /// `MEMORY_ONLY`).
    pub const OFF_HEAP: Self = Self {
        use_disk: true,
        use_memory: true,
        use_off_heap: true,
        deserialized: false,
        replication: 1,
    };

    /// The level Spark's `StorageLevel.fromString` names `name` (exact,
    /// upper-case names: callers upper-case first where Spark does).
    #[must_use]
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        let serialized = |level: Self| Self {
            deserialized: false,
            ..level
        };
        Some(match name {
            "NONE" => Self::NONE,
            "DISK_ONLY" => Self::DISK_ONLY,
            "DISK_ONLY_2" => Self::DISK_ONLY_2,
            "DISK_ONLY_3" => Self::DISK_ONLY_3,
            "MEMORY_ONLY" => Self::MEMORY_ONLY,
            "MEMORY_ONLY_2" => Self::MEMORY_ONLY_2,
            "MEMORY_ONLY_SER" => serialized(Self::MEMORY_ONLY),
            "MEMORY_ONLY_SER_2" => serialized(Self::MEMORY_ONLY_2),
            "MEMORY_AND_DISK" => Self::MEMORY_AND_DISK,
            "MEMORY_AND_DISK_2" => Self::MEMORY_AND_DISK_2,
            "MEMORY_AND_DISK_SER" => serialized(Self::MEMORY_AND_DISK),
            "MEMORY_AND_DISK_SER_2" => serialized(Self::MEMORY_AND_DISK_2),
            "OFF_HEAP" => Self::OFF_HEAP,
            _ => return None,
        })
    }

    /// `spark.sql.defaultCacheStorageLevel` of `config`.
    pub(crate) fn default_of(config: &Config) -> Self {
        config
            .get(&DEFAULT_CACHE_STORAGE_LEVEL)
            .and_then(ConfigValue::as_str)
            .and_then(Self::from_name)
            .unwrap_or(Self::MEMORY_AND_DISK)
    }
}

#[cfg(test)]
mod tests {
    use super::StorageLevel;

    #[test]
    fn names_map_to_spark_levels() {
        // Spark 4.2.0 `StorageLevel.scala` 149-161.
        let level = |d, m, o, s, r| StorageLevel {
            use_disk: d,
            use_memory: m,
            use_off_heap: o,
            deserialized: s,
            replication: r,
        };
        for (name, want) in [
            ("NONE", level(false, false, false, false, 1)),
            ("DISK_ONLY", level(true, false, false, false, 1)),
            ("DISK_ONLY_2", level(true, false, false, false, 2)),
            ("DISK_ONLY_3", level(true, false, false, false, 3)),
            ("MEMORY_ONLY", level(false, true, false, true, 1)),
            ("MEMORY_ONLY_2", level(false, true, false, true, 2)),
            ("MEMORY_ONLY_SER", level(false, true, false, false, 1)),
            ("MEMORY_ONLY_SER_2", level(false, true, false, false, 2)),
            ("MEMORY_AND_DISK", level(true, true, false, true, 1)),
            ("MEMORY_AND_DISK_2", level(true, true, false, true, 2)),
            ("MEMORY_AND_DISK_SER", level(true, true, false, false, 1)),
            ("MEMORY_AND_DISK_SER_2", level(true, true, false, false, 2)),
            ("OFF_HEAP", level(true, true, true, false, 1)),
        ] {
            assert_eq!(StorageLevel::from_name(name), Some(want), "{name}");
        }
        assert_eq!(StorageLevel::from_name("memory_only"), None);
        assert_eq!(StorageLevel::from_name("MEMORY_ONLY_3"), None);
    }
}
