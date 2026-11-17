// SPDX-License-Identifier: Apache-2.0
//! The cache arms' wire forms (design P3-T083 §3.4): Spark's `StorageLevelProtoConverter`
//! (`toStorageLevel`, `toConnectProtoType`) between the protocol's `StorageLevel` and the
//! engine's. Every field is carried both ways, the replication factor included.

use krishiv_connect::proto::v4_3 as pb;
use krishiv_session::StorageLevel;

/// The engine's storage level for the protocol's.
#[must_use]
pub(crate) fn storage_level_of(level: &pb::StorageLevel) -> StorageLevel {
    StorageLevel {
        use_disk: level.use_disk,
        use_memory: level.use_memory,
        use_off_heap: level.use_off_heap,
        deserialized: level.deserialized,
        replication: level.replication,
    }
}

/// The protocol's storage level for the engine's.
#[must_use]
pub(crate) fn proto_of(level: StorageLevel) -> pb::StorageLevel {
    pb::StorageLevel {
        use_disk: level.use_disk,
        use_memory: level.use_memory,
        use_off_heap: level.use_off_heap,
        deserialized: level.deserialized,
        replication: level.replication,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_field_survives_both_ways() {
        for level in [
            StorageLevel::NONE,
            StorageLevel::DISK_ONLY,
            StorageLevel::DISK_ONLY_2,
            StorageLevel::MEMORY_ONLY,
            StorageLevel::MEMORY_ONLY_2,
            StorageLevel::MEMORY_AND_DISK,
            StorageLevel::MEMORY_AND_DISK_2,
            StorageLevel::OFF_HEAP,
        ] {
            assert_eq!(storage_level_of(&proto_of(level)), level);
        }
        let proto = pb::StorageLevel {
            use_disk: true,
            use_memory: false,
            use_off_heap: true,
            deserialized: true,
            replication: 3,
        };
        assert_eq!(proto_of(storage_level_of(&proto)), proto);
        assert_eq!(storage_level_of(&proto).replication, 3);
    }
}
