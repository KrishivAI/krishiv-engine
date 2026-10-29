# SPDX-License-Identifier: Apache-2.0
"""Storage levels for :meth:`krishiv.sql.DataFrame.persist`, as ``pyspark.StorageLevel``."""

from __future__ import annotations

__all__ = ["StorageLevel"]


class StorageLevel:
    """How a cached DataFrame is stored.

    Every level maps to memory: the relation cache holds Arrow batches under
    the ``relation_cache`` pool, so a level asking for disk behaves like its
    memory-only counterpart for now (disk levels spill there later).
    """

    def __init__(
        self,
        useDisk: bool,
        useMemory: bool,
        useOffHeap: bool,
        deserialized: bool,
        replication: int = 1,
    ) -> None:
        self.useDisk = useDisk
        self.useMemory = useMemory
        self.useOffHeap = useOffHeap
        self.deserialized = deserialized
        self.replication = replication

    def __repr__(self) -> str:
        return (
            f"StorageLevel({self.useDisk}, {self.useMemory}, "
            f"{self.useOffHeap}, {self.deserialized}, {self.replication})"
        )

    def __str__(self) -> str:
        return (
            ("Disk " if self.useDisk else "")
            + ("Memory " if self.useMemory else "")
            + ("OffHeap " if self.useOffHeap else "")
            + ("Deserialized " if self.deserialized else "Serialized ")
            + f"{self.replication}x Replicated"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StorageLevel):
            return NotImplemented
        return (
            self.useDisk == other.useDisk
            and self.useMemory == other.useMemory
            and self.useOffHeap == other.useOffHeap
            and self.deserialized == other.deserialized
            and self.replication == other.replication
        )

    #: One in-memory copy.
    MEMORY_ONLY: StorageLevel
    #: Two in-memory copies (accepted as ``MEMORY_ONLY``).
    MEMORY_ONLY_2: StorageLevel
    #: One in-memory copy (``cache``'s level).
    MEMORY_AND_DISK: StorageLevel
    #: Two in-memory copies (accepted as ``MEMORY_AND_DISK``).
    MEMORY_AND_DISK_2: StorageLevel
    #: No memory (accepted as ``MEMORY_ONLY``).
    DISK_ONLY: StorageLevel
    #: No memory, twice (accepted as ``MEMORY_ONLY``).
    DISK_ONLY_2: StorageLevel
    #: Off-heap (accepted as ``MEMORY_ONLY``).
    OFF_HEAP: StorageLevel


#: One in-memory copy.
StorageLevel.MEMORY_ONLY = StorageLevel(False, True, False, True)
#: Two in-memory copies (accepted as ``MEMORY_ONLY``).
StorageLevel.MEMORY_ONLY_2 = StorageLevel(False, True, False, True, 2)
#: One in-memory copy (``cache``'s level).
StorageLevel.MEMORY_AND_DISK = StorageLevel(True, True, False, True)
#: Two in-memory copies (accepted as ``MEMORY_AND_DISK``).
StorageLevel.MEMORY_AND_DISK_2 = StorageLevel(True, True, False, True, 2)
#: No memory (accepted as ``MEMORY_ONLY``).
StorageLevel.DISK_ONLY = StorageLevel(True, False, False, False)
#: No memory, twice (accepted as ``MEMORY_ONLY``).
StorageLevel.DISK_ONLY_2 = StorageLevel(True, False, False, False, 2)
#: Off-heap (accepted as ``MEMORY_ONLY``).
StorageLevel.OFF_HEAP = StorageLevel(False, False, True, False)
