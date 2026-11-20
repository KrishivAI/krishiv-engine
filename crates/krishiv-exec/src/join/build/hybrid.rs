// SPDX-License-Identifier: Apache-2.0
//! Hybrid hash join: which level-0 partitions of a spilling build stay in
//! memory (`krishiv.exec.join.hybridResident`).
//!
//! Once a build spills, the old rule sent every partition to disk, so the
//! probe phase held no table that fills the budget and cannot spill. That
//! also writes and reads back every probe row (TPC-H q09 at SF1000: all
//! 1.5 B `orders` rows through the grace phase). Here the partitions that
//! fit stay resident, as long as the grace phase still has room to load one
//! spilled partition per driver beside them, and the resident tables keep to
//! a share of the memory limit, so the probe pipeline's own operators and
//! the sinks above them keep the rest.

/// The share of the memory limit resident tables may hold at most, as a
/// fraction `RESIDENT_NUM / RESIDENT_DEN`.
const RESIDENT_NUM: usize = 1;
const RESIDENT_DEN: usize = 2;
/// The share of the memory limit resident tables and the grace phase's
/// concurrent loads may hold together; the rest is left to scans (whose
/// decode gate takes up to a quarter) and the operators above the probe.
const TOGETHER_NUM: usize = 3;
const TOGETHER_DEN: usize = 4;

/// One level-0 partition of a finalizing build.
#[derive(Debug, Clone, Copy)]
pub(super) struct Partition {
    /// Rows in the partition (in memory and on disk).
    pub(super) rows: u64,
    /// `Some(bytes)`: in memory, not spilled; what keeping it resident
    /// costs (its chunks' reservation plus its index). `None`: spilled.
    pub(super) resident_cost: Option<usize>,
}

/// The level-0 partitions to keep in memory (a bit per partition), of
/// `parts` of a build that spilled at least one: the cheapest first, while
/// the kept tables stay within `RESIDENT_NUM / RESIDENT_DEN` of `budget`
/// and, with one load per driver (`drivers`) of the largest partitions left
/// on disk, within `TOGETHER_NUM / TOGETHER_DEN` of it. A partition on disk
/// is estimated at the bytes per row of the partitions in memory.
/// `budget` is the memory limit less what other consumers hold.
pub(super) fn keep(parts: &[Partition], budget: usize, drivers: usize) -> u64 {
    let (cost, rows) = parts
        .iter()
        .filter_map(|p| p.resident_cost.map(|c| (c, p.rows)))
        .fold((0usize, 0u64), |(c, r), (pc, pr)| {
            (c.saturating_add(pc), r.saturating_add(pr))
        });
    if rows == 0 {
        return 0;
    }
    // Bytes per row in memory, rounded up.
    let per_row = cost
        .div_ceil(usize::try_from(rows).unwrap_or(usize::MAX))
        .max(1);
    let estimate = |p: &Partition| {
        p.resident_cost.unwrap_or_else(|| {
            usize::try_from(p.rows)
                .unwrap_or(usize::MAX)
                .saturating_mul(per_row)
        })
    };
    let resident_cap = budget / RESIDENT_DEN * RESIDENT_NUM;
    let together_cap = budget / TOGETHER_DEN * TOGETHER_NUM;
    let drivers = drivers.max(1);
    // Candidates cheapest first; everything else is on disk.
    let mut order: Vec<usize> = (0..parts.len())
        .filter(|&p| parts[p].resident_cost.is_some() && parts[p].rows > 0)
        .collect();
    order.sort_by_key(|&p| (estimate(&parts[p]), p));
    let mut kept = 0u64;
    let mut held = 0usize;
    for (n, &p) in order.iter().enumerate() {
        let next = held.saturating_add(estimate(&parts[p]));
        if next > resident_cap {
            break;
        }
        // On disk if `p` is kept: every partition on disk now and the
        // candidates after `p`; the grace phase loads the largest at once.
        let mut disk: Vec<usize> = (0..parts.len())
            .filter(|&q| parts[q].resident_cost.is_none() && parts[q].rows > 0)
            .map(|q| estimate(&parts[q]))
            .chain(order[n + 1..].iter().map(|&q| estimate(&parts[q])))
            .collect();
        disk.sort_unstable_by(|a, b| b.cmp(a));
        let loads = disk
            .iter()
            .take(drivers)
            .fold(0usize, |s, &b| s.saturating_add(b));
        if next.saturating_add(loads) > together_cap {
            break;
        }
        kept |= 1 << p;
        held = next;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(rows: u64, cost: usize) -> Partition {
        Partition {
            rows,
            resident_cost: Some(cost),
        }
    }

    fn disk(rows: u64) -> Partition {
        Partition {
            rows,
            resident_cost: None,
        }
    }

    #[test]
    fn keeps_what_fits_beside_one_load_per_driver() {
        // 64 partitions of 100 bytes (1 byte a row), 4 on disk; budget
        // 4,000: resident <= 2,000, resident + 2 loads <= 3,000.
        let mut parts: Vec<Partition> = (0..60).map(|_| mem(100, 100)).collect();
        parts.extend((0..4).map(|_| disk(100)));
        let kept = keep(&parts, 4_000, 2);
        assert_eq!(kept.count_ones(), 20);
        assert_eq!(kept >> 60, 0, "a spilled partition is never kept");
    }

    #[test]
    fn the_grace_loads_bound_the_resident_share() {
        // 8 partitions of 1,000 bytes, one on disk, 12 drivers: keeping
        // any leaves 7 x 1,000 of loads, past 3/4 of 8,000.
        let mut parts: Vec<Partition> = (0..7).map(|_| mem(1_000, 1_000)).collect();
        parts.push(disk(1_000));
        assert_eq!(keep(&parts, 8_000, 12), 0);
        // With one driver: resident <= 4,000 and + 1,000 <= 6,000.
        assert_eq!(keep(&parts, 8_000, 1).count_ones(), 4);
    }

    #[test]
    fn the_cheapest_partitions_are_kept_first() {
        let parts = [mem(10, 900), mem(10, 100), disk(10), mem(10, 200)];
        // per row 1,200 / 30 = 40; disk estimate 400; budget 2,000:
        // resident <= 1,000, + one load <= 1,500.
        let kept = keep(&parts, 2_000, 1);
        assert_eq!(kept, 0b1010);
    }

    #[test]
    fn nothing_in_memory_keeps_nothing() {
        assert_eq!(keep(&[disk(5), disk(7)], 1 << 30, 4), 0);
        assert_eq!(keep(&[mem(0, 0), disk(7)], 1 << 30, 4), 0);
        assert_eq!(keep(&[], 1 << 30, 4), 0);
    }
}
