// SPDX-License-Identifier: Apache-2.0
//! Range coalescing: few large requests instead of many small ones.

use std::ops::Range;

/// Gap bytes a coalesced request may always read, whatever its size: below
/// this, saving a request is worth more than the bytes read for nothing.
pub(super) const SMALL_WASTE: u64 = 64 << 10;

/// Requests covering `ranges`: ranges closer than `gap` bytes (or
/// overlapping) are merged while the merged request stays within
/// `max_request` bytes (a single larger range is still one request) and
/// reads at most `max(span / 16, SMALL_WASTE)` bytes outside the ranges.
/// Returns the requests in file order and, per input range, the request
/// index that holds it.
///
/// The waste bound is a memory bound: the owners of the ranges reserve
/// their ranges' bytes, not the gaps, and a request's buffer holds the gaps
/// until its ranges are copied out or released (see `IoContext::get_ranges`).
/// Merging column chunks across unread columns of 1 MiB each made TPC-H
/// Q21's scans hold over half a GiB that no reservation covered.
#[must_use]
pub fn coalesce(
    ranges: &[Range<u64>],
    gap: u64,
    max_request: u64,
) -> (Vec<Range<u64>>, Vec<usize>) {
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_by_key(|&i| (ranges[i].start, ranges[i].end));
    let mut merged: Vec<Range<u64>> = Vec::new();
    // Per merged request: the bytes inside its ranges.
    let mut covered: Vec<u64> = Vec::new();
    let mut owner = vec![0; ranges.len()];
    for i in order {
        let r = &ranges[i];
        if let (Some(last), Some(inside)) = (merged.last_mut(), covered.last_mut()) {
            // Fully inside the previous request (a contained range).
            if r.end <= last.end {
                owner[i] = merged.len() - 1;
                continue;
            }
            let end = r.end;
            let span = end - last.start;
            let added = end - r.start.max(last.end);
            let wanted = *inside + added;
            if r.start <= last.end.saturating_add(gap)
                && span <= max_request.max(r.end - r.start)
                && span - wanted <= (span / 16).max(SMALL_WASTE)
            {
                last.end = end;
                *inside = wanted;
                owner[i] = merged.len() - 1;
                continue;
            }
        }
        merged.push(r.clone());
        covered.push(r.end - r.start);
        owner[i] = merged.len() - 1;
    }
    (merged, owner)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::coalesce;

    #[test]
    #[expect(
        clippy::single_range_in_vec_init,
        reason = "one-range requests are the cases under test"
    )]
    fn merges_near_ranges_and_respects_the_request_cap() {
        let (m, o) = coalesce(&[10..20, 0..5, 25..30, 100..110], 5, 1000);
        assert_eq!(m, vec![0..30, 100..110]);
        assert_eq!(o, vec![0, 0, 0, 1]);
        let (m, _) = coalesce(&[0..10, 12..20, 22..40], 5, 20);
        assert_eq!(m, vec![0..20, 22..40]);
        // A single range larger than the cap is still one request.
        let (m, _) = coalesce(&[0..100], 0, 10);
        assert_eq!(m, vec![0..100]);
        let (m, o) = coalesce(&[0..100, 10..20], 0, 10);
        assert_eq!((m, o), (vec![0..100], vec![0, 0]));
        assert_eq!(coalesce(&[], 1, 1), (vec![], vec![]));
    }

    #[test]
    fn a_merge_never_reads_more_than_a_sixteenth_in_gaps() {
        const MIB: u64 = 1 << 20;
        // Two 2 MiB column chunks around a 1 MiB unread one: one third of
        // the request would be gap. Not merged, whatever the gap setting.
        let (m, _) = coalesce(&[0..2 * MIB, 3 * MIB..5 * MIB], MIB, 16 * MIB);
        assert_eq!(m, vec![0..2 * MIB, 3 * MIB..5 * MIB]);
        // A 64 KiB gap between two 2 MiB chunks is within a sixteenth.
        let (m, _) = coalesce(&[0..2 * MIB, 2 * MIB + 65_536..4 * MIB], MIB, 16 * MIB);
        assert_eq!(m, vec![0..4 * MIB]);
        // Small requests may always merge across a gap of up to 64 KiB.
        let (m, _) = coalesce(&[0..100, 50_000..50_100], MIB, 16 * MIB);
        assert_eq!(m, vec![0..50_100]);
        let (m, _) = coalesce(&[0..100, 70_000..70_100], MIB, 16 * MIB);
        assert_eq!(m.len(), 2);
        // Overlapping ranges count their bytes once.
        let (m, _) = coalesce(
            &[0..MIB, MIB / 2..2 * MIB, 2 * MIB + 1000..3 * MIB],
            MIB,
            16 * MIB,
        );
        assert_eq!(m, vec![0..3 * MIB]);
    }

    proptest! {
        #[test]
        fn every_range_is_inside_its_request(
            ranges in prop::collection::vec((0u64..10_000, 1u64..500), 0..40),
            gap in 0u64..300,
            cap in 1u64..3_000,
        ) {
            let ranges: Vec<_> = ranges.into_iter().map(|(s, l)| s..s + l).collect();
            let (merged, owner) = coalesce(&ranges, gap, cap);
            for (r, &i) in ranges.iter().zip(&owner) {
                prop_assert!(merged[i].start <= r.start && r.end <= merged[i].end);
            }
            for w in merged.windows(2) {
                prop_assert!(w[0].start <= w[1].start);
            }
            prop_assert!(merged.len() <= ranges.len());
            // The gap bytes of every request stay within the waste bound.
            for (k, req) in merged.iter().enumerate() {
                let mut mine: Vec<_> = ranges.iter().zip(&owner).filter(|(_, o)| **o == k).map(|(r, _)| r.clone()).collect();
                mine.sort_by_key(|r| r.start);
                let (mut inside, mut end) = (0, req.start);
                for r in mine {
                    inside += r.end.saturating_sub(r.start.max(end));
                    end = end.max(r.end);
                }
                let span = req.end - req.start;
                prop_assert!(span - inside <= (span / 16).max(super::SMALL_WASTE));
            }
        }
    }
}
