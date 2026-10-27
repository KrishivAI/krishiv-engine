// SPDX-License-Identifier: Apache-2.0
//! Range coalescing: few large requests instead of many small ones.

use std::ops::Range;

/// How many bytes outside its ranges a coalesced request may read: at most
/// `max(span / divisor, floor)`. The callers reserve each request's whole
/// span while it is read (see `IoContext::request_bytes`), so the bound is
/// a cost bound (bytes read and briefly held for nothing), not an
/// accounting gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Waste {
    /// Gap bytes any request may read (`krishiv.io.coalesceWasteBytes`).
    pub floor: u64,
    /// A request may read up to `span / divisor` gap bytes
    /// (`krishiv.io.coalesceWasteDivisor`; at least 1).
    pub divisor: u64,
}

impl Waste {
    /// Local files: merging saves little (the kernel reads ahead), so
    /// requests read at most 1/16 or 64 KiB in gaps.
    pub const LOCAL: Self = Self {
        floor: 64 << 10,
        divisor: 16,
    };
    /// Object stores and HTTP: a request costs latency (and money), so
    /// ranges up to 1 MiB apart are merged whatever the request's size,
    /// as `object_store`'s own range coalescing does.
    pub const REMOTE: Self = Self {
        floor: 1 << 20,
        divisor: 16,
    };

    /// Whether a request of `span` bytes with `inside` bytes in its ranges
    /// reads few enough gap bytes.
    #[must_use]
    pub fn allows(self, span: u64, inside: u64) -> bool {
        span.saturating_sub(inside) <= (span / self.divisor.max(1)).max(self.floor)
    }
}

/// Coalesced requests (see [`coalesce_with`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Coalesced {
    /// The requests, in file order.
    pub requests: Vec<Range<u64>>,
    /// Per input range, the index of the request that holds it.
    pub owner: Vec<usize>,
    /// Per request, the bytes inside its ranges (overlaps counted once).
    pub covered: Vec<u64>,
}

impl Coalesced {
    /// Bytes the requests read: what to reserve while they are in flight.
    #[must_use]
    pub fn span_bytes(&self) -> u64 {
        self.requests.iter().map(|r| r.end - r.start).sum()
    }
}

/// [`coalesce_with`] for local files ([`Waste::LOCAL`]), returning the
/// requests and each range's request.
#[must_use]
pub fn coalesce(
    ranges: &[Range<u64>],
    gap: u64,
    max_request: u64,
) -> (Vec<Range<u64>>, Vec<usize>) {
    let c = coalesce_with(ranges, gap, max_request, Waste::LOCAL);
    (c.requests, c.owner)
}

/// Requests covering `ranges`: ranges closer than `gap` bytes (or
/// overlapping) are merged while the merged request stays within
/// `max_request` bytes (a single larger range is still one request) and
/// `waste` allows its gap bytes.
#[must_use]
pub fn coalesce_with(ranges: &[Range<u64>], gap: u64, max_request: u64, waste: Waste) -> Coalesced {
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_by_key(|&i| (ranges[i].start, ranges[i].end));
    let mut merged: Vec<Range<u64>> = Vec::new();
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
            let span = r.end - last.start;
            let wanted = *inside + (r.end - r.start.max(last.end));
            if r.start <= last.end.saturating_add(gap)
                && span <= max_request.max(r.end - r.start)
                && waste.allows(span, wanted)
            {
                last.end = r.end;
                *inside = wanted;
                owner[i] = merged.len() - 1;
                continue;
            }
        }
        merged.push(r.clone());
        covered.push(r.end - r.start);
        owner[i] = merged.len() - 1;
    }
    Coalesced {
        requests: merged,
        owner,
        covered,
    }
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

    /// Gap bytes of every request, from the ranges it holds.
    fn inside_bytes(ranges: &[std::ops::Range<u64>], owner: &[usize], k: usize, start: u64) -> u64 {
        let mut mine: Vec<_> = ranges
            .iter()
            .zip(owner)
            .filter(|(_, o)| **o == k)
            .map(|(r, _)| r.clone())
            .collect();
        mine.sort_by_key(|r| r.start);
        let (mut inside, mut end) = (0, start);
        for r in mine {
            inside += r.end.saturating_sub(r.start.max(end));
            end = end.max(r.end);
        }
        inside
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
                let inside = inside_bytes(&ranges, &owner, k, req.start);
                prop_assert!(super::Waste::LOCAL.allows(req.end - req.start, inside));
            }
        }

        /// Ranges of up to 2 MiB with gaps of up to 4 MiB and a request cap
        /// of up to 64 MiB: the gaps cross the 64 KiB floor and the spans
        /// the 1 MiB where `span / 16` takes over, so a merge the waste
        /// bound must refuse is generated, and merging stays reachable.
        #[test]
        fn large_gaps_never_exceed_the_waste_bound(
            ranges in prop::collection::vec((0u64..(64 << 20), 1u64..(2 << 20)), 1..30),
            gap in 0u64..(4 << 20),
            cap in (1u64 << 20)..(64 << 20),
        ) {
            let ranges: Vec<_> = ranges.into_iter().map(|(s, l)| s..s + l).collect();
            let (merged, owner) = coalesce(&ranges, gap, cap);
            for (k, req) in merged.iter().enumerate() {
                let inside = inside_bytes(&ranges, &owner, k, req.start);
                let span = req.end - req.start;
                // A request that is one range wide holds no gap by itself.
                prop_assert!(
                    super::Waste::LOCAL.allows(span, inside),
                    "span {span}, inside {inside}"
                );
            }
        }
    }
}
