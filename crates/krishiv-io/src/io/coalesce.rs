// SPDX-License-Identifier: Apache-2.0
//! Range coalescing: few large requests instead of many small ones.

use std::ops::Range;

/// Requests covering `ranges`: ranges closer than `gap` bytes (or
/// overlapping) are merged while the merged request stays within
/// `max_request` bytes (a single larger range is still one request).
/// Returns the requests in file order and, per input range, the request
/// index that holds it.
#[must_use]
pub fn coalesce(
    ranges: &[Range<u64>],
    gap: u64,
    max_request: u64,
) -> (Vec<Range<u64>>, Vec<usize>) {
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_by_key(|&i| (ranges[i].start, ranges[i].end));
    let mut merged: Vec<Range<u64>> = Vec::new();
    let mut owner = vec![0; ranges.len()];
    for i in order {
        let r = &ranges[i];
        if let Some(last) = merged.last_mut() {
            let end = last.end.max(r.end);
            if r.start <= last.end.saturating_add(gap)
                && end - last.start <= max_request.max(r.end - r.start)
            {
                last.end = end;
                owner[i] = merged.len() - 1;
                continue;
            }
            // Fully inside the previous request (a contained range).
            if r.end <= last.end {
                owner[i] = merged.len() - 1;
                continue;
            }
        }
        merged.push(r.clone());
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
        }
    }
}
