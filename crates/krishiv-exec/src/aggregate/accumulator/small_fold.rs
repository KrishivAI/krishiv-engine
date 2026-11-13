// SPDX-License-Identifier: Apache-2.0
//! The dense decimal fold of a table of few groups (study 31, K1).

use krishiv_kernels::decimal::DecimalNative;

/// The most groups a [`fold_small`] batch can have.
pub(super) const SMALL_GROUPS: usize = 8;
const _: () = assert!(SMALL_GROUPS.is_power_of_two());

/// Per-group totals of a batch over a table of at most [`SMALL_GROUPS`]
/// groups: the wrapping sum of `v` and the row count of each group, in
/// stack arrays indexed by `g & 7` (no bounds test, no per-row state
/// test). Wrapping adds commute, so the totals equal the per-row loop's.
pub(super) fn small_totals<N: DecimalNative>(
    groups: &[u32],
    v: &[N],
) -> ([i128; SMALL_GROUPS], [i64; SMALL_GROUPS]) {
    // Two lanes (even and odd rows) break the dependency chain through a
    // group's running total when consecutive rows share a group.
    let mut acc = [[0_i128; SMALL_GROUPS]; 2];
    let mut rows = [[0_i64; SMALL_GROUPS]; 2];
    let (gp, gr) = groups.as_chunks::<2>();
    let (vp, vr) = v.as_chunks::<2>();
    for (g2, v2) in gp.iter().zip(vp) {
        for k in 0..2 {
            let g = g2[k] as usize & (SMALL_GROUPS - 1);
            acc[k][g] = acc[k][g].wrapping_add(v2[k].wide());
            rows[k][g] += 1;
        }
    }
    for (&g, x) in gr.iter().zip(vr) {
        let g = g as usize & (SMALL_GROUPS - 1);
        acc[0][g] = acc[0][g].wrapping_add(x.wide());
        rows[0][g] += 1;
    }
    (
        std::array::from_fn(|g| acc[0][g].wrapping_add(acc[1][g])),
        std::array::from_fn(|g| rows[0][g] + rows[1][g]),
    )
}

/// The proven dense fold (see `DecFold::sum`) of a table of at most
/// [`SMALL_GROUPS`] groups, through [`small_totals`]; the totals are added
/// to the table at the end. `st` (`sum`'s state, no group overflowed on
/// entry) is set to "has a value" for each touched group; `count` (`avg`)
/// gets the rows per group.
fn fold_small<N: DecimalNative>(
    groups: &[u32],
    v: &[N],
    sum: &mut [i128],
    count: Option<&mut [i64]>,
    st: &mut [u8],
) {
    debug_assert!(
        groups.iter().all(|&g| (g as usize) < sum.len()),
        "a group id past the table"
    );
    let (acc, rows) = small_totals(groups, v);
    for (g, s) in sum.iter_mut().enumerate() {
        *s = s.wrapping_add(acc[g]);
    }
    match count {
        Some(c) => {
            for (g, c) in c.iter_mut().enumerate() {
                *c = c.wrapping_add(rows[g]);
            }
        }
        None => {
            for (g, s) in st.iter_mut().enumerate() {
                if rows[g] > 0 {
                    *s = 1;
                }
            }
        }
    }
}

/// The proven dense `sum` fold: raw rows, all valid, unit weights, no
/// overflow possible in this batch (an overflowed group stays NULL).
pub(super) fn dense_sum<N: DecimalNative>(
    groups: &[u32],
    v: &[N],
    sum: &mut [i128],
    st: &mut [u8],
) {
    if sum.len() <= SMALL_GROUPS && st.iter().all(|&s| s != 2) {
        return fold_small(groups, v, sum, None, st);
    }
    for (&g, x) in groups.iter().zip(v) {
        let g = g as usize;
        if st[g] != 2 {
            sum[g] = sum[g].wrapping_add(x.wide());
            st[g] = 1;
        }
    }
}

/// The proven dense `avg` fold, as [`dense_sum`].
pub(super) fn dense_avg<N: DecimalNative>(
    groups: &[u32],
    v: &[N],
    sum: &mut [i128],
    count: &mut [i64],
) {
    if sum.len() <= SMALL_GROUPS {
        return fold_small(groups, v, sum, Some(count), &mut []);
    }
    for (&g, x) in groups.iter().zip(v) {
        let g = g as usize;
        sum[g] = sum[g].wrapping_add(x.wide());
        count[g] = count[g].wrapping_add(1);
    }
}

/// The dense `count` fold: raw rows, every row counted (no NULL argument,
/// no `FILTER` mask, no running window), each with weight `k`. One group
/// (a global aggregate) adds the batch length; at most [`SMALL_GROUPS`]
/// groups count rows per group in stack arrays; more take the row loop.
/// Wrapping adds commute, so every form equals the per-row loop.
pub(super) fn dense_count(groups: &[u32], k: i64, count: &mut [i64]) {
    debug_assert!(
        groups.iter().all(|&g| (g as usize) < count.len()),
        "a group id past the table"
    );
    if let [c] = count {
        #[expect(clippy::cast_possible_wrap, reason = "wrapping, as the per-row adds")]
        let n = groups.len() as i64;
        *c = c.wrapping_add(k.wrapping_mul(n));
        return;
    }
    if count.len() <= SMALL_GROUPS {
        let mut rows = [[0_i64; SMALL_GROUPS]; 2];
        let (gp, gr) = groups.as_chunks::<2>();
        for g2 in gp {
            for k in 0..2 {
                rows[k][g2[k] as usize & (SMALL_GROUPS - 1)] += 1;
            }
        }
        for &g in gr {
            rows[0][g as usize & (SMALL_GROUPS - 1)] += 1;
        }
        for (g, c) in count.iter_mut().enumerate() {
            *c = c.wrapping_add(k.wrapping_mul(rows[0][g] + rows[1][g]));
        }
        return;
    }
    for &g in groups {
        let c = &mut count[g as usize];
        *c = c.wrapping_add(k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_count_matches_the_per_row_loop() {
        for n_groups in [1_usize, 2, 5, SMALL_GROUPS, SMALL_GROUPS + 1, 40] {
            for len in [0_usize, 1, 2, 3, 64, 101] {
                for k in [1_i64, -1, 3, i64::MAX] {
                    let groups: Vec<u32> = (0..len)
                        .map(|i| u32::try_from((i * 7 + i / 5) % n_groups).unwrap_or(0))
                        .collect();
                    let mut want = vec![i64::MAX - 2; n_groups];
                    for &g in &groups {
                        want[g as usize] = want[g as usize].wrapping_add(k);
                    }
                    let mut got = vec![i64::MAX - 2; n_groups];
                    dense_count(&groups, k, &mut got);
                    assert_eq!(want, got, "{n_groups} groups, {len} rows, weight {k}");
                }
            }
        }
    }

    /// The per-row loop `fold_small` replaces.
    fn reference(groups: &[u32], v: &[i64], sum: &mut [i128], count: &mut [i64], st: &mut [u8]) {
        for (&g, x) in groups.iter().zip(v) {
            let g = g as usize;
            sum[g] = sum[g].wrapping_add(i128::from(*x));
            count[g] += 1;
            st[g] = 1;
        }
    }

    #[test]
    fn matches_the_per_row_loop_for_every_length_and_group_count() {
        for n_groups in 1..=SMALL_GROUPS {
            for len in [0_usize, 1, 2, 3, 7, 64, 101] {
                let groups: Vec<u32> = (0..len)
                    .map(|i| u32::try_from((i * 5 + i / 3) % n_groups).unwrap_or(0))
                    .collect();
                let v: Vec<i64> = (0..len)
                    .map(|i| (i64::try_from(i).unwrap_or(0) - 40) * 1_000_003)
                    .collect();
                let (mut s1, mut c1, mut t1) = (
                    vec![7_i128; n_groups],
                    vec![2_i64; n_groups],
                    vec![0_u8; n_groups],
                );
                reference(&groups, &v, &mut s1, &mut c1, &mut t1);
                let (mut s2, mut c2) = (vec![7_i128; n_groups], vec![2_i64; n_groups]);
                fold_small(&groups, &v, &mut s2, Some(&mut c2), &mut []);
                assert_eq!(
                    (s1.clone(), c1),
                    (s2, c2),
                    "avg: {n_groups} groups, {len} rows"
                );
                let (mut s3, mut t3) = (vec![7_i128; n_groups], vec![0_u8; n_groups]);
                fold_small(&groups, &v, &mut s3, None, &mut t3);
                assert_eq!((s1, t1), (s3, t3), "sum: {n_groups} groups, {len} rows");
            }
        }
    }

    #[test]
    fn an_overflowed_group_stays_overflowed_and_big_tables_use_the_row_loop() {
        let mut sum = [5_i128, 0];
        let mut st = [1_u8, 2];
        dense_sum(&[0, 1, 0], &[1_i64, 9, 2], &mut sum, &mut st);
        assert_eq!((sum, st), ([8, 0], [1, 2]));
        let groups: Vec<u32> = (0..20).collect();
        let (mut s, mut c) = (vec![0_i128; 20], vec![0_i64; 20]);
        dense_avg(&groups, &[3_i64; 20], &mut s, &mut c);
        assert!(s.iter().all(|&x| x == 3) && c.iter().all(|&x| x == 1));
    }

    #[test]
    fn wraps_like_the_per_row_loop() {
        let v = [i64::MAX, i64::MAX, i64::MIN];
        let mut a = [i128::MAX - 5];
        fold_small(&[0, 0, 0], &v, &mut a, None, &mut [0]);
        let want = (i128::MAX - 5)
            .wrapping_add(i128::from(i64::MAX) * 2)
            .wrapping_add(i128::from(i64::MIN));
        assert_eq!(a[0], want);
    }
}
