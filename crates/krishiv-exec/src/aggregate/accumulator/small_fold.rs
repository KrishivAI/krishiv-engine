// SPDX-License-Identifier: Apache-2.0
//! The dense decimal fold of a table of few groups (study 31, K1).

use krishiv_kernels::decimal::DecimalNative;

/// The most groups a [`fold_small`] batch can have.
const SMALL_GROUPS: usize = 8;
const _: () = assert!(SMALL_GROUPS.is_power_of_two());

/// The proven dense fold (see `DecFold::sum`) of a table of at most
/// [`SMALL_GROUPS`] groups: the batch adds into stack arrays indexed by
/// `g & 7`, with no bounds test and no per-row state test, and the totals
/// are added to the table at the end. Wrapping adds commute, so the
/// result equals the per-row loop's. `st` (`sum`'s state, no group
/// overflowed on entry) is set to "has a value" for each touched group;
/// `count` (`avg`) gets the rows per group.
fn fold_small<N: DecimalNative>(
    groups: &[u32],
    v: &[N],
    sum: &mut [i128],
    count: Option<&mut [i64]>,
    st: &mut [u8],
) {
    // Two lanes (even and odd rows) break the dependency chain through a
    // group's running total when consecutive rows share a group.
    debug_assert!(
        groups.iter().all(|&g| (g as usize) < sum.len()),
        "a group id past the table"
    );
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
    let acc: [i128; SMALL_GROUPS] = std::array::from_fn(|g| acc[0][g].wrapping_add(acc[1][g]));
    let rows: [i64; SMALL_GROUPS] = std::array::from_fn(|g| rows[0][g] + rows[1][g]);
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

#[cfg(test)]
mod tests {
    use super::*;

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
