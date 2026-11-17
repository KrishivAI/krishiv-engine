// SPDX-License-Identifier: Apache-2.0
//! Key-routed pairs (design P3-A 1d): a `count(DISTINCT v)` whose pairs'
//! radix partitions come from the hash of the group keys alone, counted at
//! the merge also alone (`distinct_plan_routed(.., true)`), against the two
//! aggregations in series: the same rows for one key to 100k keys, NULL keys
//! and values, string keys, a tiny-memory spill and one slot. Without the
//! flag nothing routes.

use super::*;
use crate::aggregate::distinct::{DistinctCount, DistinctCountNode};
use crate::aggregate::{distinct_plan_routed, distinct_plan_with};
use crate::rows::RowLayout;

/// Generated `(k, v, w)` batches: `n` rows over `keys` keys, every 7th key
/// and every 5th value NULL, `v` with up to `values` values.
fn gen_input(n: i64, keys: i64, values: i64, per_batch: usize) -> Vec<Vec<ArrayRef>> {
    let rows: Vec<i64> = (0..n).collect();
    rows.chunks(per_batch)
        .map(|c| {
            let k: Vec<Option<i64>> = c
                .iter()
                .map(|&i| {
                    let k = (i * 7919) % keys.max(1);
                    (k % 7 != 3).then_some(k)
                })
                .collect();
            let v: Vec<Option<i64>> = c
                .iter()
                .map(|&i| ((i % 5) != 0).then_some((i * 31) % values.max(1)))
                .collect();
            let w: Vec<Option<i64>> = c.iter().map(|&i| Some(i % 11)).collect();
            vec![i64s(&k), i64s(&v), i64s(&w)]
        })
        .collect()
}

fn kvw() -> SchemaRef {
    schema(vec![
        ("k", DataType::Int64),
        ("v", DataType::Int64),
        ("w", DataType::Int64),
    ])
}

/// `[sum(w), count(*),] count(DISTINCT v) group by k`.
fn shape(s: &SchemaRef, plain: bool) -> (AggregateSpec, Vec<bool>) {
    let mut aggs = Vec::new();
    if plain {
        aggs.push(agg(&Sum, 2, s));
        aggs.push(AggregateExpr::count_star("c"));
    }
    aggs.push(agg(&Count, 1, s));
    let mut flags = vec![false; aggs.len()];
    flags[aggs.len() - 1] = true;
    (spec(s, vec![0], aggs), flags)
}

/// The rows of the routed plan and of the two aggregations in series.
async fn routed_and_series(
    s: &SchemaRef,
    b: &[Vec<ArrayRef>],
    sp: &AggregateSpec,
    flags: &[bool],
    o: Opts,
) -> (Vec<String>, Vec<String>) {
    let series = || distinct_plan_with(scan(s, b.to_vec()), sp.clone(), flags, false).unwrap();
    let routed =
        || distinct_plan_routed(scan(s, b.to_vec()), sp.clone(), flags, true, true).unwrap();
    let (want, _) = exec_each(series, o).await;
    let (got, _) = exec_each(routed, o).await;
    (got, want)
}

/// Whether the plan ran `DistinctCount`.
async fn counted_at_merge(plan: Arc<dyn ExecNode>, o: Opts) -> bool {
    let (_, m) = exec_with(plan, o, true).await.unwrap();
    m.operators.values().any(|e| e.name == "DistinctCount")
}

#[tokio::test]
async fn routed_pairs_count_as_the_two_aggregations_do_from_one_key_to_many() {
    let s = kvw();
    for (n, keys, values) in [
        (0, 10, 97),
        (500, 1, 97),
        (4_000, 16, 500),
        (20_000, 1_023, 97),
        (20_000, 1_024, 20_000),
        (40_000, 5_000, 300),
        (100_000, 100_000, 1_000),
    ] {
        let b = gen_input(n, keys, values, 1_024);
        for plain in [false, true] {
            let (sp, flags) = shape(&s, plain);
            let o = Opts {
                slots: 4,
                ..Opts::default()
            };
            let (got, want) = routed_and_series(&s, &b, &sp, &flags, o).await;
            assert_eq!(got, want, "n {n}, keys {keys}, plain {plain}");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, ..ProptestConfig::default() })]
    #[test]
    fn routed_pairs_count_as_the_two_aggregations_do(
        n in 0_i64..30_000,
        keys in 1_i64..100_000,
        values in 1_i64..5_000,
        batch in 64_usize..2_048,
        plain in any::<bool>(),
    ) {
        let s = kvw();
        let b = gen_input(n, keys, values, batch);
        let (sp, flags) = shape(&s, plain);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let o = Opts { slots: 4, ..Opts::default() };
        let (got, want) = rt.block_on(routed_and_series(&s, &b, &sp, &flags, o));
        prop_assert_eq!(got, want);
    }
}

/// Two string keys (the first routes) over string values with NULLs.
#[tokio::test]
async fn string_keys_and_values_count_the_same() {
    let s = schema(vec![
        ("k", DataType::Utf8View),
        ("k2", DataType::Utf8View),
        ("v", DataType::Utf8View),
    ]);
    let b: Vec<Vec<ArrayRef>> = (0..8_i64)
        .map(|j| {
            let col = |f: &dyn Fn(i64) -> Option<String>| -> ArrayRef {
                Arc::new(StringViewArray::from_iter(
                    (0..2_500).map(|i| f(j * 2_500 + i)),
                ))
            };
            vec![
                col(&|i| (i % 13 != 0).then(|| format!("a long search phrase {}", i % 3_001))),
                col(&|i| Some(format!("x{}", i % 3))),
                col(&|i| (i % 11 != 0).then(|| format!("user-{}", (i * 17) % 4_099))),
            ]
        })
        .collect();
    let aggs = vec![agg(&Count, 2, &s)];
    let sp = spec(&s, vec![0, 1], aggs);
    let o = Opts {
        slots: 4,
        ..Opts::default()
    };
    let (got, want) = routed_and_series(&s, &b, &sp, &[true], o).await;
    assert_eq!(got, want);
    assert!(got.len() > 3_000, "{}", got.len());
}

/// Many keys under a pool far smaller than their tables: routed, the pairs
/// and the key table spill and the rows are the series plan's.
#[tokio::test]
async fn routed_pairs_in_a_small_pool_spill_and_count_the_same() {
    let s = kvw();
    let b = gen_input(120_000, 120_000, 100_000, 4_096);
    for plain in [false, true] {
        let (sp, flags) = shape(&s, plain);
        let o = Opts {
            limit: 3 << 19,
            ..Opts::default()
        };
        let (_m, scope) = spill_dir();
        let mut spilled = sp.clone();
        spilled.spill = Some(scope);
        let two = exec_with(
            distinct_plan_with(scan(&s, b.clone()), spilled.clone(), &flags, false).unwrap(),
            o,
            true,
        )
        .await
        .unwrap();
        let op = DistinctCount::routed(&spilled, &flags).unwrap();
        let keys = Arc::clone(op.key_table());
        let plan = DistinctCountNode::new(scan(&s, b.clone()), op) as Arc<dyn ExecNode>;
        let at = exec_with(plan, o, true).await.unwrap();
        assert_eq!(rows(&at.0), rows(&two.0), "plain {plain}");
        assert!(
            keys.metrics().spill_runs > 0,
            "plain {plain}: the key table never spilled"
        );
    }
}

/// One slot (and `krishiv.deterministicOrder`, whose parallelism is 1)
/// keeps the two aggregations even when routed, with the same rows.
#[tokio::test]
async fn one_slot_and_deterministic_order_keep_the_two_aggregations() {
    let s = kvw();
    let b = gen_input(20_000, 4_000, 500, 512);
    let (sp, flags) = shape(&s, false);
    for (slots, deterministic) in [(1, false), (4, true)] {
        let o = Opts {
            slots,
            deterministic,
            ..Opts::default()
        };
        let routed =
            || distinct_plan_routed(scan(&s, b.clone()), sp.clone(), &flags, true, true).unwrap();
        assert!(
            !counted_at_merge(routed(), o).await,
            "{slots} slots, deterministic {deterministic}"
        );
        let (got, want) = routed_and_series(&s, &b, &sp, &flags, o).await;
        assert_eq!(got, want, "{slots} slots, deterministic {deterministic}");
    }
}

/// Without the flag a lone `count(DISTINCT)` keeps the two aggregations and
/// a counted one beside plain aggregates has unrouted pairs; with it the
/// lone count is counted at the merge over routed pairs.
#[tokio::test]
async fn only_the_flag_routes() {
    let s = kvw();
    let b = gen_input(5_000, 3, 97, 256);
    let o = Opts {
        slots: 4,
        ..Opts::default()
    };
    for (plain, routed, counted) in [
        (false, false, false),
        (false, true, true),
        (true, false, true),
        (true, true, true),
    ] {
        let (sp, flags) = shape(&s, plain);
        let plan = distinct_plan_routed(scan(&s, b.clone()), sp, &flags, true, routed).unwrap();
        assert_eq!(
            counted_at_merge(plan, o).await,
            counted,
            "plain {plain}, routed {routed}"
        );
    }
    let (sp, flags) = shape(&s, true);
    assert_eq!(DistinctCount::new(&sp, &flags).unwrap().pairs().route(), 0);
    assert_eq!(
        DistinctCount::routed(&sp, &flags).unwrap().pairs().route(),
        1
    );
    // Counting at the merge switched off routes nothing either.
    let (sp, flags) = shape(&s, false);
    let plan = distinct_plan_routed(scan(&s, b.clone()), sp, &flags, false, true).unwrap();
    assert!(!counted_at_merge(plan, o).await);
}

/// A routed hash: its top bits are those of the prefix columns' own hash
/// (one prefix value, one radix partition), its other bits those of the
/// unrouted hash of the whole key; without routing the hash is unchanged.
#[test]
fn a_routed_hash_takes_its_top_bits_from_the_prefix_alone() {
    let types = [DataType::Int64, DataType::Utf8View];
    let k = i64s(&[Some(1), Some(1), None, Some(7), Some(1)]);
    let v: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some("a"),
        Some("a much longer string value"),
        Some("a"),
        None,
        Some("b"),
    ]));
    let cols = [Arc::clone(&k), Arc::clone(&v)];
    let plain = RowLayout::new(&types, &[true, true]).unwrap();
    let routed = plain.clone().with_route(1);
    let prefix = RowLayout::new(&types[..1], &[true]).unwrap();
    let (h, r, p) = (
        plain.encode(&cols).unwrap().hashes().to_vec(),
        routed.encode(&cols).unwrap().hashes().to_vec(),
        prefix.encode(&[k]).unwrap().hashes().to_vec(),
    );
    let top = |x: u64| x >> (64 - crate::rows::ROUTE_BITS);
    let low = |x: u64| x << crate::rows::ROUTE_BITS;
    for i in 0..h.len() {
        assert_eq!(top(r[i]), top(p[i]), "row {i}");
        assert_eq!(low(r[i]), low(h[i]), "row {i}");
    }
    assert_eq!(top(r[0]), top(r[1]));
    assert_eq!(top(r[0]), top(r[4]));
    // A prefix of every column, or none, routes nothing.
    for off in [plain.clone().with_route(0), plain.with_route(2)] {
        assert_eq!(off.encode(&cols).unwrap().hashes(), h.as_slice());
    }
}
