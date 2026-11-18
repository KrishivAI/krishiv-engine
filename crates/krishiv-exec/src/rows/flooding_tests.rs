// SPDX-License-Identifier: Apache-2.0
//! Hash flooding (P2-T175; design P2-T171 §5.1, P2-T157 Appendix C).
//!
//! The stored row hash `h` is seedless (see the layout's module docs), so
//! anyone can craft keys whose `h` agree in chosen bits. The salted directory places
//! rows by the seeded `g = fin(h ^ seed)`, so crafted partial collisions
//! scatter like ordinary keys. Full 64-bit collisions of `h` (feasible only
//! by attacking the hash) stay correct, because keys are compared, and lengthen
//! probe runs only by the number of colliding keys.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::DataType;

use super::*;

/// Gathered build rows with their (possibly forged) stored hashes.
struct Src {
    rows: GatheredRows,
    hashes: Vec<u64>,
    stride: usize,
}

impl RowSource for Src {
    fn row(&self, id: u32) -> (&[u8], &[u8]) {
        let at = id as usize * self.stride;
        (
            self.rows
                .fixed
                .get(at..at + self.stride)
                .unwrap_or_default(),
            &[],
        )
    }

    fn hash(&self, id: u32) -> u64 {
        self.hashes[id as usize]
    }
}

fn layout() -> RowLayout {
    RowLayout::for_join(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"))
}

/// `keys` encoded, with the hashes of the rows `forged` replaced by `h`.
fn encode(layout: &RowLayout, keys: &[i64], forged: &[usize], h: u64) -> EncodedKeys {
    let a: ArrayRef = Arc::new(Int64Array::from(keys.to_vec()));
    let mut enc = layout.encode(&[a]).unwrap_or_else(|e| panic!("{e}"));
    for &i in forged {
        enc.hashes_mut()[i] = h;
    }
    enc
}

/// A salted directory over `keys` (see [`encode`]) with `seed`.
fn salted(
    layout: &RowLayout,
    keys: &[i64],
    forged: &[usize],
    h: u64,
    seed: u64,
) -> (JoinIndex, Src) {
    let enc = encode(layout, keys, forged, h);
    let all: Vec<usize> = (0..keys.len()).collect();
    let src = Src {
        rows: enc.gather(layout, &all).unwrap_or_else(|e| panic!("{e}")),
        hashes: enc.hashes().to_vec(),
        stride: layout.stride(),
    };
    let mut next = vec![NO_MATCH; keys.len()];
    let index = JoinIndex::build(
        layout,
        &src,
        keys.len(),
        seed,
        IndexChoice::Salted,
        &mut next,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(!index.is_flat());
    (index, src)
}

fn run(index: &JoinIndex, h: u64) -> usize {
    index
        .probe_run(h)
        .unwrap_or_else(|| panic!("a salted index"))
}

/// Looks every row of `probe` up; asserts the answers.
fn check_finds(
    layout: &RowLayout,
    index: &JoinIndex,
    src: &Src,
    probe: &EncodedKeys,
    expected: &[u32],
) {
    let which: Vec<u32> = (0..u32::try_from(probe.len()).unwrap_or(0)).collect();
    let mut found = Vec::new();
    index.find(layout, probe, &which, src, &mut found);
    assert_eq!(found, expected);
}

/// Keys whose `h` share their low 12 bits: a directory slotted by `h`
/// itself (4096 slots for 2048 keys) would hold them in one probe run of
/// 2048. Slotted by the seeded `g`, their longest run is that of
/// ordinary keys, for every seed.
#[test]
fn crafted_partial_collisions_scatter() {
    const N: usize = 2048;
    const MASK: u64 = 0xFFF;
    let layout = layout();
    let hashes = |keys: &[i64]| encode(&layout, keys, &[], 0).hashes().to_vec();
    // Search key blocks with the layout's own hash, as an attacker would.
    let mut crafted = Vec::with_capacity(N);
    for block in (0i64..).step_by(1 << 20) {
        let keys: Vec<i64> = (block..block + (1 << 20)).collect();
        let h = hashes(&keys);
        crafted.extend(
            keys.iter()
                .zip(&h)
                .filter(|&(_, h)| h & MASK == 0x5A5)
                .map(|(&k, _)| k),
        );
        if crafted.len() >= N {
            break;
        }
    }
    crafted.truncate(N);
    let ordinary: Vec<i64> = (0..i64::try_from(N).unwrap_or(0)).collect();
    let crafted_h = hashes(&crafted);
    assert!(crafted_h.iter().all(|h| h & MASK == 0x5A5));
    assert_eq!(
        salted_bytes(N, DirectoryLoad::Compact),
        4096 * 8,
        "the mask is the directory's"
    );
    for seed in [1, 0xDEAD_BEEF, 0x0123_4567_89AB_CDEF] {
        let longest = |keys: &[i64], h: &[u64]| {
            let (index, _) = salted(&layout, keys, &[], 0, seed);
            h.iter().map(|&h| run(&index, h)).max().unwrap_or(0)
        };
        let attacked = longest(&crafted, &crafted_h);
        let baseline = longest(&ordinary, &hashes(&ordinary));
        assert!(
            attacked <= 2 * baseline.max(16),
            "seed {seed:#x}: crafted run {attacked}, ordinary {baseline}"
        );
        let (index, src) = salted(&layout, &crafted, &[], 0, seed);
        let ids: Vec<u32> = (0..u32::try_from(N).unwrap_or(0)).collect();
        check_finds(
            &layout,
            &index,
            &src,
            &encode(&layout, &crafted, &[], 0),
            &ids,
        );
    }
}

/// `m` distinct keys forged to one full 64-bit `h`, among ordinary keys:
/// every key is still found, absent keys with the same `h` are not, the
/// colliding keys' probe run grows linearly in `m` (at this test's load,
/// about 1/2, the cluster takes in about as many neighbours: at most `2m`
/// plus an ordinary run; a directory near its 4/5 bound takes in more, still
/// linear in `m`, which this test does not measure); only the ordinary keys
/// next to the cluster get longer runs, so their total grows by `O(m²)`,
/// whatever the table size.
#[test]
fn full_collisions_stay_correct_and_bounded() {
    const BASE: usize = 3840;
    const H: u64 = 0x5EED_0000_C0FF_EE00;
    let base: Vec<i64> = (0..i64::try_from(BASE).unwrap_or(0)).collect();
    let layout = layout();
    let base_h = encode(&layout, &base, &[], 0).hashes().to_vec();
    let runs = |index: &JoinIndex| base_h.iter().map(|&h| run(index, h)).collect::<Vec<_>>();
    let total = |r: &[usize]| r.iter().sum::<usize>();
    for seed in [7, 0xFEED_FACE] {
        let (clean, _) = salted(&layout, &base, &[], 0, seed);
        let before = runs(&clean);
        let longest_before = before.iter().copied().max().unwrap_or(0);
        for m in [64, 256] {
            let mut keys = base.clone();
            keys.extend((0..i64::try_from(m).unwrap_or(0)).map(|i| 1_000_000 + i));
            let forged: Vec<usize> = (BASE..BASE + m).collect();
            // Same directory size with and without the colliding keys.
            assert_eq!(
                salted_bytes(BASE, DirectoryLoad::Compact),
                salted_bytes(BASE + m, DirectoryLoad::Compact)
            );
            let (index, src) = salted(&layout, &keys, &forged, H, seed);

            let collided = run(&index, H);
            assert!(collided >= m, "the forged keys share one run: {collided}");
            assert!(
                collided <= 2 * m + longest_before,
                "seed {seed:#x}: run {collided} for {m} collisions (ordinary {longest_before})"
            );
            let after = runs(&index);
            let longest_after = after.iter().copied().max().unwrap_or(0);
            assert!(longest_after <= 2 * m + longest_before);
            assert!(
                total(&after).saturating_sub(total(&before)) <= 4 * m * m,
                "seed {seed:#x}, m = {m}: ordinary runs grew from {} to {}",
                total(&before),
                total(&after)
            );

            // Every stored key is found; keys forged to `H` but absent are
            // not.
            let ids: Vec<u32> = (0..u32::try_from(BASE + m).unwrap_or(0)).collect();
            check_finds(
                &layout,
                &index,
                &src,
                &encode(&layout, &keys, &forged, H),
                &ids,
            );
            let absent: Vec<i64> = (0..64).map(|i| 2_000_000 + i).collect();
            let all: Vec<usize> = (0..absent.len()).collect();
            check_finds(
                &layout,
                &index,
                &src,
                &encode(&layout, &absent, &all, H),
                &[NO_MATCH; 64],
            );
        }
    }
}

/// The probe-run measure is the salted directory's: a flat index (direct
/// slots, no runs to aim at) has none.
#[test]
fn a_flat_index_has_no_probe_runs() {
    let layout = layout();
    let keys: Vec<i64> = (0..64).collect();
    let enc = encode(&layout, &keys, &[], 0);
    let all: Vec<usize> = (0..keys.len()).collect();
    let src = Src {
        rows: enc.gather(&layout, &all).unwrap_or_else(|e| panic!("{e}")),
        hashes: enc.hashes().to_vec(),
        stride: layout.stride(),
    };
    let mut next = vec![NO_MATCH; keys.len()];
    let index = JoinIndex::build(&layout, &src, keys.len(), 7, IndexChoice::Flat, &mut next)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(index.is_flat());
    assert_eq!(index.probe_run(enc.hashes()[0]), None);
}

/// The directory is sized for its keys at load ≤ 4/5 and never grows: at
/// key counts where 5/4 of them lands exactly on a power of two and they are
/// not a multiple of 4 (1,639: 1,639 + 409 = 2,048), rounding 5/4 down left
/// the last key past the load bound, and the directory doubled with no
/// reservation behind it (review of S5's 5/4 sizing).
#[test]
fn the_directory_holds_its_keys_without_growing() {
    let layout = layout();
    for n in [1_639_usize, 1_638, 1_640, 3_277, 6_554] {
        let keys: Vec<i64> = (0..i64::try_from(n).unwrap_or(0)).collect();
        let (index, _) = salted(&layout, &keys, &[], 0, 7);
        assert_eq!(index.len(), n);
        assert!(
            5 * n <= 4 * index.buckets(),
            "{n} keys in {}",
            index.buckets()
        );
        assert_eq!(
            index.buckets(),
            salted_bytes(n, DirectoryLoad::Compact) / 8,
            "{n} keys grew the directory"
        );
    }
}

/// Slots of each load, from the rules they implement: compact is the next
/// power of two of 5/4 of the keys (S5), roomy of twice them (the sizing
/// before S5), both at least 1,024; at SF1000's 145.8 M `orders` keys the
/// two are 2^28 and 2^29 slots (the q03 regression, perf-2).
#[test]
fn compact_and_roomy_directories_are_sized_by_their_load() {
    let pow2 = |n: usize| n.next_power_of_two().max(1 << 10) * 8;
    for n in [
        0_usize,
        1,
        7,
        819,
        820,
        1_024,
        1_639,
        2_047,
        2_048,
        2_049,
        4_096,
        100_000,
        1 << 20,
    ] {
        let compact = salted_bytes(n, DirectoryLoad::Compact);
        let roomy = salted_bytes(n, DirectoryLoad::Roomy);
        assert_eq!(compact, pow2(n + n.div_ceil(4)), "{n} compact");
        assert_eq!(roomy, pow2(2 * n), "{n} roomy");
        assert!(roomy >= compact, "{n}");
        assert!(roomy <= 2 * compact, "{n}");
    }
    assert_eq!(salted_bytes(145_800_000, DirectoryLoad::Compact), 8 << 28);
    assert_eq!(salted_bytes(145_800_000, DirectoryLoad::Roomy), 8 << 29);
    for load in [DirectoryLoad::Compact, DirectoryLoad::Roomy] {
        assert_eq!(salted_bytes(usize::MAX, load), usize::MAX, "{load:?}");
        // The flat index does not depend on the load.
        assert_eq!(
            IndexChoice::Adaptive.bytes(100, load),
            flat_bytes(100),
            "{load:?}"
        );
        assert_eq!(
            IndexChoice::Salted.bytes(100, load),
            salted_bytes(100, load)
        );
    }
}

/// Built for its keys, a directory of either load holds them at its bound
/// and never grows past what [`IndexChoice::bytes`] reserved, at the key
/// counts where its rule lands on a power of two; sized from a low estimate
/// it grows, and ends no larger than the directory sized for every row (the
/// growth allowance of `JoinTable::index_bytes_estimated` is half that).
#[test]
fn both_loads_hold_their_keys_within_their_reservation() {
    let layout = layout();
    for (load, num, den, counts) in [
        (
            DirectoryLoad::Compact,
            4,
            5,
            [1_639_usize, 1_638, 1_640, 3_277],
        ),
        (DirectoryLoad::Roomy, 1, 2, [2_048, 2_047, 2_049, 4_097]),
    ] {
        for n in counts {
            let keys: Vec<i64> = (0..i64::try_from(n).unwrap_or(0)).collect();
            let enc = encode(&layout, &keys, &[], 0);
            let all: Vec<usize> = (0..n).collect();
            let src = Src {
                rows: enc.gather(&layout, &all).unwrap_or_else(|e| panic!("{e}")),
                hashes: enc.hashes().to_vec(),
                stride: layout.stride(),
            };
            for expect in [n, 1, n / 3] {
                let mut next = vec![NO_MATCH; n];
                let index = JoinIndex::build_expecting(
                    &layout,
                    &src,
                    n,
                    expect,
                    7,
                    IndexChoice::Salted,
                    load,
                    &mut next,
                )
                .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(index.len(), n);
                assert_eq!(index.load(), Some(load));
                assert!(
                    n * den <= index.buckets() * num,
                    "{load:?}: {n} keys in {}",
                    index.buckets()
                );
                assert!(
                    index.memory_size() <= IndexChoice::Salted.bytes(n, load),
                    "{load:?}: {n} keys (expecting {expect}) took {} bytes",
                    index.memory_size()
                );
                if expect == n {
                    assert_eq!(index.buckets(), salted_bytes(n, load) / 8, "{load:?} {n}");
                }
                let ids: Vec<u32> = (0..u32::try_from(n).unwrap_or(0)).collect();
                check_finds(&layout, &index, &src, &enc, &ids);
            }
        }
    }
}
