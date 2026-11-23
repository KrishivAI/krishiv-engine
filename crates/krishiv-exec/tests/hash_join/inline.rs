// SPDX-License-Identifier: Apache-2.0
//! Inline keys in the salted index (`krishiv.exec.join.inlineKeys`): a
//! build past the flat index's size with one `BIGINT` `=` key (a one-word
//! layout) returns the same rows with the flag on and off for every join
//! type, NULL keys and duplicates included, and its index holds more bytes
//! with the flag on (16-byte slots against 8), so the flag reaches the
//! tables.

use super::*;

/// `rows` build rows over `keys` distinct keys, every 97th key NULL.
fn rows(rows: usize, keys: i64, salt: i64) -> Vec<Row> {
    (0..rows)
        .map(|i| {
            let i = i64::try_from(i).unwrap();
            let k = (i * 7 + salt) % keys;
            r((k % 97 != 0).then_some(k), Some("a"), i % 50)
        })
        .collect()
}

/// Runs `jt` on `k1` (row layout) with inline keys `on`; returns the rows
/// and the build's `index_bytes`.
async fn run_inline(jt: JoinType, probe: &[Row], build: &[Row], on: bool) -> (Vec<String>, u64) {
    let root = root(256 * MIB);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: 1024,
        join_row_layout: true,
        join_inline_keys: on,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let mut c = case(jt, 1, false, false);
    c.row_layout = true;
    let plan = join(
        c.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), 4096)),
        scan(&schema(), batches(&data(probe), 4096)),
    );
    let out = run(&runner_for(2, 256 * MIB), plan, &q).await.unwrap();
    assert_released(&root);
    (render(&out), metric(&q, "HashJoinBuild", "index_bytes"))
}

#[tokio::test(start_paused = true)]
async fn inline_keys_return_the_salted_rows_and_hold_wider_slots() {
    // 200,000 rows over 8 level-0 partitions: about 25,000 a partition,
    // past the flat index's 64 KiB.
    let build = rows(200_000, 120_000, 0);
    let probe = rows(3_000, 150_000, 11);
    for jt in TYPES {
        let (on, on_bytes) = run_inline(jt, &probe, &build, true).await;
        let (off, off_bytes) = run_inline(jt, &probe, &build, false).await;
        assert!(on == off, "{jt:?}: {} rows on, {} off", on.len(), off.len());
        assert!(on_bytes > off_bytes, "{jt:?}: {on_bytes} <= {off_bytes}");
    }
    // The inner join against a direct count of matching pairs.
    let mut counts = std::collections::HashMap::new();
    for b in &build {
        if let Some(k) = b.k1 {
            *counts.entry(k).or_insert(0usize) += 1;
        }
    }
    let pairs: usize = probe
        .iter()
        .filter_map(|p| p.k1.and_then(|k| counts.get(&k)))
        .sum();
    let (inner, _) = run_inline(JoinType::Inner, &probe, &build, true).await;
    assert_eq!(inner.len(), pairs);
    assert!(pairs > 0);
}
