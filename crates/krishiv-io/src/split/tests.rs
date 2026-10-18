// SPDX-License-Identifier: Apache-2.0
//! Split planning against partitions Spark 4.2 planned
//! (`tests/fixtures/spark-4.2.0/file-splits.json`, `tests/scripts/gen_file_splits.py`).

use krishiv_common::config::Config;
use serde_json::Value;

use super::{FileSplit, SplitConfig, SplitFile, max_split_bytes, plan};
use crate::format::{Capabilities, Splittable};
use crate::io::ByteRange;

fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/file-splits.json"
    );
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn text_source() -> Capabilities {
    Capabilities {
        splittable: Splittable::ByCodec,
        self_describing: false,
        statistics: false,
        row_index: false,
        selection: false,
        page_pruning: false,
        page_checksums: false,
        write: false,
    }
}

type Parts = Vec<Vec<(String, u64, u64)>>;

#[test]
fn partitions_match_spark_4_2_for_every_golden_case() {
    let g = golden();
    let parallelism = g["default_parallelism"].as_u64().unwrap();
    let cases = g["cases"].as_array().unwrap();
    assert!(cases.len() >= 80);
    let caps = text_source();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let mut config = Config::new();
        for (k, v) in case["conf"].as_object().unwrap() {
            config.set(k, v.as_str().unwrap()).unwrap();
        }
        let cfg = SplitConfig::from_config(&config, parallelism);
        let listing: Vec<(String, u64)> = case["listing"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| (f[0].as_str().unwrap().to_owned(), f[1].as_u64().unwrap()))
            .collect();
        let files: Vec<SplitFile> = listing
            .iter()
            .map(|(name, size)| SplitFile {
                size: *size,
                splittable: caps.is_splittable(name),
            })
            .collect();
        let got: Parts = plan(&cfg, &files)
            .iter()
            .map(|p| {
                p.splits
                    .iter()
                    .map(|s| (listing[s.file].0.clone(), s.range.start, s.range.len()))
                    .collect()
            })
            .collect();
        let want: Parts = case["partitions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                p.as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        (
                            s[0].as_str().unwrap().to_owned(),
                            s[1].as_u64().unwrap(),
                            s[2].as_u64().unwrap(),
                        )
                    })
                    .collect()
            })
            .collect();
        assert_eq!(got.len(), want.len(), "{id}: partition count");
        assert_eq!(got, want, "{id}: partitions");
    }
}

#[test]
fn config_fallbacks_and_edges() {
    // Unset minPartitionNum: leafNodeDefaultParallelism, then worker slots.
    let (c, _) = Config::from_pairs([("spark.sql.leafNodeDefaultParallelism", "4")]).unwrap();
    assert_eq!(SplitConfig::from_config(&c, 12).min_partition_num, 4);
    let (c, _) = Config::from_pairs([
        ("krishiv.sql.files.minPartitionNum", "3"),
        ("krishiv.sql.leafNodeDefaultParallelism", "4"),
        ("krishiv.sql.files.maxPartitionNum", "9"),
        ("krishiv.sql.files.maxPartitionBytes", "1m"),
        ("krishiv.sql.files.openCostInBytes", "0"),
    ])
    .unwrap();
    let cfg = SplitConfig::from_config(&c, 12);
    assert_eq!((cfg.min_partition_num, cfg.max_partition_num), (3, Some(9)));
    assert_eq!(
        (cfg.max_partition_bytes, cfg.open_cost_in_bytes),
        (1 << 20, 0)
    );
    let d = SplitConfig::from_config(&Config::new(), 0);
    assert_eq!(d.min_partition_num, 1);
    assert!(plan(&d, &[]).is_empty());
    assert_eq!(max_split_bytes(&d, &[]), 4 << 20);
    // Zero-length files are skipped; a non-splittable file stays whole.
    let files = [
        SplitFile {
            size: 0,
            splittable: true,
        },
        SplitFile {
            size: 300 << 20,
            splittable: false,
        },
    ];
    let p = plan(&d, &files);
    assert_eq!(
        p[0].splits,
        vec![FileSplit {
            file: 1,
            range: ByteRange::new(0, 300 << 20)
        }]
    );
}

#[test]
fn ten_thousand_files_cover_every_byte_once() {
    let files: Vec<SplitFile> = (0..10_000_u64)
        .map(|i| SplitFile {
            size: (i * 7919) % (300 << 20) + 1,
            splittable: i % 3 != 0,
        })
        .collect();
    let cfg = SplitConfig::from_config(&Config::new(), 16);
    let parts = plan(&cfg, &files);
    let mut covered = vec![0_u64; files.len()];
    for p in &parts {
        for s in &p.splits {
            covered[s.file] += s.range.len();
        }
    }
    assert!(covered.iter().zip(&files).all(|(c, f)| *c == f.size));
}
