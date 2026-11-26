// SPDX-License-Identifier: Apache-2.0
//! A logged append into a Spark-escaped partition directory (`p=a%3Ab`) that another append
//! overtook: its swept-file check heads each of its files by the path the log names, which must be
//! parsed as it is (re-encoding the `%` heads `p=a%253Ab`, a file that does not exist, and refuses a
//! commit whose files are all there).

use std::sync::Arc;

use super::{Env, part_batch, part_schema};
use crate::parquet::ParquetFormat;
use crate::{FileFormat, FileWriter, FormatOptions, SaveMode};

#[test]
fn an_overtaken_append_into_an_escaped_partition_still_commits() {
    let env = Env::memory();
    let o = FormatOptions::defaults(ParquetFormat.options());
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    // `a:b` is written as the directory `p=a%3Ab`, as Spark escapes it.
    let append = |uuid: &str| {
        let w = env.begin("t", SaveMode::Append, uuid).unwrap().unwrap();
        let w = env.rt.block_on(w.logged(&env.io)).unwrap();
        let w = w.partitioned(true);
        let files = env.rt.block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("write", None);
            let mut options = w.write_options(0);
            options.partition_by = vec![1];
            options.partition_names = vec!["p".to_owned()];
            let mut writer = FileWriter::new(
                &env.io,
                Arc::clone(&format),
                &part_schema(),
                o.clone(),
                options,
                krishiv_memory::Reservation::new(pool, id),
            )
            .unwrap();
            writer
                .write(&part_batch(&[1, 2], &["a:b", "a:b"]))
                .await
                .unwrap();
            writer.finish().await.unwrap()
        });
        (w, files)
    };
    let (first, first_files) = append("job-1");
    let (second, second_files) = append("job-2");
    env.rt
        .block_on(second.commit(&env.io, format.as_ref(), part_schema(), &o, second_files))
        .unwrap();
    // The first append's base is behind the log now, so it heads its files before publishing.
    env.rt
        .block_on(first.commit(&env.io, format.as_ref(), part_schema(), &o, first_files))
        .expect("every file of the overtaken append is on the store");
    let visible = env.visible("t");
    assert_eq!(visible.len(), 2, "{visible:?}");
    assert!(
        visible.iter().all(|f| f.starts_with("p=a%3Ab/")),
        "{visible:?}"
    );
    let on_store = env.names("t");
    assert!(
        !on_store.iter().any(|n| n.contains("%253A")),
        "{on_store:?}"
    );
}
