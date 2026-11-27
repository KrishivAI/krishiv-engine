// SPDX-License-Identifier: Apache-2.0
//! A file writer short of memory ends its file and goes on in a new one (SQL-33 at 6 MB): plain,
//! partitioned and bucketed writes.

use super::*;
/// A writer whose upload buffer the pool cannot hold (another consumer
/// keeps most of a small limit; one multipart part is at least 5 MiB)
/// ends the file it is writing and goes on in a new one instead of
/// failing or aborting the query (SQL-33 at a 6 MB limit, perf-1
/// 2026-11-23): every row is written once and the memory is returned.
#[test]
fn a_writer_short_of_memory_rolls_to_a_new_file() {
    use arrow_array::Int64Array;
    let env = Env::memory();
    let w = env
        .begin("tight", SaveMode::ErrorIfExists, "u")
        .unwrap()
        .unwrap();
    let pool: Arc<dyn krishiv_memory::MemoryPool> =
        krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig {
            limit: Some(4 << 20),
            ..krishiv_memory::MemoryConfig::default()
        });
    // Memory no spill takes back, as a sort's final merge budget is.
    let hog = pool.register("hog", None);
    let mut hold = krishiv_memory::Reservation::new(Arc::clone(&pool), hog);
    hold.try_grow(2 << 20).unwrap();
    let id = pool.register("write", None);
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    // Incompressible values (xorshift), 8,192 rows a batch: about 8 MB in all.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x.cast_signed()
    };
    let batches: Vec<RecordBatch> = (0..128)
        .map(|_| {
            let v: Vec<i64> = (0..8192).map(|_| next()).collect();
            RecordBatch::try_new(
                Arc::clone(&schema) as SchemaRef,
                vec![Arc::new(Int64Array::from(v))],
            )
            .unwrap()
        })
        .collect();
    let files = env.rt.block_on(async {
        let mut writer = FileWriter::new(
            &env.io,
            Arc::new(ParquetFormat),
            &(Arc::clone(&schema) as SchemaRef),
            FormatOptions::defaults(ParquetFormat.options()),
            w.write_options(0),
            krishiv_memory::Reservation::new(Arc::clone(&pool), id),
        )
        .unwrap();
        for b in &batches {
            writer.write(b).await.unwrap();
        }
        writer.finish().await.unwrap()
    });
    assert!(files.len() > 1, "the writer rolled over: {files:?}");
    let rows: u64 = files.iter().map(|f| f.rows).sum();
    assert_eq!(rows, 128 * 8192);
    assert_eq!(pool.reserved(), 2 << 20, "only the hog's memory is left");
    drop(hold);
    drop(w);
}

#[test]
fn a_partitioned_bucketed_writer_short_of_memory_rolls_within_its_partition_and_bucket() {
    use arrow_array::{Int64Array, StringArray};
    let env = Env::memory();
    let w = env
        .begin_dynamic("tight", SaveMode::ErrorIfExists, "u", false)
        .unwrap()
        .unwrap()
        .partitioned(true);
    let pool: Arc<dyn krishiv_memory::MemoryPool> =
        krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig {
            limit: Some(4 << 20),
            ..krishiv_memory::MemoryConfig::default()
        });
    let hog = pool.register("hog", None);
    let mut hold = krishiv_memory::Reservation::new(Arc::clone(&pool), hog);
    hold.try_grow(2 << 20).unwrap();
    let id = pool.register("write", None);
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, false),
        Field::new("p", DataType::Utf8, false),
        Field::new("__bucket", DataType::Int32, false),
    ]));
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x.cast_signed()
    };
    // Sorted by partition, then bucket: p=a bucket 3, then p=b bucket 1, ~4 MB each.
    let batches: Vec<RecordBatch> = (0..128)
        .map(|i| {
            let (p, bucket) = if i < 64 { ("a", 3) } else { ("b", 1) };
            let v: Vec<i64> = (0..8192).map(|_| next()).collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(v)),
                    Arc::new(StringArray::from(vec![p; 8192])),
                    Arc::new(Int32Array::from(vec![bucket; 8192])),
                ],
            )
            .unwrap()
        })
        .collect();
    let o = FormatOptions::defaults(ParquetFormat.options());
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let files = env
        .rt
        .block_on(async {
            let mut options = w.write_options(0);
            options.partition_by = vec![1];
            options.partition_names = vec!["p".to_owned()];
            let mut writer = FileWriter::new_bucketed(
                &env.io,
                Arc::clone(&format),
                &schema,
                o.clone(),
                options,
                krishiv_memory::Reservation::new(Arc::clone(&pool), id),
                Some(2),
            )?
            .bucketed_by(&schema, 2)?;
            for b in &batches {
                writer.write(b).await?;
            }
            let files = writer.finish().await?;
            w.commit(&env.io, format.as_ref(), Arc::clone(&schema), &o, files)
                .await
        })
        .unwrap();
    let mut per_partition = std::collections::BTreeMap::<String, (usize, u64)>::new();
    for f in &files {
        let path = f.path.to_string();
        let (dir, name) = path.rsplit_once('/').unwrap();
        let want_bucket = match dir {
            "tight/p=a" => 3,
            "tight/p=b" => 1,
            other => panic!("file outside its partition: {other}"),
        };
        assert_eq!(
            crate::bucket_id_of_file_name(name),
            Some(want_bucket),
            "{path}"
        );
        let e = per_partition.entry(dir.to_owned()).or_default();
        e.0 += 1;
        e.1 += f.rows;
    }
    assert_eq!(
        per_partition.values().map(|e| e.1).collect::<Vec<_>>(),
        [64 * 8192, 64 * 8192]
    );
    assert!(
        per_partition.values().all(|e| e.0 > 1),
        "each partition rolled over: {per_partition:?}"
    );
    assert_eq!(pool.reserved(), 2 << 20, "only the hog's memory is left");
    drop(hold);
}
