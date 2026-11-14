// SPDX-License-Identifier: Apache-2.0
//! Keeping a partition's in-memory chunks few: the probe side's size-tiered
//! merge ([`Part::coalesce`]) and the build side's cut into the row arena's
//! blocks of [`ARENA_ROWS`] rows ([`Part::block`], [`block_chunks`]; X3'
//! step 3 option C).

use arrow_schema::SchemaRef;
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::grace::Part;
use super::table::ARENA_ROWS;
use crate::error::{Result, internal};

impl Part {
    /// Merges runs of small in-memory chunks (each chunk carries fixed array
    /// overhead), if each merged copy can be reserved first.
    ///
    /// Size-tiered, like an LSM tree: a chunk's tier is `log16` of its bytes,
    /// and only a trailing run of at least `FANOUT` chunks no higher in tier
    /// than the newest one is merged. Merging every chunk held whenever the
    /// count reached the fan-out copied the whole partition again every 15
    /// pushes (quadratic in its size); tiered, a row is copied about once
    /// per tier it climbs.
    ///
    /// Returns the bytes of the largest run merged (each merge reserves the
    /// run again while it copies: the transient peak), 0 when none was.
    pub(crate) fn coalesce(&mut self, schema: &SchemaRef) -> Result<usize> {
        match self.memory.first_mut() {
            Some(m) => merge_tiers(&mut self.chunks, m, schema, usize::MAX, 16),
            None => Ok(0),
        }
    }

    /// [`block_chunks`] over the part's chunks, the copies reserved in its
    /// first reservation (the one its chunks are reserved in). Runs of
    /// [`BUILD_FANOUT`] chunks shorter than a block are first merged
    /// size-tiered among themselves ([`merge_tiers`]): a build fed small
    /// batches pushes chunks of a few rows (12 drivers cutting each batch
    /// into its partitions: about 20 rows each on TPC-H q10 at SF10, 29,359
    /// chunks for 573,157 rows), and finalize's accounting and tail copy then
    /// walked every one of them (finalize 9 -> 72-89 ms). Chunks of a block or
    /// more are never merged again, and chunks large enough to fill a block
    /// before [`BUILD_FANOUT`] of them are held are copied once, by the cut.
    pub(crate) fn block(&mut self, schema: &SchemaRef) -> Result<usize> {
        match self.memory.first_mut() {
            Some(m) => {
                // Cut first: a run that already fills a block is copied once,
                // into the block, not merged and then cut again.
                let cut = block_chunks(&mut self.chunks, m, schema)?;
                let merged = merge_tiers(&mut self.chunks, m, schema, ARENA_ROWS, BUILD_FANOUT)?;
                Ok(cut.max(merged))
            }
            None => Ok(0),
        }
    }
}

/// The build side's merge fan-out: 64 chunks of a block's 2^14 rows hold
/// 256 rows each on average, so only chunks smaller than that are merged
/// before the block cut copies them.
const BUILD_FANOUT: usize = 64;

/// [`Part::coalesce`]'s size-tiered merge over the trailing chunks shorter
/// than `rows` rows, `fanout` of one tier at a time, each merged copy
/// reserved in `m` first.
fn merge_tiers(
    chunks: &mut Vec<Batch>,
    m: &mut Reservation,
    schema: &SchemaRef,
    rows: usize,
    fanout: usize,
) -> Result<usize> {
    let size = |c: &Batch| c.data().get_array_memory_size();
    let tier = |c: &Batch| size(c).max(1).ilog(16);
    let mut largest = 0;
    while chunks.len() >= fanout {
        let Some(top) = chunks.last().map(tier) else {
            return Ok(largest);
        };
        let run = chunks
            .iter()
            .rev()
            .take_while(|c| c.num_rows() < rows && tier(c) <= top)
            .count();
        if run < fanout {
            return Ok(largest);
        }
        let from = chunks.len() - run;
        let old: usize = chunks[from..].iter().map(size).sum();
        if m.size() < old || m.try_grow(old).is_err() {
            return Ok(largest);
        }
        let merged = Batch::concat(schema, &chunks[from..]).map_err(|e| internal(e.to_string()));
        let merged = match merged {
            Ok(b) if size(&b) <= old => b,
            other => {
                m.shrink(old);
                return other.map(|_| largest);
            }
        };
        largest = largest.max(old);
        m.shrink(old + old - size(&merged));
        chunks.truncate(from);
        chunks.push(merged);
    }
    Ok(largest)
}

/// Cuts build rows into the arena's blocks as they come (X3' step 3
/// option C): once the trailing chunks shorter than a block hold a
/// block's rows, they are copied into one block of exactly
/// [`ARENA_ROWS`] rows (the last chunk split, its tail left as a
/// zero-copy slice), if the copy can be reserved first. Chunks of a block
/// or more stay as they are: the arena slices them. Each row is copied at
/// most once here, and finalize copies only the tails left.
///
/// Returns the bytes of the largest copy (each one reserves its chunks
/// again while it copies: the transient peak), 0 when none was made.
pub(crate) fn block_chunks(
    chunks: &mut Vec<Batch>,
    m: &mut Reservation,
    schema: &SchemaRef,
) -> Result<usize> {
    let size = |c: &Batch| c.data().get_array_memory_size();
    let mut largest = 0;
    loop {
        let run = chunks
            .iter()
            .rev()
            .take_while(|c| c.num_rows() < ARENA_ROWS)
            .count();
        let from = chunks.len() - run;
        // The chunks that make up the block: whole ones, then the head of
        // the last.
        let mut held = 0usize;
        let Some(last) = chunks[from..].iter().position(|c| {
            held += c.num_rows();
            held >= ARENA_ROWS
        }) else {
            return Ok(largest);
        };
        let last = from + last;
        let head = ARENA_ROWS - (held - chunks[last].num_rows());
        let whole = if head == chunks[last].num_rows() {
            last + 1
        } else {
            last
        };
        let freed: usize = chunks[from..whole].iter().map(size).sum();
        let old = freed
            + if whole == last {
                size(&chunks[last])
            } else {
                0
            };
        if m.size() < old {
            return Ok(largest);
        }
        if m.try_grow(old).is_err() {
            return Ok(largest);
        }
        let pieces = chunks[from..last]
            .iter()
            .cloned()
            .map(Ok)
            .chain(std::iter::once(chunks[last].slice(0, head)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .and_then(|p| Batch::concat(schema, &p))
            .map_err(|e| internal(e.to_string()));
        let n = chunks[last].num_rows();
        let tail = (whole == last).then(|| {
            chunks[last]
                .slice(head, n - head)
                .map_err(|e| internal(e.to_string()))
        });
        let (block, tail) = match (pieces, tail.transpose()) {
            (Ok(b), Ok(t)) => (b, t),
            (Err(e), _) | (_, Err(e)) => {
                m.shrink(old);
                return Err(e);
            }
        };
        // A copy larger than its sources (buffer rounding) is kept if the
        // excess can be reserved; dropping it would redo it on every push.
        let new = size(&block);
        let excess = new.saturating_sub(old);
        if excess > 0 && m.try_grow(excess).is_err() {
            m.shrink(old);
            return Ok(largest);
        }
        largest = largest.max(old);
        // Held now: the block, and the last chunk while its tail lives
        // (`old + excess >= new`).
        m.shrink(old + excess + freed - new);
        chunks.truncate(from);
        chunks.push(block);
        chunks.extend(tail);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};

    use super::*;
    use crate::join::grace::reserve_chunks;

    /// Coalescing is size-tiered: a merged chunk is not copied again by the
    /// next run of small chunks (merging all chunks every 16 pushes was
    /// quadratic), the chunk count stays bounded, and no row is lost.
    #[test]
    fn coalescing_does_not_recopy_merged_chunks() {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let id = pool.register("coalesce", None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let chunk = |i: i64| {
            let k = Int64Array::from_iter_values(i * 300..(i + 1) * 300);
            let s = StringArray::from_iter_values((0..300).map(|j| format!("row-{i}-{j}")));
            Batch::new(
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(s)]).unwrap(),
            )
        };
        let mut parts = vec![Part::with_memory(krishiv_memory::Reservation::new(
            Arc::clone(&pool),
            id,
        ))];
        let push = |parts: &mut Vec<Part>, i: i64| {
            let c = chunk(i);
            reserve_chunks(parts, &[(0, c.clone())]).unwrap();
            parts[0].chunks.push(c);
            parts[0].coalesce(&schema).unwrap()
        };
        let sizes: usize = (0..16)
            .map(|i| chunk(i).data().get_array_memory_size())
            .sum();
        let mut merged_run = 0;
        for i in 0..16 {
            merged_run = push(&mut parts, i);
        }
        assert_eq!(parts[0].chunks.len(), 1, "16 small chunks merge");
        // It reports the run it merged: the 16 chunks' bytes.
        assert_eq!(merged_run, sizes);
        let merged = Arc::clone(parts[0].chunks[0].data().column(0));
        for i in 16..31 {
            assert_eq!(push(&mut parts, i), 0, "no merge below the fan-out");
        }
        assert_eq!(parts[0].chunks.len(), 16);
        let first = parts[0].chunks[0].data().column(0);
        assert!(
            Arc::ptr_eq(first, &merged),
            "the merged chunk was copied again"
        );
        let mut most = 0;
        for i in 31..1024 {
            push(&mut parts, i);
            most = most.max(parts[0].chunks.len());
        }
        assert!(most < 3 * 16, "{most} chunks held");
        assert_eq!(parts[0].mem_rows(), 1024 * 300);
        let mut keys: Vec<i64> = parts[0]
            .chunks
            .iter()
            .flat_map(|c| {
                let k = c
                    .data()
                    .column(0)
                    .as_primitive::<arrow_array::types::Int64Type>();
                k.values().to_vec()
            })
            .collect();
        keys.sort_unstable();
        assert!(
            keys.into_iter().eq(0..1024 * 300),
            "rows lost or duplicated"
        );
        drop(parts);
        assert_eq!(pool.reserved(), 0);
    }

    /// Build chunks are cut into the arena's blocks as they come: small
    /// chunks are copied once into blocks of exactly `ARENA_ROWS` rows (the
    /// chunk that crosses a block's end split, its tail kept), chunks of a
    /// block or more are left alone, no row is lost and the reservation
    /// follows what is held.
    #[test]
    fn build_chunks_are_cut_into_blocks_once() {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let id = pool.register("block", None);
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let chunk = |from: i64, n: i64| {
            Batch::new(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from_iter_values(from..from + n))],
                )
                .unwrap(),
            )
        };
        let mut parts = vec![Part::with_memory(krishiv_memory::Reservation::new(
            Arc::clone(&pool),
            id,
        ))];
        let mut next = 0i64;
        let mut push = |parts: &mut Vec<Part>, n: i64| {
            let c = chunk(next, n);
            next += n;
            reserve_chunks(parts, &[(0, c.clone())]).unwrap();
            parts[0].chunks.push(c);
            parts[0].block(&schema).unwrap()
        };
        let block = ARENA_ROWS as i64;
        // 1,000-row chunks: a block once 17 of them are held, the 17th split.
        let mut copied = 0;
        for _ in 0..16 {
            copied = copied.max(push(&mut parts, 1000));
        }
        assert_eq!(copied, 0);
        assert!(push(&mut parts, 1000) > 0);
        let rows: Vec<usize> = parts[0].chunks.iter().map(Batch::num_rows).collect();
        assert_eq!(rows, [ARENA_ROWS, 17 * 1000 - ARENA_ROWS]);
        // A chunk of a block or more is not copied, and ends the run before it.
        assert_eq!(push(&mut parts, 2 * block + 7), 0);
        let blocked = Arc::clone(parts[0].chunks[0].data().column(0));
        for _ in 0..40 {
            push(&mut parts, 1000);
        }
        assert!(Arc::ptr_eq(parts[0].chunks[0].data().column(0), &blocked));
        assert!(
            parts[0]
                .chunks
                .iter()
                .filter(|c| c.num_rows() < ARENA_ROWS)
                .count()
                <= 17
        );
        let held: usize = parts[0]
            .chunks
            .iter()
            .map(|c| c.data().get_array_memory_size())
            .sum();
        assert!(
            parts[0].bytes() >= held,
            "{} reserved, {held} held",
            parts[0].bytes()
        );
        let mut keys: Vec<i64> = parts[0]
            .chunks
            .iter()
            .flat_map(|c| {
                let k = c
                    .data()
                    .column(0)
                    .as_primitive::<arrow_array::types::Int64Type>();
                k.values().to_vec()
            })
            .collect();
        keys.sort_unstable();
        assert!(keys.into_iter().eq(0..next), "rows lost or duplicated");
        drop(parts);
        assert_eq!(pool.reserved(), 0);
    }

    /// A run of short chunks that fills a block is cut into the block, not
    /// merged into one chunk past the block first: 63 chunks of 255 rows and
    /// one of 400 give a block and an 81-row tail (review M6 of the S5 stack).
    #[test]
    fn a_run_that_fills_a_block_is_cut_not_merged() {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let id = pool.register("cut-first", None);
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let mut parts = vec![Part::with_memory(krishiv_memory::Reservation::new(
            Arc::clone(&pool),
            id,
        ))];
        let mut next = 0i64;
        for n in std::iter::repeat_n(255i64, 63).chain([400]) {
            let c = Batch::new(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from_iter_values(next..next + n))],
                )
                .unwrap(),
            );
            next += n;
            reserve_chunks(&mut parts, &[(0, c.clone())]).unwrap();
            parts[0].chunks.push(c);
            parts[0].block(&schema).unwrap();
        }
        let rows: Vec<usize> = parts[0].chunks.iter().map(Batch::num_rows).collect();
        assert_eq!(rows, [ARENA_ROWS, 63 * 255 + 400 - ARENA_ROWS]);
        drop(parts);
        assert_eq!(pool.reserved(), 0);
    }

    /// A build fed tiny chunks (TPC-H q10 at SF10: about 20 rows each) holds
    /// few of them: runs of `BUILD_FANOUT` merge before the block cut, so
    /// finalize does not walk thousands. Rows keep their order, none is
    /// lost, and the reservation covers what is held.
    #[test]
    fn tiny_build_chunks_are_merged_before_the_block_cut() {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let id = pool.register("tiny", None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let mut parts = vec![Part::with_memory(krishiv_memory::Reservation::new(
            Arc::clone(&pool),
            id,
        ))];
        let mut next = 0i64;
        let mut most = 0;
        for _ in 0..5000 {
            let k = Int64Array::from_iter_values(next..next + 20);
            let s = StringArray::from_iter_values((next..next + 20).map(|j| format!("row-{j}")));
            next += 20;
            let c = Batch::new(
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(s)]).unwrap(),
            );
            reserve_chunks(&mut parts, &[(0, c.clone())]).unwrap();
            parts[0].chunks.push(c);
            parts[0].block(&schema).unwrap();
            most = most.max(parts[0].chunks.len());
        }
        // Without the merge: up to a block's 819 chunks of 20 rows.
        assert!(most < 2 * BUILD_FANOUT, "{most} chunks held");
        let keys: Vec<i64> = parts[0]
            .chunks
            .iter()
            .flat_map(|c| {
                let k = c
                    .data()
                    .column(0)
                    .as_primitive::<arrow_array::types::Int64Type>();
                k.values().to_vec()
            })
            .collect();
        assert!(
            keys.into_iter().eq(0..next),
            "rows lost, duplicated or reordered"
        );
        let held: usize = parts[0]
            .chunks
            .iter()
            .map(|c| c.data().get_array_memory_size())
            .sum();
        assert!(
            parts[0].bytes() >= held,
            "{} reserved, {held} held",
            parts[0].bytes()
        );
        drop(parts);
        assert_eq!(pool.reserved(), 0);
    }
}
