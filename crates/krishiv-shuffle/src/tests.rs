// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::exchange::{Exchange, ExchangeFrame, MapOutputId};
use krishiv_exec::physical::ExchangeId;
use krishiv_memory::{SpillConfig, SpillManager};

use crate::LocalShuffle;
use crate::frame::{self, FrameError, FrameHeader, FrameKind, HEADER_BYTES};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Utf8, true),
    ]))
}

fn batch(from: i64, n: i64) -> RecordBatch {
    let keys: Vec<i64> = (from..from + n).collect();
    let values: Vec<Option<String>> = keys
        .iter()
        .map(|k| (k % 3 != 0).then(|| format!("v{k}")))
        .collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(StringArray::from(values)),
        ],
    )
    .unwrap()
}

fn output(map: u32, attempt: u32) -> MapOutputId {
    MapOutputId {
        exchange: ExchangeId(7),
        stage_attempt: 0,
        map,
        attempt,
    }
}

/// Every frame of `reduce` of `map`'s committed output, `End` included.
fn read(shuffle: &LocalShuffle, map: u32, reduce: u32) -> Result<Vec<RecordBatch>, KrishivError> {
    let mut reader = shuffle.reader(ExchangeId(7), 0, map, reduce)?;
    let mut out = Vec::new();
    loop {
        match block_on(reader.next())? {
            ExchangeFrame::Data(b) => out.push(b),
            ExchangeFrame::End => return Ok(out),
        }
    }
}

fn header(kind: FrameKind) -> FrameHeader {
    FrameHeader {
        kind,
        id: 3,
        stage_attempt: 1,
        map: 2,
        attempt: 4,
        sequence: 5,
        reduce: 6,
    }
}

#[test]
fn a_frame_round_trips_with_its_header_and_batch() {
    let body = frame::batch_body(&batch(0, 10)).unwrap();
    let bytes = frame::encode(&header(FrameKind::Data), &body).unwrap();
    assert_eq!(&bytes[..4], b"KXFR");
    assert_eq!(frame::body_len(&bytes).unwrap(), body.len());
    let (h, b) = frame::decode(&bytes).unwrap();
    assert_eq!(h, header(FrameKind::Data));
    assert_eq!(frame::body_batch(b).unwrap(), batch(0, 10));
}

#[test]
fn damaged_frames_are_refused() {
    let body = frame::batch_body(&batch(0, 10)).unwrap();
    let good = frame::encode(&header(FrameKind::Data), &body).unwrap();
    let mut flipped = good.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 1;
    assert_eq!(frame::decode(&flipped), Err(FrameError::Checksum));
    let mut version = good.clone();
    version[4] = 2;
    assert_eq!(frame::decode(&version), Err(FrameError::Version(2)));
    let mut magic = good.clone();
    magic[0] = b'X';
    assert_eq!(frame::decode(&magic), Err(FrameError::Magic));
    // `Barrier` (4) is reserved for streaming.
    let mut kind = good.clone();
    kind[6] = 4;
    assert_eq!(frame::decode(&kind), Err(FrameError::Kind(4)));
    let mut lz4 = good.clone();
    lz4[7] = 1;
    assert_eq!(frame::decode(&lz4), Err(FrameError::Compression(1)));
    assert_eq!(
        frame::decode(&good[..good.len() - 1]),
        Err(FrameError::Truncated(good.len() - 1))
    );
    assert_eq!(
        frame::decode(&good[..HEADER_BYTES - 1]),
        Err(FrameError::Truncated(HEADER_BYTES - 1))
    );
    let mut longer = good;
    longer.push(0);
    assert!(matches!(frame::decode(&longer), Err(FrameError::Body(_))));
    assert!(matches!(
        frame::body_batch(b"not ipc"),
        Err(FrameError::Body(_))
    ));
}

#[test]
fn crc32c_matches_the_standard_check_value() {
    // RFC 3720 B.4 / the CRC catalogue's check value for "123456789".
    assert_eq!(frame::crc32c(b"123456789"), 0xE306_9283);
    assert_eq!(frame::crc32c(b""), 0);
}

fn write_and_commit(shuffle: &LocalShuffle, map: u32, attempt: u32, from: i64) {
    let mut w = shuffle.writer(output(map, attempt), 2, schema()).unwrap();
    w.write(0, &batch(from, 5)).unwrap();
    w.write(1, &batch(from + 5, 3)).unwrap();
    w.write(0, &batch(from + 8, 2)).unwrap();
    w.commit().unwrap();
}

fn commits_are_visible_once_and_first_wins(shuffle: &LocalShuffle) {
    // Before the commit nothing is readable.
    let mut w = shuffle.writer(output(0, 0), 2, schema()).unwrap();
    w.write(0, &batch(0, 5)).unwrap();
    let before = read(shuffle, 0, 0).unwrap_err();
    assert_eq!(before.error_class(), ErrorClass::InternalError, "{before}");
    // A writer dropped without its commit leaves nothing.
    drop(w);
    assert!(read(shuffle, 0, 0).is_err());
    assert!(shuffle.committed(ExchangeId(7), 0).is_empty());

    write_and_commit(shuffle, 0, 1, 0);
    assert_eq!(read(shuffle, 0, 0).unwrap(), vec![batch(0, 5), batch(8, 2)]);
    assert_eq!(read(shuffle, 0, 1).unwrap(), vec![batch(5, 3)]);
    // A later attempt of the same map partition is ignored, whatever it
    // wrote; another map partition is its own.
    write_and_commit(shuffle, 0, 2, 100);
    write_and_commit(shuffle, 1, 0, 200);
    assert_eq!(read(shuffle, 0, 0).unwrap(), vec![batch(0, 5), batch(8, 2)]);
    assert_eq!(read(shuffle, 1, 1).unwrap(), vec![batch(205, 3)]);
    assert_eq!(
        shuffle
            .committed(ExchangeId(7), 0)
            .into_iter()
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 0)]
    );
    // Another stage attempt has none of them.
    assert!(shuffle.reader(ExchangeId(7), 1, 0, 0).is_err());
    assert!(shuffle.reader(ExchangeId(7), 0, 0, 2).is_err());
}

#[test]
fn commit_before_read_in_memory() {
    commits_are_visible_once_and_first_wins(&LocalShuffle::in_memory());
}

#[test]
fn commit_before_read_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let shuffle = LocalShuffle::on_disk(scope.clone());
    commits_are_visible_once_and_first_wins(&shuffle);
    // One file per committed output; the dropped writer's and the losing
    // attempt's files are gone.
    let files = std::fs::read_dir(scope.dir()).unwrap().count();
    assert_eq!(files, 2);
}

#[test]
fn a_damaged_file_is_corrupt_not_wrong_rows() {
    let dir = tempfile::tempdir().unwrap();
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let shuffle = LocalShuffle::on_disk(scope.clone());
    write_and_commit(&shuffle, 0, 0, 0);
    let path = std::fs::read_dir(scope.dir())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut bytes = std::fs::read(&path).unwrap();
    // The last byte of the first frame's body.
    let first = HEADER_BYTES + frame::body_len(&bytes).unwrap();
    bytes[first - 1] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();
    let e = read(&shuffle, 0, 0).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivSpillFileCorrupt, "{e}");
    // Partition 1's frames are intact.
    assert_eq!(read(&shuffle, 0, 1).unwrap(), vec![batch(5, 3)]);
}

#[test]
fn writes_are_checked() {
    let shuffle = LocalShuffle::in_memory();
    assert!(shuffle.writer(output(0, 0), 0, schema()).is_err());
    let mut w = shuffle.writer(output(0, 0), 2, schema()).unwrap();
    assert!(w.write(2, &batch(0, 1)).is_err());
    let other = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    assert!(w.write(0, &other).is_err());
}
