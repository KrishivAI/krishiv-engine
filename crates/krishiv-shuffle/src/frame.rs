// SPDX-License-Identifier: Apache-2.0
//! Exchange frame format v1-unstable (design P2-T176 §2.4, owner question
//! 1): the unit of local shuffle files and, from P2-T176 unit 2, of the
//! wire. Versioned and checksummed from the start (ADR 0021 §4/§5, I10),
//! with no N/N−1 compatibility promise until P5 freezes it.
//!
//! A frame is a fixed 44-byte little-endian header and a body:
//!
//! | Bytes | Field |
//! |---|---|
//! | 0..4 | magic `KXFR` |
//! | 4..6 | format version (1) |
//! | 6 | kind: 1 `Data`, 2 `End`, 3 `Value` (4 `Barrier` and 5 `Watermark` reserved for P4) |
//! | 7 | body compression: 0 none (1 LZ4 and 2 ZSTD reserved) |
//! | 8..16 | exchange id (or value id) |
//! | 16..20 | stage attempt |
//! | 20..24 | map partition |
//! | 24..28 | task attempt |
//! | 28..32 | batch sequence within the map output and reduce partition |
//! | 32..36 | reduce partition |
//! | 36..40 | body length |
//! | 40..44 | CRC32C of the body |
//!
//! A `Data` body is an Arrow IPC stream of one batch (ADR 0021 §4); an
//! `End` body is empty.

use std::io::Cursor;
use std::sync::Arc;

use arrow_array::RecordBatch;

/// The magic at the start of every frame.
pub const MAGIC: [u8; 4] = *b"KXFR";
/// The format version this crate writes and reads.
pub const FORMAT_VERSION: u16 = 1;
/// The header's length in bytes.
pub const HEADER_BYTES: usize = 44;

/// What a frame carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Rows.
    Data,
    /// The end of one reduce partition of one map output.
    End,
    /// A versioned value's partial (design P2-T177).
    Value,
}

impl FrameKind {
    fn code(self) -> u8 {
        match self {
            Self::Data => 1,
            Self::End => 2,
            Self::Value => 3,
        }
    }

    fn of(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Data),
            2 => Some(Self::End),
            3 => Some(Self::Value),
            _ => None,
        }
    }
}

/// A frame's header (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// What the frame carries.
    pub kind: FrameKind,
    /// The exchange (or value) id.
    pub id: u64,
    /// The stage attempt.
    pub stage_attempt: u32,
    /// The map partition.
    pub map: u32,
    /// The task attempt.
    pub attempt: u32,
    /// The frame's place among its map output's frames of its reduce
    /// partition, from 0.
    pub sequence: u32,
    /// The reduce partition.
    pub reduce: u32,
}

/// Why a frame could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// Shorter than its header or its stated body.
    #[error("the frame is truncated ({0} bytes)")]
    Truncated(usize),
    /// Not a frame.
    #[error("the frame has no frame magic")]
    Magic,
    /// Written by another format version.
    #[error(
        "frame format version {0} is not supported (this reader reads version {FORMAT_VERSION})"
    )]
    Version(u16),
    /// A kind this reader does not know or that is reserved.
    #[error("frame kind {0} is not supported")]
    Kind(u8),
    /// A body compression this reader does not support.
    #[error("frame compression {0} is not supported")]
    Compression(u8),
    /// The body does not match its checksum.
    #[error("the frame body does not match its checksum")]
    Checksum,
    /// The body is not what its kind says.
    #[error("the frame body is malformed: {0}")]
    Body(String),
}

/// `header` and `body` as one frame.
///
/// # Errors
/// [`FrameError::Body`] for a body past 4 GiB.
pub fn encode(header: &FrameHeader, body: &[u8]) -> Result<Vec<u8>, FrameError> {
    let mut out = Vec::with_capacity(HEADER_BYTES + body.len());
    out.extend_from_slice(&header_bytes(header, body)?);
    out.extend_from_slice(body);
    Ok(out)
}

/// The header bytes of the frame of `header` and `body` (the body follows
/// them), for writing the two without copying the body.
///
/// # Errors
/// [`FrameError::Body`] for a body past 4 GiB.
pub fn header_bytes(header: &FrameHeader, body: &[u8]) -> Result<Vec<u8>, FrameError> {
    let len = u32::try_from(body.len())
        .map_err(|_| FrameError::Body(format!("a body of {} bytes", body.len())))?;
    let mut out = Vec::with_capacity(HEADER_BYTES);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.push(header.kind.code());
    out.push(0);
    out.extend_from_slice(&header.id.to_le_bytes());
    for v in [
        header.stage_attempt,
        header.map,
        header.attempt,
        header.sequence,
        header.reduce,
        len,
        crc32c(body),
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// The header of `frame` and its body, checked against the checksum.
///
/// # Errors
/// See [`FrameError`].
pub fn decode(frame: &[u8]) -> Result<(FrameHeader, &[u8]), FrameError> {
    let header = frame
        .get(..HEADER_BYTES)
        .ok_or(FrameError::Truncated(frame.len()))?;
    if header[..4] != MAGIC {
        return Err(FrameError::Magic);
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != FORMAT_VERSION {
        return Err(FrameError::Version(version));
    }
    let kind = FrameKind::of(header[6]).ok_or(FrameError::Kind(header[6]))?;
    if header[7] != 0 {
        return Err(FrameError::Compression(header[7]));
    }
    let u32_at = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let mut id = [0_u8; 8];
    id.copy_from_slice(&header[8..16]);
    let len = u32_at(36) as usize;
    let body = frame
        .get(HEADER_BYTES..HEADER_BYTES + len)
        .ok_or(FrameError::Truncated(frame.len()))?;
    if frame.len() != HEADER_BYTES + len {
        return Err(FrameError::Body(format!(
            "{} bytes after a body of {len}",
            frame.len() - HEADER_BYTES - len
        )));
    }
    if crc32c(body) != u32_at(40) {
        return Err(FrameError::Checksum);
    }
    Ok((
        FrameHeader {
            kind,
            id: u64::from_le_bytes(id),
            stage_attempt: u32_at(16),
            map: u32_at(20),
            attempt: u32_at(24),
            sequence: u32_at(28),
            reduce: u32_at(32),
        },
        body,
    ))
}

/// The body length a header states, for reading a frame in two steps.
///
/// # Errors
/// [`FrameError::Truncated`] for fewer than [`HEADER_BYTES`] bytes.
pub fn body_len(header: &[u8]) -> Result<usize, FrameError> {
    let h = header
        .get(..HEADER_BYTES)
        .ok_or(FrameError::Truncated(header.len()))?;
    Ok(u32::from_le_bytes([h[36], h[37], h[38], h[39]]) as usize)
}

/// `batch` as a `Data` body.
///
/// # Errors
/// [`FrameError::Body`] when Arrow IPC cannot encode it.
pub fn batch_body(batch: &RecordBatch) -> Result<Vec<u8>, FrameError> {
    let mut out = Vec::new();
    let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut out, &batch.schema())
        .map_err(|e| FrameError::Body(e.to_string()))?;
    writer
        .write(batch)
        .map_err(|e| FrameError::Body(e.to_string()))?;
    writer
        .finish()
        .map_err(|e| FrameError::Body(e.to_string()))?;
    drop(writer);
    Ok(out)
}

/// The batch of a `Data` body.
///
/// # Errors
/// [`FrameError::Body`] for anything but an Arrow IPC stream of one batch.
pub fn body_batch(body: &[u8]) -> Result<RecordBatch, FrameError> {
    let reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(body), None)
        .map_err(|e| FrameError::Body(e.to_string()))?;
    let mut batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| FrameError::Body(e.to_string()))?;
    match (batches.pop(), batches.is_empty()) {
        (Some(b), true) => Ok(b),
        _ => Err(FrameError::Body("not one batch".to_owned())),
    }
}

/// CRC32C (Castagnoli) of `bytes`.
#[must_use]
pub fn crc32c(bytes: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<Arc<[u32; 256]>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0_u32; 256];
        for (i, slot) in (0_u32..).zip(t.iter_mut()) {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 == 1 {
                    (c >> 1) ^ 0x82F6_3B78
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        Arc::new(t)
    });
    !bytes.iter().fold(!0_u32, |crc, &b| {
        table[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8)
    })
}
