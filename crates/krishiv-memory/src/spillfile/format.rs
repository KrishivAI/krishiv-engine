// SPDX-License-Identifier: Apache-2.0
//! The spill file layout (version 1).
//!
//! ```text
//! file   = header frame* footer
//! header = "KSPL" version:u32le                          (8 bytes)
//! frame  = kind:u8 constant:i64le len:u64le decoded:u64le crc:u32le payload[len]
//! ```
//! A data frame holds one [`Batch`](krishiv_types::batch::Batch) as a
//! complete Arrow IPC stream (schema, optional dictionaries, one record
//! batch, buffers compressed with the writer's codec). Frames are
//! self-describing, so a reader needs no codec setting, and a reader holds
//! one frame at a time. The weights travel with the batch:
//! - kind [`KIND_PLAIN`]: `Weights::Const(constant)`; the IPC stream has the
//!   data columns only;
//! - kind [`KIND_WEIGHTED`]: `Weights::Column`; the IPC stream has the
//!   weights as one extra trailing non-null `Int64` column.
//!
//! The footer frame ([`KIND_FOOTER`]) holds the number of batches and rows
//! written, so a file cut at a frame boundary is detected. `crc` is the
//! CRC-32 (IEEE) of the payload followed by the first 25 header bytes (the writer
//! streams the payload before it knows `len`, then patches the header). `decoded` is
//! an upper bound of the memory the decoded batch needs, so a reader reserves
//! `len + decoded` before it reads or decodes anything: a
//! flipped bit or a lying length is an error, never wrong data. Version 1 is
//! the only version; a reader refuses any other.

/// File magic.
pub(crate) const MAGIC: [u8; 4] = *b"KSPL";
/// Format version written. Version 1 is the only one, so the reader accepts
/// exactly it; a bump must keep reading the previous version (ADR 0029:
/// version N reads N-1).
pub(crate) const VERSION: u32 = 1;
/// Length of the file header.
pub(crate) const FILE_HEADER_LEN: usize = 8;
/// Length of a frame header.
pub(crate) const FRAME_HEADER_LEN: usize = 29;
/// Frame kind: data, `Weights::Const`.
pub(crate) const KIND_PLAIN: u8 = 0;
/// Frame kind: data, `Weights::Column` in the trailing column.
pub(crate) const KIND_WEIGHTED: u8 = 1;
/// Frame kind: the footer.
pub(crate) const KIND_FOOTER: u8 = 2;
/// Name of the extra column that carries a weights column through IPC.
pub(crate) const WEIGHT_COLUMN: &str = "__krishiv_weights";
/// Payload length of the footer frame.
pub(crate) const FOOTER_PAYLOAD_LEN: usize = 16;

pub(crate) fn file_header() -> [u8; FILE_HEADER_LEN] {
    let mut h = [0u8; FILE_HEADER_LEN];
    h[..4].copy_from_slice(&MAGIC);
    h[4..].copy_from_slice(&VERSION.to_le_bytes());
    h
}

/// Checks a file header; the error is the reason it is not acceptable.
pub(crate) fn check_file_header(h: &[u8; FILE_HEADER_LEN]) -> Result<(), String> {
    if h[..4] != MAGIC {
        return Err("not a spill file (bad magic)".to_owned());
    }
    let mut v = [0u8; 4];
    v.copy_from_slice(&h[4..]);
    let version = u32::from_le_bytes(v);
    if version == VERSION {
        Ok(())
    } else {
        Err(format!(
            "unsupported format version {version} (this build reads {VERSION})"
        ))
    }
}

/// A frame header: the fields the checksum covers, plus the checksum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameHeader {
    pub(crate) kind: u8,
    pub(crate) constant: i64,
    pub(crate) len: u64,
    /// Upper bound of the memory the decoded batch needs beyond the payload.
    pub(crate) decoded: u64,
    pub(crate) crc: u32,
}

impl FrameHeader {
    pub(crate) fn parse(b: &[u8; FRAME_HEADER_LEN]) -> Self {
        let u64_at = |i: usize| {
            let mut v = [0u8; 8];
            v.copy_from_slice(&b[i..i + 8]);
            u64::from_le_bytes(v)
        };
        let mut r = [0u8; 4];
        r.copy_from_slice(&b[25..29]);
        Self {
            kind: b[0],
            constant: i64::from_le_bytes(u64_at(1).to_le_bytes()),
            len: u64_at(9),
            decoded: u64_at(17),
            crc: u32::from_le_bytes(r),
        }
    }

    /// The 25 bytes before the checksum.
    fn fixed(&self) -> [u8; 25] {
        let mut h = [0u8; 25];
        h[0] = self.kind;
        h[1..9].copy_from_slice(&self.constant.to_le_bytes());
        h[9..17].copy_from_slice(&self.len.to_le_bytes());
        h[17..25].copy_from_slice(&self.decoded.to_le_bytes());
        h
    }

    /// Seals a header whose payload (`len` bytes) has been fed to `payload_crc`.
    pub(crate) fn seal(mut self, mut payload_crc: Crc32) -> [u8; FRAME_HEADER_LEN] {
        payload_crc.update(&self.fixed());
        self.crc = payload_crc.finish();
        let mut h = [0u8; FRAME_HEADER_LEN];
        h[..25].copy_from_slice(&self.fixed());
        h[25..].copy_from_slice(&self.crc.to_le_bytes());
        h
    }

    /// The checksum this header should carry for `payload`.
    pub(crate) fn checksum(&self, payload: &[u8]) -> u32 {
        let mut crc = Crc32::new();
        crc.update(payload);
        crc.update(&self.fixed());
        crc.finish()
    }
}

const fn crc_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut i: u32 = 0;
    while i < 256 {
        let mut c = i;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        tables[0][i as usize] = c;
        i += 1;
    }
    let mut t = 1;
    while t < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = tables[t - 1][i];
            tables[t][i] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            i += 1;
        }
        t += 1;
    }
    tables
}

/// Slice-by-8 tables: the checksum covers every byte of every spill file, so
/// it must not be the bottleneck of the I/O.
static CRC_TABLES: [[u32; 256]; 8] = crc_tables();

/// CRC-32 (IEEE 802.3), as used by zlib and gzip.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Crc32(u32);

impl Crc32 {
    pub(crate) fn new() -> Self {
        Self(u32::MAX)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        let t = &CRC_TABLES;
        let mut c = self.0;
        let (chunks, remainder) = bytes.as_chunks::<8>();
        for chunk in chunks {
            let lo = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ c;
            let hi = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            c = t[7][(lo & 0xFF) as usize]
                ^ t[6][((lo >> 8) & 0xFF) as usize]
                ^ t[5][((lo >> 16) & 0xFF) as usize]
                ^ t[4][(lo >> 24) as usize]
                ^ t[3][(hi & 0xFF) as usize]
                ^ t[2][((hi >> 8) & 0xFF) as usize]
                ^ t[1][((hi >> 16) & 0xFF) as usize]
                ^ t[0][(hi >> 24) as usize];
        }
        for &b in remainder {
            c = t[0][usize::from(((c & 0xFF) as u8) ^ b)] ^ (c >> 8);
        }
        self.0 = c;
    }

    pub(crate) fn finish(self) -> u32 {
        !self.0
    }
}
