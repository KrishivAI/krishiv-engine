// SPDX-License-Identifier: Apache-2.0
//! Hadoop compression codecs of the line-oriented sources (CSV, JSON, text),
//! by file suffix, as Spark 4.2 reads and writes them: `.gz` (gzip, members
//! may be concatenated), `.deflate` (zlib), `.bz2`, `.lz4` and `.snappy`
//! (Hadoop's block framing: a big-endian uncompressed block length, then
//! chunks of a big-endian compressed length and a raw LZ4 or Snappy block).
//!
//! Compressed files are read whole by the split that starts at byte 0 (the
//! other splits of a `.bz2` file own nothing), so line offsets count
//! uncompressed bytes. `.zst` is not a codec Spark's text sources write, and
//! reading it stays `KRISHIV_UNSUPPORTED_FEATURE`.

use std::io::Write;

use bytes::Bytes;

use crate::error::{FormatError, FormatResult, internal};
use crate::format::{Encoder, WrittenFile};
use crate::stats::Precision;

/// Largest uncompressed block a block-framed file may declare.
const MAX_BLOCK: usize = 64 << 20;
/// Compressed bytes a reader asks for at a time (bounds what one push inflates to).
pub(crate) const COMPRESSED_CHUNK: u64 = 64 << 10;
/// Uncompressed bytes per block the block-framed writers cut.
const WRITE_BLOCK: usize = 256_000;

/// A compression codec of a text file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Codec {
    Gzip,
    Deflate,
    Bzip2,
    Lz4,
    Snappy,
}

/// Names Spark accepts for `compression` / `codec` (besides none).
pub(crate) const WRITE_NAMES: &[&str] = &[
    "bzip2",
    "deflate",
    "uncompressed",
    "snappy",
    "none",
    "lz4",
    "gzip",
];

impl Codec {
    /// The codec of a file name, by Hadoop's suffix rule.
    pub(crate) fn of_path(path: &str) -> Option<Self> {
        [
            (".gz", Self::Gzip),
            (".deflate", Self::Deflate),
            (".bz2", Self::Bzip2),
            (".lz4", Self::Lz4),
            (".snappy", Self::Snappy),
        ]
        .into_iter()
        .find(|(s, _)| path.ends_with(s))
        .map(|(_, c)| c)
    }

    /// The codec a write option names: `Some(None)` for no compression,
    /// `None` for a name Spark does not know.
    pub(crate) fn parse_write(name: &str) -> Option<Option<Self>> {
        match name.to_ascii_lowercase().as_str() {
            "none" | "uncompressed" => Some(None),
            "gzip" => Some(Some(Self::Gzip)),
            "deflate" => Some(Some(Self::Deflate)),
            "bzip2" => Some(Some(Self::Bzip2)),
            "lz4" => Some(Some(Self::Lz4)),
            "snappy" => Some(Some(Self::Snappy)),
            _ => None,
        }
    }

    /// The file suffix, with its dot.
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            Self::Gzip => ".gz",
            Self::Deflate => ".deflate",
            Self::Bzip2 => ".bz2",
            Self::Lz4 => ".lz4",
            Self::Snappy => ".snappy",
        }
    }
}

fn corrupt(codec: Codec, what: impl std::fmt::Display) -> FormatError {
    FormatError::Corrupt(format!("{} file: {what}", codec.suffix()))
}

enum Dec {
    Gzip(Box<flate2::write::MultiGzDecoder<Vec<u8>>>),
    Deflate(Multi<flate2::Decompress>),
    Bzip2(Multi<bzip2::Decompress>),
    Block(BlockDec),
}

/// A streaming decompressor: compressed bytes in, plain bytes out.
pub(crate) struct Decompressor {
    codec: Codec,
    dec: Option<Dec>,
}

impl std::fmt::Debug for Decompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decompressor")
            .field("codec", &self.codec)
            .finish_non_exhaustive()
    }
}

impl Decompressor {
    pub(crate) fn new(codec: Codec) -> Self {
        let dec = match codec {
            Codec::Gzip => Dec::Gzip(Box::new(flate2::write::MultiGzDecoder::new(Vec::new()))),
            Codec::Deflate => Dec::Deflate(Multi::new(|| flate2::Decompress::new(true))),
            Codec::Bzip2 => Dec::Bzip2(Multi::new(|| bzip2::Decompress::new(false))),
            Codec::Lz4 | Codec::Snappy => Dec::Block(BlockDec::new(codec)),
        };
        Self {
            codec,
            dec: Some(dec),
        }
    }

    /// Decompresses `input`, appending the plain bytes to `out`.
    pub(crate) fn push(&mut self, input: &[u8], out: &mut Vec<u8>) -> FormatResult<()> {
        let codec = self.codec;
        let Some(dec) = self.dec.as_mut() else {
            return Err(FormatError::Other(internal("push after finish")));
        };
        let io = |e: std::io::Error| corrupt(codec, e);
        match dec {
            Dec::Gzip(d) => {
                d.write_all(input).map_err(io)?;
                if d.get_ref().len() > MAX_PUSH_OUTPUT {
                    return Err(corrupt(codec, "a chunk inflates past the size limit"));
                }
                out.append(d.get_mut());
            }
            Dec::Deflate(d) => d.run(codec, input, out)?,
            Dec::Bzip2(d) => d.run(codec, input, out)?,
            Dec::Block(d) => d.push(input, out)?,
        }
        Ok(())
    }

    /// Ends the stream; a truncated one is `Corrupt`.
    pub(crate) fn finish(&mut self, out: &mut Vec<u8>) -> FormatResult<()> {
        let codec = self.codec;
        let Some(dec) = self.dec.take() else {
            return Ok(());
        };
        let io = |e: std::io::Error| corrupt(codec, e);
        match dec {
            Dec::Gzip(d) => out.extend((*d).finish().map_err(io)?),
            Dec::Deflate(d) => d.finish(codec)?,
            Dec::Bzip2(d) => d.finish(codec)?,
            Dec::Block(d) => d.finish()?,
        }
        Ok(())
    }
}

/// One compression stream of a concatenation of streams (Hadoop's codecs
/// read every stream of a file).
trait Stream {
    /// Decompresses from `input` into the spare capacity of `out`; the bytes
    /// consumed and whether the stream ended.
    fn step(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<(usize, bool), String>;
}

impl Stream for flate2::Decompress {
    fn step(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<(usize, bool), String> {
        let before = self.total_in();
        let st = self
            .decompress_vec(input, out, flate2::FlushDecompress::None)
            .map_err(|e| e.to_string())?;
        let used = usize::try_from(self.total_in() - before).unwrap_or(input.len());
        Ok((used, st == flate2::Status::StreamEnd))
    }
}

impl Stream for bzip2::Decompress {
    fn step(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<(usize, bool), String> {
        let before = self.total_in();
        let st = self.decompress_vec(input, out).map_err(|e| e.to_string())?;
        let used = usize::try_from(self.total_in() - before).unwrap_or(input.len());
        Ok((used, st == bzip2::Status::StreamEnd))
    }
}

/// Most plain bytes one pushed chunk may inflate to: gzip and zlib cannot
/// exceed ~1032:1, so a chunk of [`COMPRESSED_CHUNK`] bytes stays within it;
/// a bomb is an error, not an allocation.
const MAX_PUSH_OUTPUT: usize = 128 << 20;

/// A decoder of consecutive streams.
struct Multi<D: Stream> {
    make: fn() -> D,
    dec: D,
    /// Whether a stream has begun and not ended.
    open: bool,
}

impl<D: Stream> Multi<D> {
    fn new(make: fn() -> D) -> Self {
        Self {
            dec: make(),
            make,
            open: false,
        }
    }

    fn run(&mut self, codec: Codec, mut input: &[u8], out: &mut Vec<u8>) -> FormatResult<()> {
        let start = out.len();
        while !input.is_empty() {
            if out.len() - start > MAX_PUSH_OUTPUT {
                return Err(corrupt(codec, "a chunk inflates past the size limit"));
            }
            self.open = true;
            let (before_out, spare) = (out.len(), out.capacity() - out.len());
            if spare < (64 << 10) {
                out.reserve(input.len().saturating_mul(4).max(256 << 10));
            }
            let (used, ended) = self.dec.step(input, out).map_err(|e| corrupt(codec, e))?;
            input = &input[used.min(input.len())..];
            if ended {
                self.dec = (self.make)();
                self.open = false;
            } else if used == 0 && out.len() == before_out {
                return Err(corrupt(codec, "the stream made no progress"));
            }
        }
        Ok(())
    }

    fn finish(self, codec: Codec) -> FormatResult<()> {
        if self.open {
            return Err(corrupt(codec, "unexpected end of input stream"));
        }
        Ok(())
    }
}

/// Hadoop's `BlockDecompressorStream` framing.
struct BlockDec {
    codec: Codec,
    buf: Vec<u8>,
    /// Plain bytes the current block still owes.
    remaining: usize,
}

impl BlockDec {
    fn new(codec: Codec) -> Self {
        Self {
            codec,
            buf: Vec::new(),
            remaining: 0,
        }
    }

    fn take_u32(&mut self) -> Option<usize> {
        let b: [u8; 4] = self.buf.get(..4)?.try_into().ok()?;
        self.buf.drain(..4);
        Some(u32::from_be_bytes(b) as usize)
    }

    fn push(&mut self, input: &[u8], out: &mut Vec<u8>) -> FormatResult<()> {
        self.buf.extend_from_slice(input);
        loop {
            if self.remaining == 0 {
                let Some(n) = self.take_u32() else {
                    return Ok(());
                };
                if n > MAX_BLOCK {
                    return Err(corrupt(self.codec, format!("block of {n} bytes")));
                }
                self.remaining = n;
                continue;
            }
            // Peek the chunk length without consuming it until the chunk is whole.
            let Some(head) = self.buf.get(..4) else {
                return Ok(());
            };
            let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
            if len > MAX_BLOCK {
                return Err(corrupt(self.codec, format!("chunk of {len} bytes")));
            }
            if self.buf.len() < 4 + len {
                return Ok(());
            }
            let chunk = &self.buf[4..4 + len];
            let plain = match self.codec {
                Codec::Lz4 => lz4_flex::block::decompress(chunk, self.remaining)
                    .map_err(|e| corrupt(self.codec, e))?,
                _ => {
                    let declared =
                        snap::raw::decompress_len(chunk).map_err(|e| corrupt(self.codec, e))?;
                    if declared > self.remaining {
                        return Err(corrupt(self.codec, "chunk longer than its block"));
                    }
                    snap::raw::Decoder::new().decompress_vec(chunk)
                }
                .map_err(|e| corrupt(self.codec, e))?,
            };
            if plain.len() > self.remaining {
                return Err(corrupt(self.codec, "chunk longer than its block"));
            }
            self.remaining -= plain.len();
            out.extend_from_slice(&plain);
            self.buf.drain(..4 + len);
        }
    }

    fn finish(self) -> FormatResult<()> {
        if self.remaining != 0 || !self.buf.is_empty() {
            return Err(corrupt(self.codec, "unexpected end of input stream"));
        }
        Ok(())
    }
}

enum Enc {
    Gzip(flate2::write::GzEncoder<Vec<u8>>),
    Deflate(flate2::write::ZlibEncoder<Vec<u8>>),
    Bzip2(bzip2::write::BzEncoder<Vec<u8>>),
    Block { pending: Vec<u8>, out: Vec<u8> },
}

/// A streaming compressor.
pub(crate) struct Compressor {
    codec: Codec,
    enc: Enc,
}

impl Compressor {
    pub(crate) fn new(codec: Codec) -> Self {
        let enc = match codec {
            Codec::Gzip => Enc::Gzip(flate2::write::GzEncoder::new(
                Vec::new(),
                flate2::Compression::default(),
            )),
            Codec::Deflate => Enc::Deflate(flate2::write::ZlibEncoder::new(
                Vec::new(),
                flate2::Compression::default(),
            )),
            Codec::Bzip2 => Enc::Bzip2(bzip2::write::BzEncoder::new(
                Vec::new(),
                bzip2::Compression::default(),
            )),
            Codec::Lz4 | Codec::Snappy => Enc::Block {
                pending: Vec::new(),
                out: Vec::new(),
            },
        };
        Self { codec, enc }
    }

    fn cut(codec: Codec, plain: &[u8], out: &mut Vec<u8>) -> FormatResult<()> {
        if plain.is_empty() {
            return Ok(());
        }
        let packed = match codec {
            Codec::Lz4 => lz4_flex::block::compress(plain),
            _ => snap::raw::Encoder::new()
                .compress_vec(plain)
                .map_err(|e| corrupt(codec, e))?,
        };
        let (plain_len, packed_len) = (
            u32::try_from(plain.len()).map_err(|e| corrupt(codec, e))?,
            u32::try_from(packed.len()).map_err(|e| corrupt(codec, e))?,
        );
        out.extend_from_slice(&plain_len.to_be_bytes());
        out.extend_from_slice(&packed_len.to_be_bytes());
        out.extend_from_slice(&packed);
        Ok(())
    }

    pub(crate) fn write(&mut self, data: &[u8]) -> FormatResult<()> {
        let codec = self.codec;
        let io = |e: std::io::Error| corrupt(codec, e);
        match &mut self.enc {
            Enc::Gzip(e) => e.write_all(data).map_err(io),
            Enc::Deflate(e) => e.write_all(data).map_err(io),
            Enc::Bzip2(e) => e.write_all(data).map_err(io),
            Enc::Block { pending, out } => {
                pending.extend_from_slice(data);
                let mut done = 0;
                while pending.len() - done >= WRITE_BLOCK {
                    Self::cut(codec, &pending[done..done + WRITE_BLOCK], out)?;
                    done += WRITE_BLOCK;
                }
                pending.drain(..done);
                Ok(())
            }
        }
    }

    /// The compressed bytes produced so far.
    pub(crate) fn take(&mut self) -> Vec<u8> {
        match &mut self.enc {
            Enc::Gzip(e) => std::mem::take(e.get_mut()),
            Enc::Deflate(e) => std::mem::take(e.get_mut()),
            Enc::Bzip2(e) => std::mem::take(e.get_mut()),
            Enc::Block { out, .. } => std::mem::take(out),
        }
    }

    /// Bytes held (pending input and unread output).
    pub(crate) fn buffered(&self) -> usize {
        match &self.enc {
            Enc::Gzip(e) => e.get_ref().capacity(),
            Enc::Deflate(e) => e.get_ref().capacity(),
            Enc::Bzip2(e) => e.get_ref().capacity(),
            Enc::Block { pending, out } => pending.capacity() + out.capacity(),
        }
    }

    /// Ends the stream: the bytes not yet taken.
    pub(crate) fn finish(self) -> FormatResult<Vec<u8>> {
        let codec = self.codec;
        let io = |e: std::io::Error| corrupt(codec, e);
        match self.enc {
            Enc::Gzip(e) => e.finish().map_err(io),
            Enc::Deflate(e) => e.finish().map_err(io),
            Enc::Bzip2(e) => e.finish().map_err(io),
            Enc::Block { pending, mut out } => {
                Self::cut(codec, &pending, &mut out)?;
                Ok(out)
            }
        }
    }
}

/// An [`Encoder`] that compresses the bytes of another.
pub(crate) struct CompressedEncoder {
    inner: Box<dyn Encoder>,
    comp: Compressor,
    /// Compressed bytes not yet taken.
    ready: Vec<u8>,
    taken: u64,
}

impl CompressedEncoder {
    pub(crate) fn new(inner: Box<dyn Encoder>, codec: Codec) -> Self {
        Self {
            inner,
            comp: Compressor::new(codec),
            ready: Vec::new(),
            taken: 0,
        }
    }

    fn drain(&mut self) -> FormatResult<()> {
        while let Some(b) = self.inner.take_bytes() {
            self.comp.write(&b)?;
        }
        self.ready.extend(self.comp.take());
        Ok(())
    }
}

impl Encoder for CompressedEncoder {
    fn write(&mut self, batch: &arrow_array::RecordBatch) -> FormatResult<()> {
        self.inner.write(batch)?;
        self.drain()
    }

    fn flush(&mut self) -> FormatResult<()> {
        self.inner.flush()?;
        self.drain()
    }

    fn take_bytes(&mut self) -> Option<Bytes> {
        let out = std::mem::take(&mut self.ready);
        self.taken += out.len() as u64;
        (!out.is_empty()).then(|| Bytes::from(out))
    }

    fn finish(self: Box<Self>) -> FormatResult<WrittenFile> {
        let Self {
            mut inner,
            mut comp,
            mut ready,
            taken,
        } = *self;
        while let Some(b) = inner.take_bytes() {
            comp.write(&b)?;
        }
        let written = inner.finish()?;
        comp.write(&written.tail)?;
        let mut stats = written.statistics;
        ready.extend(comp.take());
        ready.extend(comp.finish()?);
        stats.bytes = Precision::Exact(taken + ready.len() as u64);
        Ok(WrittenFile::new(Bytes::from(ready), stats))
    }

    fn buffered_bytes(&self) -> usize {
        self.inner.buffered_bytes() + self.comp.buffered() + self.ready.capacity()
    }

    fn written_bytes(&self) -> u64 {
        self.taken + self.ready.len() as u64
    }
}
