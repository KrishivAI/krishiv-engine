// SPDX-License-Identifier: Apache-2.0
//! Sans-IO readers of text files: lines of a split ([`LineReader`]) and
//! whole files ([`WholeReader`]).

use std::ops::Range;

use bytes::Bytes;

use super::codec::{Codec, Decompressor};
use super::{CHUNK_BYTES, LineSep};
use crate::error::{FormatError, FormatResult, internal};
use crate::io::ByteRange;

/// What a reader produced.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ReadChunk {
    /// One line without its terminator, and the file offset it starts at.
    Line(Vec<u8>, u64),
    /// Push this range, then ask again.
    NeedData(Range<u64>),
    /// No more lines in the split.
    End,
}

/// The bytes of a file pushed in ranges, from `start` on.
#[derive(Debug)]
struct Feed {
    size: u64,
    /// File offset up to which bytes were pushed.
    fetched: u64,
    buf: Vec<u8>,
    /// File offset of `buf[0]`.
    buf_pos: u64,
    /// Index in `buf` of the next unread byte.
    cursor: usize,
    /// Decompresses pushed bytes (then `size` and `fetched` count compressed bytes).
    dec: Option<Decompressor>,
}

impl Feed {
    fn new(size: u64, start: u64) -> Self {
        Self {
            size,
            fetched: start,
            buf: Vec::new(),
            buf_pos: start,
            cursor: 0,
            dec: None,
        }
    }

    fn eof(&self) -> bool {
        self.fetched >= self.size
    }

    fn rest(&self) -> &[u8] {
        &self.buf[self.cursor..]
    }

    /// The range to push next, if the file has more.
    fn want(&self) -> Option<Range<u64>> {
        let chunk = if self.dec.is_some() {
            super::codec::COMPRESSED_CHUNK
        } else {
            CHUNK_BYTES
        };
        (!self.eof()).then(|| self.fetched..(self.fetched + chunk).min(self.size))
    }

    /// Hands over the bytes of a range asked for.
    ///
    /// # Errors
    /// An internal error for bytes that do not continue the buffer.
    fn push(&mut self, range: &Range<u64>, bytes: &Bytes) -> FormatResult<()> {
        if range.start != self.fetched || bytes.len() as u64 != range.end - range.start {
            return Err(FormatError::Other(internal(format!(
                "pushed {range:?} ({} bytes), expected bytes from {}",
                bytes.len(),
                self.fetched
            ))));
        }
        self.buf.drain(..self.cursor);
        self.buf_pos += self.cursor as u64;
        self.cursor = 0;
        match self.dec.as_mut() {
            Some(d) => {
                d.push(bytes, &mut self.buf)?;
                if range.end >= self.size {
                    d.finish(&mut self.buf)?;
                }
            }
            None => self.buf.extend_from_slice(bytes),
        }
        self.fetched = range.end;
        Ok(())
    }
}

/// The lines of one split of a file (see the module docs of
/// [`crate::text`]): a split starting after byte 0 skips its first line,
/// and lines are read while they start at or before the split's end.
#[derive(Debug)]
pub(crate) struct LineReader {
    feed: Feed,
    end: u64,
    sep: LineSep,
    /// Lines starting at or before this offset belong to the previous split.
    skip_to: Option<u64>,
    finished: bool,
}

/// Where a line ends in `rest`: its length and its terminator's length, or
/// `None` when more bytes are needed to tell.
fn terminator(rest: &[u8], sep: &LineSep, eof: bool) -> Option<(usize, usize)> {
    match sep {
        LineSep::Default => {
            let i = memchr::memchr2(b'\n', b'\r', rest)?;
            if rest[i] == b'\n' {
                return Some((i, 1));
            }
            match rest.get(i + 1) {
                Some(b'\n') => Some((i, 2)),
                Some(_) => Some((i, 1)),
                None if eof => Some((i, 1)),
                None => None,
            }
        }
        LineSep::Custom(sep) => memchr::memmem::find(rest, sep).map(|i| (i, sep.len())),
    }
}

impl LineReader {
    /// The lines of `split` in a file of `size` bytes. A file that cannot be
    /// split (`splittable` false) is read whole by the split at byte 0.
    pub(crate) fn new(size: u64, split: ByteRange, splittable: bool, sep: LineSep) -> Self {
        let (start, end, finished) = if splittable {
            (split.start, split.end.min(size), split.start >= size)
        } else {
            (0, u64::MAX, split.start != 0)
        };
        // A separator of several bytes may straddle the split's start: begin
        // early enough to see all of it.
        let overlap = match &sep {
            LineSep::Custom(s) => s.len().saturating_sub(1) as u64,
            LineSep::Default => 0,
        };
        let begin = if start == 0 {
            0
        } else {
            start.saturating_sub(overlap)
        };
        Self {
            feed: Feed::new(size, begin),
            end,
            sep,
            skip_to: (start != 0).then_some(start),
            finished,
        }
    }

    /// Reads a file of `codec` (if any): the whole file, decompressed, by the
    /// split at byte 0.
    #[must_use]
    pub(crate) fn compressed(mut self, codec: Option<Codec>) -> Self {
        if let Some(c) = codec {
            self.feed.dec = Some(Decompressor::new(c));
            self.end = u64::MAX;
        }
        self
    }

    /// The next line, or the bytes needed to find it.
    pub(crate) fn next_line(&mut self) -> FormatResult<ReadChunk> {
        loop {
            if self.finished {
                return Ok(ReadChunk::End);
            }
            let line_start = self.feed.buf_pos + self.feed.cursor as u64;
            let skipping = self.skip_to.is_some_and(|s| line_start <= s);
            if !skipping && line_start > self.end {
                self.finished = true;
                return Ok(ReadChunk::End);
            }
            let eof = self.feed.eof();
            let (len, term) = match terminator(self.feed.rest(), &self.sep, eof) {
                Some(found) => found,
                None if eof && self.feed.rest().is_empty() => {
                    self.finished = true;
                    return Ok(ReadChunk::End);
                }
                None if eof => (self.feed.rest().len(), 0),
                None => {
                    return match self.feed.want() {
                        Some(r) => Ok(ReadChunk::NeedData(r)),
                        None => Err(FormatError::Other(internal("text reader stalled"))),
                    };
                }
            };
            let line = self.feed.rest()[..len].to_vec();
            self.feed.cursor += len + term;
            if skipping {
                continue;
            }
            return Ok(ReadChunk::Line(line, line_start));
        }
    }

    /// Hands over the bytes of a range asked for by [`Self::next_line`].
    ///
    /// # Errors
    /// An internal error for bytes that do not continue the buffer.
    pub(crate) fn push(&mut self, range: &Range<u64>, bytes: &Bytes) -> FormatResult<()> {
        self.feed.push(range, bytes)
    }

    /// The next chunk a read will ask for, within `max_bytes`.
    pub(crate) fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        if self.finished || self.feed.fetched > self.end {
            return Vec::new();
        }
        self.feed
            .want()
            .map(|r| r.start..r.end.min(r.start.saturating_add(max_bytes)))
            .filter(|r| r.start < r.end)
            .into_iter()
            .collect()
    }

    /// Bytes held.
    pub(crate) fn buffered_bytes(&self) -> usize {
        self.feed.buf.capacity()
    }
}

/// A whole file, for records that may span lines: read only by the split
/// that starts at byte 0.
#[derive(Debug)]
pub(crate) struct WholeReader {
    feed: Feed,
    taken: bool,
}

impl WholeReader {
    /// The file of `size` bytes, if `split` starts at byte 0.
    pub(crate) fn new(size: u64, split: ByteRange) -> Self {
        Self {
            feed: Feed::new(size, 0),
            taken: split.start != 0,
        }
    }

    /// Reads a file of `codec` (if any), decompressing it.
    #[must_use]
    pub(crate) fn compressed(mut self, codec: Option<Codec>) -> Self {
        if let Some(c) = codec {
            self.feed.dec = Some(Decompressor::new(c));
        }
        self
    }

    /// The next range to push, `None` once the file is read.
    pub(crate) fn need(&self) -> Option<Range<u64>> {
        if self.taken { None } else { self.feed.want() }
    }

    /// Hands over the bytes of [`Self::need`].
    ///
    /// # Errors
    /// An internal error for bytes that do not continue the file.
    pub(crate) fn push(&mut self, range: &Range<u64>, bytes: &Bytes) -> FormatResult<()> {
        self.feed.push(range, bytes)
    }

    /// The file's bytes once all are pushed (once; later calls `None`).
    pub(crate) fn take(&mut self) -> Option<Vec<u8>> {
        if self.taken || !self.feed.eof() {
            return None;
        }
        self.taken = true;
        Some(std::mem::take(&mut self.feed.buf))
    }

    /// The next chunk a read will ask for, within `max_bytes`.
    pub(crate) fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        self.need()
            .map(|r| r.start..r.end.min(r.start.saturating_add(max_bytes)))
            .filter(|r| r.start < r.end)
            .into_iter()
            .collect()
    }

    /// Bytes held.
    pub(crate) fn buffered_bytes(&self) -> usize {
        self.feed.buf.capacity()
    }
}
