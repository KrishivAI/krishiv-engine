// SPDX-License-Identifier: Apache-2.0
//! Sans-IO readers of text files: lines of a split ([`LineReader`]) and
//! whole files ([`WholeReader`]).

use std::ops::Range;

use bytes::Bytes;

use super::codec::{Codec, Decompressor, MAX_PUSH_OUTPUT};
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
    /// The error that ended a compressed stream early. The plain bytes decoded
    /// before it stay in `buf` (Hadoop's line reader returns the lines read
    /// before its codec throws), and the feed counts as fully fetched.
    broken: Option<FormatError>,
    /// Plain bytes decoded from the last pushed range so far.
    burst: usize,
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
            broken: None,
            burst: 0,
        }
    }

    fn eof(&self) -> bool {
        self.fetched >= self.size && !self.pending()
    }

    /// Whether pushed bytes are still to be decompressed.
    fn pending(&self) -> bool {
        self.dec.as_ref().is_some_and(Decompressor::pending)
    }

    /// Bytes held: the plain buffer and the decoder's.
    fn buffered(&self) -> usize {
        self.buf.capacity() + self.dec.as_ref().map_or(0, Decompressor::buffered)
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
        (self.fetched < self.size).then(|| self.fetched..(self.fetched + chunk).min(self.size))
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
        self.compact();
        self.fetched = range.end;
        self.burst = 0;
        if self.dec.is_some() {
            return self.decode(|d, buf| d.push(bytes, buf));
        }
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// Decompresses the next piece of the bytes pushed, if any are left:
    /// whether there was one.
    fn resume(&mut self) -> FormatResult<bool> {
        if !self.pending() {
            return Ok(false);
        }
        self.compact();
        self.decode(Decompressor::resume)?;
        Ok(true)
    }

    /// Drops the bytes read from the buffer.
    fn compact(&mut self) {
        self.buf.drain(..self.cursor);
        self.buf_pos += self.cursor as u64;
        self.cursor = 0;
    }

    /// Runs one step of the decompressor, ending the stream after the last.
    fn decode(
        &mut self,
        step: impl FnOnce(&mut Decompressor, &mut Vec<u8>) -> FormatResult<()>,
    ) -> FormatResult<()> {
        let Some(d) = self.dec.as_mut() else {
            return Ok(());
        };
        let before = self.buf.len();
        let mut res = step(d, &mut self.buf);
        self.burst += self.buf.len() - before;
        if res.is_ok() {
            // One range that inflates to one record past the limit, or to a
            // whole file past it: a bomb is an error, not an allocation.
            if self.burst.min(self.buf.len()) > MAX_PUSH_OUTPUT {
                res = Err(FormatError::Corrupt(
                    "compressed file: a chunk inflates past the size limit".to_owned(),
                ));
            } else if self.fetched >= self.size && !d.pending() {
                res = d.finish(&mut self.buf);
            }
        }
        if let Err(e) = res {
            if !matches!(e, FormatError::Corrupt(_)) {
                return Err(e);
            }
            self.dec = None;
            self.broken = Some(e);
            self.fetched = self.size;
        }
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
    /// A broken compressed stream ends the file after its complete lines
    /// instead of failing (schema inference under `ignoreCorruptFiles`).
    lenient: bool,
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
            lenient: false,
        }
    }

    /// With `lenient`, a compressed stream that breaks ends the file after the
    /// lines decoded before the break, as if it ended there; otherwise the break
    /// is raised once those lines are read.
    #[must_use]
    pub(crate) fn lenient(mut self, lenient: bool) -> Self {
        self.lenient = lenient;
        self
    }

    /// The end of the lines: [`ReadChunk::End`], or the error that broke the
    /// stream. The partial line after the last terminator is dropped with it.
    fn end(&mut self) -> FormatResult<ReadChunk> {
        self.finished = true;
        match self.feed.broken.take() {
            Some(e) if self.lenient => {
                tracing::warn!(error = %e, "a compressed file ends early; kept the lines read before the break");
                Ok(ReadChunk::End)
            }
            Some(e) => Err(e),
            None => Ok(ReadChunk::End),
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
            // After a break the bytes end mid-stream: a lone `\r` there is
            // not known to end its line.
            let clean_eof = eof && self.feed.broken.is_none();
            let (len, term) = match terminator(self.feed.rest(), &self.sep, clean_eof) {
                Some(found) => found,
                None if eof && (self.feed.rest().is_empty() || !clean_eof) => return self.end(),
                None if eof => (self.feed.rest().len(), 0),
                None => {
                    if self.feed.resume()? {
                        continue;
                    }
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
        self.feed.buffered()
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
        self.feed.push(range, bytes)?;
        while self.feed.resume()? {}
        // A record may span the rest of the file: a broken stream fails it.
        match self.feed.broken.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
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
        self.feed.buffered()
    }
}
