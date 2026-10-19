// SPDX-License-Identifier: Apache-2.0
//! [`RowPages`]: rows of one [`RowLayout`] in fixed-size pages, each with
//! its own heap for long strings, addressed by `u32` row ids.

use super::layout::{EncodedKeys, INLINE_MAX, RowLayout, view_len, word};
use crate::error::{Result, internal};

/// Target bytes of a full page's row region.
const PAGE_BYTES: usize = 256 << 10;
/// Fewest rows per page.
const MIN_PAGE_ROWS: usize = 64;
/// A page whose heap passed this many bytes takes no more rows, so one
/// page (one future spill frame) stays bounded.
const MAX_PAGE_HEAP: usize = 16 << 20;

#[derive(Debug)]
struct Page {
    rows: Vec<u8>,
    heap: Vec<u8>,
    len: usize,
}

/// Rows in pages: row id `page << shift | index in page`.
#[derive(Debug)]
pub struct RowPages {
    stride: usize,
    shift: u32,
    views: Vec<usize>,
    pages: Vec<Page>,
    len: usize,
}

impl RowPages {
    /// Empty pages for rows of `layout`.
    pub fn new(layout: &RowLayout) -> Self {
        let stride = layout.stride();
        let per_page = (PAGE_BYTES / stride).max(MIN_PAGE_ROWS);
        // The largest power of two that fits.
        let shift = usize::BITS - 1 - per_page.leading_zeros();
        Self {
            stride,
            shift,
            views: layout.views().to_vec(),
            pages: Vec::new(),
            len: 0,
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn page_rows(&self) -> usize {
        1 << self.shift
    }

    /// Bytes held (capacities).
    pub fn memory_size(&self) -> usize {
        self.pages
            .iter()
            .map(|p| p.rows.capacity() + p.heap.capacity())
            .sum::<usize>()
            + self.pages.capacity() * size_of::<Page>()
    }

    /// Appends row `i` of `keys` (encoded with this layout) and returns its
    /// row id; long strings are copied into the page's heap.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` once the row ids are exhausted (`2^32 - 1` slots).
    pub fn push(&mut self, keys: &EncodedKeys, i: usize) -> Result<u32> {
        // Validate before any mutation: a short row (an out-of-range `i`,
        // or keys of another layout) or a view outside its heap would
        // otherwise shift every later row of the page.
        let stride = self.stride;
        let row = keys.row(i);
        if row.len() != stride {
            return Err(internal(format!(
                "row {i} has {} bytes for a {stride}-byte layout",
                row.len()
            )));
        }
        let mut long = Vec::new();
        for &v in &self.views {
            let len = view_len(row, v);
            if len > INLINE_MAX {
                let from = usize::try_from(word(row, v + 8)).unwrap_or(usize::MAX);
                let bytes = keys
                    .heap()
                    .get(from..from.saturating_add(len))
                    .ok_or_else(|| internal("view outside its heap".to_owned()))?;
                long.push((v, bytes));
            }
        }
        let full = self
            .pages
            .last()
            .is_none_or(|p| p.len == self.page_rows() || p.heap.len() >= MAX_PAGE_HEAP);
        if full {
            let next = self.pages.len();
            if u32::try_from((next + 1) << self.shift).is_err() {
                return Err(internal("row ids exhausted in one row table".to_owned()));
            }
            self.pages.push(Page {
                rows: Vec::with_capacity(self.page_rows() * stride),
                heap: Vec::new(),
                len: 0,
            });
        }
        let page_index = self.pages.len() - 1;
        let shift = self.shift;
        let Some(page) = self.pages.last_mut() else {
            return Err(internal("no page to write".to_owned()));
        };
        let id = u32::try_from((page_index << shift) | page.len)
            .map_err(|_| internal("row id out of range".to_owned()))?;
        let start = page.rows.len();
        page.rows.extend_from_slice(row);
        for (v, bytes) in long {
            let at = page.heap.len() as u64;
            page.heap.extend_from_slice(bytes);
            page.rows[start + v + 8..start + v + 16].copy_from_slice(&at.to_le_bytes());
        }
        page.len += 1;
        self.len += 1;
        Ok(id)
    }

    /// Row `id`'s bytes and its page's heap (both empty if `id` is unknown).
    #[inline]
    pub fn row(&self, id: u32) -> (&[u8], &[u8]) {
        let id = id as usize;
        let at = (id & (self.page_rows() - 1)) * self.stride;
        match self.pages.get(id >> self.shift) {
            Some(p) => (
                p.rows.get(at..at + self.stride).unwrap_or_default(),
                &p.heap,
            ),
            None => (&[], &[]),
        }
    }

    /// Row `id`'s stored hash.
    #[inline]
    pub fn hash(&self, id: u32) -> u64 {
        word(self.row(id).0, 0)
    }

    /// Every row id in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.pages.iter().enumerate().flat_map(move |(p, page)| {
            (0..page.len).filter_map(move |r| u32::try_from((p << self.shift) | r).ok())
        })
    }
}
