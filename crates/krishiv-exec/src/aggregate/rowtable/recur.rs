// SPDX-License-Identifier: Apache-2.0
//! [`Recur`]: whether an append-only table's run starts are keys it has seen.
//!
//! An append-only table's [`Hll`](crate::rows::Hll) cannot say whether one
//! interval's run starts are new keys: its error over millions of keys is as
//! large as an interval's keys. TPC-H q18's `l_orderkey` never recurs (the
//! table stays append-only); TPC-DS q22's items recur every week.

/// Slots of the set (a quarter stay empty, so a probe ends).
const SLOTS: usize = 1 << 12;
/// `SLOTS - 1`, for a hash.
const SLOT_MASK: u64 = (1 << 12) - 1;
/// Keys the set holds before it samples more sparsely.
const FULL: usize = SLOTS * 3 / 4;
/// At level 0 one run start in 2^6 = 64 is sampled, by hash bits 16..
/// (which the HLL and the radix partitions do not use).
const BASE_BITS: u32 = 6;
/// The sparsest level: one run start in 2^(6 + 20).
const MAX_LEVEL: u32 = 20;
/// Sampled run starts an interval needs before it can call keys recurring:
/// at level 0 about 2,000 run starts.
const MIN_SAMPLED: usize = 32;

/// An exact set of sampled run-start hashes and, per adaptation interval,
/// how many sampled run starts it had already seen.
///
/// The set allocates its slots at [`start`](Self::start) (a table's first
/// [`SKETCH_ROWS`](super::SKETCH_ROWS) rows sample nothing). When it holds
/// [`FULL`] keys it samples at half the rate from then on and keeps only
/// the keys that rate would have taken (about half), so keys that stopped
/// coming give way to new ones: a driver whose first keys never recur
/// still learns the keys that do, and a driver with more recurring keys
/// than the set holds samples them sparsely enough to hold them.
#[derive(Debug)]
pub(super) struct Recur {
    slots: Box<[u64]>,
    len: usize,
    /// The sample rate is one run start in 2^(6 + level).
    level: u32,
    /// Sampled run starts in this interval, and those already in the set.
    pub(super) sampled: usize,
    pub(super) hits: usize,
}

impl Recur {
    /// Bytes of a started set.
    pub(super) const BYTES: usize = SLOTS * size_of::<u64>();

    pub(super) fn new() -> Self {
        Self {
            slots: Box::default(),
            len: 0,
            level: 0,
            sampled: 0,
            hits: 0,
        }
    }

    /// Whether the slots are allocated.
    pub(super) fn started(&self) -> bool {
        !self.slots.is_empty()
    }

    /// Allocates the slots; sampling starts.
    pub(super) fn start(&mut self) {
        if !self.started() {
            self.slots = vec![0; SLOTS].into_boxed_slice();
        }
    }

    pub(super) fn bytes(&self) -> usize {
        self.slots.len() * size_of::<u64>()
    }

    /// The sample mask over hash bits 16.. at the current level.
    pub(super) fn mask(&self) -> u64 {
        (1_u64 << (BASE_BITS + self.level)) - 1
    }

    /// Samples the run starts of `hashes` (`last`: the previous batch's
    /// last hash); nothing before [`start`](Self::start). At level 0 one
    /// hash in 64 is sampled, so a chunk of 8 holds none about 7 times in 8
    /// and is skipped after a vectorizable test (an all-rows scalar loop
    /// cost TPC-H q18 2% more instructions). The mask only grows stricter,
    /// so a chunk tested at the mask of its start misses no hash.
    pub(super) fn add_runs(&mut self, hashes: &[u64], last: Option<u64>) {
        const CHUNK: usize = 8;
        if !self.started() {
            return;
        }
        for (c, chunk) in hashes.chunks(CHUNK).enumerate() {
            let mask = self.mask();
            let sampled = |h: u64| (h >> 16) & mask == 0;
            if !chunk.iter().fold(false, |any, &h| any | sampled(h)) {
                continue;
            }
            for (j, &h) in chunk.iter().enumerate() {
                if !sampled(h) {
                    continue;
                }
                let i = c * CHUNK + j;
                let prev = match i.checked_sub(1) {
                    Some(p) => hashes.get(p).copied(),
                    None => last,
                };
                if prev != Some(h) {
                    self.add(h);
                }
            }
        }
    }

    /// Counts a run start, if the current level samples it.
    pub(super) fn add(&mut self, h: u64) {
        if !self.started() || (h >> 16) & self.mask() != 0 {
            return;
        }
        self.sampled += 1;
        if !self.insert(h) && self.level < MAX_LEVEL {
            self.sparser();
            if (h >> 16) & self.mask() == 0 {
                self.insert(h);
            }
        }
    }

    /// Looks `h` up and inserts it when new: false when the set is full.
    fn insert(&mut self, h: u64) -> bool {
        // 0 marks an empty slot.
        let key = h.max(1);
        let mut i = usize::try_from((h ^ (h >> 29)) & SLOT_MASK).unwrap_or(0);
        loop {
            match self.slots.get(i).copied() {
                Some(s) if s == key => {
                    self.hits += 1;
                    return true;
                }
                Some(0) => {
                    if self.len >= FULL {
                        return false;
                    }
                    self.slots[i] = key;
                    self.len += 1;
                    return true;
                }
                Some(_) => i = (i + 1) & usize::try_from(SLOT_MASK).unwrap_or(0),
                None => return true,
            }
        }
    }

    /// One level sparser: the keys the new rate does not sample go.
    fn sparser(&mut self) {
        self.level += 1;
        let mask = self.mask();
        let kept: Vec<u64> = self
            .slots
            .iter()
            .copied()
            .filter(|&k| k != 0 && (k >> 16) & mask == 0)
            .collect();
        self.slots.fill(0);
        self.len = 0;
        let (sampled, hits) = (self.sampled, self.hits);
        for k in kept {
            self.insert(k);
        }
        (self.sampled, self.hits) = (sampled, hits);
    }

    /// Over half of this interval's sampled run starts were seen before
    /// (and at least [`MIN_SAMPLED`] were sampled).
    pub(super) fn recurs(&self) -> bool {
        self.sampled >= MIN_SAMPLED && self.hits * 2 > self.sampled
    }

    /// Starts a new interval.
    pub(super) fn next_interval(&mut self) {
        (self.sampled, self.hits) = (0, 0);
    }

    #[cfg(test)]
    pub(super) fn state(&self) -> (usize, usize, usize, u32, &[u64]) {
        (self.sampled, self.hits, self.len, self.level, &self.slots)
    }
}
