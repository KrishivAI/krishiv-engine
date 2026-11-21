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
/// [`FULL`] keys it first drops the keys last sampled before the previous
/// interval, so keys that stopped coming give way to new ones whatever the
/// driver read before. Only when that frees less than a quarter of [`FULL`]
/// (the last two intervals alone hold more sampled keys) does it sample at
/// half the rate from then on, keeping the keys that rate would have taken.
/// The rate so follows the distinct keys of two intervals, not the driver's
/// history: a unique stream of an interval of 1M run starts settles near
/// level 4 (about 1,000 sampled keys an interval), where an interval of
/// 4-row runs still samples over 200 run starts. The level never falls: after
/// intervals of about 1M distinct run starts, recurring keys in runs longer
/// than about 32 rows are rarely sampled and their table stays append-only.
#[derive(Debug)]
pub(super) struct Recur {
    slots: Box<[u64]>,
    /// The interval each slot's key was last sampled in.
    stamps: Box<[u16]>,
    /// This interval's stamp (never 0, which no slot holds once filled).
    /// It wraps to 1 after 65,535 intervals: a key last stamped 65,535
    /// intervals ago then reads as stamped in this one (counted as touched
    /// and kept as fresh). At a million rows an interval that is 65 billion
    /// rows of one driver, so it is left as is.
    epoch: u16,
    /// Distinct keys of the set sampled in this interval.
    touched: usize,
    len: usize,
    /// The sample rate is one run start in 2^(6 + level).
    level: u32,
    /// Sampled run starts in this interval, and those already in the set.
    pub(super) sampled: usize,
    pub(super) hits: usize,
}

impl Recur {
    /// Bytes of a started set.
    pub(super) const BYTES: usize = SLOTS * (size_of::<u64>() + size_of::<u16>());

    pub(super) fn new() -> Self {
        Self {
            slots: Box::default(),
            stamps: Box::default(),
            epoch: 1,
            touched: 0,
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
            self.stamps = vec![0; SLOTS].into_boxed_slice();
        }
    }

    pub(super) fn bytes(&self) -> usize {
        self.slots.len() * size_of::<u64>() + self.stamps.len() * size_of::<u16>()
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
        if !self.insert(h, self.epoch) {
            self.make_room();
            if (h >> 16) & self.mask() == 0 {
                self.insert(h, self.epoch);
            }
        }
    }

    /// The previous interval's stamp.
    fn previous(&self) -> u16 {
        match self.epoch.wrapping_sub(1) {
            0 => u16::MAX,
            e => e,
        }
    }

    /// A full set drops the keys last sampled before the previous interval;
    /// when that frees less than a quarter of [`FULL`], it also goes one
    /// level sparser and drops the keys the new rate does not sample.
    fn make_room(&mut self) {
        let (epoch, previous) = (self.epoch, self.previous());
        let fresh = |s: u16| s == epoch || s == previous;
        let kept = self
            .slots
            .iter()
            .zip(self.stamps.iter())
            .filter(|&(&k, &s)| k != 0 && fresh(s))
            .count();
        let sparser = self.len - kept < FULL / 4 && self.level < MAX_LEVEL;
        if sparser {
            self.level += 1;
        }
        let mask = self.mask();
        self.retain(|k, s| fresh(s) && (!sparser || (k >> 16) & mask == 0));
    }

    /// The slot a key's probe starts at.
    fn home(key: u64) -> usize {
        usize::try_from((key ^ (key >> 29)) & SLOT_MASK).unwrap_or(0)
    }

    /// Keeps the keys `keep` takes (by key and stamp), in place: each
    /// removed key's slot is refilled by backward-shift deletion, so the
    /// probes of the keys left stay unbroken and no second set is needed.
    fn retain(&mut self, keep: impl Fn(u64, u16) -> bool) {
        let mask = usize::try_from(SLOT_MASK).unwrap_or(0);
        // Starting just after an empty slot, no probe run crosses the start,
        // and a shift only moves a later key of a run into the slot being
        // looked at, which is looked at again.
        let Some(empty) = self.slots.iter().position(|&k| k == 0) else {
            return;
        };
        let mut i = (empty + 1) & mask;
        for _ in 0..SLOTS {
            loop {
                let key = self.slots[i];
                if key == 0 || keep(key, self.stamps[i]) {
                    break;
                }
                self.remove(i);
            }
            i = (i + 1) & mask;
        }
        self.touched = self
            .stamps
            .iter()
            .zip(self.slots.iter())
            .filter(|&(&s, &k)| k != 0 && s == self.epoch)
            .count();
    }

    /// Removes the key in slot `i` (backward-shift deletion).
    fn remove(&mut self, mut i: usize) {
        let mask = usize::try_from(SLOT_MASK).unwrap_or(0);
        let mut j = i;
        loop {
            j = (j + 1) & mask;
            let key = self.slots[j];
            if key == 0 {
                break;
            }
            // The key at `j` may move to `i` when its home is not in the
            // cyclic range (i, j].
            let k = Self::home(key);
            let stays = if i <= j {
                i < k && k <= j
            } else {
                i < k || k <= j
            };
            if !stays {
                self.slots[i] = key;
                self.stamps[i] = self.stamps[j];
                i = j;
            }
        }
        self.slots[i] = 0;
        self.stamps[i] = 0;
        self.len -= 1;
    }

    /// Looks `h` up and inserts it when new, stamped `stamp`: false when the
    /// set is full. A key first stamped in this interval counts as touched.
    fn insert(&mut self, h: u64, stamp: u16) -> bool {
        // 0 marks an empty slot.
        let key = h.max(1);
        // The slot comes from the stored key, so a hash of 0 (stored as 1)
        // probes where the key is found again.
        let mut i = Self::home(key);
        loop {
            match self.slots.get(i).copied() {
                Some(s) if s == key => {
                    self.hits += 1;
                    self.stamp(i, stamp);
                    return true;
                }
                Some(0) => {
                    if self.len >= FULL {
                        return false;
                    }
                    self.slots[i] = key;
                    self.len += 1;
                    self.stamp(i, stamp);
                    return true;
                }
                Some(_) => i = (i + 1) & usize::try_from(SLOT_MASK).unwrap_or(0),
                None => return true,
            }
        }
    }

    fn stamp(&mut self, i: usize, stamp: u16) {
        if let Some(s) = self.stamps.get_mut(i)
            && *s != stamp
        {
            *s = stamp;
            if stamp == self.epoch {
                self.touched += 1;
            }
        }
    }

    /// Over half of this interval's sampled run starts were seen before
    /// (and at least [`MIN_SAMPLED`] were sampled).
    pub(super) fn recurs(&self) -> bool {
        self.sampled >= MIN_SAMPLED && self.hits * 2 > self.sampled
    }

    /// The distinct keys of this interval, estimated from those sampled.
    pub(super) fn interval_keys(&self) -> usize {
        self.touched << (BASE_BITS + self.level)
    }

    /// The keys recur ([`recurs`](Self::recurs)) and this interval's
    /// distinct keys fit in `keys`: an index that cannot hold them clears
    /// all the time (TPC-H q13 at SF100: ~10M keys, each about once a
    /// driver's million rows), and appending costs less.
    ///
    /// Limit: it reads one interval's keys. Keys that recur only over a
    /// longer period (2M keys in 4-row runs, 250K of them an interval, each
    /// back every 8 intervals) pass it, though an index that fits one
    /// interval's keys holds none of them when they come back; the table then
    /// pays for lookups that miss (about q13's 1.05x) until the keys recur
    /// within an interval or stop recurring. No measured query does this.
    pub(super) fn recurs_within(&self, keys: usize) -> bool {
        self.recurs() && self.interval_keys() <= keys
    }

    /// Starts a new interval.
    pub(super) fn next_interval(&mut self) {
        (self.sampled, self.hits, self.touched) = (0, 0, 0);
        self.epoch = self.epoch.wrapping_add(1).max(1);
    }

    #[cfg(test)]
    pub(super) fn state(&self) -> (usize, usize, usize, u32, &[u64]) {
        (self.sampled, self.hits, self.len, self.level, &self.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::{FULL, Recur};

    /// Before `start` the set has no slots: `retain` keeps nothing to keep and an insert
    /// stores nothing, so an unstarted set stays empty and costs no memory.
    #[test]
    fn an_unstarted_set_stays_empty() {
        let mut r = Recur::new();
        r.retain(|_, _| false);
        assert!(r.insert(7, 1));
        assert_eq!((r.len, r.hits, r.touched, r.bytes()), (0, 0, 0, 0));
        assert!(!r.started());
    }

    /// After an in-place `retain` every kept key is found again where its
    /// probe starts (a hit) and no dropped key is (a new insert), with keys
    /// that wrap around the end of the slots and the hash 0 among them.
    #[test]
    fn retain_keeps_every_kept_key_reachable() {
        let h = |k: u64| {
            let mut z = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z ^ (z >> 31)
        };
        for seed in 0..20_u64 {
            let mut r = Recur::new();
            r.start();
            let mut keys: Vec<u64> = (0..u64::try_from(FULL).unwrap())
                .map(|k| h(k ^ (seed << 32)))
                .collect();
            keys[0] = 0;
            for (n, &k) in keys.iter().enumerate() {
                assert!(r.insert(k, 1 + u16::try_from(n % 3).unwrap()));
            }
            assert_eq!(r.len, FULL);
            r.retain(|k, s| s != 2 && k % 5 != 0);
            let kept: Vec<u64> = keys
                .iter()
                .enumerate()
                .filter(|&(n, &k)| n % 3 != 1 && k.max(1) % 5 != 0)
                .map(|(_, &k)| k)
                .collect();
            assert_eq!(r.len, kept.len(), "seed {seed}");
            let hits = r.hits;
            for &k in &kept {
                assert!(r.insert(k, 1));
            }
            assert_eq!(
                r.hits - hits,
                kept.len(),
                "seed {seed}: a kept key was lost"
            );
            assert_eq!(
                r.len,
                kept.len(),
                "seed {seed}: a kept key was stored twice"
            );
        }
    }
}
