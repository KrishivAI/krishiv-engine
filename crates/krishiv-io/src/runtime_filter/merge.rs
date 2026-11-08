// SPDX-License-Identifier: Apache-2.0
//! [`KeyFilter::merge`]: the filter of a producer of several partitions,
//! from the filters its partitions published (design P2-T177 §2.4).
//!
//! The merge keeps a superset of every input's keys, so a row either input
//! keeps passes the result:
//!
//! | part | rule |
//! |---|---|
//! | range | union; none when either has none |
//! | statistics `IN` list | union while it fits `max_values`, else none |
//! | exact sets | union while it fits `exact_max_values`; past it a Bloom filter of the planned size, else the range |
//! | Bloom filters | OR when both have the planned size (`KeyFilterOptions::bloom_bytes`), else the range; an exact set's members are added to the other side's Bloom |
//! | the range form (dense integers, or no other form) | absorbs the other side's form; a merged set that fills its range stays a set |
//! | NULLs pass | either |
//! | empty | both |
//!
//! A merged Bloom filter that passes more than `bloom_max_fpp` of random
//! hashes, measured from its bits, falls back to the range (Trino's rule
//! for a domain past its planned size). Every rule is commutative and
//! associative in the keys it passes, so partials merge in any order.

use super::bloom::RuntimeBloom;
use super::filter::{KeyFilter, KeyFilterOptions};
use super::key::KeyValue;
use super::member::{Class, Member};

impl KeyFilter {
    /// The filter of the union of `self`'s and `other`'s keys (see the
    /// module docs), within the caps of `options`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the two compare in different key types, or
    /// either holds a top-N's bound (bounds tighten, never merge).
    pub fn merge(&self, other: &Self, options: &KeyFilterOptions) -> crate::Result<Self> {
        self.clone().merge_into(other, options)
    }

    /// [`Self::merge`], taking `self` as the accumulator of a fold over many
    /// partials: its Bloom filter takes `other`'s keys in place, so folding
    /// `P` partials copies no Bloom of the planned size.
    ///
    /// # Errors
    /// As [`Self::merge`].
    pub fn merge_into(self, other: &Self, options: &KeyFilterOptions) -> crate::Result<Self> {
        if self.key_type != other.key_type {
            return Err(crate::error::internal(format!(
                "runtime filters of {} and {} merged",
                self.key_type, other.key_type
            )));
        }
        if self.bound.is_some() || other.bound.is_some() {
            return Err(crate::error::internal(
                "a top-N bound in a runtime filter merge".to_owned(),
            ));
        }
        let nulls_pass = self.nulls_pass || other.nulls_pass;
        let keys = self.keys.saturating_add(other.keys);
        if other.empty {
            return Ok(Self {
                nulls_pass,
                keys,
                ..self
            });
        }
        if self.empty {
            return Ok(Self {
                nulls_pass,
                keys,
                ..other.clone()
            });
        }
        let range = match (&self.range, &other.range) {
            (Some(a), Some(b)) => a.union(b),
            _ => None,
        };
        let values = match (&self.values, &other.values) {
            (Some(a), Some(b)) => {
                let mut v: Vec<KeyValue> = a.iter().chain(b).cloned().collect();
                v.sort_unstable();
                v.dedup();
                (v.len() <= options.max_values).then_some(v)
            }
            _ => None,
        };
        let class = Class::of(&self.key_type)
            .ok_or_else(|| crate::error::internal("unsupported runtime filter key".to_owned()))?;
        let key_type = self.key_type;
        let member = merge_members(self.member, &other.member, class, options);
        // A merged exact set stays one even when it fills its range: as the
        // range form it would absorb the next merge's sets, and the result
        // would depend on the order of the merges.
        let member = match member {
            Member::Bloom(b) if b.pass_rate() > options.bloom_max_fpp => Member::Range,
            m => m,
        };
        Ok(Self {
            key_type,
            range,
            values,
            member,
            nulls_pass,
            empty: false,
            keys,
            bound: None,
            // Partitioned builds plan no shared array (design P2b-R1 M5):
            // their partials hold no memory to carry.
            held: super::filter::Held::default(),
        })
    }
}

/// The planned Bloom size in bytes, in whole blocks as a builder makes it;
/// `None` when the producer sizes its Bloom for its own keys.
fn planned_bytes(options: &KeyFilterOptions) -> Option<usize> {
    // A shared array's size is one producer's (design P2b-R1 M5): partials
    // of several are never built, or merged into a Bloom, at that size.
    if options.shared_bloom_bytes().is_some() {
        return None;
    }
    let planned = usize::try_from(options.bloom_bytes?).unwrap_or(usize::MAX);
    Some(RuntimeBloom::with_bytes(planned.min(options.bloom_max_bytes)).num_bytes())
}

/// The row form of the union of `a`'s and `b`'s keys.
///
/// Only Bloom filters of the planned size take part: one sized for its own
/// keys (no plan, or a producer that dropped it) counts as the range form.
/// Otherwise whether a Bloom could take another side's keys would depend
/// on the order of the merges.
fn merge_members(a: Member, b: &Member, class: Class, options: &KeyFilterOptions) -> Member {
    let planned = planned_bytes(options);
    let unplanned = |m: &Member| matches!(m, Member::Bloom(b) if Some(b.num_bytes()) != planned);
    if unplanned(&a) || unplanned(b) {
        return Member::Range;
    }
    let add = |mut bloom: RuntimeBloom, set: &Member| {
        for (h, _) in set.members(class).unwrap_or_default() {
            bloom.insert_hash(h);
        }
        Member::Bloom(bloom)
    };
    match (a, b) {
        (Member::Range, _) | (_, Member::Range) => Member::Range,
        (Member::Bloom(mut bloom), Member::Bloom(y)) => {
            if bloom.union_with(y) {
                Member::Bloom(bloom)
            } else {
                Member::Range
            }
        }
        (Member::Bloom(bloom), set) => add(bloom, set),
        (set, Member::Bloom(bloom)) => add(bloom.clone(), &set),
        (x, y) => {
            let mut all = x.members(class).unwrap_or_default();
            all.extend(y.members(class).unwrap_or_default());
            all.sort_unstable_by(|p, q| p.1.cmp(&q.1));
            all.dedup_by(|p, q| p.1 == q.1);
            if all.len() <= options.exact_max_values {
                return Member::exact(class, all).unwrap_or(Member::Range);
            }
            match planned {
                Some(bytes) => {
                    let mut bloom = RuntimeBloom::with_bytes(bytes);
                    for (h, _) in &all {
                        bloom.insert_hash(*h);
                    }
                    Member::Bloom(bloom)
                }
                None => Member::Range,
            }
        }
    }
}

#[cfg(test)]
mod tests;
