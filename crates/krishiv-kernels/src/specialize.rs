// SPDX-License-Identifier: Apache-2.0
//! Opt-in has-nulls specialization, for kernels whose **work** depends on
//! NULLs.
//!
//! Elementwise arithmetic and comparisons compute through NULL slots, so a
//! has-nulls axis buys nothing there (measured ~1.0x on the same data) and
//! they do not use it. Use this module when NULL rows change what a kernel
//! does: per-row work that is costly or must not run on a NULL row (string
//! functions, regexes, casts that parse, division that must skip NULL
//! divisors before choosing a path). Implement [`NullsKernel`] once with a
//! `const NULLS: bool` parameter and call [`by_nulls`]; it picks the
//! monomorph once per batch.
//!
//! ```
//! use arrow_array::{Array, Int64Array};
//! use arrow_buffer::NullBuffer;
//! use krishiv_kernels::specialize::{NullsKernel, by_nulls};
//!
//! /// Sums an expensive per-row function over the non-NULL rows only.
//! struct SumValid<'a>(&'a [i64]);
//!
//! impl NullsKernel for SumValid<'_> {
//!     type Output = i64;
//!     fn run<const NULLS: bool>(self, nulls: Option<&NullBuffer>) -> i64 {
//!         let cost = |v: i64| v.wrapping_mul(v); // stands in for real work
//!         match nulls.filter(|_| NULLS) {
//!             // Visit only valid rows.
//!             Some(n) => n.valid_indices().map(|i| cost(self.0[i])).sum(),
//!             // No validity at all: a straight loop.
//!             None => self.0.iter().map(|&v| cost(v)).sum(),
//!         }
//!     }
//! }
//!
//! let a = Int64Array::from(vec![Some(1), None, Some(3)]);
//! assert_eq!(by_nulls(a.nulls(), SumValid(a.values())), 10);
//! let b = Int64Array::from(vec![1, 2, 3]);
//! assert_eq!(by_nulls(b.nulls(), SumValid(b.values())), 14);
//! ```

use arrow_buffer::NullBuffer;

/// A kernel body with a has-nulls axis.
pub trait NullsKernel {
    /// The kernel's result.
    type Output;
    /// Runs the kernel. With `NULLS == false`, `nulls` is `None`; with
    /// `NULLS == true` it is the validity, which has at least one NULL.
    fn run<const NULLS: bool>(self, nulls: Option<&NullBuffer>) -> Self::Output;
}

/// Runs `kernel` with `NULLS = true` when `nulls` has a NULL, else with
/// `NULLS = false` (an all-valid bitmap counts as no NULLs).
pub fn by_nulls<K: NullsKernel>(nulls: Option<&NullBuffer>, kernel: K) -> K::Output {
    match nulls.filter(|n| n.null_count() > 0) {
        Some(n) => kernel.run::<true>(Some(n)),
        None => kernel.run::<false>(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reports which monomorph ran.
    struct Which;

    impl NullsKernel for Which {
        type Output = (bool, bool);
        fn run<const NULLS: bool>(self, nulls: Option<&NullBuffer>) -> (bool, bool) {
            (NULLS, nulls.is_some())
        }
    }

    #[test]
    fn picks_the_monomorph_once_per_batch() {
        assert_eq!(by_nulls(None, Which), (false, false));
        let all_valid = NullBuffer::new_valid(3);
        assert_eq!(by_nulls(Some(&all_valid), Which), (false, false));
        let some_null = NullBuffer::from(vec![true, false]);
        assert_eq!(by_nulls(Some(&some_null), Which), (true, true));
    }
}
