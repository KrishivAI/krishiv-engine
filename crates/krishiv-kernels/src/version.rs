// SPDX-License-Identifier: Apache-2.0
//! The Spark version whose observable behaviour kernels reproduce (ADR 0008).
//!
//! Values are identical across the supported versions; error classes and
//! messages are not. Each difference below was observed by running the
//! listed Spark releases:
//!
//! | Situation | 4.0 | 4.1 | 4.2 |
//! |---|---|---|---|
//! | `ARITHMETIC_OVERFLOW` text of `int`/`bigint` `+ - *`, unary `-`, `abs` | `integer overflow` / `long overflow` (`byte`/`short` for unary) | as 4.0 | `overflow` |
//! | `%`, `pmod`, `try_mod` by zero | `DIVIDE_BY_ZERO` | `REMAINDER_BY_ZERO` | `REMAINDER_BY_ZERO` |
//! | `upper`/`lower` of the 56 code points that gained a case mapping in Unicode 17 | unchanged | unchanged | mapped |

/// A supported Spark release line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[non_exhaustive]
pub enum SparkVersion {
    /// Spark 4.0.x.
    V4_0,
    /// Spark 4.1.x.
    V4_1,
    /// Spark 4.2.x (the default; the vendored error registry's version).
    #[default]
    V4_2,
}

/// The width of a Spark integral type, for version-specific messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntWidth {
    /// `tinyint`.
    Byte,
    /// `smallint`.
    Short,
    /// `int`.
    Int,
    /// `bigint`.
    Long,
}

impl SparkVersion {
    /// Every supported version, oldest first.
    pub const ALL: [Self; 3] = [Self::V4_0, Self::V4_1, Self::V4_2];

    /// The `ARITHMETIC_OVERFLOW` message of an exact integral operation on
    /// `width` (`MathUtils.*Exact` / `Math.*Exact`): the JDK's text before
    /// 4.2, canonicalized to `overflow` since.
    #[must_use]
    pub const fn overflow_message(self, width: IntWidth) -> &'static str {
        match (self, width) {
            (Self::V4_2, _) => "overflow",
            (_, IntWidth::Byte) => "byte overflow",
            (_, IntWidth::Short) => "short overflow",
            (_, IntWidth::Int) => "integer overflow",
            (_, IntWidth::Long) => "long overflow",
        }
    }

    /// Whether a remainder by zero raises `REMAINDER_BY_ZERO` (4.1+) rather
    /// than `DIVIDE_BY_ZERO`.
    #[must_use]
    pub const fn has_remainder_by_zero(self) -> bool {
        !matches!(self, Self::V4_0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_per_version() {
        use IntWidth::{Byte, Int, Long, Short};
        let v40: Vec<_> = [Byte, Short, Int, Long]
            .map(|w| SparkVersion::V4_0.overflow_message(w))
            .into();
        assert_eq!(
            v40,
            [
                "byte overflow",
                "short overflow",
                "integer overflow",
                "long overflow"
            ]
        );
        assert_eq!(SparkVersion::V4_1.overflow_message(Long), "long overflow");
        assert_eq!(SparkVersion::default().overflow_message(Int), "overflow");
        let rbz: Vec<_> = SparkVersion::ALL
            .map(SparkVersion::has_remainder_by_zero)
            .into();
        assert_eq!(rbz, [false, true, true]);
    }
}
