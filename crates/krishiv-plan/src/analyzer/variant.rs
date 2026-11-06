// SPDX-License-Identifier: Apache-2.0
//! The `VARIANT` rules of the analyzer: which casts into and out of a
//! variant resolve (Spark 4.2's `Cast.canCast` / `canAnsiCast`:
//! `VariantGet.checkDataType`, structs and maps only out of a variant).

use krishiv_functions::registry::variant_castable;
use krishiv_types::DataType;

use super::coerce::BadCast;

/// Whether `from` casts to `to` when either is `VARIANT` (and they differ):
/// out of a variant to any type `variant_get` returns, into a variant from
/// an atomic type or an array of them (a struct or map needs
/// `to_variant_object`).
pub(super) fn can_cast(from: &DataType, to: &DataType) -> Result<(), BadCast> {
    let ok = if *from == DataType::Variant {
        variant_castable(to, true)
    } else {
        variant_castable(from, false)
    };
    if ok { Ok(()) } else { Err(BadCast::Without) }
}

/// The atomic types a `STRING` widens to when it meets them (ANSI
/// `findWiderTypeForString`'s `(StringType, AtomicType)` case for the
/// non-numeric types; `VARIANT` is an `AtomicType` in Spark).
pub(super) fn string_promotes_to(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Boolean
            | DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
            | DataType::Binary
            | DataType::Time { .. }
            | DataType::Variant
    )
}
