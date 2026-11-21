// SPDX-License-Identifier: Apache-2.0
//! Spark's up-cast rule (`UpCastRule.canUpCast`): what a permanent view's read and a metric
//! view's dimension may convert to their stored types without an error.

use krishiv_types::DataType;

/// Spark's `UpCastRule.canUpCast`: `from` converts to `to` with no truncation, precision loss or runtime failure.
pub fn can_up_cast(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    // `numericPrecedence`: the integral and floating types widen in this order.
    let rank = |t: &T| match t {
        T::Byte => Some(0),
        T::Short => Some(1),
        T::Integer => Some(2),
        T::Long => Some(3),
        T::Float => Some(4),
        T::Double => Some(5),
        _ => None,
    };
    // `DecimalType.forType` of an integral type.
    let integral = |t: &T| match t {
        T::Byte => Some((3u8, 0u8)),
        T::Short => Some((5, 0)),
        T::Integer => Some((10, 0)),
        T::Long => Some((20, 0)),
        _ => None,
    };
    let fits = |from_null: bool, to_null: bool| !from_null || to_null;
    match (from, to) {
        _ if from == to => true,
        // `to.isWiderThan(from)`.
        (T::Decimal(a), T::Decimal(b)) => {
            b.precision() - b.scale() >= a.precision() - a.scale() && b.scale() >= a.scale()
        }
        (f, T::Decimal(d)) if integral(f).is_some() => {
            integral(f).is_some_and(|(p, s)| d.precision() - d.scale() >= p - s && d.scale() >= s)
        }
        // `from.isTighterThan(to)`.
        (T::Decimal(d), t) if integral(t).is_some() => {
            integral(t).is_some_and(|(p, s)| d.precision() - d.scale() <= p - s && d.scale() <= s)
        }
        (f, t) if rank(f).zip(rank(t)).is_some_and(|(f, t)| f < t) => true,
        (T::Date | T::TimestampNtz | T::Long, T::Timestamp)
        | (T::Date | T::Timestamp, T::TimestampNtz)
        | (T::Timestamp, T::Long)
        | (T::Null, _) => true,
        // An atomic type to a plain string.
        (f, T::String) => !f.is_nested(),
        (T::Array(a), T::Array(b)) => {
            fits(a.contains_null(), b.contains_null())
                && can_up_cast(a.element_type(), b.element_type())
        }
        (T::Map(a), T::Map(b)) => {
            fits(a.value_contains_null(), b.value_contains_null())
                && can_up_cast(a.key_type(), b.key_type())
                && can_up_cast(a.value_type(), b.value_type())
        }
        (T::Struct(a), T::Struct(b)) => {
            a.fields().len() == b.fields().len()
                && a.fields().iter().zip(b.fields()).all(|(f, t)| {
                    fits(f.nullable(), t.nullable()) && can_up_cast(f.data_type(), t.data_type())
                })
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_types::DecimalType;

    fn dec(p: u32, s: u32) -> DataType {
        DataType::Decimal(DecimalType::try_new(p, s).unwrap())
    }

    #[test]
    fn up_casts_follow_sparks_rule() {
        use DataType as T;
        assert!(can_up_cast(&T::Integer, &T::Long));
        assert!(can_up_cast(&T::Short, &T::Integer));
        assert!(!can_up_cast(&T::Long, &T::Integer));
        assert!(!can_up_cast(&T::String, &T::Integer));
        assert!(can_up_cast(&T::Integer, &T::String));
        assert!(can_up_cast(&T::Long, &T::Double));
        assert!(!can_up_cast(&T::Double, &T::Float));
        assert!(can_up_cast(&T::Integer, &dec(10, 0)));
        assert!(!can_up_cast(&T::Integer, &dec(9, 0)));
        assert!(can_up_cast(&dec(5, 0), &T::Integer));
        assert!(!can_up_cast(&dec(5, 1), &T::Integer));
        assert!(can_up_cast(&dec(5, 2), &dec(6, 2)));
        assert!(!can_up_cast(&dec(5, 2), &dec(5, 1)));
        assert!(can_up_cast(&T::Date, &T::Timestamp));
        assert!(!can_up_cast(&T::Timestamp, &T::Date));
        assert!(can_up_cast(&T::Null, &T::Integer));
    }

    #[test]
    fn nested_types_up_cast_element_wise() {
        use DataType as T;
        use krishiv_types::{ArrayType, MapType, StructField, StructType};
        let array = |t: T, n| T::Array(std::sync::Arc::new(ArrayType::new(t, n)));
        assert!(can_up_cast(
            &array(T::Integer, false),
            &array(T::Long, true)
        ));
        assert!(!can_up_cast(
            &array(T::Integer, true),
            &array(T::Long, false)
        ));
        assert!(!can_up_cast(
            &array(T::Long, true),
            &array(T::Integer, true)
        ));
        let map = |k: T, v: T, n| T::Map(std::sync::Arc::new(MapType::new(k, v, n)));
        assert!(can_up_cast(
            &map(T::Integer, T::Integer, true),
            &map(T::Long, T::Long, true)
        ));
        assert!(!can_up_cast(
            &map(T::Long, T::Integer, true),
            &map(T::Integer, T::Integer, true)
        ));
        assert!(!can_up_cast(
            &map(T::Integer, T::Integer, true),
            &map(T::Integer, T::Integer, false)
        ));
        let strukt = |fields: Vec<(&str, T)>| {
            T::Struct(StructType::new(
                fields
                    .into_iter()
                    .map(|(n, t)| StructField::new(n, t, true)),
            ))
        };
        assert!(can_up_cast(
            &strukt(vec![("a", T::Integer)]),
            &strukt(vec![("a", T::Long)])
        ));
        assert!(!can_up_cast(
            &strukt(vec![("a", T::Integer)]),
            &strukt(vec![("a", T::Integer), ("b", T::Integer)])
        ));
        assert!(!can_up_cast(&array(T::Integer, true), &T::String));
    }
}
