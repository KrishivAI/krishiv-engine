// SPDX-License-Identifier: Apache-2.0
//! The collation key of a string (ADR 0043, design `P2-collations.md` section 3.3): bytes whose
//! equality and byte order are the collation's, so every operator that compares bytes
//! (hash, sort, join, set operations) runs on the key and carries the original value.
//!
//! `UTF8_LCASE` is Spark's `CollationAwareUTF8String.lowerCaseCodePoints`: the simple lower case
//! of each code point (ICU's, not the JDK's; generated from Spark, `case_table.rs`), `U+0130` to
//! `i` + `U+0307`, final sigma to sigma, with a byte path for ASCII. `_RTRIM` first drops the trailing
//! spaces (`U+0020` only). `UTF8_BINARY` is the bytes. ICU collations have no key here (unit 6): the
//! analyzer refuses them before a key is asked for.

use std::sync::Arc;

use arrow_array::builder::BinaryViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, MapArray, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as ArrowType, Field, Fields};
use krishiv_types::{Collation, CollationFamily, DataType};

use super::case_table::COLLATION_LOWER;
use crate::error::KernelError;

/// The code point that lowers to two code points.
const I_DOT: char = '\u{130}';

/// Spark's `lowerCaseCodePoints` of one code point, appended to `out`.
fn push_lower(c: char, out: &mut Vec<u8>) {
    let cp = u32::from(c);
    if c == I_DOT {
        out.extend_from_slice("i\u{307}".as_bytes());
        return;
    }
    let at = COLLATION_LOWER.partition_point(|(first, ..)| *first <= cp);
    let mapped = at
        .checked_sub(1)
        .and_then(|i| COLLATION_LOWER.get(i))
        .filter(|(first, last, step, _)| {
            cp <= *last && cp.wrapping_sub(*first).checked_rem(*step) == Some(0)
        })
        .and_then(|(.., delta)| cp.checked_add_signed(*delta))
        .and_then(char::from_u32)
        .unwrap_or(c);
    let mut buf = [0; 4];
    out.extend_from_slice(mapped.encode_utf8(&mut buf).as_bytes());
}

/// Appends the key of `s` under `collation`.
///
/// # Errors
/// `INTERNAL_ERROR` for an ICU collation, which has no key here (the analyzer refuses it first).
pub fn push_collation_key(
    collation: Collation,
    s: &str,
    out: &mut Vec<u8>,
) -> Result<(), KernelError> {
    let s = if collation.rtrim() {
        s.trim_end_matches(' ')
    } else {
        s
    };
    match collation.family() {
        CollationFamily::Utf8Binary => out.extend_from_slice(s.as_bytes()),
        CollationFamily::Utf8Lcase => {
            if s.is_ascii() {
                out.extend(s.bytes().map(|b| b.to_ascii_lowercase()));
            } else {
                s.chars().for_each(|c| push_lower(c, out));
            }
        }
        CollationFamily::Icu { ci, ai } => icu_key(collation, ci, ai, s, out)?,
    }
    Ok(())
}

/// The ICU sort key of `s`: the collator of the collation's locale at Spark's strength (`_CI`
/// secondary, `_AI` primary, `_CS_AI` primary with the case level on), kept per thread.
#[cfg(feature = "collations-icu")]
fn icu_key(
    collation: Collation,
    ci: bool,
    ai: bool,
    s: &str,
    out: &mut Vec<u8>,
) -> Result<(), KernelError> {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use icu_collator::options::{CaseLevel, CollatorOptions, Strength};
    use icu_collator::{Collator, CollatorBorrowed, CollatorPreferences};
    use icu_locale_core::Locale;

    thread_local! {
        static COLLATORS: RefCell<HashMap<u32, Option<CollatorBorrowed<'static>>>> = RefCell::new(HashMap::new());
    }
    COLLATORS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let collator = cache.entry(collation.id()).or_insert_with(|| {
            let name = collation.icu_locale()?;
            // Spark spells the locale `language[_Script][_ALPHA3]`; ICU reads the alpha-2 region
            // (`fr_CAN` is `fr-CA`, whose French sorts the accents backwards).
            let locale = if name == "UNICODE" {
                Locale::UNKNOWN
            } else {
                let mut tag = String::new();
                for (i, part) in name.split('_').enumerate() {
                    if i > 0 {
                        tag.push('-');
                    }
                    if part.len() == 3 && i > 0 && part.chars().all(|c| c.is_ascii_uppercase()) {
                        tag.push_str(region_alpha2(part)?);
                    } else {
                        tag.push_str(part);
                    }
                }
                tag.parse::<Locale>().ok()?
            };
            let mut options = CollatorOptions::default();
            let (strength, case_level) = match (ci, ai) {
                (false, false) => (Strength::Tertiary, CaseLevel::Off),
                (true, false) => (Strength::Secondary, CaseLevel::Off),
                (true, true) => (Strength::Primary, CaseLevel::Off),
                (false, true) => (Strength::Primary, CaseLevel::On),
            };
            options.strength = Some(strength);
            options.case_level = Some(case_level);
            Collator::try_new(CollatorPreferences::from(&locale), options).ok()
        });
        let collator = collator.as_ref().ok_or_else(|| {
            KernelError::internal(format!("no ICU data for {}", collation.name()))
        })?;
        collator
            .write_sort_key_to(s, out)
            .map_err(|_| KernelError::internal("ICU sort key".to_owned()))
    })
}

/// The alpha-2 code of a country Spark's ICU collation names carry (`collations()`).
#[cfg(feature = "collations-icu")]
fn region_alpha2(alpha3: &str) -> Option<&'static str> {
    Some(match alpha3 {
        "SAU" => "SA",
        "AUT" => "AT",
        "USA" => "US",
        "AFG" => "AF",
        "CAN" => "CA",
        "ISR" => "IL",
        "IDN" => "ID",
        "NOR" => "NO",
        "IND" => "IN",
        "BIH" => "BA",
        "MNE" => "ME",
        "SRB" => "RS",
        "CHN" => "CN",
        "SGP" => "SG",
        "HKG" => "HK",
        "MAC" => "MO",
        "TWN" => "TW",
        _ => return None,
    })
}

/// Without the `collations-icu` feature an ICU collation has no key.
#[cfg(not(feature = "collations-icu"))]
fn icu_key(
    collation: Collation,
    _ci: bool,
    _ai: bool,
    _s: &str,
    _out: &mut Vec<u8>,
) -> Result<(), KernelError> {
    Err(KernelError::internal(format!(
        "no collation key for {}",
        collation.name()
    )))
}

/// The Arrow type of the key of a value of Arrow type `t`: strings become `BinaryView` (the engine's binary carrier), inside
/// arrays, maps and structs too. A map's values keep their type when `keys_only`.
#[must_use]
pub fn collation_key_type(t: &ArrowType) -> ArrowType {
    match t {
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => ArrowType::BinaryView,
        ArrowType::List(f) => ArrowType::List(key_field(f)),
        ArrowType::LargeList(f) => ArrowType::LargeList(key_field(f)),
        ArrowType::Map(f, sorted) => ArrowType::Map(key_field(f), *sorted),
        ArrowType::Struct(fs) => {
            ArrowType::Struct(fs.iter().map(|f| key_field(f)).collect::<Fields>())
        }
        other => other.clone(),
    }
}

fn key_field(f: &Field) -> Arc<Field> {
    Arc::new(Field::new(
        f.name(),
        collation_key_type(f.data_type()),
        f.is_nullable(),
    ))
}

/// The key of every row of the string column `a`.
fn string_keys(collation: Collation, a: &ArrayRef) -> Result<ArrayRef, KernelError> {
    let rows = a.len();
    let mut out = BinaryViewBuilder::with_capacity(rows);
    let mut buf = Vec::new();
    let mut row = |s: Option<&str>| -> Result<(), KernelError> {
        match s {
            None => out.append_null(),
            Some(s) => {
                buf.clear();
                push_collation_key(collation, s, &mut buf)?;
                out.append_value(&buf);
            }
        }
        Ok(())
    };
    match a.data_type() {
        ArrowType::Utf8 => a.as_string::<i32>().iter().try_for_each(&mut row)?,
        ArrowType::LargeUtf8 => a.as_string::<i64>().iter().try_for_each(&mut row)?,
        ArrowType::Utf8View => a.as_string_view().iter().try_for_each(&mut row)?,
        other => return Err(KernelError::internal(format!("collation key of {other}"))),
    }
    Ok(Arc::new(out.finish()))
}

/// The key of `a`, a column of type `t`: string leaves become their key under the collation `t`
/// gives them (a plain string under `UTF8_BINARY`), of the shape [`collation_key_type`] says.
/// Null rows stay null.
///
/// # Errors
/// `INTERNAL_ERROR` when `t` does not describe `a`, or names an ICU collation.
pub fn collation_key(t: &DataType, a: &ArrayRef) -> Result<ArrayRef, KernelError> {
    let mismatch = || KernelError::internal(format!("collation key of {} as {t:?}", a.data_type()));
    match (t, a.data_type()) {
        (DataType::CollatedString(c), _) => string_keys(*c, a),
        (DataType::String, _) => string_keys(Collation::UTF8_BINARY, a),
        (DataType::Array(e), ArrowType::List(_)) => {
            let l = a.as_list::<i32>();
            let values = collation_key(e.element_type(), l.values())?;
            let f = key_field(field_of(a.data_type()).ok_or_else(mismatch)?);
            Ok(Arc::new(arrow_array::ListArray::new(
                f,
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )))
        }
        (DataType::Array(e), ArrowType::LargeList(_)) => {
            let l = a.as_list::<i64>();
            let values = collation_key(e.element_type(), l.values())?;
            let f = key_field(field_of(a.data_type()).ok_or_else(mismatch)?);
            Ok(Arc::new(arrow_array::LargeListArray::new(
                f,
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )))
        }
        (DataType::Struct(st), ArrowType::Struct(fs)) if st.fields().len() == fs.len() => {
            let s = a.as_struct();
            let mut columns = Vec::with_capacity(fs.len());
            for (field, col) in st.fields().iter().zip(s.columns()) {
                columns.push(collation_key(field.data_type(), col)?);
            }
            Ok(Arc::new(StructArray::new(
                fs.iter().map(|f| key_field(f)).collect(),
                columns,
                s.nulls().cloned(),
            )))
        }
        (DataType::Map(m), ArrowType::Map(entries, sorted)) => {
            let map = a.as_map();
            let ArrowType::Struct(kv) = entries.data_type() else {
                return Err(mismatch());
            };
            let (key, value) = (
                collation_key(m.key_type(), map.keys())?,
                collation_key(m.value_type(), map.values())?,
            );
            let entry = StructArray::new(
                kv.iter().map(|f| key_field(f)).collect(),
                vec![key, value],
                None,
            );
            Ok(Arc::new(MapArray::new(
                key_field(entries),
                map.offsets().clone(),
                entry,
                map.nulls().cloned(),
                *sorted,
            )))
        }
        _ if a
            .logical_nulls()
            .is_some_and(|n: NullBuffer| n.null_count() == a.len()) =>
        {
            Ok(arrow_array::new_null_array(
                &collation_key_type(a.data_type()),
                a.len(),
            ))
        }
        // A type without a string leaf is its own key; one with a string leaf that did not
        // match the column above is a planner bug, never the raw bytes taken as a key.
        _ if !t.any_type(&DataType::is_string_type) => Ok(Arc::clone(a)),
        _ => Err(mismatch()),
    }
}

/// The Arrow type of [`keyed_map`] of a map of type `t`: the keys become their key type.
#[must_use]
pub fn keyed_map_type(t: &ArrowType) -> ArrowType {
    let ArrowType::Map(entries, sorted) = t else {
        return t.clone();
    };
    let ArrowType::Struct(kv) = entries.data_type() else {
        return t.clone();
    };
    let fields: Fields = kv
        .iter()
        .enumerate()
        .map(|(i, f)| if i == 0 { key_field(f) } else { Arc::clone(f) })
        .collect();
    ArrowType::Map(
        Arc::new(Field::new(
            entries.name(),
            ArrowType::Struct(fields),
            entries.is_nullable(),
        )),
        *sorted,
    )
}

/// The map `a` of type `t` with each key replaced by its collation key and the values untouched:
/// a lookup in it with a keyed needle compares as the collation does (`m[k]`, `element_at`).
///
/// # Errors
/// `INTERNAL_ERROR` when `t` does not describe `a`.
pub fn keyed_map(t: &DataType, a: &ArrayRef) -> Result<ArrayRef, KernelError> {
    let (DataType::Map(m), ArrowType::Map(..)) = (t, a.data_type()) else {
        return Err(KernelError::internal(format!(
            "keyed map of {} as {t:?}",
            a.data_type()
        )));
    };
    let ArrowType::Map(entries, sorted) = keyed_map_type(a.data_type()) else {
        return Err(KernelError::internal("a keyed map type".to_owned()));
    };
    let ArrowType::Struct(kv) = entries.data_type() else {
        return Err(KernelError::internal("a keyed map entry type".to_owned()));
    };
    let map = a.as_map();
    let entry = StructArray::new(
        kv.clone(),
        vec![
            collation_key(m.key_type(), map.keys())?,
            Arc::clone(map.values()),
        ],
        None,
    );
    Ok(Arc::new(MapArray::new(
        entries,
        map.offsets().clone(),
        entry,
        map.nulls().cloned(),
        sorted,
    )))
}

fn field_of(t: &ArrowType) -> Option<&Field> {
    match t {
        ArrowType::List(f) | ArrowType::LargeList(f) => Some(f),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{BinaryArray, BinaryViewArray, ListArray, StringArray, StringViewArray};
    use krishiv_types::ArrayType;

    use super::*;

    fn key(c: &str, s: &str) -> Vec<u8> {
        let mut out = Vec::new();
        push_collation_key(Collation::parse(c).unwrap(), s, &mut out).unwrap();
        out
    }

    /// Every code point against what Spark's own function returned
    /// (`tests/scripts/gen_collation_lowercase.py`).
    #[test]
    #[cfg_attr(miri, ignore = "reads a fixture file, which miri's isolation forbids")]
    fn lowercase_is_sparks_for_every_code_point() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/spark-4.2.0/collation-lowercase.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        // {"lower": {"304": [105, 775], ...}, ...}: a flat format, read without a JSON dependency.
        let body = text.split("\"lower\": {").nth(1).unwrap();
        let mut expected = std::collections::HashMap::new();
        for item in body.split("], ") {
            let (cp, targets) = item.split_once(": [").unwrap();
            let cp: u32 = cp.trim().trim_matches('"').parse().unwrap();
            let end = targets.find(']').unwrap_or(targets.len());
            let targets: String = targets[..end]
                .split(", ")
                .map(|t| char::from_u32(t.parse().unwrap()).unwrap())
                .collect();
            expected.insert(cp, targets);
        }
        assert_eq!(expected.len(), 1489);
        for c in (0..=0x10_FFFF_u32).filter_map(char::from_u32) {
            let got = String::from_utf8(key("UTF8_LCASE", &c.to_string())).unwrap();
            let want = expected
                .get(&u32::from(c))
                .cloned()
                .unwrap_or_else(|| c.to_string());
            assert_eq!(got, want, "U+{:04X}", u32::from(c));
        }
    }

    #[test]
    fn collations_strip_and_lower_as_spark() {
        assert_eq!(key("UTF8_LCASE", "Aß"), "aß".as_bytes());
        assert_eq!(key("UTF8_LCASE", "ς"), "σ".as_bytes());
        assert_eq!(key("UTF8_LCASE", "İ"), "i\u{307}".as_bytes());
        assert_eq!(key("UTF8_BINARY_RTRIM", "A  "), b"A");
        // Only U+0020: a tab or a no-break space stays.
        assert_eq!(key("UTF8_BINARY_RTRIM", "a\t "), b"a\t");
        assert_eq!(key("UTF8_LCASE_RTRIM", "Ab  "), b"ab");
        // Leading spaces stay.
        assert_eq!(key("UTF8_LCASE_RTRIM", " a "), b" a");
    }

    #[cfg(not(feature = "collations-icu"))]
    #[test]
    fn icu_has_no_key_here() {
        let mut out = Vec::new();
        let c = Collation::parse("UNICODE").unwrap();
        assert!(push_collation_key(c, "a", &mut out).is_err());
    }

    /// `repartition(7, c)` of a real Spark 4.2.0: `pmod(murmur3(key bytes, 42), 7)`, which is
    /// what `CollationAwareMurmur3Hash` computes.
    #[test]
    fn hashing_the_key_gives_sparks_partition_ids() {
        let check = |collation: &str, rows: &[(&str, i32)]| {
            let c = Collation::parse(collation).unwrap();
            for (s, id) in rows {
                let mut buf = Vec::new();
                push_collation_key(c, s, &mut buf).unwrap();
                let col = BinaryArray::from_vec(vec![buf.as_slice()]);
                let h = crate::hash::hash(&[&col], 1).unwrap().value(0);
                assert_eq!(h.rem_euclid(7), *id, "{collation} {s:?}");
            }
        };
        let lcase = [
            ("", 5),
            ("A", 3),
            ("ABC", 6),
            ("B", 2),
            ("a", 3),
            ("abc", 6),
            ("b", 2),
            ("ß", 2),
            ("İ", 4),
            ("Σ", 2),
            ("ς", 2),
            ("σ", 2),
            ("日本", 4),
        ];
        check("UTF8_LCASE", &lcase);
        check("UTF8_LCASE_RTRIM", &lcase);
        check("UTF8_LCASE_RTRIM", &[("a  ", 3), ("A  ", 3)]);
        check(
            "UTF8_BINARY_RTRIM",
            &[
                ("", 5),
                ("A  ", 5),
                ("ABC", 0),
                ("a  ", 3),
                ("ς", 3),
                ("İ", 1),
            ],
        );
    }

    fn lcase() -> DataType {
        DataType::collated_string(Collation::parse("UTF8_LCASE").unwrap())
    }

    #[test]
    fn columns_keep_their_nulls_on_any_carrier() {
        for a in [
            Arc::new(StringArray::from(vec![Some("aB"), None])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("aB"), None])),
        ] {
            let out = collation_key(&lcase(), &a).unwrap();
            let out = out.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            assert_eq!(out.value(0), b"ab");
            assert!(out.is_null(1));
        }
    }

    #[test]
    fn nested_keys_have_the_shape_of_the_value() {
        let t = DataType::Array(Arc::new(ArrayType::new(lcase(), true)));
        let a: ArrayRef = Arc::new(ListArray::from_iter_primitive::<
            arrow_array::types::Int32Type,
            _,
            _,
        >(vec![Some(vec![Some(1)])]));
        // Not a list of strings: the mismatch is reported, not guessed at.
        assert!(collation_key(&t, &a).is_err());
    }

    /// A type with a string leaf that does not describe the column is a planner bug: it is
    /// `INTERNAL_ERROR`, never the raw bytes passed through as if they were a key.
    #[test]
    fn a_type_that_does_not_describe_the_column_is_refused() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![Some("aB")]));
        let map = DataType::Map(Arc::new(krishiv_types::MapType::new(
            lcase(),
            DataType::Integer,
            true,
        )));
        assert!(collation_key(&map, &strings).is_err());
        let st = DataType::Struct(krishiv_types::StructType::new([
            krishiv_types::StructField::new("a", lcase(), true),
        ]));
        assert!(collation_key(&st, &strings).is_err());
        // A leaf without a string is its own key.
        let ints: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![Some(1)]));
        let out = collation_key(&DataType::Integer, &ints).unwrap();
        assert_eq!(out.data_type(), &ArrowType::Int32);
    }
    /// The `collations-basic` ICU statements of Spark 4.2 (`'a' collate unicode < 'A'` ...): each
    /// comparison is true, here by the sort key.
    #[cfg(feature = "collations-icu")]
    #[test]
    #[cfg_attr(miri, ignore = "the ICU data is large for miri")]
    fn icu_keys_order_as_the_spark_goldens_say() {
        assert!(key("unicode", "a") < key("unicode", "A"));
        assert_eq!(key("unicode_ci", "a"), key("unicode_ci", "A"));
        assert_eq!(key("unicode_ai", "a"), key("unicode_ai", "\u{e5}"));
        assert_eq!(key("unicode_ci_ai", "a"), key("unicode_ci_ai", "\u{c5}"));
        assert!(key("en", "a") < key("en", "A"));
        assert_eq!(key("en_ci", "a"), key("en_ci", "A"));
        assert_eq!(key("en_ai", "a"), key("en_ai", "\u{e5}"));
        assert_eq!(key("en_ci_ai", "a"), key("en_ci_ai", "\u{c5}"));
        assert!(key("sv", "Kypper") < key("sv", "K\u{f6}pfe"));
        assert!(key("de", "Kypper") > key("de", "K\u{f6}pfe"));
        assert_eq!(key("tr_ci", "I"), key("tr_ci", "\u{131}"));
        // `fr_CAN` is Canadian French: the accents order backwards.
        assert!(key("fr_CAN", "c\u{f4}te") < key("fr_CAN", "cot\u{e9}"));
        assert!(key("fr", "cot\u{e9}") < key("fr", "c\u{f4}te"));
    }
}
