// SPDX-License-Identifier: Apache-2.0
//! The metrics mode of each column a write records (P3-T027, PR-36a; design
//! note §6.1), as Java's `MetricsConfig.from(props, schema, sortOrder)`
//! (apache-iceberg 1.12.0) decides it:
//!
//! 1. `write.metadata.metrics.default`, when set, is the default for every
//!    column (an invalid value falls back to `truncate(16)`).
//! 2. Otherwise the default is `truncate(16)` when the schema has at most
//!    `write.metadata.metrics.max-inferred-column-defaults` (100) projected
//!    ids (struct ids included); a wider schema gives `truncate(16)` to the
//!    first that many eligible columns (each struct's primitive fields, then
//!    its children, in order) and `none` to the rest.
//! 3. Columns of the sort order whose transform preserves order get
//!    `truncate(16)` when the default is `none` or `counts`.
//! 4. `write.metadata.metrics.column.<name>` overrides a column by its full
//!    name (an invalid value falls back to the default).
//!
//! The read side keeps its own reading of the same properties
//! (`plan::metrics::WriterMetrics`): it only decides how far bounds found in a
//! manifest are trusted.

use std::collections::{HashMap, HashSet};

use iceberg::spec::{Schema, SortOrder, Type};

use crate::plan::metrics::{METRICS_COLUMN_PREFIX, METRICS_DEFAULT, MetricsMode};

/// `write.metadata.metrics.max-inferred-column-defaults`.
pub(crate) const METRICS_MAX_INFERRED: &str = "write.metadata.metrics.max-inferred-column-defaults";

const MAX_INFERRED_DEFAULT: usize = 100;

const DEFAULT_MODE: MetricsMode = MetricsMode::Truncate(16);

/// The metrics mode of each column of one write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WriteMetricsConfig {
    default: MetricsMode,
    columns: HashMap<i32, MetricsMode>,
}

impl WriteMetricsConfig {
    /// The modes for a write into a table with these properties, schema and
    /// sort order (see the module docs).
    pub(crate) fn from_table(
        props: &HashMap<String, String>,
        schema: &Schema,
        order: Option<&SortOrder>,
    ) -> Self {
        let mut columns = HashMap::new();
        let default = match props.get(METRICS_DEFAULT) {
            Some(text) => parse(text).unwrap_or(DEFAULT_MODE),
            None => {
                let max = max_inferred(props);
                if projected_ids(schema) <= max {
                    DEFAULT_MODE
                } else {
                    for id in limit_field_ids(schema, max) {
                        columns.insert(id, DEFAULT_MODE);
                    }
                    MetricsMode::None
                }
            }
        };
        let sorted_mode = match default {
            MetricsMode::None | MetricsMode::Counts => MetricsMode::Truncate(16),
            other => other,
        };
        for f in order.map_or(&[][..], |o| &o.fields[..]) {
            if f.transform.preserves_order() && schema.name_by_field_id(f.source_id).is_some() {
                columns.insert(f.source_id, sorted_mode);
            }
        }
        // Overrides in key order, so two spellings of one column resolve the
        // same way on every run (Java's order is its map's).
        let mut keys: Vec<&String> = props.keys().collect();
        keys.sort();
        for key in keys {
            if let Some(name) = key.strip_prefix(METRICS_COLUMN_PREFIX)
                && let Some(field) = schema.field_by_name(name)
            {
                columns.insert(field.id, parse(&props[key]).unwrap_or(default));
            }
        }
        Self { default, columns }
    }

    /// The mode of the column with field id `id`.
    pub(crate) fn mode(&self, id: i32) -> MetricsMode {
        self.columns.get(&id).copied().unwrap_or(self.default)
    }
}

/// Java's `MetricsModes.fromString`: `none`, `counts` and `full` in any case,
/// `truncate(<n>)` with `n` a positive int; anything else is `None`.
fn parse(text: &str) -> Option<MetricsMode> {
    if text.eq_ignore_ascii_case("none") {
        return Some(MetricsMode::None);
    }
    if text.eq_ignore_ascii_case("counts") {
        return Some(MetricsMode::Counts);
    }
    if text.eq_ignore_ascii_case("full") {
        return Some(MetricsMode::Full);
    }
    let lower = text.to_lowercase();
    let digits = lower.strip_prefix("truncate(")?.strip_suffix(')')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: i32 = digits.parse().ok()?;
    (n > 0).then(|| MetricsMode::Truncate(usize::try_from(n).unwrap_or(usize::MAX)))
}

/// Java's `maxInferredColumnDefaults`: the property as an int, the default
/// when it is negative; a value that is not an int fails Java's write, and
/// is read as the default here.
fn max_inferred(props: &HashMap<String, String>) -> usize {
    props
        .get(METRICS_MAX_INFERRED)
        .and_then(|v| v.trim().parse::<i32>().ok())
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(MAX_INFERRED_DEFAULT)
}

fn eligible(ty: &Type) -> bool {
    matches!(ty, Type::Primitive(_) | Type::Variant(_))
}

/// The size of Java's `TypeUtil.getProjectedIds(schema)`: struct, primitive
/// and variant fields; a list's element when it is not nested; a map's key
/// and value when the value is not nested.
fn projected_ids(schema: &Schema) -> usize {
    fn visit(ty: &Type, ids: &mut HashSet<i32>) {
        match ty {
            Type::Struct(s) => {
                for f in s.fields() {
                    visit(&f.field_type, ids);
                    if matches!(*f.field_type, Type::Struct(_)) || eligible(&f.field_type) {
                        ids.insert(f.id);
                    }
                }
            }
            Type::List(l) => {
                visit(&l.element_field.field_type, ids);
                if eligible(&l.element_field.field_type) {
                    ids.insert(l.element_field.id);
                }
            }
            Type::Map(m) => {
                visit(&m.key_field.field_type, ids);
                visit(&m.value_field.field_type, ids);
                if eligible(&m.value_field.field_type) {
                    ids.insert(m.key_field.id);
                    ids.insert(m.value_field.id);
                }
            }
            Type::Primitive(_) | Type::Variant(_) => {}
        }
    }
    let mut ids = HashSet::new();
    visit(&Type::Struct(schema.as_struct().clone()), &mut ids);
    ids.len()
}

/// Java's `MetricsConfig.limitFieldIds`: up to `limit` eligible ids, each
/// struct's own primitive fields before its children's.
fn limit_field_ids(schema: &Schema, limit: usize) -> Vec<i32> {
    fn visit(ty: &Type, ids: &mut Vec<i32>, limit: usize) {
        match ty {
            Type::Struct(s) => {
                for f in s.fields() {
                    if ids.len() >= limit {
                        return;
                    }
                    if eligible(&f.field_type) {
                        ids.push(f.id);
                    }
                }
                for f in s.fields() {
                    if ids.len() >= limit {
                        return;
                    }
                    visit(&f.field_type, ids, limit);
                }
            }
            Type::List(l) => {
                if ids.len() < limit && eligible(&l.element_field.field_type) {
                    ids.push(l.element_field.id);
                }
                visit(&l.element_field.field_type, ids, limit);
            }
            Type::Map(m) => {
                if ids.len() < limit && eligible(&m.key_field.field_type) {
                    ids.push(m.key_field.id);
                }
                if ids.len() < limit && eligible(&m.value_field.field_type) {
                    ids.push(m.value_field.id);
                }
                visit(&m.key_field.field_type, ids, limit);
                visit(&m.value_field.field_type, ids, limit);
            }
            Type::Primitive(_) | Type::Variant(_) => {}
        }
    }
    let mut ids = Vec::new();
    visit(&Type::Struct(schema.as_struct().clone()), &mut ids, limit);
    ids
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use iceberg::spec::{
        ListType, NestedField, NullOrder, PrimitiveType, SortDirection, SortField, StructType,
        Transform,
    };

    use super::*;

    fn long(id: i32, name: &str) -> NestedField {
        NestedField::optional(id, name, Type::Primitive(PrimitiveType::Long))
    }

    fn schema(fields: Vec<NestedField>) -> Schema {
        Schema::builder()
            .with_fields(fields.into_iter().map(Arc::new))
            .build()
            .unwrap()
    }

    fn props(kv: &[(&str, &str)]) -> HashMap<String, String> {
        kv.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn modes_parse_as_java() {
        assert_eq!(parse("NONE"), Some(MetricsMode::None));
        assert_eq!(parse("Counts"), Some(MetricsMode::Counts));
        assert_eq!(parse("full"), Some(MetricsMode::Full));
        assert_eq!(parse("TRUNCATE(8)"), Some(MetricsMode::Truncate(8)));
        for bad in [
            " full",
            "truncate(0)",
            "truncate( 8)",
            "truncate(-1)",
            "truncate(99999999999)",
            "x",
        ] {
            assert_eq!(parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn default_override_and_fallbacks() {
        let s = schema(vec![long(1, "a"), long(2, "b")]);
        let c = WriteMetricsConfig::from_table(&HashMap::new(), &s, None);
        assert_eq!(c.mode(1), MetricsMode::Truncate(16));
        let c = WriteMetricsConfig::from_table(
            &props(&[
                (METRICS_DEFAULT, "counts"),
                ("write.metadata.metrics.column.a", "full"),
                ("write.metadata.metrics.column.b", "bogus"),
                ("write.metadata.metrics.column.zz", "full"),
            ]),
            &s,
            None,
        );
        assert_eq!(c.mode(1), MetricsMode::Full);
        // An invalid column mode falls back to the default.
        assert_eq!(c.mode(2), MetricsMode::Counts);
        // An invalid default falls back to truncate(16).
        let c = WriteMetricsConfig::from_table(&props(&[(METRICS_DEFAULT, "x")]), &s, None);
        assert_eq!(c.mode(2), MetricsMode::Truncate(16));
    }

    /// A schema wider than the limit gets metrics on its first columns only,
    /// a struct's own primitives before its children's; sorted columns are
    /// promoted from `none`.
    #[test]
    fn wide_schemas_limit_inferred_columns() {
        let inner = StructType::new(vec![Arc::new(long(11, "x")), Arc::new(long(12, "y"))]);
        let s = schema(vec![
            NestedField::optional(10, "s", Type::Struct(inner)),
            long(1, "a"),
            NestedField::optional(
                20,
                "l",
                Type::List(ListType::new(Arc::new(long(21, "element")))),
            ),
            long(2, "b"),
        ]);
        // Projected ids: s, x, y, a, element, b = 6.
        assert_eq!(projected_ids(&s), 6);
        assert_eq!(limit_field_ids(&s, 3), [1, 2, 11]);
        let p = props(&[(METRICS_MAX_INFERRED, "3")]);
        let order = SortOrder::builder()
            .with_order_id(1)
            .with_sort_field(SortField {
                source_id: 21,
                transform: Transform::Identity,
                direction: SortDirection::Ascending,
                null_order: NullOrder::First,
            })
            .build_unbound()
            .unwrap();
        let c = WriteMetricsConfig::from_table(&p, &s, Some(&order));
        assert_eq!(c.mode(1), MetricsMode::Truncate(16));
        assert_eq!(c.mode(11), MetricsMode::Truncate(16));
        assert_eq!(c.mode(12), MetricsMode::None);
        assert_eq!(c.mode(21), MetricsMode::Truncate(16));
        // At the limit, the default applies everywhere.
        let c = WriteMetricsConfig::from_table(&props(&[(METRICS_MAX_INFERRED, "6")]), &s, None);
        assert_eq!(c.mode(12), MetricsMode::Truncate(16));
        // A negative limit is the default (100).
        let c = WriteMetricsConfig::from_table(&props(&[(METRICS_MAX_INFERRED, "-1")]), &s, None);
        assert_eq!(c.mode(12), MetricsMode::Truncate(16));
    }
}
