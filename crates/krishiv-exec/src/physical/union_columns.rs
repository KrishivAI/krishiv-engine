// SPDX-License-Identifier: Apache-2.0
//! Whether two union inputs carry the same columns: names, nullability, type tags and types at
//! every level, whatever their other field metadata (a Delta scan's column-mapping labels).

use arrow_schema::{DataType, Field, Schema};

/// The field metadata keys that are part of a column's type for Krishiv: the Spark type tag
/// and the Arrow extension name (a Variant). Other keys (a Delta scan's column-mapping ids and
/// physical names) are labels a union may disagree on.
const TYPE_TAGS: [&str; 2] = [
    krishiv_types::arrow::SPARK_TYPE_METADATA_KEY,
    "ARROW:extension:name",
];

/// Whether fields `x` and `y` are the same column for a union: name, nullability, the type tags
/// ([`TYPE_TAGS`]) and the type, at every nesting level, whatever their other metadata.
fn same_field(x: &Field, y: &Field) -> bool {
    fn tags(f: &Field) -> [Option<&String>; 2] {
        TYPE_TAGS.map(|k| f.metadata().get(k))
    }
    x.name() == y.name()
        && x.is_nullable() == y.is_nullable()
        && tags(x) == tags(y)
        && same_type(x.data_type(), y.data_type())
}

fn same_type(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::Struct(x), DataType::Struct(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| same_field(p, q))
        }
        (DataType::List(x), DataType::List(y))
        | (DataType::LargeList(x), DataType::LargeList(y))
        | (DataType::ListView(x), DataType::ListView(y))
        | (DataType::LargeListView(x), DataType::LargeListView(y)) => same_field(x, y),
        (DataType::FixedSizeList(x, n), DataType::FixedSizeList(y, m)) => {
            n == m && same_field(x, y)
        }
        (DataType::Map(x, s), DataType::Map(y, t)) => s == t && same_field(x, y),
        _ => a == b,
    }
}

/// Whether the union inputs `a` and `b` carry the same columns ([`same_field`]): a Delta
/// scan's columns carry their column-mapping metadata, a new table's do not (a CTAS unions
/// the two). Instantiation relabels an input whose fields differ only so.
pub(crate) fn same_columns(a: &Schema, b: &Schema) -> bool {
    a.fields().len() == b.fields().len()
        && a.fields()
            .iter()
            .zip(b.fields().iter())
            .all(|(x, y)| same_field(x, y))
}

#[cfg(test)]
mod same_columns_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn field_metadata_does_not_make_union_inputs_differ() {
        let tagged = Field::new("a", DataType::Int64, true).with_metadata(
            [("delta.columnMapping.id".to_owned(), "1".to_owned())]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
        );
        let plain = Schema::new(vec![Field::new("a", DataType::Int64, true)]);
        assert!(same_columns(&Schema::new(vec![tagged]), &plain));
        for other in [
            Field::new("b", DataType::Int64, true),
            Field::new("a", DataType::Int32, true),
            Field::new("a", DataType::Int64, false),
        ] {
            assert!(!same_columns(&Schema::new(vec![other]), &plain));
        }
    }

    #[test]
    fn nested_mapping_metadata_is_a_label_but_type_tags_are_not() {
        let meta = |k: &str, v: &str| {
            [(k.to_owned(), v.to_owned())]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>()
        };
        let st = |f: Field| {
            Schema::new(vec![Field::new(
                "s",
                DataType::Struct(vec![f].into()),
                true,
            )])
        };
        let plain = st(Field::new("x", DataType::Int32, true));
        let mapped = st(Field::new("x", DataType::Int32, true)
            .with_metadata(meta("delta.columnMapping.id", "5")));
        assert!(same_columns(&mapped, &plain));
        let list = |f: Field| Schema::new(vec![Field::new("l", DataType::List(Arc::new(f)), true)]);
        assert!(same_columns(
            &list(
                Field::new("item", DataType::Int32, true)
                    .with_metadata(meta("delta.columnMapping.id", "7"))
            ),
            &list(Field::new("item", DataType::Int32, true)),
        ));
        // Krishiv's own tags are part of the type: a binary-safe string on binary storage is
        // not a plain binary column.
        let tagged = Schema::new(vec![
            Field::new("b", DataType::BinaryView, true).with_metadata(meta(
                krishiv_types::arrow::SPARK_TYPE_METADATA_KEY,
                "string",
            )),
        ]);
        let binary = Schema::new(vec![Field::new("b", DataType::BinaryView, true)]);
        assert!(!same_columns(&tagged, &binary));
        let variant = st(Field::new("x", DataType::Int32, true)
            .with_metadata(meta("ARROW:extension:name", "arrow.parquet.variant")));
        assert!(!same_columns(&variant, &plain));
    }
}
