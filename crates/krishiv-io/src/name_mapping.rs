// SPDX-License-Identifier: Apache-2.0
//! Name mappings: field ids for a file written without them, by the names of
//! its columns (Iceberg's `schema.name-mapping.default`, read by
//! [`WithoutFieldIds::Names`](crate::WithoutFieldIds::Names)). The table
//! format parses its own text into a [`NameMapping`]; this module only applies
//! one, as Java's `ApplyNameMapping` does: every file field whose path of
//! names is mapped gets the mapped id, every other field gets none (and is
//! then not read). Names compare exactly. A list's element is the mapping's
//! `element`, a map's key and value its `key` and `value`, whatever Arrow
//! names those children.

use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};

use crate::FIELD_ID_KEY;

/// One mapped field: the names a file may use for it, its id (none: the
/// field is matched but carries no id) and its children.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MappedField {
    /// The field id, if the mapping gives one.
    pub id: Option<i32>,
    /// Every name the field had.
    pub names: Vec<String>,
    /// The mapping of a struct's fields, a list's `element`, or a map's
    /// `key` and `value`.
    pub fields: Vec<MappedField>,
}

/// The fields of one level of a mapping, by name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NameMapping {
    fields: Vec<MappedField>,
}

impl NameMapping {
    /// The mapping of the top-level `fields`.
    #[must_use]
    pub fn new(fields: Vec<MappedField>) -> Self {
        Self { fields }
    }

    /// The mapped top-level fields.
    #[must_use]
    pub fn fields(&self) -> &[MappedField] {
        &self.fields
    }

    /// `schema` (a file's) with the mapped ids on its fields at every depth;
    /// fields the mapping does not name lose any id they had.
    #[must_use]
    pub fn apply(&self, schema: &Schema) -> SchemaRef {
        Arc::new(Schema::new_with_metadata(
            apply_level(schema.fields(), &self.fields),
            schema.metadata().clone(),
        ))
    }
}

fn find<'a>(level: &'a [MappedField], name: &str) -> Option<&'a MappedField> {
    level.iter().find(|m| m.names.iter().any(|n| n == name))
}

fn apply_level(fields: &Fields, level: &[MappedField]) -> Fields {
    fields
        .iter()
        .map(|f| apply_field(f, find(level, f.name())))
        .collect()
}

/// `f` with the id of `mapped` (none when unmapped), its children mapped by
/// `mapped`'s children.
fn apply_field(f: &FieldRef, mapped: Option<&MappedField>) -> FieldRef {
    let children: &[MappedField] = mapped.map_or(&[], |m| m.fields.as_slice());
    let child = |name: &str| find(children, name);
    let data_type = match f.data_type() {
        DataType::Struct(c) => DataType::Struct(apply_level(c, children)),
        DataType::List(e) => DataType::List(apply_field(e, child("element"))),
        DataType::LargeList(e) => DataType::LargeList(apply_field(e, child("element"))),
        DataType::ListView(e) => DataType::ListView(apply_field(e, child("element"))),
        DataType::LargeListView(e) => DataType::LargeListView(apply_field(e, child("element"))),
        DataType::FixedSizeList(e, n) => {
            DataType::FixedSizeList(apply_field(e, child("element")), *n)
        }
        DataType::Map(entries, sorted) => {
            let entries = match entries.data_type() {
                DataType::Struct(kv) if kv.len() == 2 => {
                    let kv: Fields = vec![
                        apply_field(&kv[0], child("key")),
                        apply_field(&kv[1], child("value")),
                    ]
                    .into();
                    Arc::new(
                        entries
                            .as_ref()
                            .clone()
                            .with_data_type(DataType::Struct(kv)),
                    )
                }
                _ => Arc::clone(entries),
            };
            DataType::Map(entries, *sorted)
        }
        other => other.clone(),
    };
    let mut metadata = f.metadata().clone();
    metadata.remove(FIELD_ID_KEY);
    if let Some(id) = mapped.and_then(|m| m.id) {
        metadata.insert(FIELD_ID_KEY.to_owned(), id.to_string());
    }
    Arc::new(
        Field::new(f.name(), data_type, f.is_nullable())
            .with_metadata(metadata)
            .with_dict_is_ordered(f.dict_is_ordered().unwrap_or(false)),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn id(f: &Field) -> Option<&str> {
        f.metadata().get(FIELD_ID_KEY).map(String::as_str)
    }

    fn mapped(id: Option<i32>, names: &[&str], fields: Vec<MappedField>) -> MappedField {
        MappedField {
            id,
            names: names.iter().map(|n| (*n).to_owned()).collect(),
            fields,
        }
    }

    #[test]
    fn ids_follow_names_at_every_depth() {
        let element = Field::new("item", DataType::Int32, true);
        let kv = Fields::from(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, true),
        ]);
        let schema = Schema::new(vec![
            Field::new("old_name", DataType::Int64, true),
            Field::new(
                "s",
                DataType::Struct(Fields::from(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("unmapped", DataType::Int32, true),
                ])),
                true,
            ),
            Field::new("l", DataType::List(Arc::new(element)), true),
            Field::new(
                "m",
                DataType::Map(
                    Arc::new(Field::new("entries", DataType::Struct(kv), false)),
                    false,
                ),
                true,
            ),
            Field::new("extra", DataType::Int32, true)
                .with_metadata(HashMap::from([(FIELD_ID_KEY.to_owned(), "99".to_owned())])),
        ]);
        let mapping = NameMapping::new(vec![
            mapped(Some(1), &["id", "old_name"], vec![]),
            mapped(Some(2), &["s"], vec![mapped(Some(3), &["a"], vec![])]),
            mapped(Some(4), &["l"], vec![mapped(Some(5), &["element"], vec![])]),
            mapped(
                Some(6),
                &["m"],
                vec![
                    mapped(Some(7), &["key"], vec![]),
                    mapped(Some(8), &["value"], vec![]),
                ],
            ),
        ]);
        let out = mapping.apply(&schema);
        let f = out.fields();
        assert_eq!(id(&f[0]), Some("1"));
        assert_eq!(id(&f[1]), Some("2"));
        let DataType::Struct(s) = f[1].data_type() else {
            unreachable!()
        };
        assert_eq!((id(&s[0]), id(&s[1])), (Some("3"), None));
        let DataType::List(e) = f[2].data_type() else {
            unreachable!()
        };
        assert_eq!((id(&f[2]), id(e)), (Some("4"), Some("5")));
        let DataType::Map(entries, _) = f[3].data_type() else {
            unreachable!()
        };
        let DataType::Struct(kv) = entries.data_type() else {
            unreachable!()
        };
        assert_eq!((id(&kv[0]), id(&kv[1])), (Some("7"), Some("8")));
        // A field the mapping does not name has no id, whatever it had.
        assert_eq!(id(&f[4]), None);
    }
}
