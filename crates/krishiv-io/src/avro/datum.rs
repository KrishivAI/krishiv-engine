// SPDX-License-Identifier: Apache-2.0
//! Avro datums outside a container file: the binary encoding of one value
//! with a schema, as Spark's `from_avro` and `to_avro` read and write it.
//!
//! [`DatumDecoder`] turns a column of datum bytes into one Arrow array with
//! the schema mapping of the file reader (unions, decimals, timestamps,
//! `member0..N` structs); [`DatumDecoder::decode`] fails on the first bad
//! datum (`FAILFAST`) or yields NULL for it (`PERMISSIVE`: a record becomes a
//! struct of all-NULL fields, as Spark's `from_avro` does). It reuses the
//! file decoder's resolution and builders, so a value reads the same in a
//! file and in a column.

use apache_avro::Schema as AvroSchema;
use apache_avro::reader::datum::GenericDatumReader;
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Fields};

use super::decode::{append, append_null, corrupt, finish_node, node_of, resolve};
use super::encode::put_cell;
use super::schema::{UnionShape, field_of, schema_json_of_column, union_shape};
use crate::error::{FormatError, FormatResult, unsupported};

/// Decodes a column of Avro datums (see the module documentation).
#[derive(Debug, Clone)]
pub struct DatumDecoder {
    writer: AvroSchema,
    reader: AvroSchema,
    field: Field,
    permissive: bool,
}

/// `dt` with every field nullable, all the way down (Spark's
/// `DataType.asNullable`, which `PERMISSIVE` applies to the result).
fn nullable_type(dt: &DataType) -> DataType {
    let nf = |f: &Field| {
        f.clone()
            .with_data_type(nullable_type(f.data_type()))
            .with_nullable(true)
    };
    match dt {
        DataType::Struct(fs) => DataType::Struct(fs.iter().map(|f| nf(f)).collect::<Fields>()),
        DataType::List(f) => DataType::List(std::sync::Arc::new(nf(f))),
        DataType::Map(f, sorted) => {
            let DataType::Struct(es) = f.data_type() else {
                return dt.clone();
            };
            let key = es[0].as_ref().clone();
            let value = nf(&es[1]);
            let entries = Field::new(
                f.name().clone(),
                DataType::Struct(vec![key, value].into()),
                false,
            );
            DataType::Map(std::sync::Arc::new(entries), *sorted)
        }
        other => other.clone(),
    }
}

impl DatumDecoder {
    /// A decoder of datums written with `writer_json`, read as `reader_json`
    /// (the writer schema when `None`); `permissive` selects `PERMISSIVE`.
    ///
    /// # Errors
    /// [`FormatError::Corrupt`] for schema text Avro cannot parse;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a schema with no Spark type or a
    /// top-level union of several non-null types.
    pub fn new(
        writer_json: &str,
        reader_json: Option<&str>,
        permissive: bool,
    ) -> FormatResult<Self> {
        let parse =
            |s: &str| AvroSchema::parse_str(s).map_err(|e| FormatError::Corrupt(format!("{e}")));
        let writer = parse(writer_json)?;
        let reader = match reader_json {
            Some(r) => parse(r)?,
            None => writer.clone(),
        };
        if let AvroSchema::Union(u) = &reader
            && matches!(union_shape(u.variants()), UnionShape::Members)
        {
            return Err(FormatError::Other(unsupported(
                "a top-level Avro union of several non-null types (it cannot be read as one value)"
                    .to_owned(),
            )));
        }
        let mut field = field_of("value", &reader)?.with_nullable(true);
        if permissive {
            let dt = nullable_type(field.data_type());
            field = field.with_data_type(dt);
        }
        Ok(Self {
            writer,
            reader,
            field,
            permissive,
        })
    }

    /// The Arrow type of the decoded column.
    #[must_use]
    pub fn data_type(&self) -> &DataType {
        self.field.data_type()
    }

    /// Decodes `data` (NULL stays NULL) into one array of [`Self::data_type`].
    ///
    /// # Errors
    /// [`FormatError::Corrupt`] with the decoder's reason for the first bad
    /// datum when not permissive.
    pub fn decode(&self, data: &[Option<&[u8]>]) -> FormatResult<ArrayRef> {
        let datum = GenericDatumReader::builder(&self.writer)
            .build()
            .map_err(|e| corrupt(format!("an unreadable schema: {e}")))?;
        let mut node = node_of(&self.field, &self.reader)?;
        for bytes in data {
            let Some(bytes) = bytes else {
                append_null(&mut node)?;
                continue;
            };
            match self.one(&datum, bytes, &mut node) {
                Ok(()) => {}
                Err(e) if self.permissive => {
                    let _ = e;
                    self.append_bad(&mut node)?;
                }
                Err(e) => return Err(e),
            }
        }
        finish_node(node, &self.field)
    }

    /// Reads one datum; nothing is appended when it fails.
    fn one(
        &self,
        datum: &GenericDatumReader<'_>,
        bytes: &[u8],
        node: &mut super::decode::Node,
    ) -> FormatResult<()> {
        let mut cursor = std::io::Cursor::new(bytes);
        // `apache-avro` can panic on malformed bytes instead of returning an
        // error; the unwind becomes the same typed failure.
        let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            datum.read_value(&mut cursor)
        }))
        .map_err(|_| corrupt("an undecodable datum"))?
        .map_err(|e| corrupt(format!("an undecodable datum: {e}")))?;
        let resolved = resolve(&self.writer, &self.reader, value, false)?;
        append(node, &self.reader, &resolved, true)?;
        append(node, &self.reader, &resolved, false)
    }

    /// What `PERMISSIVE` yields for a bad datum: NULL, or for a record a
    /// struct whose fields are all NULL.
    fn append_bad(&self, node: &mut super::decode::Node) -> FormatResult<()> {
        super::decode::append_null_fields(node).or_else(|_| append_null(node))
    }
}

/// Encodes a column as Avro datums (see the module documentation).
#[derive(Debug, Clone)]
pub struct DatumEncoder {
    schema: AvroSchema,
    json: String,
}

impl DatumEncoder {
    /// An encoder for a column of `field`'s type with the schema `to_avro`
    /// infers: the type, or `[T, "null"]` when the column is nullable.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a type with no Avro type.
    pub fn infer(field: &Field) -> FormatResult<Self> {
        Self::with_schema(&schema_json_of_column(field)?)
    }

    /// An encoder writing `schema_json`.
    ///
    /// # Errors
    /// [`FormatError::Corrupt`] for schema text Avro cannot parse.
    pub fn with_schema(schema_json: &str) -> FormatResult<Self> {
        let schema =
            AvroSchema::parse_str(schema_json).map_err(|e| FormatError::Corrupt(format!("{e}")))?;
        Ok(Self {
            schema,
            json: schema_json.to_owned(),
        })
    }

    /// The schema's JSON text.
    #[must_use]
    pub fn schema_json(&self) -> &str {
        &self.json
    }

    /// The datum of each row of `array`; a NULL row is `None` (`to_avro` of
    /// NULL is NULL).
    ///
    /// # Errors
    /// [`FormatError::WriteFailed`] for a value the schema cannot hold.
    pub fn encode(&self, array: &ArrayRef) -> FormatResult<Vec<Option<Vec<u8>>>> {
        (0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    return Ok(None);
                }
                let mut out = Vec::new();
                put_cell(&mut out, &self.schema, array, row)?;
                Ok(Some(out))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int32Array, StringArray, StructArray};

    use super::*;

    fn rows(v: &[Option<Vec<u8>>]) -> Vec<Option<&[u8]>> {
        v.iter().map(|x| x.as_deref()).collect()
    }

    #[test]
    fn a_non_null_int_encodes_as_a_zigzag_long_and_a_nullable_one_as_union_branch_zero() {
        let plain = Field::new("v", DataType::Int32, false);
        let enc = DatumEncoder::infer(&plain).unwrap();
        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, -1]));
        assert_eq!(enc.encode(&a).unwrap(), vec![Some(vec![2]), Some(vec![1])]);
        let nullable = Field::new("v", DataType::Int32, true);
        let enc = DatumEncoder::infer(&nullable).unwrap();
        assert_eq!(enc.schema_json(), r#"["int","null"]"#);
        let a: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        assert_eq!(enc.encode(&a).unwrap(), vec![Some(vec![0, 2]), None]);
    }

    #[test]
    fn strings_round_trip_and_null_stays_null() {
        let schema = r#""string""#;
        let enc = DatumEncoder::with_schema(schema).unwrap();
        let a: ArrayRef = Arc::new(StringArray::from(vec![Some("abc"), None]));
        let bytes = enc.encode(&a).unwrap();
        assert_eq!(bytes[0].as_deref(), Some(&[6, b'a', b'b', b'c'][..]));
        let dec = DatumDecoder::new(schema, None, false).unwrap();
        let out = dec.decode(&rows(&bytes)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.is_null(1));
    }

    #[test]
    fn failfast_raises_and_permissive_nulls_a_bad_datum() {
        let schema = r#""int""#;
        let bad: Vec<Option<&[u8]>> = vec![Some(&[0x80][..]), Some(&[2][..])];
        let strict = DatumDecoder::new(schema, None, false).unwrap();
        assert!(strict.decode(&bad).is_err());
        let lax = DatumDecoder::new(schema, None, true).unwrap();
        let out = lax.decode(&bad).unwrap();
        let ints = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert!(ints.is_null(0));
        assert_eq!(ints.value(1), 1);
    }

    #[test]
    fn a_bad_record_under_permissive_is_a_struct_of_nulls() {
        let schema = r#"{"type":"record","name":"r","fields":[{"name":"a","type":"int"},{"name":"b","type":"string"}]}"#;
        let lax = DatumDecoder::new(schema, None, true).unwrap();
        let out = lax.decode(&[Some(&[0x80][..])]).unwrap();
        let s = out.as_any().downcast_ref::<StructArray>().unwrap();
        assert!(!s.is_null(0));
        assert!(s.column(0).is_null(0) && s.column(1).is_null(0));
    }

    #[test]
    fn a_top_level_union_of_two_types_is_refused() {
        assert!(DatumDecoder::new(r#"["int","string"]"#, None, false).is_err());
    }
}
