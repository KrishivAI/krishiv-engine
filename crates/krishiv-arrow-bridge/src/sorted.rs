// SPDX-License-Identifier: Apache-2.0
//! Arrow 58's C Data Interface export drops a map's keys-sorted flag: the field
//! export (`arrow-schema` 58.4.1 `ffi.rs:801-815`) replaces the flags the type export
//! set, so `Map(_, true)` crosses as `Map(_, false)`. Arrow 60 keeps it
//! (`ffi.rs:892`). This module puts the flag back from the Arrow 58 types, in safe
//! code: the schema's types, and the imported array data retyped to match.

use std::sync::Arc;

use arrow_data::ArrayData;
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};

/// `new` (imported from `old` over the C Data Interface) with every map's
/// keys-sorted flag as `old` has it.
pub(crate) fn restore_type(old: &arrow_schema_58::DataType, new: &DataType) -> DataType {
    use arrow_schema_58::DataType as Old;
    match (old, new) {
        (Old::Struct(o), DataType::Struct(n)) => DataType::Struct(restore_fields(o, n)),
        (Old::List(o), DataType::List(n)) => DataType::List(Arc::new(restore_field(o, n))),
        (Old::LargeList(o), DataType::LargeList(n)) => {
            DataType::LargeList(Arc::new(restore_field(o, n)))
        }
        (Old::FixedSizeList(o, _), DataType::FixedSizeList(n, len)) => {
            DataType::FixedSizeList(Arc::new(restore_field(o, n)), *len)
        }
        (Old::Map(o, sorted), DataType::Map(n, _)) => {
            DataType::Map(Arc::new(restore_field(o, n)), *sorted)
        }
        (Old::Dictionary(_, o), DataType::Dictionary(k, n)) => {
            DataType::Dictionary(k.clone(), Box::new(restore_type(o, n)))
        }
        _ => new.clone(),
    }
}

fn restore_field(old: &arrow_schema_58::Field, new: &Field) -> Field {
    new.clone()
        .with_data_type(restore_type(old.data_type(), new.data_type()))
}

fn restore_fields(old: &arrow_schema_58::Fields, new: &Fields) -> Fields {
    old.iter()
        .zip(new.iter())
        .map(|(o, n)| restore_field(o, n))
        .collect()
}

/// `new` with the map flags of `old` restored on every field.
pub(crate) fn restore_schema(old: &arrow_schema_58::Schema, new: &Schema) -> Schema {
    let fields = restore_fields(old.fields(), new.fields());
    Schema::new_with_metadata(fields, new.metadata().clone())
}

/// `data` relabelled as `target`, which differs from its type at most in map
/// keys-sorted flags; the buffers are shared, and the result is validated.
pub(crate) fn retype(data: ArrayData, target: &DataType) -> Result<ArrayData, ArrowError> {
    if data.data_type() == target {
        return Ok(data);
    }
    let child_targets: Vec<DataType> = match target {
        DataType::Struct(fields) => fields.iter().map(|f| f.data_type().clone()).collect(),
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => vec![f.data_type().clone()],
        DataType::Dictionary(_, values) => vec![values.as_ref().clone()],
        _ => Vec::new(),
    };
    let children = data
        .child_data()
        .iter()
        .zip(&child_targets)
        .map(|(c, t)| retype(c.clone(), t))
        .collect::<Result<Vec<_>, _>>()?;
    data.into_builder()
        .data_type(target.clone())
        .child_data(children)
        .build()
}
