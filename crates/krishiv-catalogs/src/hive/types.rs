// SPDX-License-Identifier: Apache-2.0
//! The Hive metastore structs Krishiv reads, by field id from `hive_metastore.thrift` (apache/hive master and
//! `rel/release-4.0.0`, identical for these structs). Only the fields a read-only catalog needs are kept; every
//! other field is skipped, so a newer metastore's additions never break a read. A field whose wire type is not
//! the one the IDL gives is skipped too, never misread. Sizes and nesting are bounded by the protocol's
//! configuration (`max_message_size`, `max_container_size`, `max_recursion_depth`), set by the client.

use std::collections::BTreeMap;

use thrift::Result;
use thrift::protocol::{TInputProtocol, TType};

use super::limits::Budget;

/// Reads one struct: calls `field(protocol, id, type)` for each field, which returns whether it read it; a field
/// it did not read is skipped.
pub(crate) fn read_fields(
    p: &mut dyn TInputProtocol,
    mut field: impl FnMut(&mut dyn TInputProtocol, i16, TType) -> Result<bool>,
) -> Result<()> {
    p.read_struct_begin()?;
    loop {
        let f = p.read_field_begin()?;
        if f.field_type == TType::Stop {
            break;
        }
        if !field(p, f.id.unwrap_or(0), f.field_type)? {
            p.skip(f.field_type)?;
        }
        p.read_field_end()?;
    }
    p.read_struct_end()
}

/// A `list<T>` of `expected` elements, `item` read for each. The elements are spent from the call's budget before
/// anything is allocated, a header of the wrong element type is refused, and a negative length is a protocol error.
pub(crate) fn read_list<T>(
    p: &mut dyn TInputProtocol,
    budget: &Budget,
    expected: TType,
    mut item: impl FnMut(&mut dyn TInputProtocol) -> Result<T>,
) -> Result<Vec<T>> {
    let header = p.read_list_begin()?;
    let n = usize::try_from(header.size).map_err(|_| negative_size())?;
    if n > 0 && header.element_type != expected {
        return Err(wrong_type("a list"));
    }
    budget.take_elements(n)?;
    let mut out = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        out.push(item(p)?);
    }
    p.read_list_end()?;
    Ok(out)
}

/// A `list<string>`.
pub(crate) fn read_strings(p: &mut dyn TInputProtocol, budget: &Budget) -> Result<Vec<String>> {
    read_list(p, budget, TType::String, |p| p.read_string())
}

/// A `map<string, string>`.
pub(crate) fn read_string_map(
    p: &mut dyn TInputProtocol,
    budget: &Budget,
) -> Result<BTreeMap<String, String>> {
    let header = p.read_map_begin()?;
    let n = usize::try_from(header.size).map_err(|_| negative_size())?;
    if n > 0 && (header.key_type != Some(TType::String) || header.value_type != Some(TType::String))
    {
        return Err(wrong_type("a map"));
    }
    budget.take_elements(n)?;
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let key = p.read_string()?;
        let value = p.read_string()?;
        out.insert(key, value);
    }
    p.read_map_end()?;
    Ok(out)
}

fn negative_size() -> thrift::Error {
    thrift::Error::Protocol(thrift::ProtocolError::new(
        thrift::ProtocolErrorKind::NegativeSize,
        "a container with a negative length",
    ))
}

fn wrong_type(what: &str) -> thrift::Error {
    thrift::Error::Protocol(thrift::ProtocolError::new(
        thrift::ProtocolErrorKind::InvalidData,
        format!("{what} of an unexpected element type"),
    ))
}

/// `FieldSchema`: 1 name, 2 type, 3 comment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldSchema {
    /// The column name.
    pub name: String,
    /// The Hive type string (`int`, `array<string>`, `decimal(10,2)`).
    pub type_name: String,
    /// The column comment.
    pub comment: Option<String>,
}

impl FieldSchema {
    pub(crate) fn read(p: &mut dyn TInputProtocol) -> Result<Self> {
        let mut out = Self::default();
        read_fields(p, |p, id, t| {
            match (id, t) {
                (1, TType::String) => out.name = p.read_string()?,
                (2, TType::String) => out.type_name = p.read_string()?,
                (3, TType::String) => out.comment = Some(p.read_string()?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(out)
    }
}

/// `SerDeInfo`: 1 name, 2 serializationLib, 3 parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SerDeInfo {
    /// The serde's name.
    pub name: Option<String>,
    /// The serde class (`org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe`).
    pub serialization_lib: Option<String>,
    /// The serde parameters (`field.delim`, `serialization.format`).
    pub parameters: BTreeMap<String, String>,
}

impl SerDeInfo {
    pub(crate) fn read(p: &mut dyn TInputProtocol, b: &Budget) -> Result<Self> {
        let mut out = Self::default();
        read_fields(p, |p, id, t| {
            match (id, t) {
                (1, TType::String) => out.name = Some(p.read_string()?),
                (2, TType::String) => out.serialization_lib = Some(p.read_string()?),
                (3, TType::Map) => out.parameters = read_string_map(p, b)?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(out)
    }
}

/// `StorageDescriptor`: 1 cols, 2 location, 3 inputFormat, 4 outputFormat, 5 compressed, 6 numBuckets, 7 serdeInfo,
/// 8 bucketCols, 10 parameters (`sortCols`, 9, is not needed to read a table).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StorageDescriptor {
    /// The data columns.
    pub cols: Vec<FieldSchema>,
    /// Where the files are.
    pub location: Option<String>,
    /// The Hadoop input format class.
    pub input_format: Option<String>,
    /// The Hadoop output format class.
    pub output_format: Option<String>,
    /// Whether the files are compressed.
    pub compressed: bool,
    /// The bucket count (`-1` when not bucketed).
    pub num_buckets: i32,
    /// The serde.
    pub serde_info: Option<SerDeInfo>,
    /// The bucketing columns.
    pub bucket_cols: Vec<String>,
    /// The storage parameters.
    pub parameters: BTreeMap<String, String>,
}

impl StorageDescriptor {
    pub(crate) fn read(p: &mut dyn TInputProtocol, b: &Budget) -> Result<Self> {
        let mut out = Self::default();
        read_fields(p, |p, id, t| {
            match (id, t) {
                (1, TType::List) => {
                    out.cols = read_list(p, b, TType::Struct, FieldSchema::read)?;
                }
                (2, TType::String) => out.location = Some(p.read_string()?),
                (3, TType::String) => out.input_format = Some(p.read_string()?),
                (4, TType::String) => out.output_format = Some(p.read_string()?),
                (5, TType::Bool) => out.compressed = p.read_bool()?,
                (6, TType::I32) => out.num_buckets = p.read_i32()?,
                (7, TType::Struct) => out.serde_info = Some(SerDeInfo::read(p, b)?),
                (8, TType::List) => out.bucket_cols = read_strings(p, b)?,
                (10, TType::Map) => out.parameters = read_string_map(p, b)?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(out)
    }
}

/// `Table`: 1 tableName, 2 dbName, 3 owner, 4 createTime, 7 sd, 8 partitionKeys, 9 parameters, 10 viewOriginalText,
/// 11 viewExpandedText, 12 tableType.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    /// The table name.
    pub table_name: String,
    /// The database.
    pub db_name: String,
    /// The owner.
    pub owner: Option<String>,
    /// Creation time, seconds since the epoch.
    pub create_time: i32,
    /// Where and how the data is stored; absent for some views.
    pub sd: Option<StorageDescriptor>,
    /// The partition columns, in order.
    pub partition_keys: Vec<FieldSchema>,
    /// The table properties (`EXTERNAL`, `transient_lastDdlTime`, `spark.sql.sources.provider`).
    pub parameters: BTreeMap<String, String>,
    /// A view's text as written.
    pub view_original_text: Option<String>,
    /// A view's text expanded.
    pub view_expanded_text: Option<String>,
    /// `MANAGED_TABLE`, `EXTERNAL_TABLE`, `VIRTUAL_VIEW`, `MATERIALIZED_VIEW`.
    pub table_type: Option<String>,
}

impl Table {
    pub(crate) fn read(p: &mut dyn TInputProtocol, b: &Budget) -> Result<Self> {
        let mut out = Self::default();
        read_fields(p, |p, id, t| {
            match (id, t) {
                (1, TType::String) => out.table_name = p.read_string()?,
                (2, TType::String) => out.db_name = p.read_string()?,
                (3, TType::String) => out.owner = Some(p.read_string()?),
                (4, TType::I32) => out.create_time = p.read_i32()?,
                (7, TType::Struct) => out.sd = Some(StorageDescriptor::read(p, b)?),
                (8, TType::List) => {
                    out.partition_keys = read_list(p, b, TType::Struct, FieldSchema::read)?;
                }
                (9, TType::Map) => out.parameters = read_string_map(p, b)?,
                (10, TType::String) => out.view_original_text = Some(p.read_string()?),
                (11, TType::String) => out.view_expanded_text = Some(p.read_string()?),
                (12, TType::String) => out.table_type = Some(p.read_string()?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(out)
    }
}

/// `Database`: 1 name, 2 description, 3 locationUri, 4 parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Database {
    /// The database name.
    pub name: String,
    /// Its description.
    pub description: Option<String>,
    /// Where its tables live by default.
    pub location_uri: Option<String>,
    /// Its properties.
    pub parameters: BTreeMap<String, String>,
}

impl Database {
    pub(crate) fn read(p: &mut dyn TInputProtocol, b: &Budget) -> Result<Self> {
        let mut out = Self::default();
        read_fields(p, |p, id, t| {
            match (id, t) {
                (1, TType::String) => out.name = p.read_string()?,
                (2, TType::String) => out.description = Some(p.read_string()?),
                (3, TType::String) => out.location_uri = Some(p.read_string()?),
                (4, TType::Map) => out.parameters = read_string_map(p, b)?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(out)
    }
}
