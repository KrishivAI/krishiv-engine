// SPDX-License-Identifier: Apache-2.0
//! `DESCRIBE ... AS JSON` of a stored table (`DescribeTableJsonCommand`'s object).

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::{created_time_iso, json_list, json_str, redact, unsupported};
use crate::store::{CatalogPartition, CatalogTable};

/// A JSON object of string values, keys in the order given, values redacted as `DESCRIBE` redacts them.
fn json_map<'a>(items: impl IntoIterator<Item = (&'a String, &'a String)>) -> String {
    format!(
        "{{{}}}",
        items
            .into_iter()
            .map(|(k, v)| format!("{}:{}", json_str(k), json_str(redact(k, v))))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// A column type as `DESCRIBE ... AS JSON` writes it (`DescribeTableJsonCommand`'s type objects).
fn json_type(t: &DataType) -> Result<String, KrishivError> {
    let name = match t {
        DataType::Boolean => "boolean",
        DataType::Byte => "byte",
        DataType::Short => "short",
        DataType::Integer => "int",
        DataType::Long => "long",
        DataType::Float => "float",
        DataType::Double => "double",
        DataType::Binary => "binary",
        DataType::Date => "date",
        DataType::Timestamp => "timestamp_ltz",
        DataType::TimestampNtz => "timestamp_ntz",
        DataType::String => return Ok(r#"{"name":"string","collation":"UTF8_BINARY"}"#.to_owned()),
        DataType::Decimal(d) => {
            return Ok(format!(
                r#"{{"name":"decimal","precision":{},"scale":{}}}"#,
                d.precision(),
                d.scale()
            ));
        }
        DataType::Array(_) | DataType::Map(_) | DataType::Struct(_) => {
            return Err(unsupported("DESCRIBE ... AS JSON of a nested column type"));
        }
        other => {
            return Ok(format!(
                r#"{{"name":{}}}"#,
                json_str(&other.catalog_string())
            ));
        }
    };
    Ok(format!(r#"{{"name":"{name}"}}"#))
}

/// `DESCRIBE EXTENDED [... PARTITION] ... AS JSON` of a stored table: one JSON object whose keys come in
/// the order Spark's `DescribeTableJsonCommand` writes them (a partition's differ: its values and location
/// come first, its statistics follow `last_access`, and the bucket keys sit after `provider`, before the comment).
pub(super) fn describe_json(
    t: &CatalogTable,
    p: Option<&CatalogPartition>,
    created_by: &str,
) -> Result<String, KrishivError> {
    if t.view.is_some() {
        return Err(unsupported("DESCRIBE ... AS JSON of a view"));
    }
    let columns = t
        .columns
        .iter()
        .map(|c| {
            let mut col = format!(
                r#"{{"name":{},"type":{},"nullable":{}"#,
                json_str(&c.name),
                json_type(&c.data_type)?,
                c.nullable
            );
            if let Some(comment) = &c.comment {
                col.push_str(&format!(r#","comment":{}"#, json_str(comment)));
            }
            if let Some(default) = &c.default {
                col.push_str(&format!(r#","default":{}"#, json_str(default)));
            }
            col.push('}');
            Ok(col)
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    let mut f: Vec<String> = vec![
        format!(r#""table_name":{}"#, json_str(&t.name)),
        format!(r#""catalog_name":{}"#, json_str(crate::catalog::CATALOG)),
        format!(r#""namespace":{}"#, json_list([t.namespace.as_str()])),
        format!(r#""schema_name":{}"#, json_str(&t.namespace)),
        format!(r#""columns":[{}]"#, columns.join(",")),
    ];
    let buckets = t.layout.bucket.as_ref().map(|b| {
        [
            format!(r#""num_buckets":{}"#, b.num_buckets),
            format!(
                r#""bucket_columns":{}"#,
                json_list(b.columns.iter().map(String::as_str))
            ),
            format!(
                r#""sort_columns":{}"#,
                json_list(b.sort_columns.iter().map(String::as_str))
            ),
        ]
    });
    let location = match p {
        Some(p) => {
            let values = p
                .spec
                .iter()
                .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
                .collect::<Vec<_>>()
                .join(",");
            f.push(format!(r#""partition_values":{{{values}}}"#));
            &p.location
        }
        None => {
            f.extend(buckets.iter().flatten().cloned());
            &t.location
        }
    };
    f.push(format!(
        r#""location":{}"#,
        json_str(&krishiv_exec::files::file_display(location))
    ));
    let options = &t.options;
    if !options.is_empty() {
        f.push(format!(
            r#""storage_properties":{}"#,
            json_map(options.iter())
        ));
    }
    if let Some(p) = p.filter(|p| !p.parameters.is_empty()) {
        f.push(format!(
            r#""partition_parameters":{}"#,
            json_map(p.parameters.iter())
        ));
    }
    let created = p.map_or(t.created_ms, |p| p.created_ms);
    f.push(format!(
        r#""created_time":{}"#,
        json_str(&created_time_iso(created))
    ));
    f.push(r#""last_access":"UNKNOWN""#.to_owned());
    if let Some(stats) = p.and_then(|p| p.stats) {
        f.push(format!(
            r#""partition_statistics":{}"#,
            json_str(&crate::partitions::stats_text(stats))
        ));
    }
    f.push(format!(r#""created_by":{}"#, json_str(created_by)));
    f.push(format!(
        r#""type":"{}""#,
        if t.managed { "MANAGED" } else { "EXTERNAL" }
    ));
    f.push(format!(r#""provider":{}"#, json_str(&t.provider_written)));
    if p.is_some() {
        f.extend(buckets.iter().flatten().cloned());
    }
    if let Some(c) = &t.comment {
        f.push(format!(r#""comment":{}"#, json_str(c)));
    }
    if !t.properties.is_empty() {
        f.push(format!(
            r#""table_properties":{}"#,
            json_map(t.properties.iter())
        ));
    }
    if t.is_partitioned() {
        f.push(r#""partition_provider":"Catalog""#.to_owned());
        f.push(format!(
            r#""partition_columns":{}"#,
            json_list(t.layout.partition_columns.iter().map(String::as_str))
        ));
    }
    f.push(r#""collation":"UTF8_BINARY""#.to_owned());
    Ok(format!("{{{}}}", f.join(",")))
}
