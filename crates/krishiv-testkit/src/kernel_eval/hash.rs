// SPDX-License-Identifier: Apache-2.0
//! The hash functions of the kernel evaluator (P2-T027): `hash`, `xxhash64`,
//! `md5`, `sha1`/`sha`, `sha2` and `crc32`, plus [`check_case`], which runs a
//! case of the Spark hash fixture (`tests/fixtures/spark-4.2.0/hash.json`)
//! through the kernels, nested values included.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, ListArray, MapArray, StructArray, cast::AsArray};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Fields};
use krishiv_kernels::hash as kh;
use krishiv_kernels::{KernelCtx, Operand};
use krishiv_sql::ast::{self, ExprKind, Literal};

use super::{
    Binder, EvalError, Evaluator, Node, Shape, Skip, is_integral, simple_string, spec_type,
};
use crate::hash_fixture::HashCase;

/// A hash function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HashFn {
    Hash,
    Xxhash64,
    Md5,
    Sha1,
    Sha2,
    Crc32,
}

/// The legacy switch that lets `hash` take a `MAP`.
const MAP_CONF: &str = "spark.sql.legacy.allowHashOnMapType";

/// `NULL` rows of `hash`/`xxhash64` hash to the seed.
pub(super) const SEED: i64 = 42;

impl Binder {
    /// The hash function `name` applied to `args`, or `None` for another name.
    pub(super) fn hash_function(
        self,
        name: &str,
        args: &[&ast::Expr],
    ) -> Option<Result<Node, Skip>> {
        let f = match name {
            "hash" => HashFn::Hash,
            "xxhash64" => HashFn::Xxhash64,
            "md5" => HashFn::Md5,
            "sha1" | "sha" => HashFn::Sha1,
            "sha2" => HashFn::Sha2,
            "crc32" => HashFn::Crc32,
            _ => return None,
        };
        Some(self.hash_call(name, f, args))
    }

    fn hash_call(self, name: &str, f: HashFn, args: &[&ast::Expr]) -> Result<Node, Skip> {
        let arity_ok = match f {
            HashFn::Hash | HashFn::Xxhash64 => !args.is_empty(),
            HashFn::Sha2 => args.len() == 2,
            _ => args.len() == 1,
        };
        if !arity_ok {
            return Err(format!("analysis: {name} with {} arguments", args.len()));
        }
        let mut nodes = args
            .iter()
            .map(|e| self.bind(e))
            .collect::<Result<Vec<_>, _>>()?;
        let ty = match f {
            HashFn::Hash => DataType::Int32,
            HashFn::Xxhash64 | HashFn::Crc32 => DataType::Int64,
            _ => DataType::Utf8,
        };
        if !matches!(f, HashFn::Hash | HashFn::Xxhash64) {
            // A string is its UTF-8 bytes; only NULL needs a type.
            let first = nodes.remove(0);
            let first = match first.ty() {
                DataType::Binary | DataType::Utf8 => first,
                DataType::Null => self.implicit(first, &DataType::Binary),
                t => return Err(format!("analysis: {name} of {}", simple_string(&t))),
            };
            nodes.insert(0, first);
        }
        if f == HashFn::Sha2 {
            let bits = nodes.remove(1);
            let bits = match bits.ty() {
                DataType::Int32 => bits,
                DataType::Null | DataType::Utf8 => self.implicit(bits, &DataType::Int32),
                t if is_integral(&t) => self.implicit(bits, &DataType::Int32),
                t => {
                    return Err(format!(
                        "analysis: sha2 bit length of {}",
                        simple_string(&t)
                    ));
                }
            };
            nodes.push(bits);
        }
        Ok(Node::Hash(f, nodes, ty))
    }
}

impl Evaluator {
    /// Evaluates hash function `f` over `args`.
    pub(super) fn hash_fn(&self, f: HashFn, args: &[Node]) -> Result<ArrayRef, EvalError> {
        let v = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = v.first().map_or(1, Array::len);
        let cols: Vec<&dyn Array> = v.iter().map(AsRef::as_ref).collect();
        Ok(match f {
            HashFn::Hash => Arc::new(kh::hash(&cols, rows)?),
            HashFn::Xxhash64 => Arc::new(kh::xxhash64(&cols, rows)?),
            HashFn::Md5 => Arc::new(kh::md5(cols[0])?),
            HashFn::Sha1 => Arc::new(kh::sha1(cols[0])?),
            HashFn::Crc32 => Arc::new(kh::crc32(cols[0])?),
            HashFn::Sha2 => {
                let bits = v[1].as_primitive::<Int32Type>();
                Arc::new(kh::sha2(cols[0], Operand::Array(bits))?)
            }
        })
    }
}

/// The error class of an analysis-time rejection of hashing `types`, if any:
/// Spark refuses a `MAP` (also nested) unless the legacy switch is on.
#[must_use]
pub fn analysis_error(types: &[DataType], conf: &BTreeMap<String, String>) -> Option<&'static str> {
    fn has_map(t: &DataType) -> bool {
        match t {
            DataType::Map(..) => true,
            DataType::List(f) | DataType::LargeList(f) => has_map(f.data_type()),
            DataType::Struct(fs) => fs.iter().any(|f| has_map(f.data_type())),
            _ => false,
        }
    }
    let allowed = conf.get(MAP_CONF).is_some_and(|v| v == "true");
    (!allowed && types.iter().any(has_map)).then_some("DATATYPE_MISMATCH.HASH_MAP_TYPE")
}

/// Spark's `typeof` text of a type, nested types included.
fn type_string(t: &DataType) -> String {
    match t {
        DataType::List(f) => format!("array<{}>", type_string(f.data_type())),
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(kv) if kv.len() == 2 => {
                format!(
                    "map<{},{}>",
                    type_string(kv[0].data_type()),
                    type_string(kv[1].data_type())
                )
            }
            other => other.to_string(),
        },
        DataType::Struct(fs) => {
            let fields: Vec<String> = fs
                .iter()
                .map(|f| format!("{}:{}", f.name(), type_string(f.data_type())))
                .collect();
            format!("struct<{}>", fields.join(","))
        }
        other => simple_string(other),
    }
}

/// A `CAST` target type, nested types included.
fn full_type(t: &ast::DataTypeSpec) -> Result<DataType, Skip> {
    Ok(match &t.kind {
        ast::DataTypeKind::Array(Some(el)) => {
            DataType::List(Arc::new(Field::new("item", full_type(el)?, true)))
        }
        ast::DataTypeKind::Map(Some(kv)) => map_type(full_type(&kv.0)?, full_type(&kv.1)?),
        ast::DataTypeKind::Struct(Some(fields)) => DataType::Struct(
            fields
                .iter()
                .map(|f| Ok(Field::new(&f.name.value, full_type(&f.data_type)?, true)))
                .collect::<Result<Vec<_>, Skip>>()?
                .into(),
        ),
        _ => spec_type(t)?,
    })
}

fn map_type(key: DataType, value: DataType) -> DataType {
    DataType::Map(Arc::new(entries_field(key, value)), false)
}

fn entries_field(key: DataType, value: DataType) -> Field {
    Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", key, false),
            Field::new("value", value, true),
        ])),
        false,
    )
}

/// The argument expressions of `hash(<args>)` as one-row columns.
fn build_args(args: &str, conf: &BTreeMap<String, String>) -> Result<Vec<ArrayRef>, Skip> {
    let opts = krishiv_sql::ParserOptions::from_conf(|k| conf.get(k).map(String::as_str));
    let q = krishiv_sql::parse_query(&format!("SELECT {args}"), &opts)
        .map_err(|e| format!("does not parse: {e}"))?;
    let ast::SetExprKind::Select(s) = &q.body.kind else {
        return Err("not a SELECT".to_owned());
    };
    s.projection.iter().map(|p| build(&p.expr)).collect()
}

fn leaf(e: &ast::Expr) -> Result<ArrayRef, Skip> {
    let binder = Binder {
        ansi: true,
        allow_loss: true,
        try_: false,
    };
    let node = binder.bind(e)?;
    let ev = Evaluator {
        ctx: KernelCtx::new(),
        ansi: true,
        shape: Shape::Arrays,
        invalid_utf8: std::cell::Cell::new(false),
    };
    ev.eval(&node).map_err(|e| match e {
        EvalError::Kernel(k) => k.to_string(),
        EvalError::Other(s) => s,
    })
}

/// A one-row column from a constructor call, a nested cast, or a scalar.
fn build(e: &ast::Expr) -> Result<ArrayRef, Skip> {
    match &e.kind {
        ExprKind::Nested(inner) => build(inner),
        ExprKind::Function(f) => {
            let name = f
                .name
                .parts
                .last()
                .map(|p| p.value.to_ascii_lowercase())
                .unwrap_or_default();
            let args: Vec<&ast::Expr> = f.args.iter().map(|a| &a.expr).collect();
            match name.as_str() {
                "array" => array_of(&args),
                "named_struct" => struct_of(&args),
                "map" => map_of(&args),
                _ => leaf(e),
            }
        }
        ExprKind::Literal(Literal::Typed { type_name, value })
            if type_name.value.eq_ignore_ascii_case("timestamp") && value.ends_with("+00:00") =>
        {
            // A UTC offset is the session zone (UTC): drop it.
            let local = value.trim_end_matches("+00:00");
            match super::timestamp_lit(local, super::ltz())? {
                Node::Lit(a) => Ok(a),
                _ => Err("timestamp literal".to_owned()),
            }
        }
        ExprKind::Cast {
            expr, data_type, ..
        } if matches!(&data_type.kind, ast::DataTypeKind::Primitive { name, .. } if name == "binary") =>
        {
            // `CAST(string AS BINARY)` is the UTF-8 bytes.
            let inner = build(expr)?;
            match inner.data_type() {
                DataType::Utf8 => Ok(Arc::new(
                    inner
                        .as_string::<i32>()
                        .iter()
                        .map(|s| s.map(str::as_bytes))
                        .collect::<arrow_array::BinaryArray>(),
                )),
                DataType::Null => Ok(arrow_array::new_null_array(&DataType::Binary, 1)),
                t => Err(format!("cast {} to binary", type_string(t))),
            }
        }
        ExprKind::Cast {
            expr, data_type, ..
        } if !matches!(data_type.kind, ast::DataTypeKind::Primitive { .. }) => {
            let to = full_type(data_type)?;
            let inner = build(expr)?;
            if inner.data_type() == &DataType::Null {
                return Ok(arrow_array::new_null_array(&to, 1));
            }
            retype_empty(&inner, &to)
        }
        _ => leaf(e),
    }
}

/// An empty `array()`/`map()` cast to its element types.
fn retype_empty(inner: &ArrayRef, to: &DataType) -> Result<ArrayRef, Skip> {
    if inner.data_type() == to {
        return Ok(Arc::clone(inner));
    }
    match (inner.data_type(), to) {
        (DataType::List(_), DataType::List(f)) if inner.as_list::<i32>().values().is_empty() => {
            Ok(Arc::new(ListArray::new(
                Arc::clone(f),
                OffsetBuffer::from_lengths([0]),
                arrow_array::new_empty_array(f.data_type()),
                None,
            )))
        }
        (DataType::Map(..), DataType::Map(f, ordered)) if inner.as_map().entries().is_empty() => {
            let DataType::Struct(kv) = f.data_type() else {
                return Err("map entries".to_owned());
            };
            let cols = kv
                .iter()
                .map(|c| arrow_array::new_empty_array(c.data_type()))
                .collect();
            Ok(Arc::new(MapArray::new(
                Arc::clone(f),
                OffsetBuffer::from_lengths([0]),
                StructArray::new(kv.clone(), cols, None),
                None,
                *ordered,
            )))
        }
        (a, b) => Err(format!("cast {} to {}", type_string(a), type_string(b))),
    }
}

/// Concatenates one-row columns, giving NULL elements the type of the rest.
fn concat_rows(rows: &[ArrayRef]) -> Result<(DataType, ArrayRef), Skip> {
    let ty = rows
        .iter()
        .map(Array::data_type)
        .find(|t| **t != DataType::Null)
        .cloned()
        .unwrap_or(DataType::Null);
    let typed: Vec<ArrayRef> = rows
        .iter()
        .map(|r| {
            if r.data_type() == &DataType::Null {
                Ok(arrow_array::new_null_array(&ty, r.len()))
            } else if r.data_type() == &ty {
                Ok(Arc::clone(r))
            } else {
                Err(format!(
                    "elements of {} and {}",
                    type_string(&ty),
                    type_string(r.data_type())
                ))
            }
        })
        .collect::<Result<_, _>>()?;
    if typed.is_empty() {
        return Ok((
            DataType::Null,
            arrow_array::new_empty_array(&DataType::Null),
        ));
    }
    let refs: Vec<&dyn Array> = typed.iter().map(AsRef::as_ref).collect();
    let all = arrow_select::concat::concat(&refs).map_err(|e| e.to_string())?;
    Ok((ty, all))
}

fn array_of(args: &[&ast::Expr]) -> Result<ArrayRef, Skip> {
    let rows = args
        .iter()
        .map(|e| build(e))
        .collect::<Result<Vec<_>, _>>()?;
    let (ty, values) = concat_rows(&rows)?;
    Ok(Arc::new(ListArray::new(
        Arc::new(Field::new("item", ty, true)),
        OffsetBuffer::from_lengths([rows.len()]),
        values,
        None,
    )))
}

fn string_arg(e: &ast::Expr) -> Result<String, Skip> {
    match &e.kind {
        ExprKind::Literal(Literal::String(s)) => Ok(s.clone()),
        _ => Err("named_struct field name".to_owned()),
    }
}

fn struct_of(args: &[&ast::Expr]) -> Result<ArrayRef, Skip> {
    let mut fields = Vec::new();
    let mut cols = Vec::new();
    for pair in args.chunks(2) {
        let [name, value] = pair else {
            return Err("named_struct arity".to_owned());
        };
        let col = build(value)?;
        fields.push(Field::new(string_arg(name)?, col.data_type().clone(), true));
        cols.push(col);
    }
    Ok(Arc::new(StructArray::new(Fields::from(fields), cols, None)))
}

fn map_of(args: &[&ast::Expr]) -> Result<ArrayRef, Skip> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for pair in args.chunks(2) {
        let [k, v] = pair else {
            return Err("map arity".to_owned());
        };
        keys.push(build(k)?);
        values.push(build(v)?);
    }
    let (kt, keys) = concat_rows(&keys)?;
    let (vt, values) = concat_rows(&values)?;
    let entries = entries_field(kt, vt);
    let DataType::Struct(kv) = entries.data_type().clone() else {
        return Err("map entries".to_owned());
    };
    Ok(Arc::new(MapArray::new(
        Arc::new(entries),
        OffsetBuffer::from_lengths([keys.len()]),
        StructArray::new(kv, vec![keys, values], None),
        None,
        false,
    )))
}

/// Runs one fixture case: the argument types must be Spark's, and `hash` and
/// `xxhash64` must equal Spark's values. A case Spark rejects under its
/// configuration must be rejected by [`analysis_error`] instead.
///
/// # Errors
/// What differed.
pub fn check_case(case: &HashCase) -> Result<(), String> {
    let cols = build_args(&case.expr, &case.conf)?;
    let types: Vec<DataType> = cols.iter().map(|c| c.data_type().clone()).collect();
    let fields: Vec<String> = types
        .iter()
        .enumerate()
        .map(|(i, t)| format!("col{}:{}", i + 1, type_string(t)))
        .collect();
    let ty = format!("struct<{}>", fields.join(","));
    if ty != case.spark_type {
        return Err(format!("type {ty} vs Spark {}", case.spark_type));
    }
    if let Some(class) = analysis_error(&types, &case.conf) {
        return Err(format!(
            "{class}, but Spark accepted it under {:?}",
            case.conf
        ));
    }
    let refs: Vec<&dyn Array> = cols.iter().map(AsRef::as_ref).collect();
    let h = kh::hash(&refs, 1).map_err(|e| e.to_string())?.value(0);
    let x = kh::xxhash64(&refs, 1).map_err(|e| e.to_string())?.value(0);
    if h != case.hash || x != case.xxhash64 {
        return Err(format!(
            "hash {h} / xxhash64 {x} vs Spark {} / {}",
            case.hash, case.xxhash64
        ));
    }
    Ok(())
}

/// The error class `hash(<expr>)` raises at analysis under the default
/// configuration, if the kernels' evaluator can tell.
///
/// # Errors
/// Why the expression could not be built.
pub fn default_conf_error(expr: &str) -> Result<Option<&'static str>, String> {
    let cols = build_args(expr, &BTreeMap::new())?;
    let types: Vec<DataType> = cols.iter().map(|c| c.data_type().clone()).collect();
    Ok(analysis_error(&types, &BTreeMap::new()))
}
