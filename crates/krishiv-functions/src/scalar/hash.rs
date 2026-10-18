// SPDX-License-Identifier: Apache-2.0
//! Hash functions (P2-T027 kernels): `hash` (Murmur3), `xxhash64`, `md5`,
//! `sha`/`sha1`, `sha2` and `crc32`.
//!
//! `hash` and `xxhash64` take any number of columns of any hashable type.
//! The digests take a `binary` column (the analyzer casts a string to it:
//! the UTF-8 bytes) and `sha2` an `int` bit length.

use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_kernels::hash as kh;

use super::args::{Inputs, is_binary, is_string};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["hash", "xxhash64", "md5", "sha", "sha1", "sha2", "crc32"];

/// Binds the hash function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let bytes = |t: &DataType| is_binary(t) || is_string(t);
    let n = types.len();
    let binding = match (listed, types.as_slice()) {
        ("hash" | "xxhash64", _) if n > 0 => {
            let wide = listed == "xxhash64";
            let out = if wide {
                DataType::Int64
            } else {
                DataType::Int32
            };
            Binding::new(out, move |x: &Inputs<'_>| {
                let cols = (0..n)
                    .map(|i| x.plain_column(i))
                    .collect::<Result<Vec<_>, _>>()?;
                let refs: Vec<&dyn Array> = cols.iter().map(AsRef::as_ref).collect();
                Ok(if wide {
                    Arc::new(kh::xxhash64(&refs, x.rows())?) as ArrayRef
                } else {
                    Arc::new(kh::hash(&refs, x.rows())?) as ArrayRef
                })
            })
        }
        ("md5" | "sha" | "sha1", [t]) if bytes(t) => {
            let md5 = listed == "md5";
            Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                let c = x.plain_column(0)?;
                Ok(if md5 {
                    Arc::new(kh::md5(c.as_ref())?) as ArrayRef
                } else {
                    Arc::new(kh::sha1(c.as_ref())?) as ArrayRef
                })
            })
        }
        ("crc32", [t]) if bytes(t) => Binding::new(DataType::Int64, |x: &Inputs<'_>| {
            Ok(Arc::new(kh::crc32(x.plain_column(0)?.as_ref())?) as ArrayRef)
        }),
        ("sha2", [t, DataType::Int32]) if bytes(t) => {
            Binding::new(DataType::Utf8, |x: &Inputs<'_>| {
                let bits = x.prim::<Int32Type>(1)?;
                Ok(Arc::new(kh::sha2(x.plain_column(0)?.as_ref(), bits.operand())?) as ArrayRef)
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
