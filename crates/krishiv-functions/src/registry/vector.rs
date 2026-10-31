// SPDX-License-Identifier: Apache-2.0
//! The vector scalar signatures: `vector_norm`, `vector_normalize`,
//! `vector_cosine_similarity`, `vector_inner_product` and
//! `vector_l2_distance` (Spark 4.2's `VectorFunctions`). Only `ARRAY<FLOAT>`
//! is a vector and the degree a `FLOAT`: nothing is cast, any other type is
//! `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` (an untyped `NULL` included,
//! quoted as `VOID`). Checked against the `vector-norm.sql` and
//! `vector-distance.sql` goldens and a Spark 4.2.0 run.

use krishiv_common::error::{DatatypeMismatchUnexpectedInputType, KrishivError};
use krishiv_types::{ArrayType, DataType};

use super::defs::{Spec, scalar};
use super::hof::{call_sql, quoted};
use crate::signature::{ArgType, CallTypes, Inputs};

/// Whether `t` is `ARRAY<FLOAT>` (either `containsNull`).
fn is_vector(t: Option<&DataType>) -> bool {
    matches!(t, Some(DataType::Array(a)) if *a.element_type() == DataType::Float)
}

/// Checks the arguments: `ARRAY<FLOAT>` for each of the first `vectors`, then
/// a `FLOAT` degree.
fn check_all(name: &str, c: &CallTypes<'_>, vectors: usize) -> Result<(), KrishivError> {
    for i in 0..c.args().len() {
        let (ok, required) = if i < vectors {
            (is_vector(c.data_type(i)), "\"ARRAY<FLOAT>\"")
        } else {
            (matches!(c.data_type(i), Some(DataType::Float)), "\"FLOAT\"")
        };
        if !ok {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql(name, c),
                param_index: ["first", "second"][i.min(1)].to_owned(),
                required_type: required.to_owned(),
                input_sql: format!("\"{}\"", c.sql(i).unwrap_or_default()),
                input_type: quoted(c.data_type(i)),
            }
            .into());
        }
    }
    Ok(())
}

macro_rules! rule {
    ($fn:ident, $name:literal, $vectors:literal, $result:expr) => {
        fn $fn(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            check_all($name, c, $vectors)?;
            Ok($result)
        }
    };
}

rule!(norm, "vector_norm", 1, ArgType::new(DataType::Float, true));
rule!(
    normalize,
    "vector_normalize",
    1,
    ArgType::new(
        DataType::Array(ArrayType::new(DataType::Float, false).into()),
        true
    )
);
rule!(
    cosine,
    "vector_cosine_similarity",
    2,
    ArgType::new(DataType::Float, true)
);
rule!(
    inner,
    "vector_inner_product",
    2,
    ArgType::new(DataType::Float, true)
);
rule!(
    l2,
    "vector_l2_distance",
    2,
    ArgType::new(DataType::Float, true)
);

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["vector_norm"], &Inputs::Any, norm);
    out.extend(scalar(&["vector_normalize"], &Inputs::Any, normalize));
    out.extend(scalar(&["vector_cosine_similarity"], &Inputs::Any, cosine));
    out.extend(scalar(&["vector_inner_product"], &Inputs::Any, inner));
    out.extend(scalar(&["vector_l2_distance"], &Inputs::Any, l2));
    out
}
