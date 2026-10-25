// SPDX-License-Identifier: Apache-2.0
//! Spark-compatible JSON functions over Arrow arrays (layer L2).
//!
//! The kernels behind `krishiv-functions`' `json` family: `from_json`
//! (schema-directed, streaming into Arrow builders), `to_json` (Arrow to
//! JSON text), `schema_of_json` (type inference), `get_json_object`
//! (JSONPath lookup), `json_array_length` and `json_object_keys`
//! (structural queries). The text parser is shared: a strict JSON lexer
//! with the `JSONOptions` flags Spark's JSON reader has (comments, single
//! quotes, leading zeros, ...). Malformed input is a `NULL` (Spark's
//! `PERMISSIVE` mode), never an error; only unreachable states (a kernel
//! called with the wrong Arrow type) raise.
//!
//! The streaming reader never builds a document tree: values go straight
//! from the lexer into Arrow builders, and a nested failure poisons only
//! the value that holds it (a struct field catches it and is `NULL`; an
//! array element or map value nulls the whole array or map, as Spark's
//! `JacksonParser` does). The navigation queries (`get_json_object`,
//! inference) walk the text with the same lexer and only materialize the
//! matched values.
//!
//! Dates and timestamps use the session time zone and Spark's default
//! grammars ([`crate::text`]), or the `dateFormat` / `timestampFormat` /
//! `timestampNTZFormat` patterns when the options name them, exactly like
//! the JSON data source.

mod base64;
mod infer;
mod options;
mod parse;
mod path;
mod read;
mod write;

pub use infer::infer_json;
pub use options::JsonOptions;
pub use path::{get_json_object, json_array_length, json_object_keys};
pub use read::read_json;
pub use write::write_json;
