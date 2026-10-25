// SPDX-License-Identifier: Apache-2.0
//! Native Rust scalar UDFs (layer SDK).
//!
//! A scalar UDF is a [`ScalarUdf`] implementation registered on a session
//! (`Session::register_udf` in `krishiv-session`): SQL
//! and DataFrame calls of its name resolve to it, the analyzer inserts the
//! implicit casts its [`Signature`] asks for, and the executor invokes it
//! once per batch. Aggregate UDFs do not exist yet.
//!
//! The boundary is plain data: a UDF sees Arrow arrays and an
//! [`InvokeContext`] of plain values (partition index, ANSI flag, session
//! time zone). It never sees a session or catalog handle, never loads code,
//! and never casts its own arguments (the analyzer already did). Plans
//! reference a UDF by [`name`](ScalarUdf::name) and
//! [`version`](ScalarUdf::version) only, so a plan stays pure data that any
//! worker holding the same registration can run.
//!
//! ```
//! use std::sync::Arc;
//!
//! use arrow_array::{ArrayRef, Int32Array};
//! use krishiv_types::DataType;
//! use krishiv_udf_sdk::{
//!     ArgField, InvokeContext, ReturnField, ScalarUdf, Signature, UdfArg, UdfError,
//!     UdfInputType,
//! };
//!
//! #[derive(Debug)]
//! struct Negate;
//!
//! impl ScalarUdf for Negate {
//!     fn name(&self) -> &str {
//!         "negate_i32"
//!     }
//!
//!     fn version(&self) -> u32 {
//!         1
//!     }
//!
//!     fn signature(&self) -> &Signature {
//!         use std::sync::OnceLock;
//!         static SIG: OnceLock<Signature> = OnceLock::new();
//!         SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)]))
//!     }
//!
//!     fn return_field(&self, args: &[ArgField]) -> Result<ReturnField, UdfError> {
//!         let _ = args;
//!         Ok(ReturnField {
//!             data_type: DataType::Integer,
//!             nullable: true,
//!         })
//!     }
//!
//!     fn invoke(
//!         &self,
//!         args: &[UdfArg<'_>],
//!         rows: usize,
//!         ctx: &InvokeContext,
//!     ) -> Result<ArrayRef, UdfError> {
//!         let _ = (rows, ctx);
//!         let input = args[0].array();
//!         let input = input
//!             .as_any()
//!             .downcast_ref::<Int32Array>()
//!             .ok_or_else(|| UdfError::new("negate_i32 needs an INT argument"))?;
//!         Ok(Arc::new(input.iter().map(|v| v.map(|v| v.wrapping_neg())).collect::<Int32Array>()))
//!     }
//! }
//! ```

#![forbid(unsafe_code)]

mod library;
mod udf;

pub use library::UdfLibrary;
pub use udf::{
    ArgField, InvokeContext, NullHandling, ReturnField, ScalarUdf, Signature, UdfArg, UdfError,
    UdfInputType, UdfProperties,
};
