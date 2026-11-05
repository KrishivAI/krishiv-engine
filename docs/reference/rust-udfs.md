<!-- SPDX-License-Identifier: Apache-2.0 -->
# Native Rust scalar UDFs

An application that embeds Krishiv can add its own scalar functions, written in Rust and compiled into the
same binary. A UDF implements `ScalarUdf` from `krishiv-udf-sdk` (re-exported by `krishiv` and
`krishiv-session`), is registered on a session, and is then called from SQL and from the DataFrame API like
a built-in. Aggregate UDFs are not available yet.

## A complete example

```rust
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int32Array};
use krishiv::functions::{call_udf, col};
use krishiv::{
    ArgField, InvokeContext, ReturnField, ScalarUdf, Session, Signature, UdfArg, UdfError,
    UdfInputType,
};
use krishiv_types::DataType;

#[derive(Debug)]
struct AddOne;

impl ScalarUdf for AddOne {
    fn name(&self) -> &str {
        "add_one"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)]))
    }

    fn return_field(&self, _args: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField { data_type: DataType::Integer, nullable: true })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        _rows: usize,
        _ctx: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        let input = args[0].array();
        let ints = input
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| UdfError::new("add_one needs an INT argument"))?;
        let out: Int32Array = ints
            .iter()
            .map(|v| v.map(|v| v.checked_add(1)).transpose())
            .collect::<Option<_>>()
            .ok_or_else(|| UdfError::new("add_one overflowed"))?;
        Ok(Arc::new(out))
    }
}

let session = Session::new()?;
session.register_udf(AddOne)?;

// SQL
let df = session.sql("SELECT add_one(id) FROM people").await?;

// DataFrame: call_udf (Spark's name) and call_function resolve the same way
let df = session.table("people").select([call_udf("add_one", [col("id")])]);
```

## What the engine does for you

- **Typing.** `Signature` says how arguments are taken: `Exact` (implicit casts as for a built-in with the
  same declared inputs), `Variadic` (a minimum count, every argument cast to one type) or `Coercible` (no
  implicit casts: a wrong type is `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`). A wrong argument count is
  `WRONG_NUM_ARGS`. The UDF never casts its own arguments.
- **Result.** `return_field` gives the type and nullability for the argument types after casting. It must
  be pure: the analyzer calls it once per call site, and the executor checks every result against it.
- **Nulls.** With the default `NullHandling::Strict`, a row with any NULL argument is NULL without calling
  the UDF. `Custom` hands the UDF every row.
- **Literals.** An argument is a `UdfArg::Column` of `rows` values or a `UdfArg::Scalar` one-row array. A
  deterministic UDF (the default) whose arguments are all literals is folded once at analysis.
- **Strings and binaries.** Arguments may arrive in any UTF-8 carrier (`Utf8`, `Utf8View`, `LargeUtf8`) or
  binary carrier; return the plain `Utf8` or `Binary` arrays.

## Name resolution

Function names are case-insensitive. Built-ins come first on the search path, as in Spark: a UDF named
`upper` registers, but `upper(x)` still calls the built-in; `system.session.upper(x)` (or `session.upper(x)`)
reaches the UDF. An unknown name is `UNRESOLVED_ROUTINE`. Registering a name again replaces the UDF, and
every registration invalidates plans analyzed before it.

## Versions and plans

Plans refer to a UDF by name and `version`, never by code. Bump `version` whenever the function's
behaviour changes. A worker that does not hold that name and version refuses the plan with
`KRISHIV_UDF_NOT_AVAILABLE` instead of running other code. Queries that call a session UDF are never served
from the result cache.

## Errors

A `UdfError` returned from `invoke`, a result of the wrong type, length or nullability, and a panic in any
trait method all fail the query with Spark's `FAILED_EXECUTE_UDF` (SQLSTATE 39000), naming the UDF. A
`UdfError` from `return_field` fails the call at analysis. Don't panic: the engine catches it, but only
after the worker thread has unwound.

## Stability

`krishiv-udf-sdk` follows semver. Its reviewed public surface is `crates/krishiv-udf-sdk/public-api.txt`,
checked nightly. New trait methods are only added with default implementations.
