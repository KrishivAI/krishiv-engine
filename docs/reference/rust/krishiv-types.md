**Version:** 0.2.0-alpha

# Module `krishiv_types`

Internal type representations.

[`DataType`] models Spark SQL data types with Spark's renderings
(`simpleString`, `catalogString`, `sql`, `typeName`, `json`), maps them to and from
Arrow ([`arrow`]), and parses Spark DDL type strings ([`parse_data_type`]).
[`batch::Batch`] is the engine's unit of data: Arrow columns plus row weights.

## Modules

## Module `arrow`

Spark data type ↔ Arrow mapping.

| Spark | Canonical Arrow type | Also accepted as storage |
|---|---|---|
| `boolean` | `Boolean` | |
| `tinyint` / `smallint` / `int` / `bigint` | `Int8` / `Int16` / `Int32` / `Int64` | |
| `float` / `double` | `Float32` / `Float64` | |
| `string` | `Utf8View` | `Utf8`, `LargeUtf8`; `BinaryView`, `Binary`, `LargeBinary` for bytes that are not UTF-8 |
| `binary` | `BinaryView` | `Binary`, `LargeBinary` |
| `decimal(p,s)` | `Decimal64(p, s)` for `p <= 18`, else `Decimal128(p, s)`; `Decimal64(1, 0)` for `decimal(0,0)` | `Decimal32`, `Decimal128` |
| `date` | `Date32` | |
| `timestamp` | `Timestamp(Microsecond, "UTC")` | |
| `timestamp_ntz` | `Timestamp(Microsecond, None)` | |
| `time(p)` | `Time64(Nanosecond)`, tagged when `p < 6` | |
| `interval year to month` | `Interval(YearMonth)`, tagged for other fields | |
| `interval day to second` | `Duration(Microsecond)`, tagged for other fields | |
| `interval` | `Interval(MonthDayNano)` | |
| `geometry(s)` / `geography(s)` | `Struct<srid: Int32, wkb: BinaryView>`, both children non-null, always tagged | |
| `array<e>` | `List` of a field `element` (nullable when `containsNull`) | `LargeList`, any child name |
| `map<k,v>` | `Map` of a non-null struct `entries` of `key` (non-null) and `value` (nullable when `valueContainsNull`), unsorted | any child names |
| `struct<...>` | `Struct` of the fields | |
| `variant` | `Struct<metadata: BinaryView, value: BinaryView>`, both children non-null and marked with [`VARIANT_CHILD_METADATA_KEY`]; the field also carries the Arrow extension name `arrow.parquet.variant` | |

Nested children follow Spark's `ArrowUtils` names. Each child is a field
built as [`DataType::to_arrow_field`] builds a column, so it carries its
own type tag when it needs one, and a struct field's non-empty metadata
rides in [`SPARK_METADATA_KEY`]. Nesting deeper than
[`DataType::MAX_NESTING_DEPTH`] is refused.

The plan schema (Spark [`DataType`]) is the source of truth; Arrow types
and the [`SPARK_TYPE_METADATA_KEY`] field tag only carry it through storage
and edge round trips.

- **Binary-safe strings.** A string column whose bytes are not valid UTF-8
  is stored as a binary Arrow type, and its field carries the Spark type
  under [`SPARK_TYPE_METADATA_KEY`] so it reads back as `string`.
- **Decimals.** The carrier follows the declared precision alone
  ([`DECIMAL64_MAX_PRECISION`]), so one Spark type has one
  carrier between operators, in spill files and in shuffle. Client edges
  (Connect/Flight, Python, UDFs) widen to `Decimal128`, what Spark sends.
  Arrow rejects precision 0, so `decimal(0,0)` (whose only value is 0) is
  stored with precision 1 and the field tag; use the field form for it.
- **Timestamps.** `timestamp` values are UTC instants. An input Arrow
  timestamp with any zone label is a `timestamp`; re-label it to `"UTC"` at
  ingest ([`normalize_ingest_type`]). Client egress re-labels to the session
  time zone and converts strings to `Utf8`, as Spark does.
- **Times.** A `time` value is nanoseconds since midnight, the long Spark
  hashes and sends over Spark Connect; the precision lives in the field
  tag only.
- **Intervals.** The fields of an ANSI interval live in the field tag only,
  as Spark's `ArrowUtils` loses them too. A calendar interval's nanoseconds
  are always a whole number of microseconds.
- **Spatial values.** Each row holds its SRID and its WKB bytes
  (little-endian, as [`crate::geo::rewrite`] writes it for
  [`crate::geo::ByteOrder::Ndr`]), so a `geometry(any)`
  column can mix SRIDs. The struct alone does not say geometry or
  geography, so these fields always carry the tag.
- **Variants.** Each row holds its own metadata dictionary and value bytes
  (Parquet Variant v1, unshredded: shredding exists only in files). A
  null struct slot is SQL `NULL`; a Variant null is the one-byte value
  `0x00`. The field carries the canonical Arrow extension
  name ([`VARIANT_EXTENSION_NAME`]) so Arrow consumers can recognise it;
  the marked children make the tag unnecessary.

```rust
pub mod arrow { /* ... */ }
```

### Functions

#### Function `is_variant_struct`

**Attributes:**

- `MustUse { reason: None }`

Whether `arrow` is the struct of a variant value.

```rust
pub fn is_variant_struct(arrow: &arrow_schema::DataType) -> bool { /* ... */ }
```

#### Function `normalize_ingest_type`

The Arrow type an ingested column should carry: a zoned microsecond
timestamp is re-labelled to `"UTC"` (the values are already UTC instants);
every other type is returned unchanged.

```rust
pub fn normalize_ingest_type(arrow: &arrow_schema::DataType) -> arrow_schema::DataType { /* ... */ }
```

#### Function `canonical_arrow_type`

**Attributes:**

- `MustUse { reason: None }`

`arrow` with each decimal on the canonical carrier of its precision: what an
Arrow value of any carrier becomes when it enters a query.

```rust
pub fn canonical_arrow_type(arrow: &arrow_schema::DataType) -> arrow_schema::DataType { /* ... */ }
```

#### Function `client_arrow_type`

**Attributes:**

- `MustUse { reason: None }`

`arrow` with each decimal on `Decimal128`: what a client reads.

```rust
pub fn client_arrow_type(arrow: &arrow_schema::DataType) -> arrow_schema::DataType { /* ... */ }
```

### Constants and Statics

#### Constant `SPARK_TYPE_METADATA_KEY`

Arrow field-metadata key holding the Spark type (its `catalogString`)
when the field's Arrow type alone does not determine it.

```rust
pub const SPARK_TYPE_METADATA_KEY: &str = "krishiv.spark_type";
```

#### Constant `SPARK_METADATA_KEY`

Arrow field-metadata key holding a struct field's Spark `Metadata` (its
JSON), when it has any.

```rust
pub const SPARK_METADATA_KEY: &str = "krishiv.spark_metadata";
```

#### Constant `TIMESTAMP_ARROW_TIME_ZONE`

Time zone of the canonical Arrow type for `timestamp` values, which are
UTC instants; the session time zone is applied only at the edges.

```rust
pub const TIMESTAMP_ARROW_TIME_ZONE: &str = "UTC";
```

#### Constant `SPATIAL_SRID_FIELD`

Name of the SRID child (`Int32`) of a spatial value's struct.

```rust
pub const SPATIAL_SRID_FIELD: &str = "srid";
```

#### Constant `SPATIAL_WKB_FIELD`

Name of the WKB child of a spatial value's struct.

```rust
pub const SPATIAL_WKB_FIELD: &str = "wkb";
```

#### Constant `VARIANT_METADATA_FIELD`

Name of the metadata-dictionary child of a variant value's struct.

```rust
pub const VARIANT_METADATA_FIELD: &str = "metadata";
```

#### Constant `VARIANT_VALUE_FIELD`

Name of the value child of a variant value's struct.

```rust
pub const VARIANT_VALUE_FIELD: &str = "value";
```

#### Constant `VARIANT_CHILD_METADATA_KEY`

Arrow field-metadata key on the two children of a variant struct (value
`metadata` / `value`), so the struct's Arrow type alone says `variant`
and never reads as a user `struct<metadata: binary, value: binary>`.

```rust
pub const VARIANT_CHILD_METADATA_KEY: &str = "krishiv.variant";
```

#### Constant `VARIANT_EXTENSION_NAME`

The canonical Arrow extension name of a variant field.

```rust
pub const VARIANT_EXTENSION_NAME: &str = "arrow.parquet.variant";
```

#### Constant `DECIMAL64_MAX_PRECISION`

The largest precision on the `Decimal64` carrier; above it a decimal is on
`Decimal128`. Setting this to 0 puts every decimal back on `Decimal128`.

```rust
pub const DECIMAL64_MAX_PRECISION: u8 = 18;
```

## Module `batch`

The engine's unit of data: an Arrow [`RecordBatch`] plus per-row
[`Weights`].

A [`Batch`] is plain Arrow. There is no selection vector and no parallel
vector-encoding enum: dictionary and run-end encodings are Arrow's own
array types. Filtering copies the selected rows into new arrays
(copy-on-filter); projecting and slicing share buffers.

Weights make a batch a Z-set delta, so one type serves batch, streaming
and incremental view maintenance ([`ChangeBatch`]). A batch query is an
insert-only stream and carries [`Weights::Const`]`(1)`. Operators must
route that case to the same code path as "no weights" (see
[`Batch::is_unweighted`]); the operations here never turn a constant
weight into a column, except [`Batch::concat`] of batches whose constants
differ.

```
use std::sync::Arc;

use arrow_array::{BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_types::batch::{Batch, Weights};

let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
let data = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))])?;

// A batch query's rows: weight 1 each, no weight column.
let batch = Batch::new(data.clone());
assert!(batch.is_unweighted());

// A change stream: insert row 0, retract row 1, insert row 2 twice.
let change = Batch::with_weights(data, Weights::Column(Int64Array::from(vec![1, -1, 2])))?;
let kept = change.filter(&BooleanArray::from(vec![false, true, true]))?;
assert_eq!(kept.num_rows(), 2);
assert_eq!(kept.weights(), &Weights::Column(Int64Array::from(vec![-1, 2])));
# Ok::<(), Box<dyn std::error::Error>>(())
```

```rust
pub mod batch { /* ... */ }
```

### Types

#### Enum `Weights`

The weight (multiplicity) of each row of a [`Batch`].

A positive weight `w` inserts the row `w` times; a negative one retracts
it. A weight of zero is allowed (the row has no effect) and is removed by
consolidation, not by the operations here.

The weight kind is known statically from the source and the plan:
operators never re-detect a constant inside a [`Weights::Column`].

Weights are `i64`. Operations that multiply or sum them (join
multiplicity, consolidation) must use checked arithmetic and raise an
error on overflow; the operations here only copy them.

The enum is exhaustive on purpose: a new encoding must be a compile error
at every operator that matches on it.

`PartialEq` is structural equality of the representation, not Z-set
equality: `Const(1)` over two rows differs from `Column([1, 1])`.

```rust
pub enum Weights {
    Const(i64),
    Column(arrow_array::Int64Array),
}
```

##### Variants

###### `Const`

Every row has this weight. `Const(1)` is the "no weights" fast path
that batch queries and insert-only streams use.

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `i64` |  |

###### `Column`

One weight per row, for mixed-sign change streams. The array has the
batch's row count and no nulls.

Fields:

| Index | Type | Documentation |
|-------|------|---------------|
| 0 | `arrow_array::Int64Array` |  |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **CastableFrom**
- **Clone**
  - ```rust
    fn clone(self: &Self) -> Weights { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Instrument**
- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &Weights) -> bool { /* ... */ }
    ```

- **Read**
- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnsafeUnpin**
- **UnwindSafe**
- **WithSubscriber**
#### Struct `Batch`

A [`RecordBatch`] whose rows carry [`Weights`].

The row count is the [`RecordBatch`]'s, which Arrow keeps even for a batch
with no columns.

Invariants, upheld by every constructor and operation:
- a [`Weights::Column`] has exactly [`Self::num_rows`] values and no nulls.

`PartialEq` is structural equality (same data, schema and weight
representation), not Z-set equality; compare Z-sets after consolidation.

```rust
pub struct Batch {
    // Some fields omitted
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| *private fields* | ... | *Some fields have been omitted* |

##### Implementations

###### Methods

- ```rust
  pub fn new(data: RecordBatch) -> Self { /* ... */ }
  ```
  An unweighted batch: every row has weight 1 ([`Weights::Const`]`(1)`).

- ```rust
  pub fn with_weights(data: RecordBatch, weights: Weights) -> Result<Self> { /* ... */ }
  ```
  A batch with explicit weights.

- ```rust
  pub fn num_rows(self: &Self) -> usize { /* ... */ }
  ```
  The number of rows.

- ```rust
  pub fn schema(self: &Self) -> SchemaRef { /* ... */ }
  ```
  The Arrow schema of the data columns (the weights are not a column).

- ```rust
  pub fn data(self: &Self) -> &RecordBatch { /* ... */ }
  ```
  The data columns.

- ```rust
  pub fn weights(self: &Self) -> &Weights { /* ... */ }
  ```
  The row weights.

- ```rust
  pub fn is_unweighted(self: &Self) -> bool { /* ... */ }
  ```
  Whether every row has weight 1 ([`Weights::Const`]`(1)`).

- ```rust
  pub fn into_parts(self: Self) -> (RecordBatch, Weights) { /* ... */ }
  ```
  Splits the batch into its data and weights.

- ```rust
  pub fn filter(self: &Self, mask: &BooleanArray) -> Result<Self> { /* ... */ }
  ```
  The rows where `mask` is true, with their weights (copy-on-filter).

- ```rust
  pub fn project(self: &Self, indices: &[usize]) -> Result<Self> { /* ... */ }
  ```
  The columns at `indices`, in that order, with the same rows and

- ```rust
  pub fn slice(self: &Self, offset: usize, len: usize) -> Result<Self> { /* ... */ }
  ```
  `len` rows starting at `offset`, with their weights. Zero-copy: the

- ```rust
  pub fn concat(schema: &SchemaRef, batches: &[Self]) -> Result<Self> { /* ... */ }
  ```
  The rows of `batches`, in order, as one batch with `schema` (copies,

###### Trait Implementations

- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **CastableFrom**
- **Clone**
  - ```rust
    fn clone(self: &Self) -> Batch { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Instrument**
- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &Batch) -> bool { /* ... */ }
    ```

- **Read**
- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnsafeUnpin**
- **UnwindSafe**
- **WithSubscriber**
#### Type Alias `ChangeBatch`

The change batch of streaming and incremental view maintenance: the same
type as [`Batch`], so batch, streaming and IVM share one data path.

```rust
pub type ChangeBatch = Batch;
```

## Module `error`

Error constructors for type parsing and mapping.

Every error is a [`KrishivError`] with Spark 4.2's error class for the
same situation:

| Situation | Error class |
|---|---|
| precision above 38 | `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION` |
| scale above precision | `_LEGACY_ERROR_TEMP_1228` |
| negative scale (from Arrow) | `NEGATIVE_SCALE_DISALLOWED` |
| malformed type string | `PARSE_SYNTAX_ERROR` |
| empty type string | `PARSE_EMPTY_STATEMENT` |
| unterminated `/*` comment | `UNCLOSED_BRACKETED_COMMENT` |
| non-ASCII unquoted name | `INVALID_IDENTIFIER` |
| unknown type name | `UNSUPPORTED_DATATYPE` |
| time precision above 6 | `UNSUPPORTED_TIME_PRECISION` |
| an SRID Spark does not support | `ST_INVALID_SRID_VALUE` |
| malformed WKB ([`crate::geo`]) | `WKB_PARSE_ERROR` |
| a Spark type Krishiv does not support yet | `KRISHIV_UNSUPPORTED_FEATURE` |
| bare `ARRAY`, `MAP` or `STRUCT` | `INCOMPLETE_TYPE_DEFINITION.*` |
| a type nested deeper than [`crate::DataType::MAX_NESTING_DEPTH`] | `FAILED_TO_PARSE_TOO_COMPLEX` |
| JSON that is not a Spark type | `INVALID_JSON_DATA_TYPE` |
| a [`crate::batch::Batch`] invariant broken by the caller (an engine bug) | `INTERNAL_ERROR` |

```rust
pub mod error { /* ... */ }
```

### Types

#### Type Alias `Result`

Result alias for this crate.

```rust
pub type Result<T, E = krishiv_common::error::KrishivError> = std::result::Result<T, E>;
```

## Module `geo`

Well-known binary (WKB) for `geometry` and `geography` values: the
validation Spark's `ST_GeomFromWKB` / `ST_GeogFromWKB` apply, and the
re-encoding `ST_AsBinary` does.

This is a library only for now: no SQL function calls it yet, so
`ST_GeomFromWKB` and `ST_GeogFromWKB` do not raise `WKB_PARSE_ERROR`
from SQL (the `st-functions` golden statements stay in the `time-geo-types`
known-failure category, until the functions are registered).

The input is ISO WKB: the seven OGC geometry types (point to geometry
collection) in XY, XYZ (type + 1000), XYM (+ 2000) or XYZM (+ 3000),
either byte order, which may change from one nested geometry to the
next. Extended WKB flags (SRID, Z or M bits) are not accepted. The
rules follow Spark 4.2:

- every nested geometry has its collection's dimensions, and the
  elements of a `Multi*` collection have its element type;
- ordinates are never infinite, and NaN only in points (an empty point
  is NaN, NaN);
- a line string has 0 or at least 2 points; a ring at least 4, and its
  last point repeats its first (X and Y);
- a count that is negative, or larger than the bytes left allow, is
  refused before any element is read;
- for a geography, every point has `|x| <= 180` and `|y| <= 90`;
- bytes after the geometry are ignored, and dropped from the output.

Errors are `WKB_PARSE_ERROR` with Spark's message and byte position.
Nesting is walked with an explicit stack, so a deeply nested input
cannot overflow the call stack (Spark fails on such inputs, with no
error class); the output is never larger than the input.

```rust
pub mod geo { /* ... */ }
```

### Types

#### Enum `ByteOrder`

The byte order of encoded WKB.

```rust
pub enum ByteOrder {
    Ndr,
    Xdr,
}
```

##### Variants

###### `Ndr`

Little-endian (`NDR`), the storage form and `ST_AsBinary`'s default.

###### `Xdr`

Big-endian (`XDR`).

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **CastableFrom**
- **Clone**
  - ```rust
    fn clone(self: &Self) -> ByteOrder { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Copy**
- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Eq**
- **Equivalent**
  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Hash**
  - ```rust
    fn hash<__H: $crate::hash::Hasher>(self: &Self, state: &mut __H) { /* ... */ }
    ```

- **Instrument**
- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &ByteOrder) -> bool { /* ... */ }
    ```

- **Read**
- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnsafeUnpin**
- **UnwindSafe**
- **WithSubscriber**
#### Enum `SpatialKind`

Which spatial type a value belongs to.

```rust
pub enum SpatialKind {
    Geometry,
    Geography,
}
```

##### Variants

###### `Geometry`

`geometry`: Cartesian coordinates, any range.

###### `Geography`

`geography`: longitude and latitude in degrees.

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **CastableFrom**
- **Clone**
  - ```rust
    fn clone(self: &Self) -> SpatialKind { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Copy**
- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Eq**
- **Equivalent**
  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Hash**
  - ```rust
    fn hash<__H: $crate::hash::Hasher>(self: &Self, state: &mut __H) { /* ... */ }
    ```

- **Instrument**
- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &SpatialKind) -> bool { /* ... */ }
    ```

- **Read**
- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnsafeUnpin**
- **UnwindSafe**
- **WithSubscriber**
### Functions

#### Function `rewrite`

Validates `wkb` as a value of `kind` and re-encodes it in `order`.

```
use krishiv_types::geo::{rewrite, ByteOrder, SpatialKind};

// POINT (1 2), big-endian in, little-endian out.
let xdr = [0, 0, 0, 0, 1, 0x3F, 0xF0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0, 0];
let ndr = rewrite(&xdr, SpatialKind::Geometry, ByteOrder::Ndr).unwrap();
assert_eq!(ndr[..5], [1, 1, 0, 0, 0]);
assert!(rewrite(&[1], SpatialKind::Geometry, ByteOrder::Ndr).is_err());
```

# Errors
`WKB_PARSE_ERROR` with Spark's message and position (see the module
docs for the rules).

```rust
pub fn rewrite(wkb: &[u8], kind: SpatialKind, order: ByteOrder) -> crate::error::Result<Vec<u8>> { /* ... */ }
```

#### Function `to_ewkt`

The text Spark prints a spatial value as (extended well-known text):
`POINT(1 2)`, with `SRID=<n>;` in front unless the SRID is 0.

Only a two-dimensional point is printed so far; any other geometry is
`KRISHIV_UNSUPPORTED_FEATURE`.

```
use krishiv_types::geo::{rewrite, to_ewkt, ByteOrder, SpatialKind};

let xdr = [0, 0, 0, 0, 1, 0x3F, 0xF0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0, 0];
let ndr = rewrite(&xdr, SpatialKind::Geometry, ByteOrder::Ndr).unwrap();
assert_eq!(to_ewkt(4326, &ndr).unwrap(), "SRID=4326;POINT(1 2)");
assert_eq!(to_ewkt(0, &ndr).unwrap(), "POINT(1 2)");
```

# Errors
`KRISHIV_UNSUPPORTED_FEATURE` for anything but a two-dimensional point.

```rust
pub fn to_ewkt(srid: i32, wkb: &[u8]) -> crate::error::Result<String> { /* ... */ }
```

#### Function `holds_spatial`

**Attributes:**

- `MustUse { reason: None }`

Whether `t` is, or holds anywhere inside it, a geometry or geography.

```
use krishiv_types::{geo::holds_spatial, DataType, Srid};

assert!(holds_spatial(&DataType::Geometry { srid: Srid::Any }));
assert!(!holds_spatial(&DataType::Binary));
```

```rust
pub fn holds_spatial(t: &crate::DataType) -> bool { /* ... */ }
```

## Module `interval`

Spark's text forms of interval values (`IntervalUtils.toYearMonthIntervalString`,
`toDayTimeIntervalString` in `ANSI_STYLE`, and `CalendarInterval.toString`).

```rust
pub mod interval { /* ... */ }
```

### Functions

#### Function `year_month_string`

`INTERVAL '-1-2' YEAR TO MONTH`: a year-month value in Spark's ANSI form.

```rust
pub fn year_month_string(months: i32, start: crate::YearMonthField, end: crate::YearMonthField) -> String { /* ... */ }
```

#### Function `write_year_month`

[`year_month_string`], appended to `out`.

```rust
pub fn write_year_month(out: &mut String, months: i32, start: crate::YearMonthField, end: crate::YearMonthField) { /* ... */ }
```

#### Function `day_time_string`

`INTERVAL '-1 02:03:04.5' DAY TO SECOND`: a day-time value in Spark's ANSI
form. The leading field is unbounded and, unless it is DAY, at least two
digits; the others stay in their range.

```rust
pub fn day_time_string(micros: i64, start: crate::DayTimeField, end: crate::DayTimeField) -> String { /* ... */ }
```

#### Function `write_day_time`

[`day_time_string`], appended to `out`.

```rust
pub fn write_day_time(out: &mut String, micros: i64, start: crate::DayTimeField, end: crate::DayTimeField) { /* ... */ }
```

#### Function `calendar_string`

`CalendarInterval.toString`: the non-zero units, `0 seconds` for none,
e.g. `1 years 2 months 3 days 4.5 seconds`.

```rust
pub fn calendar_string(months: i32, days: i32, micros: i64) -> String { /* ... */ }
```

## Module `interval_parse`

Spark's `IntervalUtils` string parsers that run at both plan and run time:
`stringToInterval` ([`parse_string`], a calendar interval, also the
`INTERVAL 'text'` literal) and the strings a cast reads as an ANSI interval
([`cast_to_year_month`], [`cast_to_day_time`]), which are not the literal
grammar.

```rust
pub mod interval_parse { /* ... */ }
```

### Types

#### Struct `Calendar`

Spark's `CalendarInterval`.

```rust
pub struct Calendar {
    pub months: i32,
    pub days: i32,
    pub micros: i64,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `months` | `i32` | Months. |
| `days` | `i32` | Days. |
| `micros` | `i64` | Microseconds. |

##### Implementations

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **CastableFrom**
- **Clone**
  - ```rust
    fn clone(self: &Self) -> Calendar { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Copy**
- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Default**
  - ```rust
    fn default() -> Calendar { /* ... */ }
    ```

- **Eq**
- **Equivalent**
  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Instrument**
- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &Calendar) -> bool { /* ... */ }
    ```

- **Read**
- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnsafeUnpin**
- **UnwindSafe**
- **WithSubscriber**
#### Struct `Named`

The fields an interval literal names; Spark types the literal by them,
whatever its values (`INTERVAL 0 MONTH 1 DAY` mixes fields).

```rust
pub struct Named {
    pub year_month: Option<(crate::YearMonthField, crate::YearMonthField)>,
    pub day_time: Option<(crate::DayTimeField, crate::DayTimeField)>,
}
```

##### Fields

| Name | Type | Documentation |
|------|------|---------------|
| `year_month` | `Option<(crate::YearMonthField, crate::YearMonthField)>` | The lowest and the highest year-month field. |
| `day_time` | `Option<(crate::DayTimeField, crate::DayTimeField)>` | The lowest and the highest day-time field. |

##### Implementations

###### Methods

- ```rust
  pub fn of_range(from: &str, to: &str) -> Self { /* ... */ }
  ```
  The fields of the range `from TO to`, in Spark's unit names.

- ```rust
  pub const fn mixed(self: Self) -> bool { /* ... */ }
  ```
  Whether it names fields of both families.

###### Trait Implementations

- **Allocation**
- **Any**
  - ```rust
    fn type_id(self: &Self) -> TypeId { /* ... */ }
    ```

- **Borrow**
  - ```rust
    fn borrow(self: &Self) -> &T { /* ... */ }
    ```

- **BorrowMut**
  - ```rust
    fn borrow_mut(self: &mut Self) -> &mut T { /* ... */ }
    ```

- **CastableFrom**
- **Clone**
  - ```rust
    fn clone(self: &Self) -> Named { /* ... */ }
    ```

- **CloneToUninit**
  - ```rust
    unsafe fn clone_to_uninit(self: &Self, dest: *mut u8) { /* ... */ }
    ```

- **Copy**
- **Debug**
  - ```rust
    fn fmt(self: &Self, f: &mut $crate::fmt::Formatter<''_>) -> $crate::fmt::Result { /* ... */ }
    ```

- **Default**
  - ```rust
    fn default() -> Named { /* ... */ }
    ```

- **Eq**
- **Equivalent**
  - ```rust
    fn equivalent(self: &Self, key: &K) -> bool { /* ... */ }
    ```

- **Freeze**
- **From**
  - ```rust
    fn from(t: T) -> T { /* ... */ }
    ```
    Returns the argument unchanged.

- **Instrument**
- **Into**
  - ```rust
    fn into(self: Self) -> U { /* ... */ }
    ```
    Calls `U::from(self)`.

- **PartialEq**
  - ```rust
    fn eq(self: &Self, other: &Named) -> bool { /* ... */ }
    ```

- **Read**
- **RefUnwindSafe**
- **Send**
- **StructuralPartialEq**
- **Sync**
- **ToOwned**
  - ```rust
    fn to_owned(self: &Self) -> T { /* ... */ }
    ```

  - ```rust
    fn clone_into(self: &Self, target: &mut T) { /* ... */ }
    ```

- **TryFrom**
  - ```rust
    fn try_from(value: U) -> Result<T, <T as TryFrom<U>>::Error> { /* ... */ }
    ```

- **TryInto**
  - ```rust
    fn try_into(self: Self) -> Result<U, <U as TryFrom<T>>::Error> { /* ... */ }
    ```

- **Unpin**
- **UnsafeUnpin**
- **UnwindSafe**
- **WithSubscriber**
### Functions

#### Function `cast_to_year_month`

`CAST(string AS <year-month type>)`, in months.

# Errors
`INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING` for another form, and
`.INTERVAL_PARSING` for a field out of range or a value past the `int` range.

```rust
pub fn cast_to_year_month(input: &str, start: crate::YearMonthField, end: crate::YearMonthField) -> Result<i32, krishiv_common::error::KrishivError> { /* ... */ }
```

#### Function `cast_to_day_time`

`CAST(string AS <day-time type>)`, in microseconds. The first field takes
at most 9 digits for a day, 10 for an hour, 12 for a minute and 13 for a
second, leading zeros included, and the others stay in their range. One
field other than seconds wraps, as Spark's `Long` arithmetic does
(`CAST('106751992' AS INTERVAL DAY)` is negative); seconds and a range of
fields fail past the `long` range instead.

# Errors
`INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING_WITH_NOTICE` for another
form, and `.INTERVAL_PARSING` for a field or a sum out of range. For most of
those Spark raises a bare `IllegalArgumentException` or `ArithmeticException`;
the class is Krishiv's.

```rust
pub fn cast_to_day_time(input: &str, start: crate::DayTimeField, end: crate::DayTimeField) -> Result<i64, krishiv_common::error::KrishivError> { /* ... */ }
```

#### Function `trim_all`

Spark's `UTF8String.trimAll`: whitespace and control characters.

```rust
pub fn trim_all(s: &str) -> &str { /* ... */ }
```

#### Function `parse_string`

Spark's `IntervalUtils.stringToInterval`, with the units the string names:
`[interval] [+|-]value unit ...`, units `year`, `month`, `week`, `day`,
`hour`, `minute`, `second`, `millisecond`, `microsecond` (each optionally
plural); only seconds take a fraction (up to nanoseconds).

# Errors
Spark's `INVALID_INTERVAL_FORMAT.*`.

```rust
pub fn parse_string(input: &str) -> Result<(Calendar, Named), krishiv_common::error::KrishivError> { /* ... */ }
```

## Module `variant`

Parquet Variant binary codec, version 1 (Parquet `VariantEncoding.md`).

[`VariantRef`] is a validated borrow over a `(metadata, value)` pair:
construction checks every reachable header, count, id and offset, and
each child access re-validates its own bounds, so malformed bytes surface
as `MALFORMED_VARIANT` instead of panicking. [`VariantBuilder`] turns a
JSON document into exactly the bytes Spark 4.2.0's `parse_json` produces
(same dictionary order, widths, number rules and error classes).

[`VariantWriter`] builds values from typed parts (casts to `VARIANT`,
`variant_get` re-encoding) with Spark's `VariantBuilder` byte choices.
`DataType::Variant` stores these bytes; the kernels print and cast them.

# Verified behaviour (live Spark 4.2.0 probes, 2026)

* Metadata header `0x01 | (sorted_strings << 4) | ((offset_width - 1) << 6)`
  (bit 5 reserved); object headers carry
  the offset width, field-id width and `is_large` bit; arrays likewise
  without field ids. Sorted-flagged dictionaries hold strictly increasing
  keys.
* Object field ids are ordered by key (Java `String` order) with values in input order;
  offsets are per-field starts plus the total, and are *not* monotonic.
* Integers prefer the smallest int8/16/32/64 that fits; larger integers
  become decimals (scale 0) up to 38 digits, then doubles.
* Fractions become decimals when `precision <= 38` and `scale <= 38`
  (width by `max(precision, scale)`), doubles otherwise; any exponent
  means double, with overflow to infinity.
* The value buffer is capped at [`VARIANT_SIZE_LIMIT`] bytes with
  `VARIANT_SIZE_LIMIT` (size limit rendered as `128.0 MiB`).
* Deliberate divergences from Spark are documented on the items involved:
  the reader rejects trailing bytes and out-of-range ids that Spark's lax
  validator accepts; the builder shares Spark's 1000-level nesting cap;
  non-ASCII literal terminators approximate Java's
  identifier rule (ASCII, `$` included, is exact).

# Unverified

Primitive ids 17-19 (time, nanosecond timestamps) follow the encoding
text; Spark 4.2.0 reserves them and never emits them, so they are covered
only by hand-built byte strings here.

```rust
pub mod variant { /* ... */ }
```

### Functions

#### Function `key_order`

**Attributes:**

- `MustUse { reason: None }`

Java's `String.compareTo` of two UTF-8 keys: UTF-16 code-unit order, the
order Spark sorts object fields in and searches them by. It differs from
byte order only between supplementary characters and `U+E000..=U+FFFF`;
keys that are not UTF-8 compare by bytes.

```rust
pub fn key_order(a: &[u8], b: &[u8]) -> core::cmp::Ordering { /* ... */ }
```

### Constants and Statics

#### Constant `VARIANT_SIZE_LIMIT`

Maximum Variant value-buffer size in bytes (128 MiB).

Verified against Spark 4.2.0's `VariantUtil.SIZE_LIMIT`; exceeding it in
[`VariantBuilder::from_json`] raises `VARIANT_SIZE_LIMIT`.

```rust
pub const VARIANT_SIZE_LIMIT: usize = 134_217_728;
```

### Re-exports

#### Re-export `VariantBuilder`

```rust
pub use builder::VariantBuilder;
```

#### Re-export `VariantKind`

```rust
pub use reader::VariantKind;
```

#### Re-export `VariantRef`

```rust
pub use reader::VariantRef;
```

#### Re-export `FieldEntry`

```rust
pub use writer::FieldEntry;
```

#### Re-export `VariantWriter`

```rust
pub use writer::VariantWriter;
```

#### Re-export `digits`

```rust
pub use writer::digits;
```

#### Re-export `strip_trailing_zeros`

```rust
pub use writer::strip_trailing_zeros;
```

## Re-exports

### Re-export `CatalogEntry`

```rust
pub use collation::CatalogEntry;
```

### Re-export `Collation`

```rust
pub use collation::Collation;
```

### Re-export `CollationFamily`

```rust
pub use collation::CollationFamily;
```

### Re-export `DataType`

```rust
pub use data_type::DataType;
```

### Re-export `DayTimeField`

```rust
pub use data_type::DayTimeField;
```

### Re-export `DecimalStorage`

```rust
pub use data_type::DecimalStorage;
```

### Re-export `DecimalType`

```rust
pub use data_type::DecimalType;
```

### Re-export `Srid`

```rust
pub use data_type::Srid;
```

### Re-export `YearMonthField`

```rust
pub use data_type::YearMonthField;
```

### Re-export `parse_data_type`

```rust
pub use ddl::parse_data_type;
```

### Re-export `parse_metadata`

```rust
pub use json::parse_metadata;
```

### Re-export `ArrayType`

```rust
pub use nested::ArrayType;
```

### Re-export `MapType`

```rust
pub use nested::MapType;
```

### Re-export `Metadata`

```rust
pub use nested::Metadata;
```

### Re-export `StructField`

```rust
pub use nested::StructField;
```

### Re-export `StructType`

```rust
pub use nested::StructType;
```

### Re-export `equals_ignore_case`

```rust
pub use nested::equals_ignore_case;
```
