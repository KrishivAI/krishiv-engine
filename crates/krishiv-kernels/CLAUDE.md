# krishiv-kernels

**Layer:** L2 (may depend only on crates in L2 or lower; `cargo xtask layering`).  
**Real code arrives:** P2; unsafe kernels: Opus.

## Purpose
Vectorised compute kernels (SIMD) with Spark semantics, Spark-observable Murmur3 (seed 42) and internal fast hashing.

## Layout (kernel specialization framework, P2-T020)
- `policy` (4 overflow policies as ZSTs), `ops` (`Operand`, `BinaryOp`), `dispatch` (`Target`, `KernelCtx`).
- `chunk`: the generic 64-row chunk engine; monomorphs `<L_SCALAR, R_SCALAR>` × policy, chosen once per batch. No
  has-nulls axis (compute-through-nulls makes it ~1.0x); `specialize` is the opt-in `const NULLS` axis for kernels
  whose work changes with NULLs (division, strings).
  A `LaneOp` reports a `Fault` (overflow, divide-by-zero); the policy maps each fault to wrap / NULL / raise.
- `unsafe_dispatch`: the only `unsafe` (calling `#[target_feature]` clones); `xtask source-guard` rejects `unsafe`
  in any other file here and `xtask budgets` counts unsafe sites.
- `version`: the Spark version profile (`KernelCtx::with_spark_version`); error classes/messages differ by version.
- Kernels: `integer` (tinyint..bigint `+ - * %`, bigint `div`, unary `-`/`abs`), `decimal` (`+ - * / % div`, unary; Spark result types
  `result_type` incl. allowPrecisionLoss; exact `+ - *` on 128/64-bit lanes, the rest exact in 128 then 256 bits, HALF_UP),
  `float` (`+ - * / %`, unary), `compare` (`= != < <= > >=` for primitives incl. date/timestamp/decimal with Spark
  float order, strings/binary in byte order, booleans; `null_safe` builds `<=>` from `=`), `cast` (Spark CAST on Arrow
  carriers: integral/float/decimal, string -> integral/decimal/float/date, date/timestamp/timestamp_ntz, all -> string;
  `CastMode` Ansi/Legacy/Try; session zone via `TimeZoneRules`, `FixedOffset` only here; parsers in `cast/text.rs`). Dividing lanes keep a per-row flag word so NULL slots with a zero divisor never take
  the slow path (measured: `div` with 10% NULLs 302 -> 262 µs, = the no-NULL time; decimal `/` with 10% NULLs is
  1.08x of the 90%-rows bound, so no kernel takes the `by_nulls` axis: below the 1.2x bar).
- `string` (P2-T023a): Spark string functions on `Utf8`/`LargeUtf8` (and `Binary` where Spark defines them): `upper`/`lower`
  (ASCII fast path; Unicode data keyed by `SparkVersion`: the 56 Unicode 17 additions are unmapped before 4.2),
  `char_length`/`byte_length`, `trim` (spaces only by default), `match_str`/`match_binary` (`startswith`/`endswith`/
  `contains`), `replace`, `pad`/`pad_binary`, `substring`/`substring_binary`, `concat`, `like` (`string/like.rs`:
  literal-shaped patterns take string comparisons, the rest a linear backtracking matcher; errors only on non-NULL rows)
  and `split_literal` (`string/split.rs`: delimiters without regex metacharacters; a regex delimiter is
  `KRISHIV_UNSUPPORTED_FEATURE` until the regex engine). Operands are columns or literals (`StrOperand`, `BinOperand`,
  `Operand<Int32Type>`); NULL rows are skipped, so these kernels have no has-nulls axis. Output offsets are checked.
- `hash` (P2-T027): `hash` (Murmur3 x86_32, seed 42; the tail bytes are mixed one sign-extended byte at a time, as Spark does), `xxhash64`
  (XXH64, seed 42; `hashInt`/`hashLong` written out), `md5`/`sha1`/`sha2`/`crc32` (`hash/digest.rs`; CRC table in-house, digests from
  RustCrypto). `hash`/`xxhash64` take any number of columns of any type and chain per row (NULL leaves the hash unchanged; arrays hash
  their elements, structs their fields, maps each key then value). A column is resolved to a `Col` plan once, rows only match on the variant. The
  analyzer's MAP rejection (`DATATYPE_MISMATCH.HASH_MAP_TYPE` without `spark.sql.legacy.allowHashOnMapType`) is the caller's check.
  Unsupported types (dictionary, 256-bit decimal, time of day, month-day-nano interval) are `KRISHIV_UNSUPPORTED_FEATURE`.
- `reference`: row-at-a-time oracle for property tests (`tests/props.rs`), the fuzz target (`fuzz/`) and benches.

## Invariants
- Spark float ordering (NaN, -0.0) in every compare/sort/hash kernel, nested types included.
- Murmur3_x86_32 seed 42 implemented in-house; fast hash (foldhash/xxh3) only where not observable.
- Generics inside kernels; every specialisation is tested against `reference` on every available CPU target.
- A NULL row never raises: overflow flags are masked with the validity word; ANSI errors name the first failing row.
- `#![deny(clippy::arithmetic_side_effects)]`: only explicit `wrapping_*`/`overflowing_*`/`checked_*` arithmetic.
- `unsafe` allowed only in `unsafe_*.rs` modules, each block with `// SAFETY:`; Miri on every PR touching this
  crate (`miri-kernels`), nightly ASan and differential fuzzing (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0008, 0009, 0010, 0011, 0029, 0034

## Test
`cargo nextest run -p krishiv-kernels --no-tests=pass` · `cargo clippy -p krishiv-kernels --all-targets -- -D warnings`
`KRISHIV_REQUIRE_TARGETS=baseline,x86_64v3` makes a skipped CPU target a test failure (set in CI) ·
`cargo +nightly miri test -p krishiv-kernels` · `cargo bench -p krishiv-kernels` ·
`(cd crates/krishiv-kernels/fuzz && cargo +nightly fuzz run differential -- -max_total_time=60)`
