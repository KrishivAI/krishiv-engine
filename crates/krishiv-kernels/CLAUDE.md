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
- `Dispatch` (sealed): `KernelCtx` picks the CPU target at run time (three clones of each loop, plus scalar-operand
  loops); `Baseline` runs the build-baseline loop only, a scalar repeated into an array (one loop per kernel), for
  one-row/constant evaluation such as plan-time folding. Dispatching entry points take `ctx: &impl Dispatch`.
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
  (ASCII fast path; Unicode data keyed by `SparkVersion`: the 56 Unicode 17 additions are unmapped before 4.2; std maps only
  the code points of `string/case_table.rs`, generated from the Spark 4.2 fixture by `tests/scripts/gen_case_table.py`, so a
  toolchain with newer Unicode data does not change the output),
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
- `map` (P2-T029): the map functions over `MapArray`/`ListArray`/`StructArray` (offsets and child values, never
  boxed): `map`, `map_concat`, `map_contains_key`, `map_keys`, `map_values`, `map_entries`, `map_from_arrays`,
  `map_from_entries` and `str_to_map`. Keys compare with Spark's equality and hash with the observable Murmur3
  (NaN equals NaN, `-0.0` equals `0.0`); a null key is `NULL_MAP_KEY`, a repeat `DUPLICATED_MAP_KEY` under
  `EXCEPTION` and last-wins under `LAST_WIN`.
- `datetime` (P2-T025a): `date_add`/`date_sub`/`date_diff` (ANSI: `ARITHMETIC_OVERFLOW` past the `int` range, a documented divergence from Spark; legacy: Java `int` wrap; `try_*`: NULL),
  field extraction (`DatePart`: `year`, `month`, `day`, `extract`/`date_part` incl. ISO week and `SECOND` as
  `decimal(8,6)`), `date`/`timestamp` ± interval (`date_add_months` clamps the day; `timestamp_add_day_time` adds whole
  days as calendar days in the zone and keeps the start's offset in a DST overlap, like `ZonedDateTime.plusDays`), and
  `current_date` from the query-start instant. The calendar functions live in `datetime/calendar.rs`: `monthname`/`dayname`,
  `unix_date`, `last_day`, `next_day` (`ILLEGAL_DAY_OF_WEEK` under ANSI, NULL without; the two-letter names and the lack of
  trimming are assumed, unverified against a Spark run), `trunc`/`date_trunc` (`TruncUnit`:
  Spark's unit names and aliases; an hour or a day is assumed to keep the start's offset in a DST overlap, the date levels start the local
  day), `months_between` (31-day months, whole months for equal or last days; the time of day in whole seconds counted from the
  instant the local day starts is matrix-pinned in UTC and assumed across DST), `make_date` (`DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION` with the `java.time` message under
  ANSI, NULL without) and the zone conversions `from_utc_timestamp`/`to_utc_timestamp`/`convert_timezone` (`Zones`: one zone, or
  one per row with `None` for a NULL zone; `DATETIME_OVERFLOW` under ANSI, NULL without, like `next_day`'s bad name and
  `make_date`'s bad field). `last_day`/`next_day` int-wrap at the range ends is assumed, unverified against a Spark run
  (no golden or matrix cell reaches it). Field accessors (`dayofweek`, `weekday`, `hour`, ...) are `extract_date`/
  `extract_timestamp` with a `DatePart` (`extract_timestamp` keeps its pre-existing always-raise convention). Zones come in as `TimeZoneRules`, never as names. The cast module also
  parses `string -> TIMESTAMP[_NTZ]` (`cast/timestamp.rs`: the full Spark grammar; a region id in the text and a bare
  time of day are `KRISHIV_UNSUPPORTED_FEATURE` until the tz database / query clock reach the kernels).
- `math` (P2-T026): the `double` functions (`UnaryFn`/`BinaryFn`: `sqrt` .. `signum`, `pow`, `atan2`, `hypot`, `log`; pure-Rust `libm`
  plus the fdlibm ports of `math/fdlibm.rs` for `cbrt`/`log10`/`hypot`/`acosh`/`tanh`, so results do not depend on the CPU or libc;
  `asinh`/`acosh` follow the `SparkVersion`), `round`/`bround`/`ceil`/`floor` (`math/round.rs`: integral with ANSI overflow, `double` through
  the shortest decimal text like `BigDecimal.valueOf`, exact decimals with `round_decimal_type`, `ceil`/`floor` of doubles to `bigint`),
  `pmod` (`math/pmod.rs`: Java `int`/`long` wrap in `(r + n) % n`), `greatest`/`least` (`math/extremum.rs`: NULLs skipped, NaN greatest,
  `-0.0 == 0.0`). The module docs list which functions are bit-identical to Spark's JVM results and the share of inputs where
  `sin`/`cos`/`tan`/`ln`/`atan2`/`log10` differ in the last bit (the JVM's intrinsics are not fdlibm); `tests/math_differential.rs` in
  `krishiv-testkit` holds that bound.
- `reference`: row-at-a-time oracle for property tests (`tests/props.rs`), the fuzz target (`fuzz/`) and benches.

## Key files
Paths are relative to this crate.
- `src/lib.rs` module map and crate docs
- `src/chunk.rs` generic 64-row chunk engine; `src/policy.rs` overflow policies (wrap, NULL, raise)
- `src/dispatch.rs` `Target` and `KernelCtx`; `src/unsafe_dispatch.rs` the only `unsafe` module
- `src/integer.rs`, `src/decimal.rs`, `src/float.rs` arithmetic kernels; `src/compare.rs` comparisons
- `src/cast.rs` Spark CAST on Arrow carriers; `src/string.rs` string kernels
- `src/hash.rs` and `src/hash/murmur3.rs` Murmur3 (seed 42), `src/hash/xxh64.rs` xxhash64
- `src/map.rs` map functions over nested arrays
- `src/datetime.rs` date and timestamp kernels; `src/math.rs` double functions, `src/math/fdlibm.rs` ports
- `src/reference.rs` row-at-a-time oracle for property tests

## Extension points
- New kernel: a module with a `reference` oracle entry, a property test in `tests/props.rs`, and a bench in `benches/`.
- New CPU-specific path: the `unsafe` clone goes in `src/unsafe_dispatch.rs` only, with a `// SAFETY:` comment.

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

## Tests
`cargo nextest run -p krishiv-kernels --no-tests=pass` · `cargo clippy -p krishiv-kernels --all-targets -- -D warnings`
`KRISHIV_REQUIRE_TARGETS=baseline,x86_64v3` makes a skipped CPU target a test failure (set in CI) ·
`cargo +nightly miri test -p krishiv-kernels` · `cargo bench -p krishiv-kernels` ·
`(cd crates/krishiv-kernels/fuzz && cargo +nightly fuzz run differential -- -max_total_time=60)`
