# sequence(start, stop[, step]) over DATE / TIMESTAMP / TIMESTAMP_NTZ, and schema_of_xml(xml, options) goldens (SQL-6)

Status target: `compatible` for every rule below. The "Quirks" list is Spark behaviour that looks like a bug (recorded,
not endorsed; `incompatible` candidates, opt-in via `krishiv.expr.<name>.allowIncompatible`).

Layout: `inputs/<name>.sql` (`;`-terminated statements, one session, so SET and views carry over),
`results/spark-<ver>/<name>.sql.out` (output of a real run), `run.sh <ver> [name]`. The runner is `../run_oracle.py`.
Every expected value was produced by executing Spark; none is transcribed. Hand-written expectations: none.

| Case file | Queries | Covers |
|---|---|---|
| `sequence_dates` | 84 | DATE: calendar-style `interval 1 month`, ANSI year-month and day-time steps, default step, descending, step 0, wrong direction, NULL, month-end clamping, type errors, ANSI off |
| `sequence_timestamps` | 83 | TIMESTAMP and TIMESTAMP_NTZ in UTC: same matrix, sub-second steps, default step, ANSI off |
| `sequence_tz` | 60 | `America/Los_Angeles` across 2018-03-11 (spring forward) and 2018-11-04 (fall back), step `1 day` vs `24 hours`; NTZ is zone independent; `Australia/Lord_Howe`, `Asia/Kolkata` |
| `sequence_calendar` | 78 | true CALENDAR interval steps (`make_interval`, `cast('..' as interval)`, `spark.sql.legacy.interval.enabled=true` literals), in UTC and in the DST zone |
| `sequence_errors` | 62 | exact error classes/messages, maximum array length, column inputs, CASE/IF/COALESCE/WHERE conditional evaluation |
| `schema_of_xml` | 204 | options, nested/repeated/attributes-only/empty/mixed/invalid XML, types, namespaces, ANSI off, time zone, conditional evaluation |

## How the .out files were made

`apache/spark:<ver>` container `krishiv-s7sql456-oracle-<ver>` (`--rm`, removed on exit), `spark-submit ../run_oracle.py`,
session time zone UTC unless the file `SET`s it, ANSI default on (files also `SET spark.sql.ansi.enabled=false` for a
block). Runs: **4.2.0 and 4.3.0**. 3.5.7, 4.0.0 and 4.1.0 were NOT run (`unverified-by-run`; `schema_of_xml`/`from_xml`
are new in 4.0, so 3.5 is expected `UNRESOLVED_ROUTINE`). The `.out` files are root-owned (written from the container).
Source read at tags `v4.2.0` and `v4.3.0` via raw.githubusercontent.com; the fetch tool truncated the large files and gave no
line numbers, so each rule cites file + declaration (line numbers `unavailable`).

Sources: `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/collectionOperations.scala`
(`Sequence`, `InternalSequenceBase`, `IntegralSequenceImpl`, `TemporalSequenceImpl`, `DurationSequenceImpl`,
`PeriodSequenceImpl`, `getSequenceLength`), `.../catalyst/xml/XmlOptions.scala`, `.../catalyst/xml/XmlInferSchema.scala`,
`.../expressions/xml/xmlExpressions.scala` (`SchemaOfXml`).

## sequence rules (identical on 4.2.0 and 4.3.0)

| Rule | Source | Case file |
|---|---|---|
| Result type `array<date>` for DATE bounds, `array<timestamp>` for TIMESTAMP, TIMESTAMP_NTZ gives NTZ array; mixed DATE and TIMESTAMP/NTZ bounds coerce to a timestamp array (`2018-01-01 00:00:00`...) | run | dates, timestamps |
| Arity 2 or 3, else `WRONG_NUM_ARGS.WITHOUT_SUGGESTION` (42605, `expectedNum` `[2, 3]`) | run | dates, errors |
| Step types accepted for temporal bounds: CALENDAR interval, INTERVAL YEAR TO MONTH (any `year`/`month` field), INTERVAL DAY TO SECOND (any fields). An integral, string, NULL-literal (untyped) or boolean step => `DATATYPE_MISMATCH.SEQUENCE_WRONG_INPUT_TYPES` (42K09); a typed NULL interval (`cast(NULL as interval day)`) => result NULL | `Sequence.checkInputDataTypes`; run | dates, timestamps |
| Any NULL start, stop or (typed) step => NULL | run | dates, timestamps |
| Default step (omitted): `+1 day` if start <= stop, `-1 day` otherwise (`TemporalSequenceImpl` default `CalendarInterval(0,1,0)`; usage text "1 day and -1 day"). So `sequence(ts, ts + 3s)` is a one-element array and `sequence(date, earlier date)` counts down by day | run | dates, timestamps |
| A step is split into (months, days, microseconds). YEAR TO MONTH step => (n, 0, 0); DAY TO SECOND step => days = whole days of the duration, micros = remainder (so `interval 24 hours` == `interval 1 day`, see quirk 1); CALENDAR => its own three fields | `splitStep` in `DurationSequenceImpl`/`PeriodSequenceImpl`/`TemporalSequenceImpl` | tz, calendar |
| Element i is `start + i*step` evaluated from the START each time (not accumulated), so Jan 31 + 1 month: `Jan 31, Feb 28, Mar 31, Apr 30, May 31, Jun 30`; leap years: `2020-01-31`... `Feb 29`; `2020-02-29 + 1 year` => `2021-02-28 ... 2024-02-29` | `addInterval`; run | dates, timestamps |
| Inclusive of stop when it lands exactly; stops before overshoot; start == stop => `[start]` for any non-zero step | run | dates |
| DATE bounds: stepping is done on DATE (days); the step must have a non-zero month or day part, else `_LEGACY_ERROR_TEMP_3242` (`SparkIllegalArgumentException`, SQLSTATE none) message `sequence step must be an interval <type> of day granularity if start and end values are dates`, where `<type>` is `day to second`, `year to month` or `interval` (CALENDAR, and under legacy.interval literals). This fires for step 0 and for sub-day steps (`interval 12 hours`, `1 microsecond`, `make_interval(0,0,0,0,24)`) but a day plus remainder (`1 day 12 hours`, `1 day 1 microsecond`) is accepted and truncates to days: `1 day 12 hours` => `Jan 1, Jan 2, Jan 4` | `InternalSequenceBase.eval` (`scale == MICROS_PER_DAY && stepMonths == 0 && stepDays == 0`); run | dates, calendar, errors |
| TIMESTAMP/NTZ bounds with step 0 or wrong-direction step: `java.lang.IllegalArgumentException` `requirement failed: Illegal sequence boundaries: <start micros> to <stop micros> by <step>` (no error class). `<step>` is the raw step: months as a bare integer for YM steps (`by -1`), micros for DT steps (`by 3600000000`), `by 1 months` for CALENDAR steps, `by 0` for a zero CALENDAR step. DATE wrong direction: `Illegal sequence boundaries: 17532 to 17536 by -1` (epoch days) for a day step but micros for a month step. In a column / non-constant-folded path the same error is `SparkIllegalArgumentException _LEGACY_ERROR_TEMP_3243` with params `start`,`stop`,`step`, message `Illegal sequence boundaries: ...` (no `requirement failed:` prefix) | `getSequenceLength` `require`; generated code `_LEGACY_ERROR_TEMP_3243`; run | errors (view `sq`), timestamps, dates |
| Maximum array length: `SparkRuntimeException` `COLLECTION_SIZE_LIMIT_EXCEEDED.PARAMETER` SQLSTATE 54000, params `functionName` `` `sequence` ``, `maxRoundedArrayLength` `2147483632`, `numberOfElements` (exact count, computed in BigInt on overflow), `parameter` `` `count` ``. Applies to integral (`sequence(1, 2147483647)` => 2147483647 elements) and temporal (`1 microsecond` over a year => `31449600000001`). 2147483632 elements is not rejected up front (a JVM heap OOM in the 1 GB oracle, so not recorded) | `Sequence.sequenceLength` / `MAX_ROUNDED_ARRAY_LENGTH`; run | errors |
| `size(sequence(date 0001-01-01, 9999-12-31, 1 day))` = 3652059 | run | errors |
| Session time zone only matters for TIMESTAMP (and DATE->micros conversions): months/days are added in local wall-clock time via the session zone, remainder micros are added as elapsed time. Spring forward (LA 2018-03-11): `1 day` and `24 hours` from `03-10 00:00` give `03-10, 03-11, 03-12, 03-13 00:00`; from `03-10 02:30` `1 day` gives `03-11 03:30` (gap resolved forward) then `03-12 02:30`; `sequence(ts 03-11 02:30, 03-12 02:30, 1 day)` is `[03-11 03:30]` (the nonexistent start 02:30 is already shifted to 03:30, so the next element 03-12 03:30 is past stop 03-12 02:30). `1 hour` over the gap skips `02:00` (`00,01,03,04,05`). Fall back (2018-11-04): `1 hour` repeats `01:00` (two equal elements); `1 day` stays on midnight / the same wall time | `timestampAddInterval`; run | tz |
| TIMESTAMP_NTZ ignores the session zone (`1 hour` across the LA gap gives `00,01,02,03,04,05`) | `timestampNTZAddInterval`; run | tz |
| CALENDAR interval `24 hours` is real elapsed time, unlike the ANSI DT interval: LA `make_interval(0,0,0,0,24)` from `03-10 00:00` gives `03-10 00:00, 03-11 00:00, 03-12 01:00`, and across fall back `11-03 00:00, 11-04 00:00, 11-04 23:00, 11-05 23:00`. `make_interval(0,0,0,1)` (1 day) stays on wall-clock midnight | run | calendar |
| `interval 1 month 2 days` (mixed fields) is a parse error `_LEGACY_ERROR_TEMP_0029` (ParseException, params `literal`) with ANSI on **and off**; it is a CALENDAR interval only with `spark.sql.legacy.interval.enabled=true`. Calendar steps are otherwise built with `make_interval` or `cast('1 month 2 days' as interval)`. Month+day step: `(2018-01-01, 2018-04-05, 1 month 2 days)` => `01-01, 02-03, 03-05` (start + i*(1 month, 2 days) each from start); negative day part allowed (`make_interval(0,1,0,-2)` => `01-01, 01-30, 02-25`): the direction check is on the combined step | `AstBuilder` interval literal; run | calendar, errors |
| `interval '1' year to month` etc. with a malformed string literal is `INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING` (22006); `interval 9999999999 days` overflows the interval literal: `INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION` (22006) | run | dates, errors |
| ANSI on/off: no difference in results or error classes for any numeric/interval/type-error case, EXCEPT that string bounds are coerced only with ANSI on: `sequence(date, '2018-01-05', interval 1 day)` works on (result is dates) and raises `SEQUENCE_WRONG_INPUT_TYPES` off (quirk 4) | run | dates |
| Conditional evaluation: `CASE`/`IF`/`COALESCE` and `WHERE id = 1 AND size(sequence(..))` do not raise on unselected rows (view `sq`, a step-0 row and a wrong-direction row); but `WHERE id <> 2 AND size(sequence(a, b, s)) = 3` still evaluates the row with the wrong direction (id 3) and raises (not a conditional guard). `CASE WHEN s > interval 0 month THEN sequence(...)` with a NULL-bounds row gives NULL | run | errors |

## schema_of_xml rules

`schema_of_xml(xml_literal [, map(...)])` returns a STRING (a DDL-like `STRUCT<...>` text, uppercase types, backticks around
names needing quotes, fields sorted by name in code-point order: `_VALUE` < `_attr` < lower-case names). Equal on 4.2.0 and
4.3.0 except the single difference noted.

| Rule | Source | Case file |
|---|---|---|
| Issue example: `schema_of_xml('<p><a attr="2">1</a><a>3</a></p>', map('excludeAttribute','true'))` => `STRUCT<a: ARRAY<BIGINT>>`; without options => `STRUCT<a: ARRAY<STRUCT<_VALUE: BIGINT, _attr: BIGINT>>>` (the first `a` has an attribute so both elements are structs; the second has `_attr` only in the schema, not a value) | run | schema_of_xml |
| Input must be a foldable STRING: NULL (typed or not) => `DATATYPE_MISMATCH.UNEXPECTED_NULL` (42K09, param `exprName` `xml`); non-STRING (INT, BINARY) => `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`; a column => `DATATYPE_MISMATCH.NON_FOLDABLE_INPUT`; arity 1 or 2 (`expectedNum [1, 2]`) | `SchemaOfXml.checkInputDataTypes`; run | schema_of_xml |
| Options must be a literal `map()` of string => string: NULL/`named_struct`/string => `INVALID_OPTIONS.NON_MAP_FUNCTION` (42K06); `map()` with no entries, a non-STRING value, or a NULL value => `INVALID_OPTIONS.NON_STRING_TYPE` (42K06); non-foldable keys/values (a column) => `INTERNAL_ERROR` `Cannot evaluate expression` (quirk 6) | run | schema_of_xml |
| Option names are case-insensitive; unknown options are ignored | `XmlOptions` (CaseInsensitiveMap); run | schema_of_xml |
| `excludeAttribute` (default false, `getBool`: only true/false, else `_LEGACY_ERROR_TEMP_2147` `excludeAttribute flag can be true or false.`): drops all attribute fields; element-with-only-attributes then has `STRUCT<>`; text+attributes collapses to `_VALUE` only or the primitive | `XmlOptions`; `XmlInferSchema`; run | schema_of_xml |
| `attributePrefix` default `_` (any string, incl. empty `''` => bare names, and `@` => `` `@attr` `` quoted); `valueTag` default `_VALUE`, `''` => `IllegalArgumentException` `requirement failed: 'valueTag' option should not be empty string.`; the two are not forced to differ in this path (`attributePrefix=''` with `valueTag='attr'` gives two `attr` fields) | `XmlOptions` | schema_of_xml |
| `valueTag` field exists only when an element has text AND attributes or child elements; text interleaved with children (`text<a/>more`) makes it `ARRAY<STRING>` (pieces) | `XmlInferSchema.inferObject` | schema_of_xml |
| Repeated element names => `ARRAY`; element types merged (`<a>1</a><a>1.5</a>` => `ARRAY<DOUBLE>`; `1` + `x` => `ARRAY<STRING>`; struct+struct merge fields; primitive+struct => `ARRAY<STRING>`) | `compatibleType` | schema_of_xml |
| Inference order Long(BIGINT) -> Decimal (only `prefersDecimal=true`) -> Double -> Boolean (`true`/`TRUE`) -> Date -> Timestamp -> String. `+5`, `00123` are BIGINT; `1e3`, `-0.0`, 20-digit integers are DOUBLE; `NaN`/`Infinity` are STRING; `prefersDecimal=true`: `1.5` => `DECIMAL(2,1)`, `99999999999999999999` => `DECIMAL(20,0)`, beyond 38 digits => DOUBLE, mixed `1.5` + `2` => `DECIMAL(21,1)`; `prefersDecimal='maybe'` => `IllegalArgumentException: For input string: "maybe"` (uses `toBoolean`, not `getBool`) | `inferFrom`, `compatibleType`; run | schema_of_xml |
| `inferSchema=false` => every leaf STRING (`inferSchema='maybe'` => `_LEGACY_ERROR_TEMP_2147`) | `inferFrom` | schema_of_xml |
| `timestampFormat`/`timestampNTZFormat`/`dateFormat`: `2020-01-02T03:04:05` is TIMESTAMP (never NTZ here, even with `timestampNTZFormat`); `dateFormat='yyyy/MM/dd'` makes `2020-01-02` TIMESTAMP and `2020/01/02` DATE. Session time zone has no effect on the printed schema | run | schema_of_xml |
| `ignoreSurroundingSpaces` default true (`'  1  '` => BIGINT); false => STRING | `XmlOptions` | schema_of_xml |
| `ignoreNamespace=true` strips prefixes from element/attribute names (`ns:a` => `a`, merging with `a` into `ARRAY<..>`); default keeps them as `` `ns:a` `` and exposes the declaration as `` `_xmlns:ns`: STRING``. With `ignoreNamespace=true` the declaration attribute is still present but named `_ns`; a default `xmlns="..."` stays `_xmlns` | run | schema_of_xml |
| `samplingRatio`: parsed as a double, must be `> 0` (`requirement failed: samplingRatio (0.0) should be greater than 0`); values > 1 accepted; non-numeric => `NumberFormatException`. No effect on a single literal | `XmlOptions` | schema_of_xml |
| `rowTag` is ignored by `schema_of_xml` (the whole document root is the row; root element name never appears) | run | schema_of_xml |
| Parse failures: default `mode=PERMISSIVE` returns `STRUCT<_corrupt_record: STRING>` (session `columnNameOfCorruptRecord`, `columnNameOfCorruptRecord` option not honoured: stays `_corrupt_record`), covers `''`, `' '`, `not xml`, unclosed or mismatched tags, bare `<`, unknown entity, `a=1` unquoted, duplicate attribute. `mode=FAILFAST` => `SparkException` `MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION` (22023, `badRecord` `_corrupt_record`, `failFastMode` FAILFAST) with a cause; `DROPMALFORMED` => `PARSE_MODE_UNSUPPORTED` (42601, "Acceptable modes are PERMISSIVE and FAILFAST"); an unknown mode string behaves as PERMISSIVE. Trailing content after the first root element is ignored (`</p></p>`, `garbage`); the XML declaration, comments and CDATA are accepted | `handleXmlErrorsByParseMode`; run | schema_of_xml |
| `timeZone` option validated: `Not/AZone` => `INVALID_TIMEZONE` (22009); `locale=de-DE` with prefersDecimal still parses `1,5` as `DECIMAL(2,1)` | run | schema_of_xml |
| The schema is directly usable: `from_xml(x, schema_of_xml(x))` => `{[{1, 2}, {3, null}]}` and with `excludeAttribute` on both `{[1, 3]}` | run | schema_of_xml |
| ANSI on/off and conditional evaluation (CASE/IF/COALESCE/WHERE with AND false around a bad `schema_of_xml('<p>')`): no difference, no raise | run | schema_of_xml |

## Per-version differences

- `sequence`: 4.2.0 and 4.3.0 outputs are byte identical apart from the header line.
- `schema_of_xml`: exactly one difference, in `nullValue` handling. `schema_of_xml('<p><c>x</c><c>1</c></p>', map('nullValue','x'))`
  is `STRUCT<c: ARRAY<STRING>>` on 4.2.0 and `STRUCT<c: ARRAY<BIGINT>>` on 4.3.0; likewise
  `<c>NULL</c><c>1</c>` with `nullValue='NULL'`. So 4.3.0 treats a value equal to `nullValue` as null during inference.
  Krishiv targeting 4.2 should keep 4.2 behaviour behind the version key.
- 3.5.x / 4.0.x / 4.1.x: not run (`unverified-by-run`).

## Quirks to decide (Spark behaviour that looks like a bug)

1. `interval 24 hours` (DAY TO SECOND) == `interval 1 day` for TIMESTAMP bounds (whole days are split out), so it does not
   mean 24 elapsed hours across DST, while the equivalent CALENDAR `make_interval(0,0,0,0,24)` does.
2. The wrong-direction / zero-step error is a bare `java.lang.IllegalArgumentException` (`requirement failed:` prefix, no
   error class) in the interpreted path but `_LEGACY_ERROR_TEMP_3243` in the generated/column path; the step is printed raw.
3. `_LEGACY_ERROR_TEMP_3242` for a DATE sub-day or zero step names the interval type but has no SQLSTATE.
4. String bounds coerce to dates only under ANSI on; no coercion under ANSI off.
5. `interval 1 month 2 days` is a parse error even with ANSI off (only the legacy flag allows it).
6. `schema_of_xml` with option map values from a column dies with `INTERNAL_ERROR Cannot evaluate expression`.
7. `valueTag`/`attributePrefix` can collide (`attr` + `attr` duplicate field names printed).
8. `timestampNTZFormat` never makes `schema_of_xml` infer `TIMESTAMP_NTZ`; `dateFormat` of another shape makes ISO dates TIMESTAMP.
9. `columnNameOfCorruptRecord` option is not honoured by `schema_of_xml` (always `_corrupt_record`).

## Not covered / open

- `sequence(0, 2147483631)` (largest permitted integral) hit a JVM heap OOM in the default container and was dropped.
- 3.5.7/4.0.0/4.1.0 not run; source line numbers not obtained (fetch tool returned no numbering).
- `schema_of_xml` over multiple rows / `samplingRatio` on files is a reader concern (`spark.read.xml`), out of scope here.
