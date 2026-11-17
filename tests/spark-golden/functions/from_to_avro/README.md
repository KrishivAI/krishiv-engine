# from_avro / to_avro goldens (P3 PR34 step 1)

Status target: `compatible` for the documented rules; the Spark quirks listed under "Quirks to decide" are open
(`incompatible` candidates, each opt-in via `krishiv.expr.<name>.allowIncompatible`).

Layout: `inputs/<name>.sql` (one statement per `;`-terminated line group), `results/spark-<ver>/<name>.sql.out`
(output of a real run), `../run_oracle.py` (the runner). Every expected value in `results/` was produced by executing
Spark; none is transcribed. Hand-written expectations: none (error classes are also from the run).

## How the .out files were made

`apache/spark:<ver>` container (own container `krishiv-s7pr34-oracle-*`, `--rm`; `krishiv-bench-spark` untouched),
`spark-submit --packages org.apache.spark:spark-avro_2.13:<ver>` (`_2.12` for 3.5.7), runner `../run_oracle.py`,
session time zone UTC unless the file says otherwise, ANSI default (on in 4.x; 3.5.7 has no SQL function at all).
Runs: 3.5.7, 4.0.0, 4.1.0, 4.2.0, 4.3.0 (4.3.0 = the `apache/spark:4.3.0` image; source read at tag `v4.3.0`).

## Rules and sources (tag v4.2.0 unless noted; v4.3.0 is identical apart from docs text)

| Rule | Source | Case |
|---|---|---|
| SQL `from_avro(child, jsonSchema[, options])` and `to_avro(child[, jsonSchema])` exist since 4.0.0 (registered `FunctionRegistry.scala:1023`); 3.5.x has only the Scala/Python API, SQL gives `UNRESOLVED_ROUTINE` | `avroSqlFunctions.scala:58,155`; `results/spark-3.5.7/*` | all |
| Arity: from_avro 2-3 args, to_avro 1-2 args, else `WRONG_NUM_ARGS.WITHOUT_SUGGESTION` (42605) | `avroSqlFunctions.scala:62,157` | errors_args |
| Schema arg must be a foldable STRING or NULL; options a foldable `MAP<STRING,STRING>`, `MAP<NULL,NULL>` or NULL, else `DATATYPE_MISMATCH.TYPE_CHECK_FAILURE_WITH_HINT` (42K09) with the exact hint text | `avroSqlFunctions.scala:72-98,168-181` | errors_args |
| Foldable-but-computed args (`concat(...)`) are accepted; a `map_concat(...)` options arg is rejected with the third-argument error (observed) | run | errors_args |
| NULL / empty-string / non-JSON schema text raises `org.apache.avro.SchemaParseException` (not a Spark error class) | `AvroUtils.parseAvroSchema`; run | basic, modes_options |
| from_avro result type = Avro schema mapped to SQL; PERMISSIVE forces all-nullable (`dt.asNullable`) | `AvroDataToCatalyst.scala:44-52` | basic (typeof cases) |
| Default parse mode FAILFAST: any decode failure raises `SparkException` "Malformed records are detected in record parsing. Current parse Mode: FAILFAST..." with the decoder exception as cause | `AvroDataToCatalyst.scala:107-122` | modes_options |
| PERMISSIVE: bad bytes give NULL (scalar) or a struct of all-NULL fields (record, struct itself non-null) | `AvroDataToCatalyst.scala:88-100,115` | modes_options |
| `mode` DROPMALFORMED (any value other than PERMISSIVE/FAILFAST that parses) raises `PARSE_MODE_UNSUPPORTED` (42601) at analysis; an unrecognised mode string (`bogus`, empty) silently behaves as PERMISSIVE (`ParseMode.fromString` fallback; verified by run: NULL result) | `AvroDataToCatalyst.scala:74-80`; run | modes_options |
| Options are case-insensitive (`MODE`, `mode`, `PERMISSIVE`/`permissive`) | run | modes_options |
| `avroSchema` option = reader (evolved) schema; resolution failure is FAILFAST malformed | `AvroDataToCatalyst.scala:63-66` | modes_options |
| Top-level non-nullable union cannot be deserialized (`Attempting to treat union[int, string] as a RECORD`); a union inside a record gives `struct<member0,member1>` (or `member_<type>` / custom prefix with `enableStableIdentifiersForUnionType`, `stableIdentifierPrefixForUnionType`) | `AvroDataToCatalyst.scala:44-47`; run | basic, modes_options |
| `recursiveFieldMaxDepth`: default (-1) on a recursive schema raises `IncompatibleSchemaException`; 1..15 truncates; 0 raises; >15 raises `STDS_INVALID_OPTION_VALUE.WITH_MESSAGE` | run | modes_options |
| to_avro with no schema infers one: nullable input -> union `[T, "null"]` (branch 0 = value), non-nullable literal -> plain `T`; decimals -> fixed; timestamp -> timestamp-micros; timestamp_ntz -> local-timestamp-micros | `CatalystDataToAvro.scala:37-38` (`SchemaConverters.toAvroType(dt, nullable)`); run | basic |
| to_avro wire format is Avro binary (zigzag varints, IEEE little-endian floats; NaN `00..F87F`, -0.0 `00..0080` preserved bit-exact) | `CatalystDataToAvro.scala:50-56`; run | basic |
| to_avro NULL input -> NULL; schema NULL == no schema | `CatalystDataToAvro.scala:48`; `avroSqlFunctions.scala:183-190` | basic |
| to_avro with an incompatible explicit schema raises `org.apache.spark.sql.avro.IncompatibleSchemaException` (exact message text per case in the `.out`) | run | basic |
| Invalid UTF-8 in a Spark string passes through to_avro byte-for-byte (`02FF`); from_avro of such bytes yields the same bytes back | run | basic, ansi_tz |
| Conditional evaluation: CASE/IF/COALESCE/AND/OR do not evaluate from_avro on unselected rows (FAILFAST error is not raised for them) | run | conditional |
| ANSI on/off makes no difference to either function (outputs equal; diff by `spark.sql.ansi.enabled`) | run | ansi_tz |
| Session time zone never changes the bytes; only the rendering of a decoded `timestamp-micros`/`-millis` (local-timestamp-* is NTZ) | run | ansi_tz |
| Decimal: scale/precision mismatch with the explicit schema raises `IncompatibleSchemaException` (no silent rescale) | run | basic |
| Spark's own SQLQueryTestSuite has no Avro module, so `to_from_avro.sql` there expects `AVRO_NOT_LOADED_SQL_FUNCTIONS_UNUSABLE` (22KD3); Krishiv does not copy that (known-failure category `divergence-avro-not-loaded`, owner decision 2026-11-06). The argument-check statements of that file (lines 6-8 of the input) must pass once the functions exist | `tests/spark-golden/v4.2.0/results/to_from_avro.sql.out` | n/a |

## Per-version differences (from diffing the five runs)

- 3.5.7: every SQL call is `UNRESOLVED_ROUTINE` (42883). Not supported; Krishiv should register for 3.5 as `unsupported` (clear error), or match the Spark error.
- 4.0.0 vs 4.2.0: (a) `enableStableIdentifiersForUnionType='maybe'` raises raw `java.lang.IllegalArgumentException: For input string: "maybe"` in 4.0.0 and 4.1.0, `STDS_INVALID_OPTION_VALUE.WITH_MESSAGE` (42616) from 4.2.0; (b) an unknown named Avro schema (`'"bogus"'`) is `INTERNAL_ERROR` (XX000, analysis phase, NPE cause) in 4.0.0 but a plain `SchemaParseException` in 4.1.0+; (c) a decoder cause class differs (`IndexOutOfBoundsException` vs `ArrayIndexOutOfBoundsException`), cause-text only.
- 4.1.0 vs 4.2.0: only (a).
- 4.2.0 vs 4.3.0: no output differences in any of the 5 files; source diff is only `arguments = ...` doc text.

## Quirks to decide (Spark behaviour that looks like a bug; recorded, not endorsed)

1. `from_avro(NULL-literal, ...)`, `from_avro(1, ...)`, `from_avro('abc', ...)`, `from_avro(array(1), ...)` raise `INTERNAL_ERROR` (XX000, "Cannot resolve the runtime replaceable expression") in 4.0-4.3 instead of a type-mismatch; `from_avro(cast('abc' as binary), ...)` works. Krishiv should likely raise a proper `DATATYPE_MISMATCH`/`UNEXPECTED_INPUT_TYPE` instead (`incompatible`).
2. `from_avro(X'00', '["null","int"]')` returns `0`, not NULL (null branch of a root nullable union loses its nullness through the row updater; `... is null` is false). Cases: basic.sql.
3. `from_avro(..., map(cast(null as string), cast(null as string)))` raises `NULL_MAP_KEY` (2200E) from the map constructor, not from from_avro.
4. Non-UTF-8 string from `cast(X'ff' as string)` is written as-is by to_avro; Krishiv's string type must allow arbitrary bytes there.

## Not covered / open

- Wire cases for Avro `bytes`-with-decimal inferred schemas (fixed size) are only shown through `hex(to_avro(decimal))`; Krishiv must derive the fixed size exactly (`000000007B` for `decimal(10,2)` 1.23, nullable union prefix `00`).
- Avro file source/sink and `schema_of_avro` are out of scope; two `schema_of_avro` queries are included only as type-mapping references.
- `datetimeRebaseMode` LEGACY/EXCEPTION only exercised on epoch day 0 (no pre-1582 dates).
