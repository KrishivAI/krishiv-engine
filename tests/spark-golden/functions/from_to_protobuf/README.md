# from_protobuf / to_protobuf goldens (P3 PR34 step 1)

Layout: `fixtures/krishiv_golden.proto` (Krishiv-authored; scalars of every wire type, enum, nested, repeated, map,
optional, oneof, recursive `Node`) and `fixtures/krishiv_golden.desc` (built with
`protoc --include_imports --descriptor_set_out=krishiv_golden.desc krishiv_golden.proto`, libprotoc from the host);
`inputs/<name>.sql`; `results/spark-<ver>/<name>.sql.out` (real runs); `../run_oracle.py`.
In the container the golden tree is mounted read-only at `/golden`, so the descriptor path in the SQL is
`/golden/from_to_protobuf/fixtures/krishiv_golden.desc` (the testkit must remap it). Hand-built wire bytes in
`from_protobuf` inputs were produced with `protoc --encode`; every expected value is from the Spark run.
Run set: 3.5.7, 4.0.0, 4.1.0, 4.2.0, 4.3.0 with `spark-protobuf_2.13:<ver>` (`_2.12` on 3.5.7); see the avro README for the setup.

## Rules and sources (v4.2.0; v4.3.0 differs only in docs text and the descriptor reader class name `ProtobufDescriptorFileReader`)

| Rule | Source | Case |
|---|---|---|
| SQL functions exist since 4.0.0; on 3.5.7 `UNRESOLVED_ROUTINE` | `toFromProtobufSqlFunctions.scala:65,202`; `results/spark-3.5.7` | all |
| Forms: `(data, messageName)`, `(data, messageName, descFilePathOrOptions)` (a map third arg = options, a string/binary literal third arg = descriptor path/bytes), `(data, messageName, descFilePath, options)`; 5+ or <2 args: `WRONG_NUM_ARGS.WITHOUT_SUGGESTION` | `toFromProtobufSqlFunctions.scala:70-90` | errors_args |
| messageName must be a foldable STRING (a NULL literal is rejected with the same TYPE_CHECK_FAILURE_WITH_HINT text); descFilePath a foldable STRING/BINARY/NULL; options a foldable `MAP<STRING,STRING>`/NULL; hint texts in the `.out` | `:104-134`, `:241-285` | errors_args |
| `to_protobuf` first arg must be STRUCT: `DATATYPE_MISMATCH.NON_STRUCT_TYPE` (42K09) incl. for NULL literal (`VOID`) | `:241-251` | basic, errors_args |
| No descriptor (or empty string/empty binary/NULL) means messageName is a Java class name: `CANNOT_LOAD_PROTOBUF_CLASS` (42K03) here; a non-string/binary third arg (e.g. `42`) also falls to that path | `ProtobufUtils.scala:188-204` | basic, errors_args |
| messageName matches the short or the full name (first match in the set) ; unknown -> `PROTOBUF_MESSAGE_NOT_FOUND` (42K0G); an enum name is not a message | `ProtobufUtils.scala:241-255` | basic |
| `to_protobuf` differs from `from_protobuf` for an empty descriptor: `''` path -> `CANNOT_PARSE_PROTOBUF_DESCRIPTOR` (cause text differs: 4.2 `Is a directory`, 4.3 `Can not create a Path from an empty string`), `X''` -> `PROTOBUF_MESSAGE_NOT_FOUND`; `from_protobuf` treats both as no descriptor | `toFromProtobufSqlFunctions.scala:145-148` vs `:295-298`; run | modes_options |
| Missing file `PROTOBUF_DESCRIPTOR_FILE_NOT_FOUND` (42K0G); garbage bytes `CANNOT_PARSE_PROTOBUF_DESCRIPTOR` (22018) | `ProtobufUtils.scala:265-272` | basic |
| from_protobuf result is a struct of ALL message fields (proto3 plain scalars nullable, absent -> NULL; repeated -> empty array; map -> empty map; oneof members all fields, unset NULL); `emit.default.values=true` renders type zero values for plain proto3 scalar fields (`COLOR_UNSPECIFIED` for enums) but not for `optional`, message-typed or unset oneof fields | `ProtobufOptions.scala:92-176`; run | basic |
| Unsigned ints: `uint32`/`fixed32` -> INT (wraps negative), `uint64`/`fixed64` -> BIGINT (wraps); `upcast.unsigned.ints=true` -> BIGINT / `DECIMAL(20,0)` | `ProtobufOptions`; run | basic |
| enums: string name by default (`800101` RED, `800102` GREEN); `enums.as.ints=true` -> INT column; a truncated enum varint (`8001ff`) is malformed. Unknown numeric enum values are not exercised | run | basic |
| bytes -> BINARY, invalid UTF-8 in a proto `string` field -> malformed (FAILFAST error / PERMISSIVE NULL); to_protobuf of invalid-UTF-8 Spark string writes U+FFFD (`EFBFBD`), not the raw byte | run | basic, modes_options, ansi_tz |
| Recursive message without `recursive.fields.max.depth` -> `RECURSIVE_PROTOBUF_SCHEMA` (42K0G); depth 1 keeps no `next`, 2 and 3 nest one/two levels (only 1-3 exercised); 0 and 11 raise it again (observed) | run | basic |
| Parse mode default FAILFAST: `MALFORMED_PROTOBUF_MESSAGE` (XX000, SparkException); PERMISSIVE gives NULL (not a struct of NULLs, unlike Avro); DROPMALFORMED raises `PARSE_MODE_UNSUPPORTED` only on the malformed path (valid input decodes), an unrecognised mode string (`bogus`) silently behaves as PERMISSIVE (verified by run: NULL) | `ProtobufDataToCatalyst.scala:81-96,129` | modes_options |
| A known field number with a different wire type is "unknown field matching a regular field" -> malformed (`protobufFieldTypeMismatchError`) | `ProtobufDataToCatalyst.scala:105-116` | modes_options |
| to_protobuf field mapping is strict: struct field type must equal the proto type (int->int32 only; long, tinyint, decimal, string-for-int are `UNABLE_TO_CONVERT_TO_PROTOBUF_MESSAGE_TYPE` 42K0G); unknown struct field names also fail; missing struct fields are omitted; zero/proto3-default and NULL values are not written (empty bytes); `optional` zero IS written; enums accept the name or the int; bad name `CANNOT_CONVERT_SQL_VALUE_TO_PROTOBUF_ENUM_TYPE` (42846) | `ProtobufSerializer`; run | basic |
| to_protobuf NULL struct (typed) -> NULL; map entries with NULL value are written as zero | run | basic |
| Option value parse errors: from 4.2.0 `STDS_INVALID_OPTION_VALUE.WITH_MESSAGE` (42616); 4.0.0/4.1.0 raw `IllegalArgumentException`/`NumberFormatException` | `ProtobufOptions.scala:38-52`; run | modes_options |
| Unknown options ignored | run | modes_options |
| Conditional evaluation: CASE/IF/COALESCE/AND/OR skip unselected rows (no FAILFAST error); to_protobuf type errors are analysis errors so they fire even in an unselected branch | run | conditional |
| ANSI on/off: no differences for either function | run | ansi_tz |
| Session time zone never changes the bytes (timestamps cannot be written to int64 fields directly: `TIMESTAMP` -> 42K0G; use `unix_micros`); `date` into `int32` is accepted (days) | run | ansi_tz |

## Per-version differences

- 3.5.7: all SQL calls `UNRESOLVED_ROUTINE` (42883).
- 4.0.0 = 4.1.0 (identical outputs). 4.2.0 = 4.3.0 except one cause text (see the empty-descriptor row).
- 4.0/4.1 vs 4.2: option value errors only (`enums.as.ints`, `emit.default.values`, `recursive.fields.max.depth` with a non-parsable value): raw Java exception -> `STDS_INVALID_OPTION_VALUE.WITH_MESSAGE`.

## Quirks to decide

1. `from_protobuf(1|'abc'|array|NULL literal, ...)` raise `INTERNAL_ERROR` (XX000) instead of a type error (same as from_avro). Proposed `incompatible`.
2. `to_protobuf(named_struct('nums', array(1, null, 3)), ...)` hits an optimizer-phase `INTERNAL_ERROR` in Spark; recorded, do not copy.
3. `from_protobuf(..., 42)` (third arg int) is not a type error but falls into the Java-class path (`CANNOT_LOAD_PROTOBUF_CLASS`).
4. Java-class mode (`messageName` = generated class on the classpath) cannot be reproduced without a JVM; propose `unsupported` with `CANNOT_LOAD_PROTOBUF_CLASS`.

## Not covered / open

- `convert.any.fields.to.json`, `unwrap.primitive.wrapper.types`, `retain.empty.message.types`: accepted only (typeof checked); no `Any`/wrapper message in the descriptor. A follow-up with a second descriptor is needed if those options are in scope.
- Proto2 messages, extensions (`spark.sql.function.protobufExtensions.enabled`) and well-known types (Timestamp, Duration) are not covered.
