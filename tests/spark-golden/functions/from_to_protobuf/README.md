# from_protobuf / to_protobuf goldens (P3 PR34 steps 1 and 2)

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

## Step 2 (P3-T087 second half): type mapping, options, proto2, well-known types

New fixtures: `fixtures/krishiv_golden_types.{proto,desc}` (proto3: Timestamp, Duration, Any, all wrappers, Empty, empty
messages, self/mutual/only-recursive messages, repeated/map/oneof/optional/nested-enum/packed kinds, aliased and negative
enums) and `fixtures/krishiv_golden_p2.{proto,desc}` (proto2: `required`, defaults, group, extensions, case-clashing names).
New inputs (generated by `gen_inputs_types.py`, wire bytes from `protoc --encode`; no expected value is generated there):
`types_wkt`, `types_any`, `types_empty`, `types_recursion`, `types_kinds`, `to_types`, `proto2`, `types_tz`.
`results/spark-<ver>/<name>.sql.out` come from real runs of 3.5.7, 4.0.0, 4.1.0, 4.2.0, 4.3.0 (`apache/spark:<ver>` own
`--rm` containers, `--packages org.apache.spark:spark-protobuf_2.13:<ver>`, `_2.12` on 3.5.7; session tz UTC, ANSI default).
`schema_nullability.out` (runner `run_schema_nullability.py`, DataFrame API) is the only place the exact nullability of the
result schema is recorded (`simpleString` hides it). **3.5.7 has no SQL function**, so its step-2 `.out` files come from
`run_oracle_df.py`, which rebuilds each call through `pyspark.sql.protobuf.functions` (result columns are named `__pbN`;
values and errors are comparable, and 3.5 errors carry `sqlState: None`). Zero hand-written expectations.

### Registration and signatures (rule: answers "SQL name or DataFrame only?")

- SQL `from_protobuf` / `to_protobuf` exist since 4.0.0 (`FunctionRegistry.scala:1027-1028`, `toFromProtobufSqlFunctions.scala`
  `since = "4.0.0"`); the spark-protobuf module is not on the default classpath of `spark-sql` (class looked up reflectively,
  `CLASSNOTFOUND` -> `PROTOBUF_NOT_LOADED_SQL_FUNCTIONS_UNUSABLE`, `:146,:295`). Krishiv always has it loaded.
- DataFrame API (Scala `org.apache.spark.sql.protobuf.functions`, Python `pyspark.sql.protobuf.functions`; `@Experimental`),
  source `sql/api/src/main/scala/org/apache/spark/sql/protobuf/functions.scala` at v4.2.0 (read through a page fetch, line
  numbers not recorded): `from_protobuf(data, messageName, descFilePath[, options])` since 3.4.0,
  `from_protobuf(data, messageName, binaryFileDescriptorSet: Array[Byte][, options])` since 3.5.0, `from_protobuf(data,
  messageClassName[, options])`; `to_protobuf` has the same six shapes. The SQL form `(data, messageName[, descPathOrBinary|options][, options])`
  maps onto the same expression (`ProtobufDataToCatalyst` / `CatalystDataToProtobuf`). Java-class form (no descriptor) is
  `incompatible`/`unsupported` in Krishiv (no JVM).
- `to_protobuf` ignores `options` (`CatalystDataToProtobuf.scala` never reads them): `to_protobuf(..., map('enums.as.ints','true'))`
  still accepts the enum NAME or number (`types_kinds`), and `retain.empty.message.types`/`recursive.fields.max.depth` do not
  restrict the struct (`types_empty`, `types_recursion`). Options are only validated, never parsed, on the to side.

### Type mapping, protobuf -> Spark (from_protobuf; `SchemaConverters.scala` v4.2.0)

| Rule | Source | Case |
|---|---|---|
| The result is a struct of ALL message fields in declaration order, nullable root (`SchemaType(..., nullable = true)`), non-repeated fields nullable unless `required` (proto2 `required` -> `NOT NULL`), repeated -> `array<T NOT NULL>`, map -> `map<K,V NOT NULL>` (`valueContainsNull = false`); a missing repeated/map is `[]`/`{}` not NULL | `SchemaConverters.scala:60-70,243-244`; `schema_nullability.out` | schema_nullability, types_kinds |
| int32/sint32/sfixed32 -> INT, int64/sint64/sfixed64 -> BIGINT, uint32/fixed32 -> INT (wraps), uint64/fixed64 -> BIGINT (wraps); `upcast.unsigned.ints=true` -> BIGINT for `uint32` and `DECIMAL(20,0)` for `uint64` ONLY; `fixed32`/`fixed64` have wire type FIXED32/FIXED64, are not unsigned-upcast and stay `int`/`bigint` (Scalars schema with the option on: `u32:bigint, u64:decimal(20,0), f32:int, f64:bigint`) | `SchemaConverters.scala:84-95`; `schema_nullability.out` (Scalars upcast) | schema_nullability, types_kinds |
| enum -> STRING (value name), `enums.as.ints` -> INT; an enum number not in the (open, proto3) enum renders as `UNKNOWN_ENUM_VALUE_<Enum>_<n>` (negative: `..._-5`); with `enums.as.ints` the raw number; aliases render the first-declared name (`W_ONE` for 1) | `ProtobufDeserializer.scala:414-425`; run | types_kinds |
| proto2 closed enum with an unknown number goes to unknown fields; because a known field number is then present in the unknown set, the message is "malformed" (`PROTOBUF_FIELD_TYPE_MISMATCH`, FAILFAST error). The error names the wrong field: the code does `getFields.get(number)` (an INDEX), so field number 5 reports field `b` | `ProtobufDataToCatalyst.scala:105-116`; run | proto2 |
| `google.protobuf.Timestamp` -> TIMESTAMP (micros = seconds*1000 via `multiplyExact`, plus `NANOSECONDS.toMicros(nanos)` truncating toward zero; `seconds` > 9223372036854 overflows -> malformed; negative/over-range `nanos` are simply added, no normalisation) matched BY MESSAGE NAME `Timestamp` with fields `seconds`,`nanos` in that order (any package) | `SchemaConverters.scala:109-113`; `ProtobufDeserializer.scala:277-285` | types_wkt, types_tz |
| `google.protobuf.Duration` -> `INTERVAL DAY TO SECOND` (same arithmetic: `seconds*1000` millis -> micros + nanos truncated to micros; mixed signs just add, so `{1,-500000000}` is 0.5 s) matched by name `Duration` | `SchemaConverters.scala:103-107`; `ProtobufDeserializer.scala:287-295` | types_wkt |
| `google.protobuf.Any` -> `struct<type_url:string,value:binary>`; `convert.any.fields.to.json=true` -> STRING with `JsonFormat` (`omittingInsignificantWhitespace`, `preservingProtoFieldNames`, enums as ints when `enums.as.ints`), type registry = whole descriptor set; an unresolvable `type_url` or malformed payload is a record-level failure (FAILFAST error / PERMISSIVE NULL); an empty `Any` prints `{}`; `emit.default.values` does not add JSON defaults; repeated Any -> `array<string>` | `SchemaConverters.scala:114-117`; `ProtobufDataToCatalyst.scala:62-69`; `ProtobufDeserializer.scala:82-95,297-302` | types_any |
| Primitive wrappers (`Int32Value` ... `BytesValue`) -> `struct<value:T>` by default; `unwrap.primitive.wrapper.types=true` -> the primitive (UInt32Value -> INT, or BIGINT with `upcast.unsigned.ints`; UInt64Value -> BIGINT / `DECIMAL(20,0)`). A wrapper present but empty gives `{NULL}` (struct) / NULL (unwrapped); with `emit.default.values`+unwrap an empty wrapper yields the zero value (`0`, `''`, `false`), without unwrap `{0}` | `SchemaConverters.scala:118-153`; `ProtobufDeserializer.scala:304-398` | types_wkt |
| `google.protobuf.Empty` and any other message with no fields: field DROPPED from the parent (struct omitted); a top-level empty message gives `struct<>`; `retain.empty.message.types=true` keeps it as `struct<__dummy_field_in_empty_struct:string>` (value `{NULL}`, struct itself non-null when present). A message that becomes empty only because every field was dropped by the recursion limit is treated the same | `SchemaConverters.scala:60-62,220-232,248-253` | types_empty |
| `repeated`/map entries of a message type whose type is dropped (recursion limit) drop the whole container field; map key/value `Rec` at depth limit drops the map | `SchemaConverters.scala:155-184` | types_recursion |
| `oneof` members are all ordinary nullable fields; setting two members on the wire keeps the last one only (protobuf semantics) | run | types_kinds |
| `optional` (proto3 presence) and message fields are never filled by `emit.default.values`; singular proto3 scalars, enums and `string`/`bytes` are (`SHADE_UNSPECIFIED`, `''`, `0`); inside any PRESENT nested message (singular, repeated element or map value) `emit.default.values` also fills that message's plain scalars (`{, 0}`) | `ProtobufDeserializer.scala:493-512` | types_kinds |
| map: duplicate keys are NOT merged (`{1 -> a, 1 -> b}`), wire order is preserved, a missing value is the type's default (`''`, `{null,null}` struct of NULLs; with `emit.default.values` zero fields), a missing key is key 0/`''` | `ProtobufDeserializer.scala:139-180` | types_kinds |
| Field names are the proto field names (`snake_case`, `CamelCase`), never the JSON name | run | types_kinds |
| `recursive.fields.max.depth`: default -1 and 0 raise `RECURSIVE_PROTOBUF_SCHEMA` (42K0G); 1 drops every recursive field; N allows N-1 levels; 11 and 100 raise again; depth counts per message type along the path (mutual recursion MA<->MB counts each type), values parse with Scala `String.toInt` (`+2`, `02` accepted; `' 2'`, `'2 '`, `1.5`, `abc` -> `STDS_INVALID_OPTION_VALUE.WITH_MESSAGE` from 4.2.0) | `SchemaConverters.scala:187-210`; `ProtobufOptions.scala:92` | types_recursion |
| Values deeper than the limit are silently truncated (field absent from the result), not an error | run | types_recursion |
| proto2 `[default = X]` is returned for an absent field (`opt` -> `dflt`, enum default, bytes default, `inf`), a field without a default is NULL; `required` missing -> the whole record is malformed (FAILFAST error naming the field, PERMISSIVE NULL), also for a missing required field in a nested/repeated element (`ps[0].req`); group -> a struct field named after the (lower-cased) group | `ProtobufDeserializer.scala:493-512`; run | proto2 |
| Extensions: only with `spark.sql.function.protobufExtensions.enabled=true` (SQLConf, added in 4.2.0): extension fields are appended after regular fields in field-number order, for from AND to; the conf is ignored on 3.5.7/4.0/4.1 | `ProtobufUtils.scala:241-262`, `SchemaConverters.scala:50-58`; `schema_nullability.out`, `proto2` | proto2 |
| Field matching for `to_protobuf` is case-insensitive by default (`I32` -> `i32`); with two proto fields differing only by case `PROTOBUF_FIELD_MISSING` (42K0G, "gave 2 matches"); `spark.sql.caseSensitive=true` requires exact names; `from_protobuf` of such a message fails at conversion (`CANNOT_CONVERT_PROTOBUF_MESSAGE_TYPE_TO_SQL_TYPE`) | `ProtobufUtils.scala:141-158` | proto2 |

### Type mapping, Spark -> protobuf (to_protobuf; `ProtobufSerializer.scala`)

| Rule | Source | Case |
|---|---|---|
| exact type pairs only (`(IntegerType, INT)`, `(LongType, LONG)`, ...); `uint32` accepts INT (raw bits) or BIGINT (`toInt` wrap), `uint64` accepts BIGINT (raw bits) or `DECIMAL` whose unscaled value fits a long: `decimal(20,0)` 18446744073709551615 raises a raw `java.lang.ArithmeticException: BigInteger out of long range` (not a Spark error), `-1` writes the all-ones value; on 3.5.7 neither the BIGINT-for-uint32 nor the DECIMAL-for-uint64 form exists (`UNABLE_TO_CONVERT_TO_PROTOBUF_MESSAGE_TYPE`) | `ProtobufSerializer.scala:100-110` | types_kinds |
| TIMESTAMP -> `Timestamp` using `microsToMillis` (FLOOR division to ms: sub-millisecond precision is lost; `seconds = millis / 1000`, `nanos = (millis % 1000) * 1e6` with Java truncation, so `1969-12-31 23:59:59.999999` encodes `{seconds:0, nanos:-1000000}`, `23:59:59.0005` encodes `{seconds:-1}`); TIMESTAMP_NTZ, DATE, BIGINT into a `Timestamp` field are `UNABLE_TO_CONVERT_TO_PROTOBUF_MESSAGE_TYPE`; a `struct<seconds:bigint,nanos:int>` also works through the struct path | `ProtobufSerializer.scala:152-160`; run | to_types, types_tz |
| `INTERVAL DAY TO SECOND` (any start/end field) -> `Duration`, millisecond precision, seconds and nanos re-normalised to the same sign; YEAR-MONTH intervals are rejected | `ProtobufSerializer.scala:291-312` | to_types |
| primitives into wrapper fields: INT -> `Int32Value`/`UInt32Value`, BIGINT -> `Int64Value`/`UInt64Value`, STRING, BINARY, BOOLEAN, FLOAT, DOUBLE (zero values are written as an empty-but-present wrapper `2200`); a `struct<value:T>` also works; on 3.5.7 wrappers are not writable | `ProtobufSerializer.scala:188-232` | to_types |
| enum fields accept the NAME (case-sensitive) or INT `findValueByNumber`; unknown -> `CANNOT_CONVERT_SQL_VALUE_TO_PROTOBUF_ENUM_TYPE` (42846; the message lists names for the string form and numbers for the int form) | `ProtobufSerializer.scala:115-142` | types_kinds |
| array -> repeated (an array with a NULL struct element hits an optimizer-phase `INTERNAL_ERROR`, recorded not copied); map -> entry list (NULL value -> the value type's default); `oneof` members all settable, two set in one struct -> the LATER struct field wins (`DynamicMessage` clears the other member); `optional` zero/empty IS written, plain proto3 zero is NOT written; an empty `named_struct()` into a message field writes the empty message (`0A00`) | `ProtobufSerializer.scala:161-187,244-290,360-368` | types_kinds, types_empty |
| proto2: a missing `required` field in the struct is `PROTOBUF_FIELD_MISSING_IN_SQL_SCHEMA` (42K0G) at analysis; a NULL `required` scalar writes the type default (`0800`), a NULL `required` MESSAGE raises a raw `UninitializedMessageException` (no error class); explicit `[default=]` values are written when given | `ProtobufSerializer.scala:341-368`; `ProtobufUtils.scala:110-135` | proto2 |
| Struct field names resolved through the case rules above; `json_name` does not participate | `ProtobufUtils.scala` | types_kinds |

### Session settings

ANSI on/off and the session time zone never change the bytes either function reads or writes; only the rendering of a decoded
`TIMESTAMP` and the interpretation of a local timestamp literal that is encoded depend on the zone (`types_tz`, incl. both
2024 US DST edges: encoding `2024-03-10 02:30:00` in `America/Los_Angeles` resolves the gap forward). `DurOnly` decoding is zone-free.

## Per-version differences for step 2 (diff of the five runs)

- 3.5.7: DataFrame API only; no `unwrap.primitive.wrapper.types`, `upcast.unsigned.ints`, `retain.empty.message.types`
  (options silently ignored: wrappers stay structs, empty messages dropped), no extensions; the to side cannot write
  BIGINT->uint32, DECIMAL->uint64, wrapper types; every Spark error has `sqlState: None`; option parse errors are raw Java
  exceptions. `Timestamp`/`Duration`, `Any` JSON, recursion and enum behaviour equal 4.x.
- 4.0.0 = 4.1.0 (all step-2 outputs identical). 4.0/4.1 vs 4.2.0/4.3.0: (a) option value parse errors are raw
  `java.lang.IllegalArgumentException` (booleans) / `NumberFormatException` (`recursive.fields.max.depth`) in 4.0/4.1 and
  `STDS_INVALID_OPTION_VALUE.WITH_MESSAGE` (42616) from 4.2.0; (b) protobuf extensions (`spark.sql.function.protobufExtensions.enabled`) exist only from 4.2.0.
- 4.2.0 = 4.3.0 except the text of a `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` message on an interval addition used only to build an input (`to_types`).

## Quirks to decide (recorded, not endorsed)

5. `to_protobuf(TIMESTAMP)` loses sub-millisecond precision and writes `{seconds:0,nanos:-1000000}` for the last ms before the epoch (an invalid `Timestamp`); `from_protobuf` adds `nanos` without normalising (`{seconds:1,nanos:-1}` decodes to `00:00:01`, `types_wkt`); proposed `compatible` (bug-for-bug) since such bytes may already exist.
6. `to_protobuf` of `decimal(20,0)` values above `Long.MAX` into `uint64` raises a raw `ArithmeticException` (no error class); `from_protobuf` with `upcast.unsigned.ints` is the lossless direction. Proposed: raise a proper Spark error (`incompatible`, opt-in to the raw text).
7. Unknown-field mismatch errors name the field by INDEX, not field number (see proto2 row); the message text is part of the cause only. Krishiv may report the right field (`incompatible` only in cause text).
8. `required` NULL message in `to_protobuf` raises `UninitializedMessageException` (class name only, no Spark error class); Krishiv must raise an error of the same kind (a SparkException wrapping, class `INTERNAL_ERROR` is not what Spark gives).
9. `convert.any.fields.to.json` uses protobuf-java `JsonFormat` (64-bit ints as strings, `Timestamp` as RFC 3339, `Duration` as `"1s"`, bytes base64, field names preserved). Krishiv needs an equivalent JSON printer; the goldens cover message, Timestamp, Duration, Int32Value only.
10. Enum numbers outside the proto3 enum render as `UNKNOWN_ENUM_VALUE_<Enum>_<n>` (Java `findValueByNumberCreatingIfUnknown` naming).
11. Wrapper `UInt32Value` with `unwrap` and no upcast returns INT (wraps negative); `UInt64Value` BIGINT wraps.

## Krishiv divergences (recorded, deliberate)

12. Nullability of unwrapped wrapper lists: under `unwrap.primitive.wrapper.types=true` a `repeated Int32Value` is `array<int NOT NULL>` in Spark, yet an empty wrapper element (`0A00`) reads as a NULL element (`[5, NULL, 7]`). Krishiv gives the same values and declares `array<int>` (`containsNull = true`) for these arrays only, so no element is dropped. Test: `krishiv-io` `an_empty_wrapper_element_of_a_repeated_unwrapped_field_stays_a_null`.
13. In the 4-argument form, a descriptor path that is foldable but not a literal (`concat('/d/', 'x.desc')`) passes the argument check and is then refused with `KRISHIV_UNSUPPORTED_FEATURE` (reading a protobuf descriptor file by path); Spark folds it and reads the file. Test: `krishiv-session` `protobuf_access::a_foldable_non_literal_descriptor_path_is_unsupported_not_read`.
14. A malformed map entry (no key or value field) in the descriptor set is `CANNOT_PARSE_PROTOBUF_DESCRIPTOR` here; Spark's protobuf-java builds the descriptor and fails with its own exception text. Test: `krishiv-io` `a_map_entry_without_its_key_and_value_fields_does_not_parse`.

## Not covered / open

- Java-class mode (`messageClassName`): needs a JVM, listed as `unsupported`.
- `Any` inside a map/oneof, proto2 `Any`, `Struct`/`Value`/`ListValue`/`FieldMask` (no special mapping in Spark: they come out as plain messages; not exercised), `Timestamp`/`Duration` declared in a user package with the same short name (Spark matches by short name and field names only; not exercised).
- Very large messages, `max recursion` of the protobuf parser itself (depth 100 wire nesting), and non-UTF-8 in `string` map keys.
- `types_tz` runs a few statements under `spark.sql.ansi.enabled=false` only to show the functions read no ANSI setting; the ANSI-on/off pairs for cast/NULL handling are in `ansi_tz.sql`.
- `from_protobuf` binary-descriptor form (`X'...'` third argument) is exercised by step 1 only (`basic.sql`); step 2 uses the path form and, on 3.5.7, only the path form exists in the DataFrame API.
