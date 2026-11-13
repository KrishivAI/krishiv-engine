-- malformed input, default mode (FAILFAST)
select from_avro(X'ff', '"int"', map());
select from_avro(X'ff', '"int"', map('mode', 'FAILFAST'));
select from_avro(X'ff', '"int"', map('mode', 'failfast'));
select from_avro(X'ff', '"int"', map('MODE', 'FAILFAST'));
-- PERMISSIVE: the result for a scalar is NULL
select from_avro(X'ff', '"int"', map('mode', 'PERMISSIVE'));
select from_avro(X'ff', '"int"', map('mode', 'permissive'));
-- PERMISSIVE with a record: all fields NULL, even non-nullable ones
select from_avro(X'ff', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"},{"name":"b","type":"string"}]}', map('mode', 'PERMISSIVE'));
select from_avro(X'ff', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"},{"name":"b","type":"string"}]}', map('mode', 'PERMISSIVE')) is null;
select from_avro(X'ff', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"},{"name":"b","type":"string"}]}', map('mode', 'PERMISSIVE')).a;
select from_avro(X'0201', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"},{"name":"b","type":"string"}]}', map('mode', 'PERMISSIVE'));
-- unsupported modes
select from_avro(X'02', '"int"', map('mode', 'DROPMALFORMED'));
select from_avro(X'ff', '"int"', map('mode', 'DROPMALFORMED'));
select from_avro(X'02', '"int"', map('mode', 'bogus'));
-- avroSchema option: evolved reader schema
select from_avro(X'02', '"int"', map('avroSchema', '"long"'));
select from_avro(X'02', '"int"', map('avroSchema', '"string"'));
select from_avro(X'0202', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"}]}', map('avroSchema', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"},{"name":"b","type":"string","default":"dflt"}]}'));
select from_avro(X'0202', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"}]}', map('avroSchema', '{"type":"record","name":"r","fields":[{"name":"a","type":"int"},{"name":"b","type":"string"}]}'));
-- stable identifiers for unions
select from_avro(X'0002', '{"type":"record","name":"r","fields":[{"name":"u","type":["int","string"]}]}', map('enableStableIdentifiersForUnionType', 'true'));
select from_avro(X'020261', '{"type":"record","name":"r","fields":[{"name":"u","type":["int","string"]}]}', map('enableStableIdentifiersForUnionType', 'true'));
select from_avro(X'020261', '{"type":"record","name":"r","fields":[{"name":"u","type":["int","string"]}]}', map('enableStableIdentifiersForUnionType', 'true', 'stableIdentifierPrefixForUnionType', 'u_'));
select typeof(from_avro(X'0204', '["int","string"]', map('enableStableIdentifiersForUnionType', 'true')));
select typeof(from_avro(X'0204', '["int","string"]', map('enableStableIdentifiersForUnionType', 'true', 'stableIdentifierPrefixForUnionType', 'u_')));
select typeof(from_avro(X'0204', '["int","string"]', map('enableStableIdentifiersForUnionType', 'maybe')));
-- recursive schemas
select typeof(from_avro(X'00', '{"type":"record","name":"Node","fields":[{"name":"v","type":"int"},{"name":"next","type":["null","Node"]}]}', map()));
select typeof(from_avro(X'00', '{"type":"record","name":"Node","fields":[{"name":"v","type":"int"},{"name":"next","type":["null","Node"]}]}', map('recursiveFieldMaxDepth', '1')));
select typeof(from_avro(X'00', '{"type":"record","name":"Node","fields":[{"name":"v","type":"int"},{"name":"next","type":["null","Node"]}]}', map('recursiveFieldMaxDepth', '2')));
select typeof(from_avro(X'00', '{"type":"record","name":"Node","fields":[{"name":"v","type":"int"},{"name":"next","type":["null","Node"]}]}', map('recursiveFieldMaxDepth', '0')));
select typeof(from_avro(X'00', '{"type":"record","name":"Node","fields":[{"name":"v","type":"int"},{"name":"next","type":["null","Node"]}]}', map('recursiveFieldMaxDepth', '99')));
-- invalid schema
select from_avro(X'02', 'not json', map());
select from_avro(X'02', '', map());
select from_avro(X'02', '"bogus"', map());
select from_avro(X'02', '{"type":"record","name":"r"}', map());
-- datetime rebase options
select from_avro(X'00', '{"type":"int","logicalType":"date"}', map('datetimeRebaseMode', 'LEGACY'));
select from_avro(X'00', '{"type":"int","logicalType":"date"}', map('datetimeRebaseMode', 'CORRECTED'));
select from_avro(X'00', '{"type":"int","logicalType":"date"}', map('datetimeRebaseMode', 'EXCEPTION'));
select from_avro(X'00', '{"type":"int","logicalType":"date"}', map('datetimeRebaseMode', 'bogus'));
-- unknown options are ignored
select from_avro(X'02', '"int"', map('nonsense', '1'));
-- an unrecognised mode string behaves as PERMISSIVE (ParseMode.fromString falls back with a warning)
select from_avro(X'ff', '"int"', map('mode', 'bogus'));
select from_avro(X'ff', '"int"', map('mode', ''));
