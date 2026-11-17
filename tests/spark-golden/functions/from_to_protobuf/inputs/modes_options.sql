-- parse modes: default FAILFAST
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'FAILFAST'));
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('MODE', 'permissive'));
select from_protobuf(X'0a', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'0a', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'0a05', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'08', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'0801ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'0801ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'7202ff00', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'7202ff00', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'0a0100', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'0a0100', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'1001', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'1801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE')) is null;
select from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'DROPMALFORMED'));
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'DROPMALFORMED'));
select from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'bogus'));
select from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('enums.as.ints', 'maybe'));
select from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('emit.default.values', 'maybe'));
select from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('nonsense', '1'));
select typeof(from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('convert.any.fields.to.json', 'true')));
select typeof(from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('unwrap.primitive.wrapper.types', 'true')));
-- an unrecognised mode string behaves as PERMISSIVE
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'bogus'));
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'dropmalformed'));
-- to_protobuf with an empty descriptor path
select hex(to_protobuf(named_struct('i32', 1), 'Scalars', ''));
select hex(to_protobuf(named_struct('i32', 1), 'Scalars', X''));
