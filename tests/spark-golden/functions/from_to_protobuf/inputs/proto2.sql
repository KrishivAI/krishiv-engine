-- proto2: required, defaults, group, extensions (spark.sql.function.protobufExtensions.enabled)
select from_protobuf(X'', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'0801', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'0801', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('emit.default.values', 'true'));
select from_protobuf(X'08001200180028003900000000000000004000', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'0801200120024b08054c52030a016c', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'08014b4c', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select typeof(from_protobuf(X'', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('mode', 'PERMISSIVE')).grp);
select from_protobuf(X'08012807', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'08012807', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('enums.as.ints', 'true'));
-- proto2 unknown enum number: preserved as unknown field, not an enum value
select from_protobuf(X'08012807', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
-- proto2 message nesting and a repeated message with a missing required field
select from_protobuf(X'', 'krishiv.golden.p2.P2Nest', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'0a020801', 'krishiv.golden.p2.P2Nest', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'1202080112020802', 'krishiv.golden.p2.P2Nest', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'1200', 'krishiv.golden.p2.P2Nest', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'1200', 'krishiv.golden.p2.P2Nest', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'0a01781200', 'krishiv.golden.p2.P2Required', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'0a0178', 'krishiv.golden.p2.P2Required', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'0a0178', 'krishiv.golden.p2.P2Required', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('mode', 'PERMISSIVE'));
-- extensions: not in the schema by default; with the SQL conf on, extension fields are appended (4.2 source)
select from_protobuf(X'0801a00605aa060165b206030a0178b80601b80602', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
set spark.sql.function.protobufExtensions.enabled=true;
select from_protobuf(X'', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(X'0801a00605aa060165b206030a0178b80601b80602', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select from_protobuf(X'0801a00605aa060165b206030a0178b80601b80602', 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc', map('emit.default.values', 'true'));
select hex(to_protobuf(named_struct('req', 1, 'ext_i', 5, 'ext_s', 'e', 'ext_item', named_struct('label', 'x'), 'ext_r', array(1, 2)), 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
set spark.sql.function.protobufExtensions.enabled=false;
select hex(to_protobuf(named_struct('req', 1, 'ext_i', 5), 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('req', 1), 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
-- to_protobuf proto2: required fields missing / NULL; defaults are not written
select hex(to_protobuf(named_struct('opt', 'x'), 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('req', cast(null as int)), 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('req', 0, 'opt', 'dflt', 'hue', 'HUE_C', 'flag', true, 'd', cast('Infinity' as double)), 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('req', 1, 'grp', named_struct('x', 5)), 'krishiv.golden.p2.P2', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('a', 'x', 'b', named_struct()), 'krishiv.golden.p2.P2Required', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('a', 'x', 'b', cast(null as struct<label:string>)), 'krishiv.golden.p2.P2Required', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('a', 'x'), 'krishiv.golden.p2.P2Required', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('a', cast(null as string), 'b', named_struct()), 'krishiv.golden.p2.P2Required', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
-- case-insensitive field resolution and ambiguity (CaseClash has fields abc and ABC)
select from_protobuf(X'0a0131120132', 'krishiv.golden.p2.CaseClash', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc');
select hex(to_protobuf(named_struct('abc', '1'), 'krishiv.golden.p2.CaseClash', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('ABC', '2'), 'krishiv.golden.p2.CaseClash', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('Abc', '3'), 'krishiv.golden.p2.CaseClash', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
set spark.sql.caseSensitive=true;
select hex(to_protobuf(named_struct('Abc', '3'), 'krishiv.golden.p2.CaseClash', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('abc', '1', 'ABC', '2'), 'krishiv.golden.p2.CaseClash', '/golden/from_to_protobuf/fixtures/krishiv_golden_p2.desc'));
select hex(to_protobuf(named_struct('I32', 1), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
set spark.sql.caseSensitive=false;
select hex(to_protobuf(named_struct('I32', 1), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
