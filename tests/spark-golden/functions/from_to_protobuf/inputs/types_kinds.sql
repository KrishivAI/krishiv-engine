-- Kinds: type mapping of repeated/map/oneof/optional/nested enum/field names
select from_protobuf(X'', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2', 'enums.as.ints', 'true', 'upcast.unsigned.ints', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0301020010021a01611a001a01ff22017822002202c3a92a0c0000c03f0000c07f000080ff32100000000000000080000000000000f07f3a020100420201044a05ffffffff0f520affffffffffffffffff015a07080112036f6e655a0d08ffffffffffffffffff0112006206080512027a7a6a0408011002720a0a016b12050a016e10037a1108ffffffff0f10ffffffffffffffffff018a01070a056f6e656f66a201040a027031a20100aa0100b00100ba0100c20100ca01020102d00103d00104da01020801e2010163ea010173f201016a', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'0a0301020010021a01611a001a01ff22017822002202c3a92a0c0000c03f0000c07f000080ff32100000000000000080000000000000f07f3a020100420201044a05ffffffff0f520affffffffffffffffff015a07080112036f6e655a0d08ffffffffffffffffff0112006206080512027a7a6a0408011002720a0a016b12050a016e10037a1108ffffffff0f10ffffffffffffffffff018a01070a056f6e656f66a201040a027031a20100aa0100b00100ba0100c20100ca01020102d00103d00104da01020801e2010163ea010173f201016a', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2', 'enums.as.ints', 'true', 'upcast.unsigned.ints', 'true'));
select from_protobuf(X'0a0301020010021a01611a001a01ff22017822002202c3a92a0c0000c03f0000c07f000080ff32100000000000000080000000000000f07f3a020100420201044a05ffffffff0f520affffffffffffffffff015a07080112036f6e655a0d08ffffffffffffffffff0112006206080512027a7a6a0408011002720a0a016b12050a016e10037a1108ffffffff0f10ffffffffffffffffff018a01070a056f6e656f66a201040a027031a20100aa0100b00100ba0100c20100ca01020102d00103d00104da01020801e2010163ea010173f201016a', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2', 'emit.default.values', 'true'));
-- oneof: each member, none, two members on the wire (last wins)
select from_protobuf(X'900102', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'9a01026162', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'900100', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).sh;
select from_protobuf(X'8a01030a0161900102', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'9001028a01030a0161', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
-- proto3 optional: presence tracked, explicit empty/zero kept
select from_protobuf(X'b00100ba0100', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2', 'emit.default.values', 'true'));
-- enum numbers outside the declared set, negative enum numbers, aliases
select from_protobuf(X'1063', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'1063', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2', 'enums.as.ints', 'true'));
select from_protobuf(X'0a026301', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'10fbffffffffffffffff01', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'08ffffffffffffffffff011203010100', 'krishiv.golden.types.WeirdMsg', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'08ffffffffffffffffff011203010100', 'krishiv.golden.types.WeirdMsg', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('enums.as.ints', 'true'));
select from_protobuf(X'0807', 'krishiv.golden.types.WeirdMsg', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
-- repeated: packed and unpacked encodings of the same field are both accepted, interleaved occurrences concatenate
select from_protobuf(X'ca01020102', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).packed_nums;
select from_protobuf(X'c80101ca01020304c80105', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).packed_nums;
select from_protobuf(X'd00101d201020304', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).unpacked_nums;
-- map: duplicate keys, entry without value, entry without key, map order
select from_protobuf(X'5a0508011201615a050801120162', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).i2s;
select map_keys(from_protobuf(X'5a0508031201615a050801120162', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).i2s);
select from_protobuf(X'5a0408011200', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).i2s;
select from_protobuf(X'5a050800120176', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).i2s;
select from_protobuf(X'72050a016b1200', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).s2p;
select from_protobuf(X'72050a016b1200', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2', 'emit.default.values', 'true')).s2p;
select from_protobuf(X'82010d0a016b12080a017212030a0171', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).s2rec;
-- repeated message elements: an empty element message is a struct of NULLs, not NULL
select from_protobuf(X'a20100a201030a017a', 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2')).ps;
-- to_protobuf: repeated / map / oneof / optional / nested enum / enums
select hex(to_protobuf(named_struct('shades', array('LIGHT', 'DARK'), 'shade', 'DARK'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('shades', array(1, 2)), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('shades', array(1, 2)), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('enums.as.ints', 'true')));
select hex(to_protobuf(named_struct('shade', 99), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('shade', 'LIGHT'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('shade', 'light'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('bins', array(X'ff', X''), 'strs', array('a', '')), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('flts', array(1.5f, cast('NaN' as float)), 'dbls', array(-0.0d), 'bools', array(true, false)), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('s64s', array(-1L, 2L), 'u32s', array(-1), 'u64s', array(-1L)), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('u32s', array(cast(4294967295 as bigint))), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('u64s', array(cast(18446744073709551615 as decimal(20,0)))), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('u64s', array(cast(-1 as decimal(20,0)))), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('u2u', map(cast(4294967295 as bigint), cast(18446744073709551615 as decimal(20,0)))), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('i2s', map(1, 'one', -1, ''), 'l2b', map(5L, X'7a'), 'b2e', map(true, 'DARK')), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('s2p', map('k', named_struct('name', 'n', 'id', 3))), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('p', named_struct('name', 'o')), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('p', named_struct('name', 'o'), 'sh', 'DARK'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('sh', 'SHADE_UNSPECIFIED'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('oshade', 'SHADE_UNSPECIFIED', 'ostr', ''), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('nested', named_struct('inside', 'INSIDE_ONE')), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('single', named_struct()), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('ps', array(named_struct('name', cast(null as string)), named_struct('name', 'z'))), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('ps', array(cast(null as struct<name:string>))), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('CAMELCASE', 'c', 'Snake_Case', 's'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('customJson', 'j'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('json_name_field', 'j'), 'krishiv.golden.types.Kinds', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
-- to_protobuf field names match case-insensitively by default, exactly with spark.sql.caseSensitive=true
select hex(to_protobuf(named_struct('I32', 1), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
