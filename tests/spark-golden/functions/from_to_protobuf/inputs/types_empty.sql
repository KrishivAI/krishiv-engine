-- empty message types: dropped by default, kept as __dummy_field_in_empty_struct with retain.empty.message.types
select from_protobuf(X'', 'krishiv.golden.types.HasEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.HasEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Empty0', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.Empty0', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true'));
select typeof(from_protobuf(X'', 'krishiv.golden.types.Empty0', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select typeof(from_protobuf(X'', 'krishiv.golden.types.Empty0', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true')));
select typeof(from_protobuf(X'', 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc').e);
select from_protobuf(X'0a0012016e1a001a0022050a016b1200', 'krishiv.golden.types.HasEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0012016e1a001a0022050a016b1200', 'krishiv.golden.types.HasEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true'));
select from_protobuf(X'0a0012016e1a001a0022050a016b1200', 'krishiv.golden.types.HasEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true', 'emit.default.values', 'true'));
select from_protobuf(X'0a00', 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a00', 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true'));
select from_protobuf(X'0a00', 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'maybe'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true', 'mode', 'PERMISSIVE')) is null;
-- to_protobuf into empty / retained-empty messages
select hex(to_protobuf(named_struct(), 'krishiv.golden.types.Empty0', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('__dummy_field_in_empty_struct', 'x'), 'krishiv.golden.types.Empty0', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('e', named_struct()), 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
select hex(to_protobuf(named_struct('e', named_struct('__dummy_field_in_empty_struct', 'x')), 'krishiv.golden.types.OnlyEmpty', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc'));
