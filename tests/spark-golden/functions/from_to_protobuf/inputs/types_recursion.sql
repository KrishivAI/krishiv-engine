-- recursive.fields.max.depth: -1 (default) rejects, 1 drops the recursive field, 2.. nests; 0 and >10 rejected
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '-1'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '0'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '3'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '4'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '5'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '10'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '11'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '100'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', 'abc'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1.5'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', ' 2'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2 '));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '+2'));
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '02'));
-- mutual recursion MA <-> MB, a message that is only a recursive field
select from_protobuf(X'', 'krishiv.golden.types.MA', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.MB', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.MA', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1'));
select from_protobuf(X'', 'krishiv.golden.types.MB', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1', 'retain.empty.message.types', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.MA', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'', 'krishiv.golden.types.MB', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2', 'retain.empty.message.types', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.MA', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '3'));
select from_protobuf(X'', 'krishiv.golden.types.MB', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '3'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '3'));
select from_protobuf(X'', 'krishiv.golden.types.OnlyRec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '3', 'retain.empty.message.types', 'true'));
-- values truncate at the depth limit
select from_protobuf(X'0a0161120d0a016212080a016312030a01641a0a0a026b3112040a026b32220d0a017812080a017612030a0177', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1'));
select from_protobuf(X'0a0161120d0a016212080a016312030a01641a0a0a026b3112040a026b32220d0a017812080a017612030a0177', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
select from_protobuf(X'0a0161120d0a016212080a016312030a01641a0a0a026b3112040a026b32220d0a017812080a017612030a0177', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '3'));
select from_protobuf(X'0a0161120d0a016212080a016312030a01641a0a0a026b3112040a026b32220d0a017812080a017612030a0177', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '4'));
select from_protobuf(X'0a0161120d0a016212080a016312030a01641a0a0a026b3112040a026b32220d0a017812080a017612030a0177', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0161120d0a016212080a016312030a01641a0a0a026b3112040a026b32220d0a017812080a017612030a0177', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '3', 'emit.default.values', 'true'));
select from_protobuf(X'0801120a08021206080312020804', 'krishiv.golden.types.MA', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '2'));
-- to_protobuf of a deeper struct than the proto allows is not limited by the option
select hex(to_protobuf(named_struct('name', 'a', 'child', named_struct('name', 'b')), 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1')));
-- recursion through repeated and map: the container field is dropped, the others stay
select from_protobuf(X'', 'krishiv.golden.types.Rec', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('recursive.fields.max.depth', '1'));
