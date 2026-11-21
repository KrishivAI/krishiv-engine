-- Timestamp and Duration map to TIMESTAMP and INTERVAL DAY TO SECOND; wrappers are structs unless unwrapped
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true', 'upcast.unsigned.ints', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('convert.any.fields.to.json', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('enums.as.ints', 'true', 'retain.empty.message.types', 'true'));
-- Timestamp values (session time zone is UTC; see ansi_tz for others)
select from_protobuf(X'0a020801', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0808011080cab5ee01', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a05080110dc0b', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a05080110e707', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a00', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a1108ffffffffffffffffff0110ff93ebdc03', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0b08ffffffffffffffffff01', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0b088092b8c398feffffff01', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0d08ff82d1ffaf0710ff93ebdc03', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0d080110ffffffffffffffffff01', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0808011080a8d6b907', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0808f7b5c1deb78c02', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0808f6b5c1deb78c02', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0808011080cab5ee01', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('emit.default.values', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('emit.default.values', 'true'));
-- Duration values
select from_protobuf(X'0a020801', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0808011080cab5ee01', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a1608ffffffffffffffffff011080b6ca91feffffffff01', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0d08011080b6ca91feffffffff01', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a1108ffffffffffffffffff011080cab5ee01', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0908cdbf0510959aef3a', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a05080110dc0b', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a00', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a0808f7b5c1deb78c02', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'0a070880bcaece9709', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select typeof(from_protobuf(X'', 'krishiv.golden.types.DurOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc').dur);
select typeof(from_protobuf(X'', 'krishiv.golden.types.TsOnly', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc').ts);
-- repeated and map of Timestamp, repeated wrappers
select from_protobuf(X'720208017205080210e8077a070a016b120208038201020805820100', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
-- wrapper types with and without unwrap; value 0 inside a wrapper, wrapper present but empty, absent
-- values
select from_protobuf(X'220208052a0608ffffffff0f320b08f9ffffffffffffffff013a0b08ffffffffffffffffff0142040a0268694a040a026162520208015a050d0000c03f6209090000000000000440', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'220208052a0608ffffffff0f320b08f9ffffffffffffffff013a0b08ffffffffffffffffff0142040a0268694a040a026162520208015a050d0000c03f6209090000000000000440', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true'));
select from_protobuf(X'220208052a0608ffffffff0f320b08f9ffffffffffffffff013a0b08ffffffffffffffffff0142040a0268694a040a026162520208015a050d0000c03f6209090000000000000440', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true', 'upcast.unsigned.ints', 'true'));
select from_protobuf(X'220208052a0608ffffffff0f320b08f9ffffffffffffffff013a0b08ffffffffffffffffff0142040a0268694a040a026162520208015a050d0000c03f6209090000000000000440', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true', 'emit.default.values', 'true'));
select from_protobuf(X'220208052a0608ffffffff0f320b08f9ffffffffffffffff013a0b08ffffffffffffffffff0142040a0268694a040a026162520208015a050d0000c03f6209090000000000000440', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('emit.default.values', 'true'));
-- empty wrappers
select from_protobuf(X'22002a0032003a0042004a0052005a006200', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'22002a0032003a0042004a0052005a006200', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true'));
select from_protobuf(X'22002a0032003a0042004a0052005a006200', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true', 'upcast.unsigned.ints', 'true'));
select from_protobuf(X'22002a0032003a0042004a0052005a006200', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true', 'emit.default.values', 'true'));
select from_protobuf(X'22002a0032003a0042004a0052005a006200', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('emit.default.values', 'true'));
-- absent
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true', 'upcast.unsigned.ints', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('unwrap.primitive.wrapper.types', 'true', 'emit.default.values', 'true'));
select from_protobuf(X'', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('emit.default.values', 'true'));
-- Empty (google.protobuf.Empty) field: dropped from the schema unless retain.empty.message.types
select from_protobuf(X'6a00', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc');
select from_protobuf(X'6a00', 'krishiv.golden.types.Wkt', '/golden/from_to_protobuf/fixtures/krishiv_golden_types.desc', map('retain.empty.message.types', 'true'));
