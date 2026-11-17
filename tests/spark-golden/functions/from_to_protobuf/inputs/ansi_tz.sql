set spark.sql.ansi.enabled=false;
select hex(to_protobuf(named_struct('i32', 1, 'str', 'a'), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('i32', 1L), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('i32', '1'), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'ff', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc', map('mode', 'PERMISSIVE'));
select from_protobuf(1, 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf('abc', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf(X'0801', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc').i32 + 1;
set spark.sql.ansi.enabled=true;
select hex(to_protobuf(named_struct('i32', 1, 'str', 'a'), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select from_protobuf(1, 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
select from_protobuf('abc', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc');
-- timestamps and durations have no native mapping in this descriptor; session time zone is irrelevant to the wire bytes
set spark.sql.session.timeZone=America/Los_Angeles;
select hex(to_protobuf(named_struct('i64', unix_micros(timestamp'2020-03-08 02:30:00')), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('i64', unix_micros(timestamp'2020-11-01 01:30:00')), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('i64', unix_micros(timestamp'2020-03-08 03:00:00-07:00')), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('i64', timestamp'2020-03-08 03:00:00'), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('i32', date'2020-01-02'), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
set spark.sql.session.timeZone=UTC;
-- decimals
select hex(to_protobuf(named_struct('i64', cast(5 as decimal(10,0))), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('dbl', cast(1.5 as decimal(5,2))), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
-- Unicode and whitespace in string fields
select hex(to_protobuf(named_struct('str', ' \t\n\u00a0\u3000'), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select hex(to_protobuf(named_struct('str', 'é😀'), 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc'));
select from_protobuf(X'7206c3a9f09f9880', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc').str;
select from_protobuf(X'7203efbfbd', 'krishiv.golden.Scalars', '/golden/from_to_protobuf/fixtures/krishiv_golden.desc').str;
