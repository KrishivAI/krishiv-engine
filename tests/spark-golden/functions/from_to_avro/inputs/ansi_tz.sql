-- ANSI off: the same decoding/encoding entry points
set spark.sql.ansi.enabled=false;
select hex(to_avro(1));
select hex(to_avro('x', '"int"'));
select hex(to_avro(1, '"string"'));
select hex(to_avro(cast(1.23 as decimal(10,2)), '{"type":"bytes","logicalType":"decimal","precision":5,"scale":1}'));
select from_avro(X'ff', '"int"', map());
select from_avro(X'ff', '"int"', map('mode', 'PERMISSIVE'));
select from_avro(1, '"int"', map());
select from_avro('abc', '"int"', map());
select from_avro(X'02', '"long"', map()) + 1;
select from_avro(to_avro(cast(300 as int)), '"int"', map());
create temporary view v as select * from values (X'02', 1), (X'ff', 2) as t(b, id);
select id, case when id = 1 then from_avro(b, '"int"', map()) else -1 end from v order by id;
select id, from_avro(b, '"int"', map('mode', 'PERMISSIVE')) from v order by id;
drop view v;
set spark.sql.ansi.enabled=true;
-- ANSI on (explicit)
select hex(to_avro(1));
select from_avro(1, '"int"', map());
select from_avro('abc', '"int"', map());
select from_avro(X'ff', '"int"', map());
-- session time zone only affects rendering of the decoded instant, never the bytes
set spark.sql.session.timeZone=America/Los_Angeles;
select hex(to_avro(timestamp'2020-03-08 02:30:00'));
select hex(to_avro(timestamp'2020-03-08 01:59:59'));
select hex(to_avro(timestamp'2020-11-01 01:30:00'));
select hex(to_avro(timestamp_ntz'2020-03-08 02:30:00'));
select hex(to_avro(timestamp'2020-03-08 03:00:00-07:00'));
select from_avro(to_avro(timestamp'2020-11-01 01:30:00'), '{"type":"long","logicalType":"timestamp-micros"}', map());
select from_avro(to_avro(timestamp'2020-03-08 03:00:00'), '{"type":"long","logicalType":"timestamp-micros"}', map());
select from_avro(to_avro(timestamp_ntz'2020-03-08 02:30:00'), '{"type":"long","logicalType":"local-timestamp-micros"}', map());
select from_avro(X'80d0b4d6c0be0a', '{"type":"long","logicalType":"timestamp-micros"}', map());
select from_avro(X'80d0b4d6c0be0a', '{"type":"long","logicalType":"timestamp-millis"}', map());
set spark.sql.session.timeZone=UTC;
select from_avro(X'80d0b4d6c0be0a', '{"type":"long","logicalType":"timestamp-micros"}', map());
select from_avro(X'80d0b4d6c0be0a', '{"type":"long","logicalType":"local-timestamp-micros"}', map());
-- unicode and whitespace in schema JSON
select from_avro(X'06616263', ' \t\n"string"  ', map());
select from_avro(X'06616263', ' "string"', map());
select hex(to_avro('é'));
select hex(to_avro('　'));
select from_avro(X'04c3a9', '"string"', map());
select from_avro(X'02ff', '"string"', map());
select hex(cast(from_avro(X'02ff', '"string"', map()) as binary));
