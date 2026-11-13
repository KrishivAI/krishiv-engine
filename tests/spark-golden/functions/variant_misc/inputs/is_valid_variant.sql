-- is_valid_variant (4.2+): true/false/NULL; ANSI default (on)
select is_valid_variant(parse_json('null'));
select is_valid_variant(parse_json('[{"b":true,"a":0}]'));
select is_valid_variant(parse_json('{"a":1,"b":[1,2,{"c":"x"}]}'));
select is_valid_variant(parse_json('1.5e300'));
select is_valid_variant(parse_json('"é😀 \u0000"'));
select is_valid_variant(parse_json('[]')), is_valid_variant(parse_json('{}'));
select typeof(is_valid_variant(parse_json('1')));
-- SQL NULL vs variant null
select is_valid_variant(cast(null as variant));
select is_valid_variant(null);
select is_valid_variant(parse_json(null));
select is_valid_variant(cast(parse_json('null') as variant)), parse_json('null') is null, is_variant_null(parse_json('null'));
-- other variant constructors
select is_valid_variant(to_variant_object(named_struct('a', 1, 'b', 'x')));
select is_valid_variant(cast(1 as variant)), is_valid_variant(cast('x' as variant)), is_valid_variant(cast(array(1,2) as variant));
select is_valid_variant(cast(cast(1.50 as decimal(5,2)) as variant)), is_valid_variant(cast(date'2020-01-01' as variant)), is_valid_variant(cast(timestamp'2020-01-01 00:00:00' as variant));
select is_valid_variant(cast(double('NaN') as variant)), is_valid_variant(cast(double('-0.0') as variant)), is_valid_variant(cast(double('Infinity') as variant));
select is_valid_variant(cast(X'ff00' as variant));
select is_valid_variant(try_parse_json('{bad'));
select is_valid_variant(try_parse_json('{"a":1}'));
select is_valid_variant(parse_json('{bad'));
-- non-variant inputs
select is_valid_variant(1);
select is_valid_variant('{"a":1}');
select is_valid_variant(X'00');
select is_valid_variant();
select is_valid_variant(parse_json('1'), parse_json('2'));
-- table input, empty input, conditional evaluation
create temporary view v as select * from values (1, '{"a":1}'), (2, null), (3, 'null'), (4, '[1,2]') as t(id, j);
select id, is_valid_variant(parse_json(j)) from v order by id;
select id, case when id = 1 then is_valid_variant(parse_json(j)) else false end from v order by id;
select id, if(j is null, null, is_valid_variant(parse_json(j))) from v order by id;
select id from v where is_valid_variant(parse_json(j)) order by id;
select count(*), count(is_valid_variant(parse_json(j))), sum(cast(is_valid_variant(parse_json(j)) as int)) from v;
select count(*) from v where false and is_valid_variant(parse_json(j));
select is_valid_variant(parse_json(j)) from v where id < 0;
select is_valid_variant(parse_json(j)), count(*) from v group by 1 order by 1;
select coalesce(is_valid_variant(parse_json(j)), false) from v order by id;
drop view v;
-- ANSI off
set spark.sql.ansi.enabled=false;
select is_valid_variant(parse_json('{"a":1}')), is_valid_variant(cast(null as variant));
select is_valid_variant(parse_json('{bad'));
select is_valid_variant(cast(X'ff00' as variant));
select is_valid_variant(1);
set spark.sql.ansi.enabled=true;
-- session time zone does not matter
set spark.sql.session.timeZone=America/Los_Angeles;
select is_valid_variant(parse_json('"2020-03-08 02:30:00"')), is_valid_variant(cast(timestamp'2020-03-08 02:30:00' as variant));
-- function lookup
describe function is_valid_variant;
describe function extended is_valid_variant;
