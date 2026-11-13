-- conditional evaluation: unselected rows must not raise
create temporary view v as select * from values (X'02', 1), (X'ff', 2) as t(b, id);
select id, case when id = 1 then from_avro(b, '"int"', map()) else -1 end from v order by id;
select id, if(id = 1, from_avro(b, '"int"', map()), -1) from v order by id;
select id, coalesce(if(id = 2, cast(null as int), from_avro(b, '"int"', map())), -1) from v order by id;
select id, id = 2 or from_avro(b, '"int"', map()) = 1 from v order by id;
select id, id = 1 and from_avro(b, '"int"', map()) = 1 from v order by id;
select id, nvl(null, from_avro(b, '"int"', map())) from v where id = 1;
select id, from_avro(b, '"int"', map()) from v order by id;
select id, from_avro(b, '"int"', map('mode', 'PERMISSIVE')) from v order by id;
select id from v where from_avro(b, '"int"', map('mode', 'PERMISSIVE')) is not null;
select id, case when id = 1 then to_avro(id, '"string"') else to_avro(id) end from v order by id;
select id, hex(case when id = 1 then to_avro(named_struct('a', id), '{"type":"record","name":"r","fields":[{"name":"a","type":"int"}]}') else cast(null as binary) end) from v order by id;
drop view v;
