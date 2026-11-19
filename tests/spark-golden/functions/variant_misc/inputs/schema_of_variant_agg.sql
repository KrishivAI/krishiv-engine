-- schema_of_variant_agg (4.0+): merged schema string; NULL rows skipped; never NULL
create temporary view v as select * from values (1, '1'), (2, '2'), (3, '3') as t(id, j);
select schema_of_variant_agg(parse_json(j)) from v;
select typeof(schema_of_variant_agg(parse_json(j))) from v;
drop view v;
create temporary view v as select * from values ('{"a":1,"b":true}'), ('{"a":2,"c":1.23}'), ('{"a":3.5}') as t(j);
select schema_of_variant_agg(parse_json(j)) from v;
select schema_of_variant(parse_json(j)) from v order by j;
drop view v;
-- mixed types merge (JsonInferSchema.compatibleType)
select schema_of_variant_agg(parse_json(j)) from values ('1'), ('"x"') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('1'), ('1.5') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('1'), ('true') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('[1]'), ('{"a":1}') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('[1,2]'), ('[1.5]'), ('[]') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('{"a":{"b":1}}'), ('{"a":{"c":"x"}}') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('1'), (null), ('2') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('null'), ('1') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('null'), ('null') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('1e400'), ('9223372036854775808'), ('0.1') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('12345678901234567890123456789012345678'), ('1') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('{"a b":1,"c.d":2,"`q`":3}') as t(j);
select schema_of_variant_agg(cast(d as variant)) from values (date'2020-01-01'), (date'2021-01-01') as t(d);
select schema_of_variant_agg(cast(d as variant)) from values (timestamp'2020-01-01 00:00:00'), (date'2021-01-01') as t(d);
select schema_of_variant_agg(cast(d as variant)) from values (double('NaN')), (double('-0.0')), (double('Infinity')) as t(d);
-- NULL / empty input
select schema_of_variant_agg(cast(null as variant));
select schema_of_variant_agg(parse_json(j)) from values (cast(null as string)) as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('1') as t(j) where j = 'zzz';
select schema_of_variant_agg(parse_json(j)) is null from values ('1') as t(j) where j = 'zzz';
-- grouping, distinct, filter, window
create temporary view g as select * from values (1,'{"a":1}'), (1,'{"b":true}'), (2,'[1,2]'), (2,null), (3,null) as t(k, j);
select k, schema_of_variant_agg(parse_json(j)) from g group by k order by k;
select schema_of_variant_agg(distinct parse_json(j)) from g;
select schema_of_variant_agg(parse_json(j)) filter (where k = 1) from g;
select schema_of_variant_agg(parse_json(j)) filter (where k = 99) from g;
select k, schema_of_variant_agg(parse_json(j)) over (partition by k) from g order by k, j;
select k, schema_of_variant_agg(parse_json(j)) over (order by k) from g order by k, j;
select schema_of_variant_agg(parse_json(j)) from g group by ();
select count(*), schema_of_variant_agg(parse_json(j)), count(j) from g;
select schema_of_variant_agg(parse_json(j)) from g group by k having k = 1;
-- non-variant / bad arity
select schema_of_variant_agg(1);
select schema_of_variant_agg('{"a":1}');
select schema_of_variant_agg();
select schema_of_variant_agg(parse_json('1'), parse_json('2'));
-- errors in input
select schema_of_variant_agg(parse_json(j)) from values ('{bad') as t(j);
select schema_of_variant_agg(try_parse_json(j)) from values ('{bad'), ('1') as t(j);
select schema_of_variant_agg(cast(X'ff00' as variant));
-- conditional: unselected rows not evaluated
select schema_of_variant_agg(case when k = 1 then parse_json(j) else parse_json('{bad') end) from g where k = 1;
drop view g;
-- ANSI off
set spark.sql.ansi.enabled=false;
select schema_of_variant_agg(parse_json(j)) from values ('1'), ('"x"') as t(j);
select schema_of_variant_agg(parse_json(j)) from values ('{bad') as t(j);
select schema_of_variant_agg(1);
set spark.sql.ansi.enabled=true;
-- schema_of_variant edge cases
select schema_of_variant(parse_json('null')), schema_of_variant(cast(null as variant)), schema_of_variant(parse_json(null));
select schema_of_variant(parse_json('[{"b":true,"a":0}]'));
select schema_of_variant(parse_json('[1,"a",true,null]'));
select schema_of_variant(parse_json('{"a b":1,"a.b":2,"a":{"x y":[]}}'));
select schema_of_variant(parse_json('1e400')), schema_of_variant(parse_json('12345678901234567890')), schema_of_variant(parse_json('1.0'));
select schema_of_variant(cast(X'ff00' as variant));
select schema_of_variant(1);
-- function lookup
describe function schema_of_variant_agg;
describe function extended schema_of_variant_agg;
