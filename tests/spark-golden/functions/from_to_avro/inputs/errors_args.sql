-- argument and type checks (analysis errors)
select from_avro(X'02', 42, map());
select from_avro(X'02', '"int"', 42);
select from_avro(X'02', '"int"', map(1, 2));
select from_avro(X'02', '"int"', map('a', 1));
select from_avro(X'02', '"int"', map('a', 'b', 'c', 'd'));
select to_avro(1, 42);
select to_avro(1, X'00');
select from_avro();
select from_avro(X'02');
select from_avro(X'02', '"int"', map(), 1);
select to_avro();
select to_avro(1, '"int"', 3);
-- child type must be BINARY for from_avro (implicit casts)
select from_avro(1, '"int"', map());
select from_avro('abc', '"int"', map());
select from_avro(cast('abc' as binary), '"int"', map());
select from_avro(array(1), '"int"', map());
-- non-foldable schema / options
create temporary view schemas as select '"int"' as s, map('mode', 'PERMISSIVE') as o, X'02' as b;
select from_avro(b, s, map()) from schemas;
select from_avro(b, '"int"', o) from schemas;
select to_avro(1, s) from schemas;
select from_avro(b, concat('"in', 't"'), map()) from schemas;
select from_avro(b, '"int"', map('mode', concat('PERMIS', 'SIVE'))) from schemas;
select from_avro(b, '"int"', map_concat(map('mode', 'PERMISSIVE'), map())) from schemas;
-- function resolution forms
select FROM_AVRO(X'02', '"int"', map());
select from_avro(X'02', '"int"', map()) as r;
select count(from_avro(b, '"int"', map())) from schemas;
select sum(from_avro(b, '"int"', map())) from schemas;
select * from schemas where from_avro(b, '"int"', map()) = 1;
drop view schemas;
