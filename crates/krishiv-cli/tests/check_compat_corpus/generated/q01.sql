-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT map_from_entries(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(map_from_entries)';
SELECT tuple_sketch_agg_integer(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_integer)';
SELECT nullifzero(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(nullifzero)';
SELECT rpad(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(rpad)';
SELECT kll_sketch_to_string_double(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
