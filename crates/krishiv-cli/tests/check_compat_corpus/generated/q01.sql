-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT map_keys(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(map_keys)';
SELECT tuple_sketch_agg_double(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';
SELECT nvl(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(nvl)';
SELECT rtrim(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(rtrim)';
SELECT kll_sketch_to_string_bigint(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_bigint)';
