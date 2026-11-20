-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT map_keys(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(map_keys)';
SELECT tuple_difference_theta_double(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_double)';
SELECT nvl2(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(nvl2)';
SELECT schema_of_csv(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(schema_of_csv)';
SELECT kll_sketch_to_string_bigint(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_bigint)';
