-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT map_values(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(map_values)';
SELECT tuple_difference_double(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(tuple_difference_double)';
SELECT octet_length(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(octet_length)';
SELECT schema_of_json(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(schema_of_json)';
SELECT kll_sketch_merge_float(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_merge_float)';
