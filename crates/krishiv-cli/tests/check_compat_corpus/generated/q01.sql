-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT monotonically_increasing_id(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(monotonically_increasing_id)';
SELECT schema_of_avro(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';
SELECT posexplode_outer(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(posexplode_outer)';
SELECT shiftright(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(shiftright)';
SELECT kll_sketch_agg_float(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_float)';
