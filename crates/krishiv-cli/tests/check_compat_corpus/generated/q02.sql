-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT map_concat(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(map_concat)';
SELECT xpath_number(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
SELECT kll_sketch_agg_float(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_float)';
SELECT to_avro(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(to_avro)';
SELECT radians(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(radians)';
