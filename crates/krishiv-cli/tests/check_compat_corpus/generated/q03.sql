-- query 3: nope(1)
SELECT tuple_intersection_agg_integer(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_agg_integer)';
SELECT xpath_double(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(xpath_double)';
SELECT sec(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(sec)';
SELECT legacy_fn_6(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';
SELECT tuple_sketch_estimate_integer(a, b) AS c5 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_estimate_integer)';
SELECT legacy_fn_16(a, b) AS c6 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
