-- query 19: nope(1)
SELECT tuple_intersection_integer(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_integer)';
SELECT tuple_intersection_agg_integer(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_agg_integer)';
SELECT xpath_float(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
SELECT legacy_fn_21(a, b) AS c4 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_21)';
SELECT legacy_fn_26(a, b) AS c5 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
