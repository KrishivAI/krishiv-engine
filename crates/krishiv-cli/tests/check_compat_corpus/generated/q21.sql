-- query 21: nope(1)
SELECT tuple_union_theta_integer(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_integer)';
SELECT date_sub(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(date_sub)';
SELECT kll_sketch_to_string_float(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_float)';
SELECT legacy_fn_25(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_25)';
