-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT kurtosis(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(kurtosis)';
SELECT xpath_number(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
SELECT try_reflect(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
SELECT tuple_sketch_summary_double(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_double)';
SELECT monotonically_increasing_id(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(monotonically_increasing_id)';
