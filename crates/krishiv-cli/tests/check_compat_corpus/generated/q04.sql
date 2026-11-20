-- query 4: nope(1)
SELECT legacy_fn_6(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';
SELECT tuple_difference_theta_integer(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_integer)';
SELECT legacy_fn_16(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT input_file_name(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(input_file_name)';
SELECT legacy_fn_35(a, b) AS c5 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT make_interval(a, b) AS c6 FROM t4 WHERE x IN (1, 2) AND y = 'x(make_interval)';
