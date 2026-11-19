-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT lpad(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(lpad)';
SELECT xpath_number(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
SELECT xpath_float(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
SELECT theta_sketch_estimate(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(theta_sketch_estimate)';
SELECT to_date(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(to_date)';
