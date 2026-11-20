-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT ltrim(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(ltrim)';
SELECT xpath_int(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT xpath_boolean(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath_boolean)';
SELECT theta_sketch_agg(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(theta_sketch_agg)';
SELECT to_json(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(to_json)';
