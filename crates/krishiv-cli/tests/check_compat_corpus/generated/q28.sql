-- query 28: nope(1)
SELECT xpath_int(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT legacy_fn_8(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(legacy_fn_8)';
SELECT last_value(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(last_value)';
SELECT legacy_fn_23(a, b) AS c4 FROM t28 WHERE x IN (1, 2) AND y = 'x(legacy_fn_23)';
SELECT python_worker_logs(a, b) AS c5 FROM t28 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
