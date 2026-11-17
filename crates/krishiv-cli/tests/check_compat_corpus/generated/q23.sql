-- query 23: nope(1)
SELECT to_binary(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(to_binary)';
SELECT legacy_fn_14(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT python_worker_logs(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
SELECT date_from_unix_date(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(date_from_unix_date)';
