-- query 19: nope(1)
SELECT python_worker_logs(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
SELECT array_join(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(array_join)';
SELECT weekday(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(weekday)';
