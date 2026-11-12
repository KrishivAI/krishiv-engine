-- query 14: nope(1)
SELECT to_protobuf(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(to_protobuf)';
SELECT substr(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(substr)';
SELECT python_worker_logs(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
SELECT array_distinct(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
