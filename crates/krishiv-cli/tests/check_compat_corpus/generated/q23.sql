-- query 23: nope(1)
SELECT reflect(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT legacy_fn_13(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT sha(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(sha)';
SELECT signum(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(signum)';
SELECT to_protobuf(a, b) AS c5 FROM t23 WHERE x IN (1, 2) AND y = 'x(to_protobuf)';
