-- query 23: nope(1)
SELECT schema_of_avro(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';
SELECT legacy_fn_13(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT session_window(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(session_window)';
SELECT sign(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(sign)';
SELECT to_number(a, b) AS c5 FROM t23 WHERE x IN (1, 2) AND y = 'x(to_number)';
