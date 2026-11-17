-- query 27: nope(1)
SELECT sin(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(sin)';
SELECT legacy_fn_11(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT try_multiply(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(try_multiply)';
