-- query 27: nope(1)
SELECT slice(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(slice)';
SELECT legacy_fn_11(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT try_to_binary(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(try_to_binary)';
