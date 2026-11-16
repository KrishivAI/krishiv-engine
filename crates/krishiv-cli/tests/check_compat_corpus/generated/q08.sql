-- query 8: nope(1)
SELECT legacy_fn_32(a, b) AS c1 FROM t8 WHERE x IN (1, 2) AND y = 'x(legacy_fn_32)';
SELECT java_method(a, b) AS c2 FROM t8 WHERE x IN (1, 2) AND y = 'x(java_method)';
SELECT months_between(a, b) AS c3 FROM t8 WHERE x IN (1, 2) AND y = 'x(months_between)';
