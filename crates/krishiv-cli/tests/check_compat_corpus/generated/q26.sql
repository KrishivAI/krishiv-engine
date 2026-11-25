-- query 26: nope(1)
SELECT st_asbinary(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(st_asbinary)';
SELECT legacy_fn_20(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT legacy_fn_34(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_34)';
SELECT bool_or(a, b) AS c4 FROM t26 WHERE x IN (1, 2) AND y = 'x(bool_or)';
