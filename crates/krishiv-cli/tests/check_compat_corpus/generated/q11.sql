-- query 11: nope(1)
SELECT least(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(least)';
SELECT try_aes_decrypt(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(try_aes_decrypt)';
SELECT legacy_fn_20(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
