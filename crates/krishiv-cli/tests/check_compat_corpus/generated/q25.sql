-- query 25: nope(1)
SELECT regr_intercept(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(regr_intercept)';
SELECT equal_null(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(equal_null)';
SELECT try_aes_decrypt(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(try_aes_decrypt)';
SELECT sequence(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(sequence)';
SELECT legacy_fn_11(a, b) AS c5 FROM t25 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
