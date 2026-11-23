-- query 7: nope(1)
SELECT legacy_fn_35(a, b) AS c1 FROM t7 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT contains(a, b) AS c2 FROM t7 WHERE x IN (1, 2) AND y = 'x(contains)';
SELECT aes_decrypt(a, b) AS c3 FROM t7 WHERE x IN (1, 2) AND y = 'x(aes_decrypt)';
SELECT legacy_fn_3(a, b) AS c4 FROM t7 WHERE x IN (1, 2) AND y = 'x(legacy_fn_3)';
SELECT theta_union_agg(a, b) AS c5 FROM t7 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
SELECT sha(a, b) AS c6 FROM t7 WHERE x IN (1, 2) AND y = 'x(sha)';
