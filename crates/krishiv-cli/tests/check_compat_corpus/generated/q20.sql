-- query 20: nope(1)
SELECT substring(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(substring)';
SELECT var_samp(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(var_samp)';
SELECT time_bucket(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(time_bucket)';
SELECT java_method(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(java_method)';
SELECT legacy_fn_11(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
