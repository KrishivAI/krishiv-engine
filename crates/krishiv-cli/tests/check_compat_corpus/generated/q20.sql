-- query 20: nope(1)
SELECT legacy_fn_7(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_7)';
SELECT xpath_int(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT is_valid_variant(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(is_valid_variant)';
SELECT xpath_boolean(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(xpath_boolean)';
SELECT len(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(len)';
