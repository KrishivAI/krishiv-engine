-- query 26: nope(1)
SELECT legacy_fn_25(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_25)';
SELECT java_method(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(java_method)';
SELECT xpath_number(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
