-- query 7: nope(1)
SELECT legacy_fn_3(a, b) AS c1 FROM t7 WHERE x IN (1, 2) AND y = 'x(legacy_fn_3)';
SELECT xpath_int(a, b) AS c2 FROM t7 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT add_months(a, b) AS c3 FROM t7 WHERE x IN (1, 2) AND y = 'x(add_months)';
