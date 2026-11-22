-- query 18: nope(1)
SELECT xpath_short(a, b) AS c1 FROM t18 WHERE x IN (1, 2) AND y = 'x(xpath_short)';
SELECT explode(a, b) AS c2 FROM t18 WHERE x IN (1, 2) AND y = 'x(explode)';
SELECT try_multiply(a, b) AS c3 FROM t18 WHERE x IN (1, 2) AND y = 'x(try_multiply)';
