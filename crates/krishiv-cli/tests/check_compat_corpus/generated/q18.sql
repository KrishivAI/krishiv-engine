-- query 18: nope(1)
SELECT xpath_number(a, b) AS c1 FROM t18 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
SELECT explode(a, b) AS c2 FROM t18 WHERE x IN (1, 2) AND y = 'x(explode)';
SELECT try_parse_json(a, b) AS c3 FROM t18 WHERE x IN (1, 2) AND y = 'x(try_parse_json)';
