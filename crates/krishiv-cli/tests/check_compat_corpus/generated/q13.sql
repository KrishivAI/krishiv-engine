-- query 13: nope(1)
SELECT current_database(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(current_database)';
SELECT octet_length(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(octet_length)';
SELECT raise_error(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(raise_error)';
SELECT legacy_fn_20(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT parse_json(a, b) AS c5 FROM t13 WHERE x IN (1, 2) AND y = 'x(parse_json)';
SELECT legacy_fn_28(a, b) AS c6 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';
