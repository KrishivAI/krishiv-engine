-- query 14: nope(1)
SELECT random(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(random)';
SELECT legacy_fn_29(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT java_method(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(java_method)';
SELECT get_json_object(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(get_json_object)';
SELECT day(a, b) AS c5 FROM t14 WHERE x IN (1, 2) AND y = 'x(day)';
SELECT struct(a, b) AS c6 FROM t14 WHERE x IN (1, 2) AND y = 'x(struct)';
