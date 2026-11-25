-- query 14: nope(1)
SELECT parse_url(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(parse_url)';
SELECT legacy_fn_29(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT approx_top_k_estimate(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(approx_top_k_estimate)';
SELECT get_json_object(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(get_json_object)';
SELECT day(a, b) AS c5 FROM t14 WHERE x IN (1, 2) AND y = 'x(day)';
SELECT soundex(a, b) AS c6 FROM t14 WHERE x IN (1, 2) AND y = 'x(soundex)';
