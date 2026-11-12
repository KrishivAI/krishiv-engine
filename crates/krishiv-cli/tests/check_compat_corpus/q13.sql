-- query 13: nope(1)
SELECT current_time(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(current_time)';
SELECT pmod(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(pmod)';
SELECT regexp_extract(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(regexp_extract)';
SELECT legacy_fn_20(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT posexplode_outer(a, b) AS c5 FROM t13 WHERE x IN (1, 2) AND y = 'x(posexplode_outer)';
SELECT legacy_fn_28(a, b) AS c6 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';
