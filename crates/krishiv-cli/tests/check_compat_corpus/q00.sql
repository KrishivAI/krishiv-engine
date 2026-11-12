-- query 0: nope(1)
SELECT soundex(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(soundex)';
SELECT regexp_count(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(regexp_count)';
SELECT legacy_fn_25(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(legacy_fn_25)';
