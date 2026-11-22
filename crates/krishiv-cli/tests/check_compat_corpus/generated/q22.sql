-- query 22: nope(1)
SELECT date_trunc(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(date_trunc)';
SELECT kll_sketch_to_string_double(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT legacy_fn_25(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(legacy_fn_25)';
SELECT power(a, b) AS c4 FROM t22 WHERE x IN (1, 2) AND y = 'x(power)';
