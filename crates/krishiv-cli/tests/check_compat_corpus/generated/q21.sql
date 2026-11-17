-- query 21: nope(1)
SELECT date_trunc(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(date_trunc)';
SELECT kll_sketch_merge_float(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(kll_sketch_merge_float)';
SELECT legacy_fn_25(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_25)';
SELECT quote(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(quote)';
