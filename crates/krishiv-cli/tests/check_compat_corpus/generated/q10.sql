-- query 10: nope(1)
SELECT make_date(a, b) AS c1 FROM t10 WHERE x IN (1, 2) AND y = 'x(make_date)';
SELECT kll_merge_agg_double(a, b) AS c2 FROM t10 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_double)';
SELECT legacy_fn_26(a, b) AS c3 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT variance(a, b) AS c4 FROM t10 WHERE x IN (1, 2) AND y = 'x(variance)';
SELECT lower(a, b) AS c5 FROM t10 WHERE x IN (1, 2) AND y = 'x(lower)';
