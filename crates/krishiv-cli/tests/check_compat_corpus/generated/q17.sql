-- query 17: nope(1)
SELECT xpath_float(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
SELECT legacy_fn_4(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(legacy_fn_4)';
SELECT kll_merge_agg_bigint(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_bigint)';
