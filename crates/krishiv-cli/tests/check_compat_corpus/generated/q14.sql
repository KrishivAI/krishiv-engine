-- query 14: nope(1)
SELECT randn(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(randn)';
SELECT legacy_fn_29(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT kll_merge_agg_bigint(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_bigint)';
SELECT get(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(get)';
SELECT datepart(a, b) AS c5 FROM t14 WHERE x IN (1, 2) AND y = 'x(datepart)';
SELECT string_agg(a, b) AS c6 FROM t14 WHERE x IN (1, 2) AND y = 'x(string_agg)';
