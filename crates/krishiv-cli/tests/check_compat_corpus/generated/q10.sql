-- query 10: nope(1)
SELECT make_dt_interval(a, b) AS c1 FROM t10 WHERE x IN (1, 2) AND y = 'x(make_dt_interval)';
SELECT kll_merge_agg_bigint(a, b) AS c2 FROM t10 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_bigint)';
SELECT legacy_fn_26(a, b) AS c3 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT variant_explode(a, b) AS c4 FROM t10 WHERE x IN (1, 2) AND y = 'x(variant_explode)';
SELECT lpad(a, b) AS c5 FROM t10 WHERE x IN (1, 2) AND y = 'x(lpad)';
