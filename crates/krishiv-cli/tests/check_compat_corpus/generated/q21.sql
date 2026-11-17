-- query 21: nope(1)
SELECT substring_index(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(substring_index)';
SELECT variant_explode(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(variant_explode)';
SELECT time_diff(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(time_diff)';
SELECT hll_union_agg(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(hll_union_agg)';
SELECT legacy_fn_11(a, b) AS c5 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
