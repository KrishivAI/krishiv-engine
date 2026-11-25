-- query 4: nope(1)
SELECT kll_merge_agg_float(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_float)';
SELECT kll_sketch_get_rank_float(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_rank_float)';
SELECT slice(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(slice)';
SELECT forall(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(forall)';
