-- query 4: nope(1)
SELECT kll_merge_agg_double(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_double)';
SELECT kll_sketch_get_rank_double(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_rank_double)';
SELECT smallint(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(smallint)';
SELECT format_number(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(format_number)';
