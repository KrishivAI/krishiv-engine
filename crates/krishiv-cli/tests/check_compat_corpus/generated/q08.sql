-- query 8: nope(1)
SELECT legacy_fn_32(a, b) AS c1 FROM t8 WHERE x IN (1, 2) AND y = 'x(legacy_fn_32)';
SELECT kll_sketch_get_rank_bigint(a, b) AS c2 FROM t8 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_rank_bigint)';
SELECT map_zip_with(a, b) AS c3 FROM t8 WHERE x IN (1, 2) AND y = 'x(map_zip_with)';
