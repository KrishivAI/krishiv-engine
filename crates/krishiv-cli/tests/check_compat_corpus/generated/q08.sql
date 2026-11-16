-- query 8: nope(1)
SELECT array_compact(a, b) AS c1 FROM t8 WHERE x IN (1, 2) AND y = 'x(array_compact)';
SELECT kll_sketch_get_rank_bigint(a, b) AS c2 FROM t8 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_rank_bigint)';
SELECT coalesce(a, b) AS c3 FROM t8 WHERE x IN (1, 2) AND y = 'x(coalesce)';
SELECT to_varchar(a, b) AS c4 FROM t8 WHERE x IN (1, 2) AND y = 'x(to_varchar)';
SELECT array_except(a, b) AS c5 FROM t8 WHERE x IN (1, 2) AND y = 'x(array_except)';
