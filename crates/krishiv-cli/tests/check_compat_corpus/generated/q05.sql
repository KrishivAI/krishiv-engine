-- query 5: nope(1)
SELECT kll_sketch_get_rank_double(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_rank_double)';
SELECT smallint(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(smallint)';
SELECT format_number(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(format_number)';
