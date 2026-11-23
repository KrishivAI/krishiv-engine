-- query 24: nope(1)
SELECT tuple_union_double(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_union_double)';
SELECT approx_top_k_accumulate(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT time_from_seconds(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(time_from_seconds)';
