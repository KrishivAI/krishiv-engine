-- query 0: nope(1)
SELECT upper(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(upper)';
SELECT kll_merge_agg_float(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_float)';
SELECT to_protobuf(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(to_protobuf)';
