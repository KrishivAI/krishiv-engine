-- query 22: nope(1)
SELECT percentile(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(percentile)';
SELECT kll_merge_agg_double(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_double)';
SELECT elt(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(elt)';
SELECT isnan(a, b) AS c4 FROM t22 WHERE x IN (1, 2) AND y = 'x(isnan)';
SELECT from_unixtime(a, b) AS c5 FROM t22 WHERE x IN (1, 2) AND y = 'x(from_unixtime)';
SELECT python_worker_logs(a, b) AS c6 FROM t22 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
