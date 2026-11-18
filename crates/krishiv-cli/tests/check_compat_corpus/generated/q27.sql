-- query 27: nope(1)
SELECT percentile_cont(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(percentile_cont)';
SELECT tuple_sketch_agg_double(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';
SELECT xpath_string(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(xpath_string)';
