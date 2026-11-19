-- query 5: nope(1)
SELECT shuffle(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(shuffle)';
SELECT schema_of_avro(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';
SELECT xpath_long(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
SELECT kll_sketch_get_quantile_bigint(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_bigint)';
SELECT approx_top_k_accumulate(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT legacy_fn_1(a, b) AS c6 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
