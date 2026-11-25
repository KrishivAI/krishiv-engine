-- query 24: nope(1)
SELECT schema_of_avro(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';
SELECT current_catalog(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(current_catalog)';
SELECT approx_top_k_accumulate(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT regexp(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(regexp)';
SELECT theta_sketch_agg(a, b) AS c5 FROM t24 WHERE x IN (1, 2) AND y = 'x(theta_sketch_agg)';
SELECT schema_of_variant_agg(a, b) AS c6 FROM t24 WHERE x IN (1, 2) AND y = 'x(schema_of_variant_agg)';
