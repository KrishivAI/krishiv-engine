-- query 5: nope(1)
SELECT vector_cosine_similarity(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(vector_cosine_similarity)';
SELECT schema_of_avro(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';
SELECT try_to_binary(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(try_to_binary)';
SELECT legacy_fn_13(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT approx_top_k(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(approx_top_k)';
SELECT legacy_fn_1(a, b) AS c6 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
