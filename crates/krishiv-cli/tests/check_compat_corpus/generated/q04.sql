-- query 4: nope(1)
SELECT kll_sketch_merge_float(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(kll_sketch_merge_float)';
SELECT tuple_union_theta_integer(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_integer)';
SELECT url_decode(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(url_decode)';
SELECT from_avro(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(from_avro)';
