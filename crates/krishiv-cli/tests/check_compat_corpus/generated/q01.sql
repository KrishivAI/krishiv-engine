-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT map_zip_with(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(map_zip_with)';
SELECT to_protobuf(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(to_protobuf)';
SELECT overlay(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(overlay)';
SELECT schema_of_variant(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(schema_of_variant)';
SELECT kll_sketch_merge_double(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_merge_double)';
