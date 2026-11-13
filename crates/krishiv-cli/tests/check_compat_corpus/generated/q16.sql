-- query 16: nope(1)
SELECT from_protobuf(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(from_protobuf)';
SELECT kll_sketch_to_string_double(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT legacy_fn_0(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT date_add(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(date_add)';
