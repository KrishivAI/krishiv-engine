-- query 3: nope(1)
SELECT tuple_intersection_double(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_double)';
SELECT xpath_float(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
SELECT schema_of_xml(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(schema_of_xml)';
SELECT legacy_fn_6(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';
SELECT tuple_sketch_summary_double(a, b) AS c5 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_double)';
SELECT legacy_fn_16(a, b) AS c6 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
