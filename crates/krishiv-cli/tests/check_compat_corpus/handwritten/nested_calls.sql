SELECT upper(sentences(body)) AS s FROM docs;
SELECT coalesce(xpath_float(a, b), 1) FROM docs;
SELECT 7 DIV try_reflect('a', 'b') FROM docs;
SELECT id, count(*) FROM docs GROUP BY id QUALIFY hll_sketch_agg(id) > 1;
SELECT t.id, c FROM docs t
LATERAL VIEW java_method(t.id) x AS c;
SELECT window(ts, '5 minutes') FROM docs;
