SELECT * FROM VALUES (1, 'a'), (2, 'b') AS t(id, label);

SELECT t.id, e.col
FROM (VALUES (1), (2)) t(id)
LATERAL VIEW explode(array(1, 2, 3)) e AS col;

SELECT * FROM range(10) r(n) WHERE n > 2;

SELECT a IS DISTINCT FROM sentences(b) AS diff,
       extract(year FROM try_reflect('java.lang.System', 'currentTimeMillis')) AS y
FROM docs;
