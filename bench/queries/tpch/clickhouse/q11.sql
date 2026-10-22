-- Source: ClickHouse/ClickHouse tests/benchmarks/tpc-h/queries/query_11.sql at a79b7c063486 (Apache-2.0).
-- Adaptations: none (verbatim). The HAVING fraction is the SF1 value 0.0001, the same constant the shared query uses at every scale factor.
SELECT
    ps_partkey,
    sum(ps_supplycost * ps_availqty) AS value
FROM partsupp, supplier, nation
WHERE (ps_suppkey = s_suppkey)
    AND (s_nationkey = n_nationkey)
    AND (n_name = 'GERMANY')
GROUP BY ps_partkey
HAVING sum(ps_supplycost * ps_availqty) > (
    SELECT sum(ps_supplycost * ps_availqty) * 0.0001
    FROM partsupp, supplier, nation
    WHERE (ps_suppkey = s_suppkey)
        AND (s_nationkey = n_nationkey)
        AND (n_name = 'GERMANY')
)
ORDER BY value DESC;
