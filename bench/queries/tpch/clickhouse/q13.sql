-- Source: ClickHouse/ClickHouse tests/benchmarks/tpc-h/queries/query_13.sql at a79b7c063486 (Apache-2.0).
-- Adaptations: none (verbatim).
SELECT
    c_count,
    count(*) AS custdist
FROM (
    SELECT
        c_custkey,
        count(o_orderkey) AS c_count
    FROM customer
    LEFT OUTER JOIN orders ON (c_custkey = o_custkey)
        AND (o_comment NOT LIKE '%special%requests%')
    GROUP BY c_custkey
) AS c_orders
GROUP BY c_count
ORDER BY
    custdist DESC,
    c_count DESC;
