-- Monthly sales report. Owner: analytics (approx_top_k_estimate(x) was dropped).
WITH monthly(region, month, total) AS (
  SELECT region,
         date_trunc('MONTH', order_ts),
         CAST(sum(amount) AS DECIMAL(18, 2))
  FROM orders
  WHERE status IN ('shipped', 'paid') AND note NOT LIKE '%sentences(1)%'
  GROUP BY region, date_trunc('MONTH', order_ts)
),
ranked AS (
  SELECT region, month, total,
         rank() OVER (PARTITION BY region ORDER BY total DESC) AS rk,
         percentile_approx(total, 0.5) OVER (PARTITION BY region) AS med
  FROM monthly
)
SELECT region, month, total, rk,
       legacy_score(total, med) AS score,
       hll_sketch_agg(customer_id) AS uniques
FROM ranked JOIN customers USING (region)
WHERE rk <= 3 /* histogram_numeric(total, 5) is not used here */
ORDER BY region, rk;
