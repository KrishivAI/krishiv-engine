SELECT count(*) AS n, sum(q.bid_cents) AS bid, sum(t.qty) AS qty
FROM trades t ASOF JOIN quotes q
  ON t.symbol = q.symbol AND t.ts >= q.ts
