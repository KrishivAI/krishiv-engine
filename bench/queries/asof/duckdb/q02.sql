SELECT count(*) AS n, sum(q.ask_cents) AS ask, sum(t.price_cents) AS price
FROM trades t ASOF JOIN quotes q
  ON t.symbol = q.symbol AND t.ts < q.ts
