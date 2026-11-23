---
description: "Compare and aggregate embedding vectors in SQL with the vector_* functions: similarity, distance, norms, normalisation and nearest-neighbour queries."
---

# Vector search with SQL

Krishiv has the Spark 4.2 `vector_*` functions. They work on a vector stored as an `ARRAY<FLOAT>` column, which is
how an embedding is usually kept in a table, and they give the same answers as Spark 4.2.0 down to the last bit. This page shows the functions, a nearest-neighbour query and
what to expect from the results.

## The functions

| Function | Returns |
|---|---|
| `vector_inner_product(a, b)` | the dot product of two vectors |
| `vector_l2_distance(a, b)` | the Euclidean distance between two vectors |
| `vector_cosine_similarity(a, b)` | the cosine of the angle between two vectors, from -1 to 1 |
| `vector_norm(v [, degree])` | the L1 (`1.0`), L2 (`2.0`, the default) or infinity (`float('inf')`) norm |
| `vector_normalize(v [, degree])` | the vector divided by its norm, so its norm is 1 |
| `vector_sum(v)` | an aggregate: the element-wise sum of the vectors of a group |
| `vector_avg(v)` | an aggregate: the element-wise mean of the vectors of a group |

Each reference page has the exact syntax and examples: [`vector_cosine_similarity`](../reference/sql/functions/vector_cosine_similarity.md),
[`vector_l2_distance`](../reference/sql/functions/vector_l2_distance.md), [`vector_inner_product`](../reference/sql/functions/vector_inner_product.md),
[`vector_norm`](../reference/sql/functions/vector_norm.md), [`vector_normalize`](../reference/sql/functions/vector_normalize.md),
[`vector_sum`](../reference/sql/functions/vector_sum.md) and [`vector_avg`](../reference/sql/functions/vector_avg.md).

```sql
SELECT vector_cosine_similarity(array(1.0F, 2.0F, 3.0F), array(4.0F, 5.0F, 6.0F));  -- 0.9746319
SELECT vector_l2_distance(array(1.0F, 2.0F, 3.0F), array(4.0F, 5.0F, 6.0F));        -- 5.196152
SELECT vector_norm(array(3.0F, 4.0F), 2.0F);                                         -- 5.0
SELECT vector_normalize(array(3.0F, 4.0F), 2.0F);                                    -- [0.6, 0.8]
```

The arguments are `ARRAY<FLOAT>`. Write a literal with the `F` suffix as above, or `CAST(... AS FLOAT)`. The two
vectors of a pair must have the same number of elements, or the call fails.

## Find the nearest rows

Order by a distance and take the first rows. A table with an `embedding ARRAY<FLOAT>` column:

```sql
SELECT id, title,
       vector_cosine_similarity(embedding, array(0.12F, -0.40F, 0.33F /* ... one value per dimension */)) AS score
FROM docs
ORDER BY score DESC
LIMIT 10;
```

Krishiv runs an `ORDER BY ... LIMIT` as a top-k, which keeps only the best rows instead of sorting the whole table.
For the Euclidean distance, order by `vector_l2_distance(...)` ascending.

The query vector, written as `array(...)` of literals, is folded to one literal before the scan, so the vector is not
rebuilt for each row. From Python, pass the same literal:

```python
query = [0.12, -0.40, 0.33]  # your embedding
literal = "array(" + ", ".join(f"{x}F" for x in query) + ")"
df = spark.sql(
    f"SELECT id, vector_cosine_similarity(embedding, {literal}) AS score FROM docs ORDER BY score DESC LIMIT 10"
)
```

This is an exact search: every row is compared, and 0.3 has no approximate-nearest-neighbour index. For a table of a few
million vectors an exact scan is often fast enough, and it returns the true nearest rows. Narrow the scan with a
`WHERE` on another column (a language, a tenant, a date) when you can; the filter runs before the distance.

## Average and sum vectors

`vector_avg` and `vector_sum` are aggregates, so they work with `GROUP BY`. A centroid per label:

```sql
SELECT label, vector_avg(embedding) AS centroid
FROM docs
GROUP BY label;
```

All the vectors of a group must have the same number of elements.

## What to expect from the results

- **Same bits as Spark.** The sums run in blocks of 8 elements, as Spark 4.2.0 runs them, because a plain left-to-right
  sum differs in the last digits from 8 dimensions up. A query gives the same floats on Krishiv and on Spark 4.2.
- **NULL.** A NULL *element* inside a vector does not hide a dimension mismatch: two vectors of different sizes still fail.
- **A zero vector.** `vector_cosine_similarity` and `vector_normalize` give NULL when the square root of the product of the
  two squared norms is smaller than the smallest normal `FLOAT` (about 1.2e-38), instead of dividing by nearly zero. That
  product is computed in single precision, so vectors with very small elements (such as `1e-12`) can give NULL although
  each norm alone is above the limit.
- **NaN.** `vector_norm(v, float('inf'))` skips a NaN element, as Spark does.
- **Single precision.** A scalar result, and each element of a vector result (`vector_normalize`, `vector_sum`, `vector_avg`), is a
  `FLOAT`. Cast to `DOUBLE` before you compare it with a threshold computed in
  double precision.

## Spark versions

The functions exist from Spark 4.2. Each reference page says which Spark profiles have it.

## Related

- [`ORDER BY ... LIMIT`](../reference/sql/statements.md) and the other statements.
- [Migrate a PySpark job](pyspark-migration.md) for the rest of what carries over from Spark.
