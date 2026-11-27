---
description: "String collations in Krishiv: what ICU adds, which builds include it, and what it changes for sorting, comparison, upper and lower."
---

# Collations and ICU

A collation decides how strings compare and sort: whether `a` and `A` are equal, whether `é` sorts with `e`, and how
letters outside ASCII are ordered. Krishiv follows Spark's collation names and the `COLLATE` function. See the
[`collate` function reference](../reference/sql/functions/collate.md) for the syntax.

## ICU support

Krishiv uses the ICU Unicode library for collation sort keys and case mapping when the `collations-icu` feature is
on. Which builds have it, from 0.3.0 on:

| Build | ICU collations |
|---|---|
| The `krishiv` crate with its default features | yes |
| The `krishiv` command-line binary: the release downloads, `cargo install krishiv-cli` and the Docker image | yes |
| The Python wheels (`pip install krishiv`, `pip install krishiv-compat`) | no |

`UTF8_BINARY`, `UTF8_LCASE` and their `_RTRIM` forms do not need ICU. In a build without it, a comparison, a sort
or grouping key, or `upper` / `lower` under an ICU collation (`UNICODE`, `UNICODE_CI`, a language such as `de`)
raises `KRISHIV_UNSUPPORTED_FEATURE`:

```sql
SELECT 'a' COLLATE UNICODE_CI = 'A';
-- a wheel: KRISHIV_UNSUPPORTED_FEATURE
-- the command-line binary: true
```

The message names the operation, not ICU. From Python, send the query to a `krishiv server` started from the binary
or the Docker image over Spark Connect to use an ICU collation.

With ICU on:

- Sort keys follow the Unicode collation rules for each strength: case-sensitive or not, and accent-sensitive or not.
- `upper` and `lower` follow the collation's case rules.
- `initcap` is refused under an ICU collation.

## Things to know

- A collated operand of `decode` is refused.
- A cast to `STRING` ends the collation: the collation is not carried over to the result.
