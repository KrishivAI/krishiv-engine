---
description: "String collations in Krishiv: what ICU adds, which builds include it, and what it changes for sorting, comparison, upper and lower."
---

# Collations and ICU

A collation decides how strings compare and sort: whether `a` and `A` are equal, whether `é` sorts with `e`, and how
letters outside ASCII are ordered. Krishiv follows Spark's collation names and the `COLLATE` function. See the
[`collate` function reference](../reference/sql/functions/collate.md) for the syntax.

## ICU support

> ICU support is not in 0.2.0. It arrives in the release after it.

Krishiv uses the ICU Unicode library for collation sort keys and case mapping when the `collations-icu` feature is
on. The default builds of the `krishiv` crate and of the `krishiv` command-line binary have it on.

The Python wheel is built without it, so an ICU collation (`unicode`, `unicode_ci`, `de`, ...) works there only
where no key is needed. A sort, a comparison, a grouping key or `upper`/`lower` over one fails with
`KRISHIV_UNSUPPORTED_FEATURE`, and the message names the missing `collations-icu` feature. `utf8_binary` and
`utf8_lcase` work in every build.

With ICU on:

- Sort keys follow the Unicode collation rules for each strength: case-sensitive or not, and accent-sensitive or not.
- `upper` and `lower` follow the collation's case rules.
- `initcap` is refused under an ICU collation.

## Things to know

- A collated operand of `decode` is refused.
- A cast to `STRING` ends the collation: the collation is not carried over to the result.
