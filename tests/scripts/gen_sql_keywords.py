#!/usr/bin/env python3
"""Generate the Spark SQL keyword classes fixture (P2-T070).

The keyword classes come from Spark's own grammar at the release tag, never from docs or memory:
`SqlBaseLexer.g4` lists every keyword token (between `--SPARK-KEYWORD-LIST-START` and `-END`) with its
text(s), and `SqlBaseParser.g4` lists `ansiNonReserved`, `strictNonReserved` and `nonReserved`. With
`spark.sql.ansi.enforceReservedKeywords`, every keyword outside `ansiNonReserved` is reserved. The
fixture records, per keyword text, its token and classes; `krishiv-sql` tests its keyword table
against it.

Usage (from repo root; needs network access to GitHub):

    python3 tests/scripts/gen_sql_keywords.py --out tests/fixtures/spark-4.2.0/sql-keywords.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

TAG = "v4.2.0"
BASE = (
    f"https://raw.githubusercontent.com/apache/spark/{TAG}/sql/api/src/main/antlr4/org/apache/spark/sql/catalyst/parser"
)


def fetch(name: str) -> str:
    with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as r:
        return r.read().decode("utf-8")


def rule(grammar: str, name: str) -> set[str]:
    body = grammar.split(f"\n{name}\n", 1)[1].split("\n    ;", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)  # marker comments such as //--ANSI-NON-RESERVED-END
    return set(re.findall(r"\b[A-Z_][A-Z_0-9]*\b", body))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/sql-keywords.json")
    args = ap.parse_args()
    lexer, parser = fetch("SqlBaseLexer.g4"), fetch("SqlBaseParser.g4")
    block = lexer.split("//--SPARK-KEYWORD-LIST-START", 1)[1].split("//--SPARK-KEYWORD-LIST-END", 1)[0]
    tokens: dict[str, list[str]] = {}
    for m in re.finditer(r"^(\w+)\s*:\s*([^;]+);", block, re.M):
        tokens[m.group(1)] = re.findall(r"'([^']+)'", m.group(2))
    ansi, strict, non = rule(parser, "ansiNonReserved"), rule(parser, "strictNonReserved"), rule(parser, "nonReserved")
    keywords = [
        {
            "text": text,
            "token": token,
            "ansi_reserved": token not in ansi,
            "strict_non_reserved": token in strict,
            "non_reserved": token in non,
            # Spark's `sql_keywords()` lists the keywords of single-text lexer rules; the extra texts of
            # a rule with alternatives (`RLIKE | REGEXP`, `TEMPORARY | TEMP`) lex as keywords but are
            # not listed.
            "listed": len(texts) == 1,
        }
        for token, texts in sorted(tokens.items())
        for text in texts
    ]
    doc = {
        "format_version": 1,
        "spark_tag": TAG,
        "generator": "tests/scripts/gen_sql_keywords.py",
        "notes": "Keyword tokens of SqlBaseLexer.g4 and the keyword rules of SqlBaseParser.g4; a keyword is "
        "reserved under spark.sql.ansi.enforceReservedKeywords when it is not in ansiNonReserved.",
        "keywords": keywords,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(keywords)} keyword texts ({len(tokens)} tokens) -> {args.out}")


if __name__ == "__main__":
    main()
