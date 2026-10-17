#!/usr/bin/env python3
"""Generate the Spark function-matrix golden fixtures (P2-T106a, ADR 0029 coverage strategy).

Every expected value is produced by executing a REAL Spark 4.2 (the `krishiv-bench` container); nothing
is transcribed from docs or memory (review I12). Inputs are derived mechanically:

1. The function list comes from Spark itself: `FunctionRegistry` + `TableFunctionRegistry` +
   `FunctionRegistry.builtinOperators` dumped by `fnmatrix/dump_registry.scala` inside the container
   (`spark-shell`), cross-checked against `SHOW FUNCTIONS` over Spark Connect (counts printed).
2. Seed statements are each function's `ExpressionInfo` examples (`> SELECT ...;`), plus the few seeds in
   `SUPPLEMENTAL_SEEDS` for registry entries without examples (cast-named functions and operators).
3. Each seed's call arguments whose type resolves on its own are substituted with typed edge values
   (null, NaN, -0.0, +/-Infinity, integral min/max, decimal(38), empty/invalid-UTF-8/unicode strings,
   binary, calendar edges, DST gap/overlap timestamps) and with one probe value per input type.
   Single-column `FROM VALUES ... AS tab(col)` aggregate seeds also get edge-valued input rows and empty input.
4. Every case runs with ANSI on and ANSI off.

Each result cell carries Spark's text rendering (`CAST(col AS STRING)`; binary as `hex`) and a typed
value collected over Arrow in the same query (exact float bits, decimal unscaled+scale, epoch micros,
days, hex bytes, nested structure with NULL distinct from "null"). Every conf that affects rendering is
pinned in `base_conf`. Errors record condition, SQLSTATE, message parameters, message (stack trace, plan
and client call-site context stripped), the phase Spark's exception type implies, and
`stability: unstable` for `_LEGACY_ERROR_TEMP_*` / unclassified / SQLSTATE-less errors.

Cases whose analyzed plan Spark reports as non-deterministic or clock/environment-dependent
(`fnmatrix/classify_cases.scala`) record schema only. Output is deterministic (no timestamps, stable
ordering, content-hashed ids, no randomness); the run fails rather than commit generator-side failures
or local paths/hostnames.

Usage (from repo root; take the bench lock, start Spark, stop it afterwards; see bench/README.md):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    flock "$KRISHIV_DATA/.bench.lock" nice -n 19 sh -c \
      'cd bench && uv run python ../tests/scripts/gen_function_matrix.py \
         --out ../tests/fixtures/spark-4.2.0/function-matrix'
    docker compose -f bench/baselines/docker-compose.yml stop spark
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from decimal import Decimal

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_function_matrix.py"
ANSI = "spark.sql.ansi.enabled"
TZ = "spark.sql.session.timeZone"
BASE_TZ = "UTC"
DST_TZ = "America/Los_Angeles"
MODES = (("ansi_on", "true"), ("ansi_off", "false"))
MAX_ROWS = 1000
QUERY_TIMEOUT_S = 60

# Seeds for registry entries that carry no examples (inputs only; expectations still come from Spark).
SUPPLEMENTAL_SEEDS = {
    "bigint": ["SELECT bigint('42')", "SELECT bigint(4.5D)"],
    "binary": ["SELECT binary('abc')"],
    "boolean": ["SELECT boolean('true')", "SELECT boolean(0)"],
    "date": ["SELECT date('2024-02-29')"],
    "decimal": ["SELECT decimal('12.345')"],
    "double": ["SELECT double('1.5')"],
    "float": ["SELECT float('1.5')"],
    "int": ["SELECT int('42')", "SELECT int(4.5D)"],
    "smallint": ["SELECT smallint('42')"],
    "string": ["SELECT string(42)"],
    "time": ["SELECT time('12:34:56')"],
    "timestamp": ["SELECT timestamp('2024-02-29 12:34:56')"],
    "tinyint": ["SELECT tinyint('42')"],
    "<>": ["SELECT 1 <> 2"],
    "!=": ["SELECT 1 != 2"],
    "||": ["SELECT 'a' || 'b'"],
    "case": ["SELECT CASE WHEN 1 > 0 THEN 'a' ELSE 'b' END"],
}

# Determinism is decided by Spark per case (fnmatrix/classify_cases.scala). Seeded rand/randn/uuid/
# shuffle are excluded too: Spark marks them non-deterministic even with a seed. The only hand-kept
# addition: Spark marks these deterministic, but with no `iv` argument they draw a random IV.
VALUE_UNSTABLE = {"aes_encrypt": "random IV when `iv` is omitted",
                  "try_aes_encrypt": "random IV when `iv` is omitted"}

# Conf keys that change how values render or parse; their effective values are read from Spark and
# pinned in base_conf (unset ones are listed in the manifest). ANSI is per mode.
RENDER_CONF_KEYS = [
    "spark.sql.ansi.doubleQuotedIdentifiers", "spark.sql.binaryOutputStyle",
    "spark.sql.decimalOperations.allowPrecisionLoss", "spark.sql.function.concatBinaryAsString",
    "spark.sql.legacy.castComplexTypesToString.enabled", "spark.sql.legacy.interval.enabled",
    "spark.sql.legacy.timeParserPolicy", "spark.sql.parser.escapedStringLiterals", "spark.sql.timestampType",
]

# Integer edges (min/max) are not substituted into size/count arguments of these functions: Spark
# would try to materialise multi-GB values or rows, which tests the JVM heap, not semantics.
SIZE_ARG_FUNCTIONS = {
    "array_repeat", "format_number", "lpad", "rpad", "randstr", "range", "repeat", "sequence", "space",
    "array_insert", "stack", "explode", "inline", "posexplode", "bloom_filter_agg", "count_min_sketch",
    "approx_percentile", "percentile_approx", "approx_top_k", "approx_top_k_accumulate",
}

# No edge/probe derivation at all (examples only): substitutions make Spark's *analyzer* (which a
# Connect interrupt cannot stop) allocate without bound, e.g. window(ts, '10 minutes', '1') builds
# 600M candidate windows and exhausts the driver heap.
DERIVE_SKIP = {"window"}

TYPE_PROBES = [
    ("boolean", "true"),
    ("tinyint", "CAST(1 AS TINYINT)"),
    ("smallint", "CAST(1 AS SMALLINT)"),
    ("int", "1"),
    ("bigint", "1L"),
    ("float", "CAST(1.5 AS FLOAT)"),
    ("double", "1.5D"),
    ("decimal_10_2", "CAST(1.5 AS DECIMAL(10,2))"),
    ("decimal_38_18", "CAST(1.5 AS DECIMAL(38,18))"),
    ("string_numeric", "'1'"),
    ("string", "'abc'"),
    ("binary", "X'01'"),
    ("date", "DATE'2024-02-29'"),
    ("timestamp", "TIMESTAMP'2024-02-29 12:34:56.123456'"),
    ("timestamp_ntz", "TIMESTAMP_NTZ'2024-02-29 12:34:56.123456'"),
    ("interval_ym", "INTERVAL '1-2' YEAR TO MONTH"),
    ("interval_dt", "INTERVAL '1 02:03:04' DAY TO SECOND"),
    ("array_int", "array(1, 2)"),
    ("map_string_int", "map('a', 1)"),
    ("struct", "named_struct('a', 1, 'b', 'x')"),
    ("void", "NULL"),
]

EDGE_CATALOG = {
    "null": "typed NULL of the argument's type",
    "nan": "floating NaN",
    "neg_zero": "floating -0.0",
    "pos_inf": "floating +Infinity",
    "neg_inf": "floating -Infinity",
    "int_min": "integral type minimum",
    "int_max": "integral type maximum",
    "decimal38_max_frac": "DECIMAL(38,18) with 20 integral and 18 fractional nines",
    "decimal38_min_int": "DECIMAL(38,0) of 38 negative nines",
    "empty_string": "''",
    "invalid_utf8": "CAST(X'C328FF' AS STRING): bytes that are not valid UTF-8",
    "unicode": "multi-byte string (accents, CJK, emoji, combining mark)",
    "whitespace": "string padded with ASCII and Unicode whitespace",
    "empty_binary": "X''",
    "binary_bytes": "X'00FF7F80'",
    "date_min": "DATE'0001-01-01'",
    "date_max": "DATE'9999-12-31'",
    "date_julian_gap": "DATE'1582-10-10' (inside the Julian->Gregorian gap)",
    "ts_pre_epoch": "TIMESTAMP'1969-12-31 23:59:59.999999' (UTC session)",
    "dst_gap": "TIMESTAMP '2024-03-10 02:30:00' in America/Los_Angeles (nonexistent local time)",
    "dst_overlap": "TIMESTAMP '2024-11-03 01:30:00' in America/Los_Angeles (ambiguous local time)",
    "ntz_dst_gap": "TIMESTAMP_NTZ'2024-03-10 02:30:00' with session zone America/Los_Angeles",
    "empty_array": "empty array of the argument's type",
    "array_of_null": "single-NULL-element array of the argument's type",
    "empty_map": "empty map of the argument's type",
    "rows_all_null": "aggregate input: two NULL rows",
    "rows_empty": "aggregate input: zero rows (WHERE false)",
    "rows_int_max": "aggregate input: two integral-maximum rows (sum overflow)",
    "rows_nan": "aggregate input: NaN and 1.0",
    "rows_signed_zero": "aggregate input: -0.0 and 0.0",
    "rows_decimal38": "aggregate input: two DECIMAL(38,18) maxima (sum overflow)",
}

INT_BOUNDS = {
    "tinyint": ("CAST(-128 AS TINYINT)", "CAST(127 AS TINYINT)"),
    "smallint": ("CAST(-32768 AS SMALLINT)", "CAST(32767 AS SMALLINT)"),
    "int": ("CAST(-2147483648 AS INT)", "CAST(2147483647 AS INT)"),
    "bigint": ("CAST(-9223372036854775808 AS BIGINT)", "CAST(9223372036854775807 AS BIGINT)"),
}
D38_MAX_FRAC = "CAST('99999999999999999999.999999999999999999' AS DECIMAL(38,18))"
D38_MIN_INT = "CAST('-99999999999999999999999999999999999999' AS DECIMAL(38,0))"


# --------------------------------------------------------------------------------------------------
# SQL text scanning (quote/paren aware; only for deriving inputs, never for expectations)
# --------------------------------------------------------------------------------------------------

def _scan(s: str):
    """Yield (index, char, depth, in_quote) for each character of `s`."""
    depth = 0
    quote = None
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            yield i, c, depth, True
            if c == "\\" and quote in "'\"":
                i += 1
                if i < len(s):
                    yield i, s[i], depth, True
            elif c == quote:
                quote = None
        elif c in "'\"`":
            quote = c
            yield i, c, depth, True
        else:
            if c == "(":
                depth += 1
            yield i, c, depth, False
            if c == ")":
                depth -= 1
        i += 1


def matching_paren(s: str, open_idx: int) -> int:
    """Index of the `)` closing the `(` at `open_idx`, or -1."""
    target = None
    for i, c, d, q in _scan(s):
        if q:
            continue
        if i == open_idx:
            target = d
        elif i > open_idx and c == ")" and d == target:
            return i
    return -1


def split_args(s: str, start: int, end: int) -> list[tuple[int, int]]:
    """Top-level comma-separated spans of s[start:end] (the text between a call's parens)."""
    inner = s[start:end]
    if not inner.strip():
        return []
    spans = []
    seg = 0
    for i, c, d, q in _scan(inner):
        if not q and d == 0 and c == ",":
            spans.append((seg, i))
            seg = i + 1
    spans.append((seg, len(inner)))
    out = []
    for a, b in spans:
        while a < b and inner[a].isspace():
            a += 1
        while b > a and inner[b - 1].isspace():
            b -= 1
        out.append((start + a, start + b))
    return out


def top_level_keyword(s: str, kw: str) -> int:
    """Index of the first top-level (depth 0, unquoted) occurrence of word `kw`, or -1."""
    pat = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    quoted = set()
    depth_at = {}
    for i, _c, d, q in _scan(s):
        if q:
            quoted.add(i)
        depth_at[i] = d
    for m in pat.finditer(s):
        i = m.start()
        if i not in quoted and depth_at.get(i, 0) == 0:
            return i
    return -1


def find_call(stmt: str, name: str) -> tuple[int, int] | None:
    """(open_paren, close_paren) of the first unquoted `name(` call in `stmt`."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return None
    quoted = {i for i, _c, _d, q in _scan(stmt) if q}
    for m in re.finditer(r"(?<![\w.`])" + re.escape(name) + r"\s*\(", stmt, re.IGNORECASE):
        if m.start() in quoted:
            continue
        open_idx = m.end() - 1
        close = matching_paren(stmt, open_idx)
        if close > 0:
            return open_idx, close
    return None


def find_infix(stmt: str, name: str) -> list[tuple[int, int]] | None:
    """Operand spans of `SELECT <lhs> OP <rhs>` / `SELECT OP <rhs>` for operator-named functions."""
    m = re.match(r"\s*SELECT\s+", stmt, re.IGNORECASE)
    if not m or top_level_keyword(stmt, "FROM") >= 0:
        return None
    body_start = m.end()
    op = re.escape(name)
    word = name.isalpha()
    pat = re.compile((r"\s+" + op + r"\s+") if word else (r"\s*" + op + r"\s*"), re.IGNORECASE)
    quoted_or_nested = {i for i, _c, d, q in _scan(stmt) if q or d > 0}
    for mm in pat.finditer(stmt, body_start):
        if mm.start() in quoted_or_nested or any(i in quoted_or_nested for i in range(mm.start(), mm.end())):
            continue
        lhs = (body_start, mm.start())
        rhs = (mm.end(), len(stmt))
        if not stmt[rhs[0]:rhs[1]].strip():
            return None
        # Reject an operator that is part of a longer operator token (e.g. `<` inside `<=>`).
        if not word:
            before = stmt[mm.start() - 1] if mm.start() > 0 else " "
            after = stmt[mm.end()] if mm.end() < len(stmt) else " "
            if before in "<>=!|&^*/%+-~" or after in "<>=!|&^*/%+-~":
                continue
        spans = [rhs] if not stmt[lhs[0]:lhs[1]].strip() else [lhs, rhs]
        if any("," in stmt[a:b] and top_level_comma(stmt[a:b]) for a, b in spans):
            return None
        return spans
    return None


def top_level_comma(s: str) -> bool:
    return any(c == "," and d == 0 and not q for _i, c, d, q in _scan(s))


def parse_examples(text: str) -> list[str]:
    """Statements of an ExpressionInfo examples block (`> stmt;` lines; continuation lines joined)."""
    stmts = []
    cur = None
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("> "):
            cur = s[2:]
        elif cur is not None:
            cur += " " + s
        else:
            continue
        if cur.rstrip().endswith(";"):
            stmts.append(cur.rstrip()[:-1].strip())
            cur = None
    return stmts


# --------------------------------------------------------------------------------------------------
# Edge and probe values per Spark type (simpleString)
# --------------------------------------------------------------------------------------------------

def edge_values(t: str) -> list[tuple[str, str, dict]]:
    """(label, sql_expr, extra_conf) edge inputs for an argument of Spark type `t`."""
    out = [("null", "NULL" if t == "void" else f"CAST(NULL AS {t})", {})]
    if t in ("float", "double"):
        T = t.upper()
        out += [("nan", f"CAST('NaN' AS {T})", {}), ("neg_zero", f"CAST('-0.0' AS {T})", {}),
                ("pos_inf", f"CAST('Infinity' AS {T})", {}), ("neg_inf", f"CAST('-Infinity' AS {T})", {})]
    elif t in INT_BOUNDS:
        lo, hi = INT_BOUNDS[t]
        out += [("int_min", lo, {}), ("int_max", hi, {})]
    elif t.startswith("decimal"):
        out += [("decimal38_max_frac", D38_MAX_FRAC, {}), ("decimal38_min_int", D38_MIN_INT, {})]
    elif t == "string" or t.startswith("string collate"):
        out += [("empty_string", "''", {}), ("invalid_utf8", "CAST(X'C328FF' AS STRING)", {}),
                ("unicode", "'Ä€日本語😀é'", {}), ("whitespace", "' \\t a\\u3000\\n '", {})]
    elif t == "binary":
        out += [("empty_binary", "X''", {}), ("binary_bytes", "X'00FF7F80'", {})]
    elif t == "date":
        out += [("date_min", "DATE'0001-01-01'", {}), ("date_max", "DATE'9999-12-31'", {}),
                ("date_julian_gap", "DATE'1582-10-10'", {})]
    elif t == "timestamp":
        out += [("ts_pre_epoch", "TIMESTAMP'1969-12-31 23:59:59.999999'", {}),
                ("dst_gap", "TIMESTAMP'2024-03-10 02:30:00'", {TZ: DST_TZ}),
                ("dst_overlap", "TIMESTAMP'2024-11-03 01:30:00'", {TZ: DST_TZ})]
    elif t == "timestamp_ntz":
        out += [("ntz_dst_gap", "TIMESTAMP_NTZ'2024-03-10 02:30:00'", {TZ: DST_TZ})]
    elif t.startswith("array<"):
        out += [("empty_array", f"CAST(array() AS {t})", {}), ("array_of_null", f"CAST(array(NULL) AS {t})", {})]
    elif t.startswith("map<"):
        out += [("empty_map", f"CAST(map() AS {t})", {})]
    return out


def agg_row_edges(t: str) -> list[tuple[str, list[str]]]:
    """(label, row values) edge inputs for a single-column aggregate over a column of type `t`."""
    null = "NULL" if t == "void" else f"CAST(NULL AS {t})"
    out = [("rows_all_null", [null, null])]
    if t in INT_BOUNDS:
        out.append(("rows_int_max", [INT_BOUNDS[t][1]] * 2))
    elif t in ("float", "double"):
        T = t.upper()
        out += [("rows_nan", [f"CAST('NaN' AS {T})", f"CAST(1.0 AS {T})"]),
                ("rows_signed_zero", [f"CAST('-0.0' AS {T})", f"CAST(0.0 AS {T})"])]
    elif t.startswith("decimal"):
        out.append(("rows_decimal38", [D38_MAX_FRAC] * 2))
    return out


# --------------------------------------------------------------------------------------------------
# Case derivation
# --------------------------------------------------------------------------------------------------

def family_of(fn: dict) -> str:
    g = fn["group"] or ("operators" if fn["registry"] == "operator" else "misc_funcs")
    return g[:-6] if g.endswith("_funcs") else g


def seeds_for(fn: dict) -> list[tuple[str, dict]]:
    """(statement, conf-from-SET) seeds of one function, in example order."""
    out = []
    conf: dict[str, str] = {}
    stmts = parse_examples(fn["examples"]) or SUPPLEMENTAL_SEEDS.get(fn["name"], [])
    for st in stmts:
        m = re.fullmatch(r"SET\s+([\w.]+)\s*=\s*(\S+)", st, re.IGNORECASE)
        if m:
            conf[m.group(1)] = m.group(2)
            continue
        out.append((st, dict(conf)))
    return out


class TypeOracle:
    """Spark-resolved type of a standalone expression (AnalyzePlan only, cached)."""

    def __init__(self, spark):
        self.spark = spark
        self.cache: dict[str, str | None] = {}

    def type_of(self, expr: str) -> str | None:
        if expr not in self.cache:
            try:
                self.cache[expr] = self.spark.sql(f"SELECT {expr}").schema.fields[0].dataType.simpleString()
            except Exception:  # noqa: BLE001 - unresolvable on its own (column refs, lambdas, ...)
                self.cache[expr] = None
        return self.cache[expr]


def derive_variants(fn: dict, stmt: str, oracle: TypeOracle, seen: set) -> list[tuple[str, str, dict]]:
    """(kind, sql, extra_conf) variants of seed `stmt` with edge/probe values substituted."""
    name = fn["name"]
    if name in DERIVE_SKIP:
        return []
    call = find_call(stmt, name)
    if call:
        spans = split_args(stmt, call[0] + 1, call[1])
    else:
        spans = find_infix(stmt, name) or []
    infix = call is None
    out = []
    for pos, (a, b) in enumerate(spans):
        arg = stmt[a:b]
        if re.search(r"=>|->|\bDISTINCT\b", arg, re.IGNORECASE) or arg.strip() == "*":
            continue
        t = oracle.type_of(arg)
        if t is None:
            continue
        subs = [(f"edge:{label}", expr, conf) for label, expr, conf in edge_values(t)]
        subs += [(f"type:{label}", expr, {}) for label, expr in TYPE_PROBES]
        for kind, expr, conf in subs:
            if kind in ("edge:int_min", "edge:int_max") and name in SIZE_ARG_FUNCTIONS:
                continue
            key = (len(spans), pos, kind, expr)
            if key in seen:
                continue
            seen.add(key)
            rep = f"({expr})" if infix else expr
            out.append((f"{kind}@arg{pos}", stmt[:a] + rep + stmt[b:], conf))
    out += derive_agg_rows(fn, stmt, oracle, seen)
    return out


AGG_VALUES = re.compile(
    r"(?P<head>.*?\bFROM\s+VALUES\s+)(?P<rows>.+?)(?P<tail>\s+(?:AS\s+)?(?P<alias>\w+)\s*\(\s*(?P<col>\w+)\s*\))(?P<rest>.*)",
    re.IGNORECASE | re.DOTALL,
)


def derive_agg_rows(fn: dict, stmt: str, oracle: TypeOracle, seen: set) -> list[tuple[str, str, dict]]:
    if fn["group"] != "agg_funcs":
        return []
    m = AGG_VALUES.fullmatch(stmt)
    if not m:
        return []
    rows = m.group("rows")
    close = matching_paren(rows, 0) if rows.startswith("(") else -1
    if close < 0 or top_level_comma(rows[1:close]):
        return []
    try:
        t = oracle.spark.sql(f"SELECT {m.group('col')} FROM VALUES {rows}{m.group('tail')}").schema.fields[0]
        t = t.dataType.simpleString()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for label, vals in agg_row_edges(t):
        key = ("rows", label, t)
        if key in seen:
            continue
        seen.add(key)
        new_rows = ", ".join(f"({v})" for v in vals)
        out.append((f"edge:{label}@rows", m.group("head") + new_rows + m.group("tail") + m.group("rest"), {}))
    rest = m.group("rest")
    if ("rows", "rows_empty") not in seen and not re.search(r"\b(WHERE|GROUP|ORDER|HAVING|LIMIT)\b", rest, re.I):
        seen.add(("rows", "rows_empty"))
        out.append(("edge:rows_empty@rows", m.group("head") + rows + m.group("tail") + " WHERE false" + rest, {}))
    return out


# A JVM identity hash in rendered output (e.g. `[B@1b2c3d4` from format_string('%s', binary)).
JVM_IDENTITY = re.compile(r"(?:\[[BCSIJFDZ]|[A-Za-z_$][\w$.]*)@[0-9a-f]{5,8}\b")


def calls_value_unstable(sql: str) -> bool:
    return any(re.search(r"(?<![\w.])" + re.escape(n) + r"\s*\(", sql, re.IGNORECASE) for n in VALUE_UNSTABLE)


def has_top_level_order_by(sql: str) -> bool:
    i = top_level_keyword(sql, "ORDER")
    return i >= 0 and re.match(r"ORDER\s+BY\b", sql[i:], re.IGNORECASE) is not None


# --------------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------------

def norm_message(msg: str) -> str:
    """Message text without the JVM stack trace or the appended logical plan; expr ids normalised."""
    msg = msg.split("\n\nJVM stacktrace:")[0]
    # Query context added by Spark/Connect: SQL fragment, or the DataFrame call site (a local file path).
    msg = re.split(r"\n== (?:SQL|DataFrame)\b", msg, maxsplit=1)[0]
    msg = re.sub(r"\s*called from\s+\S+:\d+", "", msg)
    msg = re.split(r";\n(?=['!+:\[A-Za-z])", msg, maxsplit=1)[0]
    msg = re.sub(r"\b(stage|task|TID)\s+\d+(?:\.\d+)?", r"\1 N", msg, flags=re.IGNORECASE)
    return re.sub(r"#\d+", "#N", msg).rstrip()


def spark_phase(e: Exception) -> str:
    """Phase implied by Spark's exception type: parse, analysis, or execution (runtime evaluation)."""
    from pyspark.errors.exceptions.connect import AnalysisException, ParseException
    if isinstance(e, ParseException):
        return "parse"
    return "analysis" if isinstance(e, AnalysisException) else "execution"


def error_of(e: Exception) -> dict:
    phase = spark_phase(e)
    cond = getattr(e, "getCondition", lambda: None)()
    state = getattr(e, "getSqlState", lambda: None)()
    params = getattr(e, "getMessageParameters", lambda: None)() or {}
    params = {k: re.sub(r"#\d+", "#N", str(v)) for k, v in sorted(params.items())}
    if not cond:
        from pyspark.errors.exceptions.connect import SparkConnectGrpcException
        if isinstance(e, SparkConnectGrpcException):
            # Raised by Spark without an error condition (legacy JVM exception): keep the phase.
            cond = f"UNCLASSIFIED.{type(e).__name__}"
        else:
            phase = "client"
            cond = f"CLIENT.{type(e).__name__}"
    # Unclassified JVM exceptions have unstable text (the JIT may drop it: OmitStackTraceInFastThrow).
    message = "" if cond.startswith("UNCLASSIFIED.") else norm_message(str(e))
    err = {"error_class": cond, "sqlstate": state, "message": message, "message_parameters": params,
           "phase": phase}
    if cond.startswith(("_LEGACY_ERROR_TEMP_", "UNCLASSIFIED.", "CLIENT.")) or not state:
        err["stability"] = "unstable"  # not stable API (ADR 0008): compare "errors" + phase only
    return err


def text_columns(F, types: list[str]):
    """Spark-side text rendering per column: (expression, render kind)."""
    out = []
    for i, t in enumerate(types):
        c = F.col(f"c{i}")
        if t == "binary":
            out.append((F.hex(c), "hex"))
        elif t.startswith(("geometry", "geography")):
            out.append((F.hex(F.expr(f"st_asbinary(c{i})")), "wkb_hex"))
        else:
            out.append((c.cast("string"), "cast"))
    return out


def _float_cell(v, t) -> dict:
    import numpy as np
    if pa_types().is_float32(t):
        f = np.float32(v)
        return {"f32": f"{int(f.view(np.uint32)):08x}", "repr": repr(f)}
    f = np.float64(v)
    return {"f64": f"{int(f.view(np.uint64)):016x}", "repr": repr(float(f))}


def _decimal_cell(d: Decimal, scale: int) -> dict:
    sign, digits, exp = d.as_tuple()
    n = int("".join(map(str, digits)) or "0")
    k = exp + scale
    n = n * 10 ** k if k >= 0 else n // 10 ** (-k)
    return {"unscaled": str(-n if sign else n), "scale": scale}


def pa_types():
    import pyarrow as pa
    return pa.types


def encode_typed(arr) -> list:
    """Typed JSON cells of an Arrow array (exact; NULL is JSON null, never the string "null")."""
    import pyarrow as pa
    t = arr.type
    T = pa.types
    if T.is_null(t):
        return [None] * len(arr)
    if T.is_map(t):
        keys, items = encode_typed(arr.keys), encode_typed(arr.items)
        offs = arr.offsets.to_pylist()
        vals = [{"map": [[keys[j], items[j]] for j in range(offs[i], offs[i + 1])]} for i in range(len(arr))]
    elif T.is_list(t) or T.is_large_list(t):
        child = encode_typed(arr.values)
        offs = arr.offsets.to_pylist()
        vals = [child[offs[i]:offs[i + 1]] for i in range(len(arr))]
    elif T.is_struct(t):
        names = [t.field(j).name for j in range(t.num_fields)]
        fields = [encode_typed(f) for f in arr.flatten()]
        vals = [{"struct": [[names[j], fields[j][i]] for j in range(len(names))]} for i in range(len(arr))]
    elif T.is_timestamp(t):
        key = "ts_" if t.tz else "ntz_"
        ints = arr.cast(pa.int64()).to_pylist()
        vals = [{key + t.unit: v, **({"tz": t.tz} if t.tz else {})} for v in ints]
    elif T.is_date32(t):
        vals = [{"days": v} for v in arr.cast(pa.int32()).to_pylist()]
    elif T.is_string(t) or T.is_large_string(t):
        vals = [None if b is None else _utf8_or_hex(b) for b in arr.cast(pa.binary()).to_pylist()]
    elif T.is_binary(t) or T.is_large_binary(t):
        vals = [None if b is None else {"hex": b.hex().upper()} for b in arr.to_pylist()]
    elif T.is_floating(t):
        vals = [None if v is None else _float_cell(v, t) for v in arr.to_pylist()]
    elif T.is_decimal(t):
        vals = [None if v is None else _decimal_cell(v, t.scale) for v in arr.to_pylist()]
    elif T.is_boolean(t) or T.is_integer(t):
        vals = arr.to_pylist()
    else:
        vals = [{"arrow": str(t), "text": str(v)} for v in arr.to_pylist()]
    valid = arr.is_valid().to_pylist() if arr.null_count else None
    return vals if valid is None else [v if ok else None for v, ok in zip(vals, valid)]


def _utf8_or_hex(b: bytes):
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return {"utf8_hex": b.hex().upper()}


def collect(df, F, types: list[str]) -> tuple[list, list, list | None, str | None]:
    """One query returning text and typed cells: (rows, render kinds, typed_rows, typed_unavailable)."""
    import pyarrow as pa
    text = text_columns(F, types)
    raw = [F.col(f"c{i}").alias(f"r{i}") for i in range(len(types))]
    def arrow(cols):
        try:
            return df.select(*cols).toArrow()
        except ValueError as e:
            if "non-nullable" not in str(e):
                raise
            # Spark declared a column non-nullable yet returned NULL (recorded in `schema`); the
            # client's Arrow cast rejects that, so re-select through an opaque nullable wrapper.
            return df.select(*[F.when(F.rand(0) >= 0, c) for c in cols]).toArrow()

    try:
        table, typed_reason = arrow([e for e, _ in text] + raw), None
    except Exception as e:  # noqa: BLE001 - a raw column Arrow cannot carry (interval, TIME, dup names)
        typed_reason = getattr(e, "getCondition", lambda: None)() or type(e).__name__
        table = arrow([e for e, _ in text])
    n = len(types)
    cols = [pa.concat_arrays(table.column(i).chunks) if table.column(i).num_chunks else pa.array([], table.column(i).type)
            for i in range(table.num_columns)]
    kinds = [k for _, k in text]
    text_cells = []
    for i in range(n):
        cells = [None if b is None else _utf8_or_hex(b) for b in cols[i].cast(pa.binary()).to_pylist()]
        if any(isinstance(c, dict) for c in cells):  # not valid UTF-8: whole column as hex of the bytes
            cells = [None if c is None else (c["utf8_hex"] if isinstance(c, dict) else c.encode().hex().upper())
                     for c in cells]
            kinds[i] = "utf8_hex"
        text_cells.append(cells)
    rows = [list(r) for r in zip(*text_cells)] if n else [[] for _ in range(table.num_rows)]
    typed = None
    if typed_reason is None:
        tc = [encode_typed(cols[n + i]) for i in range(n)]
        typed = [list(r) for r in zip(*tc)] if n else [[] for _ in range(table.num_rows)]
    return rows, kinds, typed, typed_reason


class Watchdog:
    """Interrupts all running operations of a session after QUERY_TIMEOUT_S."""

    def __init__(self, spark):
        self.spark = spark
        self.fired = False
        self.timer = None

    def __enter__(self):
        def fire():
            self.fired = True
            try:
                self.spark.interruptAll()
            except Exception:  # noqa: BLE001
                pass
        self.timer = threading.Timer(QUERY_TIMEOUT_S, fire)
        self.timer.start()
        return self

    def __exit__(self, *exc):
        self.timer.cancel()
        return False


def run_case(spark, F, sql: str, nondet: bool) -> dict:
    with Watchdog(spark) as wd:
        try:
            df = spark.sql(sql)
            fields = df.schema.fields
        except Exception as e:  # noqa: BLE001
            return {"timeout": True} if wd.fired else {"error": error_of(e)}
        schema = [{"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in fields]
        if nondet:
            return {"schema": schema, "nondeterministic": True}
        types = [f["type"] for f in schema]
        renamed = df.toDF(*[f"c{i}" for i in range(len(types))]).limit(MAX_ROWS + 1)
        try:
            rows, kinds, typed, typed_reason = collect(renamed, F, types)
        except Exception as e:  # noqa: BLE001
            return {"timeout": True} if wd.fired else {"schema": schema, "error": error_of(e)}
    result: dict = {"schema": schema, "render": kinds, "truncated": len(rows) > MAX_ROWS,
                    "ordered": has_top_level_order_by(sql)}
    pairs = list(zip(rows, typed if typed is not None else [None] * len(rows)))[:MAX_ROWS]
    if not result["ordered"]:
        pairs.sort(key=lambda p: json.dumps(p, ensure_ascii=False, sort_keys=True))
    if any(v is not None and JVM_IDENTITY.search(v) for r, _ in pairs for v in r):
        return {"schema": schema, "nondeterministic": True}
    result["rows"] = [r for r, _ in pairs]
    if typed is None:
        result["typed_unavailable"] = typed_reason
    else:
        result["typed_rows"] = [t for _, t in pairs]
    return result


class Session:
    def __init__(self, remote: str):
        from pyspark.sql import SparkSession
        self.spark = SparkSession.builder.remote(remote).create()
        self.conf: dict[str, str] = {}

    def apply(self, conf: dict[str, str]):
        for k in sorted(set(self.conf) - set(conf)):
            self.spark.conf.unset(k)
        for k, v in sorted(conf.items()):
            if self.conf.get(k) != v:
                self.spark.conf.set(k, v)
        self.conf = dict(conf)

    def close(self):
        try:
            self.spark.stop()
        except Exception:  # noqa: BLE001
            pass


def is_parse_error(res: dict) -> bool:
    return res.get("error", {}).get("error_class", "").startswith("PARSE_")


def case_id(fn: str, sql: str, conf: dict) -> str:
    h = hashlib.sha256(json.dumps([fn, sql, conf], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return f"{fn}#{h[:12]}"


def build_cases(fn: dict, oracle: TypeOracle) -> list[dict]:
    """All cases of one function (mode-independent part), before execution, in deterministic order.

    Each case is run once per mode in `MODES`; the conf here excludes `spark.sql.ansi.enabled`.
    """
    proto = []
    seen: set = set()
    for n, (stmt, set_conf) in enumerate(seeds_for(fn)):
        kind = "example" if fn["examples"].strip() else "seed"
        proto.append((f"{kind}{n}", stmt, set_conf))
        for vkind, sql, extra in derive_variants(fn, stmt, oracle, seen):
            proto.append((f"{kind}{n}:{vkind}", sql, {**set_conf, **extra}))
    cases = []
    ids = set()
    for kind, sql, extra in proto:
        cid = case_id(fn["name"], sql, dict(sorted({TZ: BASE_TZ, **extra}.items())))
        if cid in ids:
            continue
        ids.add(cid)
        cases.append({"id": cid, "function": fn["name"], "kind": kind, "sql": sql,
                      "conf": dict(sorted(extra.items()))})
    return cases


# --------------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------------

def spark_shell(container: str, script: str, data: str = "") -> list[str]:
    """Run a Scala script with `spark-shell -I` inside the container; returns its `@@` lines."""
    files = {"/tmp/krishiv_script.scala": script, "/tmp/krishiv_cases.jsonl": data}
    for path, text in files.items():
        subprocess.run(["docker", "exec", "-i", container, "sh", "-c", f"cat > {path}"], input=text,
                       text=True, check=True)
    cmd = ["docker", "exec", container, "sh", "-c",
           "/opt/spark/bin/spark-shell --master 'local[1]' --driver-memory 2g -I /tmp/krishiv_script.scala "
           "2>/dev/null; rm -f /tmp/krishiv_script.scala /tmp/krishiv_cases.jsonl"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=True).stdout
    lines = [line for line in out.splitlines() if line.startswith("@@")]
    if "@@END" not in lines:
        raise SystemExit("spark-shell script did not complete:\n" + out[-2000:])
    return lines


def dump_registry(container: str) -> tuple[str, list[dict]]:
    script = (Path(__file__).parent / "fnmatrix" / "dump_registry.scala").read_text(encoding="utf-8")
    lines = spark_shell(container, script)
    version = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@VERSION "))
    return version, [json.loads(line[5:]) for line in lines if line.startswith("@@FN ")]


def classify_nondeterministic(container: str, keys: list[tuple[str, dict]]) -> set[int]:
    """Indexes of (sql, conf) whose analyzed plan Spark reports as non-deterministic / CURRENT_LIKE."""
    script = (Path(__file__).parent / "fnmatrix" / "classify_cases.scala").read_text(encoding="utf-8")
    data = "".join(json.dumps({"i": i, "sql": sql, "conf": conf}) + "\n" for i, (sql, conf) in enumerate(keys))
    return {int(line.split()[1]) for line in spark_shell(container, script, data) if line.startswith("@@ND ")}


def container_image(container: str) -> str:
    try:
        return subprocess.run(["docker", "inspect", "-f", "{{.Config.Image}}", container],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")


def self_check(out: Path, by_family: dict) -> None:
    """Refuse to leave generator-side failures or local paths/hostnames in the fixtures.

    Paths are looked for in error text (where client call sites leak); Spark's own examples contain
    path-like literals (`%SystemDrive%/Users/John`), so result values are not scanned for them. The
    host and user names are looked for everywhere.
    """
    bad = [f"{c['id']}/{m}" for cases in by_family.values() for c in cases for m, _ in MODES
           if c[m].get("error", {}).get("error_class", "").startswith(("CLIENT.", "GENERATOR."))]
    if bad:
        raise SystemExit(f"generator-side failures must not be committed: {bad[:20]}")
    leaks = [f"{c['id']}/{m}" for cases in by_family.values() for c in cases for m, _ in MODES
             if LOCAL_PATH.search(json.dumps(c[m].get("error", {})))]
    if leaks:
        raise SystemExit(f"local paths in error text: {leaks[:20]}")
    user = getpass.getuser()
    words = [w for w in (socket.gethostname(), user if len(user) >= 4 else "") if w]
    for f in sorted(out.rglob("*")):
        if f.is_file():
            text = f.read_text(encoding="utf-8")
            hits = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text)]
            if hits:
                raise SystemExit(f"{f}: contains local host/user name {hits}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", default="sc://127.0.0.1:15002")
    ap.add_argument("--container", default="krishiv-bench-spark")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/function-matrix")
    ap.add_argument("--work", default="", help="per-function result cache dir; lets a crashed run resume")
    ap.add_argument("--only", default="", help="comma-separated function names (debugging; partial output)")
    args = ap.parse_args()

    from pyspark.sql import SparkSession, functions as F

    started = time.time()
    version, registry = dump_registry(args.container)
    main_session = SparkSession.builder.remote(args.remote).getOrCreate()
    if main_session.version != version:
        raise SystemExit(f"registry dump version {version} != Connect server {main_session.version}")
    shown = sorted(r[0] for r in main_session.sql("SHOW FUNCTIONS").collect())
    by_name = {}
    for fn in registry:
        by_name.setdefault(fn["name"], fn)  # function registry wins over table/operator duplicates
    missing = sorted(set(shown) - set(by_name))
    extra = sorted(set(by_name) - set(shown))
    print(f"Spark {version}: SHOW FUNCTIONS = {len(shown)}; registry dump = {len(by_name)} distinct names "
          f"({len(registry)} entries); missing from dump = {missing}; not in SHOW FUNCTIONS = {extra}",
          file=sys.stderr)
    if missing:
        raise SystemExit("every SHOW FUNCTIONS name must be in the registry dump")
    names = shown if not args.only else sorted(set(args.only.split(",")) & set(shown))
    render_conf = {k: main_session.conf.get(k) for k in RENDER_CONF_KEYS}
    base_conf = dict(sorted({TZ: BASE_TZ, **{k: v for k, v in render_conf.items() if v is not None}}.items()))
    unset_conf = sorted(k for k, v in render_conf.items() if v is None)

    # Pass 1: derive every case (inputs only). Pass 2: Spark classifies determinism. Pass 3: execute.
    builder = Session(args.remote)
    builder.apply(base_conf)
    oracle = TypeOracle(builder.spark)
    planned = {name: build_cases(by_name[name], oracle) for name in names}
    builder.close()
    keys = sorted({(c["sql"], json.dumps({**base_conf, **c["conf"]}, sort_keys=True))
                   for cases in planned.values() for c in cases})
    nd_idx = classify_nondeterministic(args.container, [(sql, json.loads(conf)) for sql, conf in keys])
    nondet = {(keys[i][0], keys[i][1]) for i in nd_idx}
    print(f"planned {sum(map(len, planned.values()))} cases; Spark marks {len(nondet)} non-deterministic",
          file=sys.stderr)

    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list[dict]] = {}
    fn_rows = []
    timeouts = []
    dropped_parse = 0
    work = Path(args.work) if args.work else None
    if work:
        work.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(names):
        fn = by_name[name]
        family = family_of(fn)
        done = work / (hashlib.sha256(name.encode()).hexdigest()[:16] + ".json") if work else None
        if done and done.exists():
            saved = json.loads(done.read_text(encoding="utf-8"))
            by_family.setdefault(family, []).extend(saved["cases"])
            fn_rows.append(saved["row"])
            dropped_parse += saved["dropped"]
            continue
        dropped_before = dropped_parse
        cases = planned[name]
        sess = Session(args.remote)
        try:
            for mode, ansi in MODES:
                for c in cases:
                    conf = {**base_conf, **c["conf"]}
                    sess.apply({ANSI: ansi, **conf})
                    nd = (c["sql"], json.dumps(conf, sort_keys=True)) in nondet or calls_value_unstable(c["sql"])
                    res = run_case(sess.spark, F, c["sql"], nd)
                    if res.get("timeout"):
                        timeouts.append(f"{c['id']}/{mode}")
                        sess.close()
                        sess = Session(args.remote)
                        res = {"error": {"error_class": "GENERATOR.TIMEOUT"}}
                    c[mode] = res
        finally:
            sess.close()
        kept = []
        for c in cases:
            if ":" in c["kind"] and all(is_parse_error(c[m]) for m, _ in MODES):
                dropped_parse += 1  # a substitution the parser rejects says nothing about the function
                continue
            if c["ansi_off"] == c["ansi_on"]:
                c["ansi_off"] = {"same_as": "ansi_on"}
            kept.append(c)
        cases = sorted(kept, key=lambda c: c["id"])
        by_family.setdefault(family, []).extend(cases)
        fn_rows.append({"name": name, "family": family, "registry": fn["registry"], "class": fn["class"],
                        "since": fn["since"], "deprecated": fn["deprecated"].strip(), "cases": len(cases),
                        "seed_source": "examples" if fn["examples"].strip() else "supplemental"})
        if done:
            done.write_text(json.dumps({"cases": cases, "row": fn_rows[-1],
                                        "dropped": dropped_parse - dropped_before}, ensure_ascii=False),
                            encoding="utf-8")
        print(f"[{idx + 1}/{len(names)}] {name}: {len(cases)} cases", file=sys.stderr)

    families = []
    for family in sorted(by_family):
        cases = sorted(by_family[family], key=lambda c: (c["function"], c["id"]))
        shard = f"cases/{family}.jsonl"
        with open(out / shard, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        families.append({"family": family, "shard": shard,
                         "functions": len({c["function"] for c in cases}), "cases": len(cases)})
    manifest = {
        "format_version": FORMAT_VERSION,
        "spark_version": version,
        "spark_image": container_image(args.container),
        "generator": GENERATOR,
        "seed": None,
        "determinism": "no randomness; content-hashed ids; sorted output; unordered results sorted; cases "
                       "Spark analyzes as non-deterministic or CURRENT_LIKE (incl. seeded rand/uuid/shuffle), "
                       "value_unstable_functions, and outputs with JVM identity hashes record schema only",
        "base_conf": base_conf,
        "unset_conf": unset_conf,
        "modes": {mode: {ANSI: ansi} for mode, ansi in MODES},
        "case_format": "one JSON object per line: id, function, kind, sql, conf (overrides of base_conf; "
                       "mode-independent); family = the shard's family; per mode (ansi_on, ansi_off) either "
                       "{schema, rows, render, ordered, truncated, typed_rows | typed_unavailable}, "
                       "{schema?, error{error_class, sqlstate, message, message_parameters, phase, "
                       "stability?}}, {schema, nondeterministic}, or {same_as: <mode>}",
        "typed_cell_format": "JSON null = SQL NULL; bool/int as JSON; string as JSON string or {utf8_hex}; "
                             "{f64|f32: IEEE bits hex, repr}; {unscaled, scale}; {ts_us, tz} / {ntz_us}; "
                             "{days}; {hex}; arrays as JSON arrays; {map: [[k, v]]}; {struct: [[name, v]]}",
        "max_rows": MAX_ROWS,
        "show_functions_count": len(shown),
        "show_functions": shown,
        "value_unstable_functions": VALUE_UNSTABLE,
        "derive_skip_functions": sorted(DERIVE_SKIP),
        "size_arg_functions": sorted(SIZE_ARG_FUNCTIONS),
        "edge_catalog": EDGE_CATALOG,
        "type_probes": {label: expr for label, expr in TYPE_PROBES},
        "total_cases": sum(f["cases"] for f in families),
        "total_mode_cases": sum(f["cases"] for f in families) * len(MODES),
        "dropped_unparseable_variants": dropped_parse,
        "families": families,
        "functions": fn_rows,
    }
    write_json(out / "manifest.json", manifest)
    self_check(out, by_family)
    print(f"wrote {manifest['total_cases']} cases x {len(MODES)} modes for {len(fn_rows)} functions in "
          f"{len(families)} families against Spark {version} ({time.time() - started:.0f}s; timeouts: {timeouts})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
