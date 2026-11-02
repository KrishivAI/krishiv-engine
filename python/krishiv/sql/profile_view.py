# SPDX-License-Identifier: Apache-2.0
"""An offline viewer for ``krishiv.query-profile`` documents (P2-T096).

:class:`ProfileView` turns the profile JSON of ``df.explain(analyze=True, json=True)``
into one self-contained HTML page: the physical plan as a tree, each node with its
operators' time, rows, blocking and spills as bars scaled to the whole query, and the
pipelines on a timeline. There is no script, no font and no network request: the page
opens from a file or a mail attachment, and ``_repr_html_`` shows the same view inline
in a notebook. Every text from the document is HTML-escaped.

The reader follows the profile format's rule (``design/P2-T140-profile-format-v1-envelope.md``):
unknown members and extras are ignored, a missing or mistyped member this viewer uses is an error.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

__all__ = ["ProfileView", "ui"]

_FORMAT = "krishiv.query-profile"

_STYLE = """
.kp{font:13px/1.45 system-ui,sans-serif;color:#1d2330;max-width:1100px}
.kp h2{font-size:15px;margin:14px 0 6px}
.kp .sum{display:flex;flex-wrap:wrap;gap:6px 18px;margin:4px 0}
.kp .sum span b{font-weight:600}
.kp ul{list-style:none;margin:0;padding-left:18px;border-left:1px solid #c9d0dc}
.kp>ul{border-left:0;padding-left:0}
.kp li{margin:3px 0}
.kp .node{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.kp .name{font-weight:600}
.kp .detail{color:#5b6475;font-family:ui-monospace,monospace;font-size:12px}
.kp .op{margin:2px 0 2px 14px;display:grid;grid-template-columns:130px 1fr;gap:2px 10px;align-items:center}
.kp .bar{background:#e8ecf3;height:9px;border-radius:2px;position:relative}
.kp .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:#3d6fd6}
.kp .bar i.io{background:#d68a3d}
.kp .bar i.mem{background:#c0494b}
.kp .num{color:#3a4252;font-variant-numeric:tabular-nums}
.kp .tl{position:relative;height:10px;background:#f1f3f8;border-radius:2px;margin:4px 0}
.kp .tl i{position:absolute;top:0;bottom:0;background:#3d6fd6;border-radius:2px;opacity:.8}
.kp table{border-collapse:collapse}
.kp td,.kp th{padding:2px 10px 2px 0;text-align:left;font-weight:400}
.kp .tag{background:#eef1f7;border-radius:3px;padding:0 5px;margin-right:4px;font-size:12px}
"""


def _ns(value: int) -> str:
    """Nanoseconds as the shortest readable duration."""
    if value >= 1_000_000_000:
        return f"{value / 1e9:.2f} s"
    if value >= 1_000_000:
        return f"{value / 1e6:.1f} ms"
    if value >= 1_000:
        return f"{value / 1e3:.1f} µs"
    return f"{value} ns"


def _bytes(value: int) -> str:
    for unit, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if value >= size:
            return f"{value / size:.1f} {unit}"
    return f"{value} B"


def _count(value: int) -> str:
    return f"{value:,}"


class ProfileView:
    """A query profile, shown as HTML.

    ``ProfileView(document)`` takes the parsed document (a ``dict``) or its JSON text. Use
    :meth:`to_html` for a full page, :meth:`save` to write it, or leave the object as the last
    expression of a notebook cell.
    """

    def __init__(self, document: dict[str, Any] | str) -> None:
        doc = json.loads(document) if isinstance(document, str) else document
        if not isinstance(doc, dict) or doc.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} document")
        if doc.get("version") != 1:
            raise ValueError(f"unsupported {_FORMAT} version {doc.get('version')!r} (this viewer reads 1)")
        for member in ("query", "pipelines", "plan", "operators"):
            if member not in doc:
                raise ValueError(f"the profile has no {member!r} member")
        self.document = doc

    def __repr__(self) -> str:
        q = self.document["query"]
        return f"ProfileView(wall={_ns(q['wall_time_ns'])}, operators={len(self.document['operators'])})"

    # ------------------------------------------------------------------ rendering
    def _fragment(self) -> str:
        doc = self.document
        query = doc["query"]
        wall = max(int(query["wall_time_ns"]), 1)
        operators = doc["operators"]
        by_node: dict[int, list[dict[str, Any]]] = {}
        loose: list[dict[str, Any]] = []
        for op in operators:
            node = op.get("node")
            if node is None:
                loose.append(op)
            else:
                by_node.setdefault(int(node), []).append(op)
        # Bars are scaled to the most time any operator spent, so the heavy ones stand out.
        scale = max(
            [int(op["metrics"]["compute_time_ns"]) for op in operators]
            + [int(op["metrics"]["blocked_on_io_ns"]) for op in operators]
            + [1]
        )
        plan = doc["plan"]

        def operator_block(op: dict[str, Any]) -> str:
            m = op["metrics"]
            rows = [
                ("compute", "", int(m["compute_time_ns"]), _ns(int(m["compute_time_ns"]))),
                ("blocked on I/O", "io", int(m["blocked_on_io_ns"]), _ns(int(m["blocked_on_io_ns"]))),
            ]
            if int(m["blocked_on_memory_ns"]):
                rows.append(
                    ("blocked on memory", "mem", int(m["blocked_on_memory_ns"]), _ns(int(m["blocked_on_memory_ns"])))
                )
            out = [
                '<div class="op"><span class="num">'
                f'<span class="tag">{escape(str(op["role"]))}</span>{escape(str(op["name"]))}</span><span></span>'
            ]
            for label, cls, value, text in rows:
                width = min(100.0, 100.0 * value / scale)
                out.append(
                    f'<span class="num">{escape(label)}</span>'
                    f'<span class="bar" title="{escape(text)}"><i class="{cls}" style="width:{width:.1f}%"></i></span>'
                )
            facts = [
                f"rows in {_count(int(m['rows_in']))}",
                f"rows out {_count(int(m['rows_out']))}",
                f"batches {_count(int(m['batches_out']))}",
                f"drivers {_count(int(m['drivers']))}",
                f"peak {_bytes(int(m['peak_reserved_bytes']))}",
            ]
            if int(m["spills"]):
                facts.append(f"spills {_count(int(m['spills']))} ({_bytes(int(m['spilled_bytes']))})")
            for extra in op.get("extra", []):
                value = extra["value"]
                shown = value if isinstance(value, str) else _count(int(value))
                facts.append(f"{extra['name']} {shown}")
            out.append(f'<span class="num">facts</span><span class="num">{escape(" · ".join(facts))}</span></div>')
            return "".join(out)

        def node_html(position: int, seen: set[int]) -> str:
            if position in seen:  # a malformed document must not loop
                return ""
            seen = seen | {position}
            node = plan[position]
            ident = node.get("id")
            head = (
                f'<div class="node"><span class="name">{escape(str(node["name"]))}</span>'
                f'<span class="detail">{escape(str(node["detail"]))}</span></div>'
            )
            ops = "".join(operator_block(op) for op in by_node.get(int(ident), [])) if ident is not None else ""
            kids = "".join(node_html(int(c), seen) for c in node.get("children", []))
            return f"<li>{head}{ops}{('<ul>' + kids + '</ul>') if kids else ''}</li>"

        summary = [
            ("wall", _ns(int(query["wall_time_ns"]))),
            ("workers", _count(int(query["workers"]))),
            ("drivers", _count(int(query["drivers"]))),
            ("batches", _count(int(query["batches"]))),
            ("yields", _count(int(query["yields"]))),
            ("longest poll", _ns(int(query["max_slice_ns"]))),
            ("stalls", _count(int(query["stalls"]))),
        ]
        if "busy_ns" in query:
            summary.append(("busy", _ns(int(query["busy_ns"]))))
        parts = [
            '<div class="kp">',
            '<div class="sum">'
            + "".join(f"<span>{escape(k)} <b>{escape(v)}</b></span>" for k, v in summary)
            + "</div>",
            "<h2>Plan</h2>",
            f"<ul>{node_html(0, set())}</ul>" if plan else "<p>(empty plan)</p>",
        ]
        if loose:
            parts.append("<h2>Result</h2>" + "".join(operator_block(op) for op in loose))
        pipelines = doc["pipelines"]
        if pipelines:
            parts.append("<h2>Pipelines</h2><table>")
            for p in pipelines:
                start, end = int(p["started_ns"]), int(p["ended_ns"])
                left = min(100.0, 100.0 * start / wall)
                width = max(0.5, min(100.0 - left, 100.0 * max(end - start, 0) / wall))
                parts.append(
                    f"<tr><th>pipeline {int(p['id'])}</th>"
                    f'<td class="num">{_ns(max(end - start, 0))}, {_count(int(p["drivers"]))} drivers, '
                    f"{_count(int(p['threads']))} threads</td>"
                    '<td style="width:520px"><div class="tl">'
                    f'<i style="left:{left:.1f}%;width:{width:.1f}%"></i></div></td></tr>'
                )
            parts.append("</table>")
        parts.append("</div>")
        return f"<style>{_STYLE}</style>" + "".join(parts)

    def _repr_html_(self) -> str:
        """The view as an HTML fragment, for notebooks."""
        return self._fragment()

    def to_html(self, title: str = "Krishiv query profile") -> str:
        """A complete, self-contained HTML page (no script, no network)."""
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{escape(title)}</title></head><body>{self._fragment()}</body></html>"
        )

    def save(self, path: str | Path, title: str = "Krishiv query profile") -> Path:
        """Writes :meth:`to_html` to ``path`` and returns it."""
        target = Path(path)
        target.write_text(self.to_html(title), encoding="utf-8")
        return target


def ui(source: Any, *, path: str | Path | None = None) -> ProfileView:
    """The viewer for ``source``: a profile document (``dict`` or JSON text), a
    :class:`ProfileView`, or a DataFrame (which runs ``EXPLAIN ANALYZE`` and views its profile).

    With ``path`` the page is also written there. In a notebook the returned view renders itself.
    """
    if isinstance(source, ProfileView):
        view = source
    elif isinstance(source, dict | str):
        view = ProfileView(source)
    elif hasattr(source, "explainString"):
        view = ProfileView(source.explainString(analyze=True, json=True))
    else:
        raise TypeError(f"cannot view a {type(source).__name__}: pass a profile document or a DataFrame")
    if path is not None:
        view.save(path)
    return view
