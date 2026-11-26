# SPDX-License-Identifier: Apache-2.0
"""Write the third-party notices file shipped in a Krishiv binary artifact.

The wheels, the CLI binaries and the container image link third-party crates
statically. Their licenses (MIT, BSD, Apache-2.0's NOTICE rule, ...) ask that
the copyright and license text travel with the binary, so every artifact
carries one text file that lists each third-party crate linked into it and
reproduces the license and notice files its published package holds, nested
ones included (a `-sys` crate's bundled C library keeps its own license in a
subdirectory).

The crate set is the dependency graph `cargo metadata` resolves for one
package, its features and the artifact's target triples: normal dependencies
only (dev and build dependencies are not linked). `cargo metadata` resolves the
whole workspace, so feature unification can add a crate the artifact does not
link; the list is an upper bound, never short. Krishiv's own workspace crates
are covered by LICENSE and NOTICE at the repository root.

Usage (stdlib only, run from the repository; cargo must be on PATH):

    python3 .github/scripts/third_party_notices.py --package krishiv-cli \\
        --cargo-args "--features server,ui" --target x86_64-unknown-linux-gnu \\
        --title "the krishiv CLI" --out out/THIRD-PARTY-NOTICES.txt

`--cargo-args` takes the same flag string the build gets (maturin's or
cargo's): only `--features`, `--no-default-features` and `--all-features` are
read from it. `--target` may repeat; the notices then cover the union.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# File names (upper-cased, before any extension) that hold license or notice text.
_STEMS = ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE", "UNLICENSE")
# Directories never searched for nested license files: they are not linked.
_SKIP_DIRS = {
    ".git",
    "target",
    "tests",
    "test",
    "benches",
    "bench",
    "examples",
    "fuzz",
    "docs",
    "doc",
}
# Packages published without a license file, and where their license text comes from: a crate of
# the same project (same repository, same license), for a bare SPDX id the standard text as
# another crate in the set carries it, or (`vendored:<file>`) the project's own license file kept
# in `notices/` beside this script, for a project none of whose crates carries one. A sibling
# counts only with the same repository and license field, a vendored file only for a crate of the
# repository it was copied from; otherwise the crate stays listed as carrying no license file.
SAME_TEXTS_AS = {
    "alloc-stdlib": "alloc-no-stdlib",  # dropbox/rust-alloc-no-stdlib, BSD-3-Clause
    "tonic-prost": "tonic",  # grpc/grpc-rust, MIT
    "flatbuffers": "spdx:Apache-2.0",  # google/flatbuffers, the Apache License 2.0 alone
    # napi-rs/napi-rs, MIT: no crate of the project ships its LICENSE (notices/README.md).
    "napi": "vendored:napi-rs.LICENSE",
    "napi-sys": "vendored:napi-rs.LICENSE",
    "napi-derive": "vendored:napi-rs.LICENSE",
    "napi-derive-backend": "vendored:napi-rs.LICENSE",
}
# The repository each vendored file was copied from: a crate borrows it only from there.
VENDORED_FROM = {"napi-rs.LICENSE": "https://github.com/napi-rs/napi-rs"}
VENDORED_DIR = Path(__file__).resolve().parent / "notices"
# The first words of the standard Apache License 2.0 text.
_APACHE_2 = "Apache License Version 2.0, January 2004"
# How deep below the package root nested license files are looked for.
_MAX_DEPTH = 3


@dataclass(frozen=True)
class Crate:
    """One third-party package in the artifact's dependency graph."""

    name: str
    version: str
    license: str
    repository: str
    root: Path
    license_file: str | None = None


@dataclass
class Notices:
    """The rendered file and what it found."""

    text: str
    crates: int
    without_files: list[str] = field(default_factory=list)


def is_notice_name(name: str) -> bool:
    """Whether a file name holds license or notice text (LICENSE-MIT, COPYING.txt, NOTICE, ...)."""
    upper = name.upper()
    return any(upper == s or upper.startswith((s + "-", s + ".", s + "_")) for s in _STEMS)


def notice_files(crate: Crate) -> list[Path]:
    """The license and notice files of one crate's package, root first, then nested, sorted."""
    found: set[Path] = set()
    if crate.license_file:
        named = crate.root / crate.license_file
        if named.is_file():
            found.add(named)
    for dirpath, dirnames, filenames in os.walk(crate.root):
        rel = Path(dirpath).relative_to(crate.root)
        depth = len(rel.parts)
        in_licenses_dir = any(p.upper() in ("LICENSES", "LICENCES") for p in rel.parts)
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in _SKIP_DIRS and depth < _MAX_DEPTH)
        for f in filenames:
            if in_licenses_dir or is_notice_name(f):
                found.add(Path(dirpath) / f)
    return sorted(
        found,
        key=lambda p: (
            len(p.relative_to(crate.root).parts),
            str(p.relative_to(crate.root)),
        ),
    )


def parse_cargo_args(args: str) -> list[str]:
    """The feature flags of a build's flag string, as `cargo metadata` flags."""
    words = shlex.split(args)
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("--no-default-features", "--all-features"):
            out.append(w)
        elif w in ("--features", "-F") and i + 1 < len(words):
            out += ["--features", words[i + 1]]
            i += 1
        elif w.startswith("--features="):
            out += ["--features", w.split("=", 1)[1]]
        i += 1
    return out


def cargo_metadata(manifest: Path, feature_flags: list[str], target: str) -> dict:
    """`cargo metadata` for one member manifest, its features and one target triple."""
    cmd = [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--manifest-path",
        str(manifest),
        "--filter-platform",
        target,
        *feature_flags,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def linked_crates(metadata: dict, package: str) -> list[Crate]:
    """The third-party packages `package` reaches through normal dependencies."""
    packages = {p["id"]: p for p in metadata["packages"]}
    roots = [p["id"] for p in metadata["packages"] if p["name"] == package and p.get("source") is None]
    if not roots:
        raise SystemExit(f"error: no workspace package named {package!r}")
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for dep in nodes[pid]["deps"]:
            if any(k.get("kind") is None for k in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    crates = []
    for pid in seen:
        p = packages[pid]
        if p.get("source") is None:  # a workspace crate: Krishiv's own
            continue
        crates.append(
            Crate(
                name=p["name"],
                version=p["version"],
                license=p.get("license") or "(no license field)",
                repository=p.get("repository") or "",
                root=Path(p["manifest_path"]).parent,
                license_file=p.get("license_file"),
            )
        )
    return crates


def _normal(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n")) + "\n"


def render(title: str, crates: list[Crate]) -> Notices:
    """The notices file: a crate list, then every distinct license or notice text once."""
    crates = sorted(set(crates), key=lambda c: (c.name, c.version))
    texts: dict[str, tuple[str, list[str]]] = {}  # digest -> (text, ["crate version: path", ...])
    refs: dict[Crate, list[str]] = {}
    for c in crates:
        ids = []
        for f in notice_files(c):
            body = _normal(f.read_text(encoding="utf-8", errors="replace"))
            digest = hashlib.sha256(body.encode()).hexdigest()[:12]
            texts.setdefault(digest, (body, []))[1].append(f"{c.name} {c.version}: {f.relative_to(c.root)}")
            if digest not in ids:
                ids.append(digest)
        refs[c] = ids
    borrowed: dict[Crate, str] = {}
    for c in crates:
        source = SAME_TEXTS_AS.get(c.name)
        if refs[c] or source is None:
            continue
        if source == "spdx:Apache-2.0":
            apache = sorted(d for d, (body, _) in texts.items() if " ".join(body.split()[:6]) == _APACHE_2)
            if apache:
                refs[c] = apache[:1]
                borrowed[c] = "the standard Apache License 2.0 text"
        elif source.startswith("vendored:"):
            name = source.removeprefix("vendored:")
            if c.repository.rstrip("/") != VENDORED_FROM[name]:
                continue
            body = _normal((VENDORED_DIR / name).read_text(encoding="utf-8"))
            digest = hashlib.sha256(body.encode()).hexdigest()[:12]
            texts.setdefault(digest, (body, []))[1].append(
                f"{c.name} {c.version}: {name} (the project's LICENSE, kept by Krishiv)"
            )
            refs[c] = [digest]
            borrowed[c] = "the project's own LICENSE file"
        else:
            sibling = [
                o
                for o in crates
                if o.name == source and refs[o] and o.repository == c.repository and o.license == c.license
            ]
            if sibling:
                refs[c] = refs[sibling[-1]]
                borrowed[c] = f"the texts of {source}, the same project"
    without = [f"{c.name} {c.version}" for c in crates if not refs[c]]
    lines = [
        f"Third-party software in {title}",
        "",
        "Krishiv is licensed under the Apache License 2.0 (LICENSE and NOTICE in the",
        f"Krishiv distribution). {title[0].upper() + title[1:]} also contains the {len(crates)} third-party Rust",
        "crates listed below, each under its own license. After the list, every license and",
        "notice file those crates' published packages carry is reproduced once; each crate",
        "names the texts that apply to it by their [id].",
        "",
        "Crates (name version, license, repository, texts):",
        "",
    ]
    for c in crates:
        ids = ", ".join(f"[{i}]" for i in refs[c]) or "(the package carries no license file)"
        if c in borrowed:
            ids += f" (the package carries no license file: {borrowed[c]})"
        repo = f" {c.repository}" if c.repository else ""
        lines.append(f"- {c.name} {c.version}, {c.license},{repo} {ids}")
    lines += ["", "License and notice texts:", ""]
    for digest, (body, used) in sorted(texts.items(), key=lambda kv: kv[1][1][0]):
        lines.append("=" * 78)
        lines.append(f"[{digest}] from:")
        lines += [f"  {u}" for u in used]
        lines.append("-" * 78)
        lines.append(body)
    return Notices(
        text="\n".join(lines).rstrip("\n") + "\n",
        crates=len(crates),
        without_files=without,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the third-party notices file of a binary artifact.")
    ap.add_argument("--package", required=True, help="workspace package the artifact is built from")
    ap.add_argument(
        "--cargo-args",
        default="",
        help="the build's flag string; feature flags are read",
    )
    ap.add_argument(
        "--target",
        action="append",
        required=True,
        help="target triple (repeatable: union)",
    )
    ap.add_argument("--title", required=True, help='what the artifact is, e.g. "the krishiv CLI"')
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: cwd)")
    a = ap.parse_args(argv)
    meta0 = cargo_metadata(a.root / "Cargo.toml", [], a.target[0])
    members = {p["name"]: Path(p["manifest_path"]) for p in meta0["packages"] if p.get("source") is None}
    if a.package not in members:
        raise SystemExit(f"error: no workspace package named {a.package!r}")
    flags = parse_cargo_args(a.cargo_args)
    crates: list[Crate] = []
    for t in a.target:
        crates += linked_crates(cargo_metadata(members[a.package], flags, t), a.package)
    notices = render(a.title, crates)
    if notices.crates == 0:
        raise SystemExit("error: no third-party crates found; the dependency walk is broken")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(notices.text, encoding="utf-8")
    print(
        f"{a.out}: {notices.crates} crates, {len(notices.text.encode()):,} bytes",
        file=sys.stderr,
    )
    if notices.without_files:
        print(
            f"no license file in {len(notices.without_files)}: {', '.join(notices.without_files)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
