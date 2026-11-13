# SPDX-License-Identifier: Apache-2.0
"""The denominator per Spark version, computed in the design's order (§2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .manifest import EXCLUSION_CAP, Entry


@dataclass(frozen=True)
class Denominator:
    version: str
    collected: int
    spark_skips: int  # unconditional skips on Spark itself; outside the cap
    oracle_env: int  # fail on the Spark oracle in our environment; inside the cap
    manifest: int  # matched by a manifest entry (and not already removed above)
    denominator: int
    cap: int  # 20% of (collected - spark_skips)
    by_reason: dict[str, int]
    # Skipped on the Spark oracle at run time under the same client (e.g. `skipIf(is_remote_only())`, "Requires JVM
    # access"): the same treatment as `spark_skips`, outside the cap, reported on its own (review pass 2 N9).
    conditional_skips: int = 0

    @property
    def excluded_share(self) -> float:
        base = self.collected - self.spark_skips - self.conditional_skips
        return (self.manifest + self.oracle_env) / base if base else 0.0

    @property
    def within_cap(self) -> bool:
        return self.manifest + self.oracle_env <= self.cap

    def as_dict(self) -> dict:
        d = asdict(self)
        d["excluded_share"] = round(self.excluded_share, 4)
        d["within_cap"] = self.within_cap
        return d


def compute(
    version: str,
    collected: set[str],
    spark_skips: set[str],
    oracle_env: set[str],
    manifest: list[Entry],
    conditional_skips: set[str] | None = None,
) -> tuple[Denominator, set[str]]:
    """Return the parts and the set of test ids that make up the denominator.

    `conditional_skips`: ids the oracle skipped at run time (see `oracle_skips`); a static skip is not counted twice.
    """
    cond = (conditional_skips or set()) & collected - spark_skips
    after_skips = collected - spark_skips - cond
    after_oracle = after_skips - oracle_env
    by_reason: dict[str, int] = {}
    excluded: set[str] = set()
    for test_id in sorted(after_oracle):
        for e in manifest:
            if e.applies(version) and e.matches(test_id):
                excluded.add(test_id)
                by_reason[e.reason_class.value] = by_reason.get(e.reason_class.value, 0) + 1
                break  # first matching entry owns the id; no double counting
    if oracle_env & after_skips:
        by_reason["ORACLE_ENV"] = len(oracle_env & after_skips)
    remaining = after_oracle - excluded
    parts = Denominator(
        version=version,
        collected=len(collected),
        spark_skips=len(collected & spark_skips),
        oracle_env=len(oracle_env & after_skips),
        manifest=len(excluded),
        denominator=len(remaining),
        cap=int(EXCLUSION_CAP * len(after_skips)),
        by_reason=dict(sorted(by_reason.items())),
        conditional_skips=len(cond),
    )
    return parts, remaining
