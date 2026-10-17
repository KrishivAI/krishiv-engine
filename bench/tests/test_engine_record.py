from krishiv_bench.adapters.base import Dataset, Engine, Query
from krishiv_bench.runner import engine_record


class _Fake(Engine):
    name = "fake"

    def __init__(self, build: dict[str, str]) -> None:
        self._build = build

    def version(self) -> str:
        return "1.2.3"

    def setup(self, dataset: Dataset) -> None:
        del dataset

    def run(self, query: Query) -> list[tuple]:
        del query
        return []

    def build_info(self) -> dict[str, str]:
        return self._build


def test_allocator_recorded_when_reported():
    rec = engine_record(_Fake({"allocator": "mimalloc"}), "fake")
    assert rec["allocator"] == "mimalloc"
    assert rec["build"] == {"allocator": "mimalloc"}


def test_no_allocator_key_when_unreported():
    rec = engine_record(_Fake({}), "fake")
    assert "allocator" not in rec
