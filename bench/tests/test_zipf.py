import numpy as np
import pyarrow.parquet as pq
import pytest

from krishiv_bench.datasets import gen_zipf, zipf_keys


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5])
def test_zipf_rank_frequency_matches_pmf(alpha):
    keys, rows = 1000, 400_000
    k = zipf_keys(rows, keys, alpha, seed=7)
    assert k.min() >= 0
    assert k.max() < keys
    counts = np.bincount(k, minlength=keys) / rows
    pmf = 1 / np.arange(1, keys + 1) ** alpha
    pmf /= pmf.sum()
    # hottest keys within 5% relative of the theoretical pmf; skew is monotone-ish
    assert counts[0] == pytest.approx(pmf[0], rel=0.05)
    assert counts[1] == pytest.approx(pmf[1], rel=0.08)
    assert counts[:10].sum() > counts[-10:].sum() * 5


def test_zipf_higher_alpha_is_more_skewed():
    def top(a):
        return np.mean(zipf_keys(100_000, 10_000, a, seed=1) == 0)

    assert top(0.8) < top(1.1) < top(1.5)


def test_zipf_deterministic_and_validates():
    assert (zipf_keys(1000, 50, 1.2, seed=3) == zipf_keys(1000, 50, 1.2, seed=3)).all()
    with pytest.raises(ValueError, match="alpha"):
        zipf_keys(10, 10, 0)


def test_gen_zipf_parquet(tmp_path):
    out = gen_zipf(25_000, 100, 1.2, out=tmp_path / "z.parquet", chunk=10_000)
    t = pq.read_table(out)
    assert t.num_rows == 25_000
    assert t.column_names[:2] == ["key", "id"]
    assert t["id"].to_pylist()[-1] == 24_999
