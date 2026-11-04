from __future__ import annotations

import pyarrow as pa

from krishiv_bench.result_check import checksum, same


def table(**cols) -> pa.Table:
    return pa.table(cols)


def test_row_order_and_column_names_do_not_matter():
    a = table(x=[1, 2, 3], s=["a", "b", None], f=[0.5, 1.5, 2.5])
    b = table(p=[3, 1, 2], q=[None, "a", "b"], r=[2.5, 0.5, 1.5])
    assert same(checksum(a), checksum(b))


def test_a_changed_value_a_dropped_row_and_a_swapped_pairing_differ():
    base = table(x=[1, 2, 3], s=["a", "b", "c"])
    assert not same(checksum(base), checksum(table(x=[1, 2, 4], s=["a", "b", "c"])))
    assert not same(checksum(base), checksum(table(x=[1, 2], s=["a", "b"])))
    # same column multisets, different rows: only the row hash tells
    assert not same(checksum(base), checksum(table(x=[1, 2, 3], s=["b", "a", "c"])))


def test_float_sums_tolerate_last_bit_noise_only():
    a = table(f=[0.1, 0.2, 0.3])
    assert same(checksum(a), checksum(table(f=[0.3, 0.2, 0.1 + 1e-17])))
    assert not same(checksum(a), checksum(table(f=[0.1, 0.2, 0.31])))


def test_decimals_dates_and_nulls():
    import datetime
    import decimal

    t = pa.table(
        {
            "d": pa.array([decimal.Decimal("1.50"), None], pa.decimal128(5, 2)),
            "t": pa.array([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)]),
        }
    )
    assert same(checksum(t), checksum(t))
    assert not same(checksum(t), checksum(t.slice(0, 1)))
