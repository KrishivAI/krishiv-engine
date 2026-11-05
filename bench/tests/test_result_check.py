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


def test_floats_moved_to_other_rows_differ():
    base = table(k=[1, 2], m=[10.0, 20.0])
    assert same(checksum(base), checksum(table(k=[2, 1], m=[20.0, 10.0])))  # the same rows in another order
    assert not same(checksum(base), checksum(table(k=[1, 2], m=[20.0, 10.0])))  # measures on the wrong keys
    # last-bit noise in each measure is still tolerated
    assert same(checksum(base), checksum(table(k=[1, 2], m=[10.0 * (1 + 1e-12), 20.0 * (1 - 1e-12)])))


def test_a_result_of_floats_only_is_still_summed():
    assert same(checksum(table(m=[1.5, 2.5])), checksum(table(m=[2.5, 1.5])))
    assert not same(checksum(table(m=[1.5, 2.5])), checksum(table(m=[1.5, 2.6])))


def test_null_and_nan_float_sums_compare_as_equal_to_themselves_only():
    nulls = table(k=[1, 2], m=[None, None], s=["a", "b"])
    assert same(checksum(nulls), checksum(table(k=[2, 1], m=[None, None], s=["b", "a"])))
    assert not same(checksum(nulls), checksum(table(k=[1, 2], m=[1.0, None], s=["a", "b"])))
    nan = table(k=[1], m=[float("nan")])
    assert same(checksum(nan), checksum(table(k=[1], m=[float("nan")])))
    assert not same(checksum(nan), checksum(table(k=[1], m=[1.0])))
