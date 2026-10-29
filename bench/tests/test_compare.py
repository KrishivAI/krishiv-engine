import datetime as dt
from decimal import Decimal as D

from krishiv_bench import compare as c


def test_scalars_exact_and_tolerant():
    assert c.values_equal(1, 1)
    assert not c.values_equal(1, 2)
    assert c.values_equal("a ", "a")
    assert not c.values_equal("a", "b")
    assert c.values_equal(dt.date(1995, 1, 1), "1995-01-01")
    # same-scale decimals must be exact: one lost lineitem (~1e3) in a 5e10 sum must fail
    assert not c.values_equal(D("56586554400.73"), D("56586553400.73"))
    assert not c.values_equal(D("1.00"), D("1.01"))
    # Spark AVG scale 6 vs wider exact value / double
    assert c.values_equal(D("0.049985"), D("0.049985295838397614"))
    assert c.values_equal(D("0.049985"), 0.049985295838397614)
    assert not c.values_equal(D("0.049985"), 0.05)
    # TPC-DS q66: Spark (and Krishiv) sum quotients rounded to scale 6, DuckDB sums doubles.
    assert c.values_equal(D("11.694972"), 11.694972872414953)
    assert not c.values_equal(D("11.694960"), 11.694972872414953)
    assert c.values_equal(1.0, 1.0000001)
    assert not c.values_equal(1.0, 1.001)
    assert c.values_equal(5, D("5.00"))
    assert not c.values_equal(5, D("5.01"))


def test_rows_order_and_multiset():
    a, b = [(1, "x"), (2, "y")], [(2, "y"), (1, "x")]
    ok, msg = c.rows_match(a, b, ordered=True)
    assert not ok
    assert "ORDER" in msg
    assert c.rows_match(a, b, ordered=False)[0]
    assert not c.rows_match(a, [(1, "x")], False)[0]
    assert "value mismatch" in c.rows_match(a, [(1, "x"), (3, "y")], True)[1]


def test_large_results_are_compared():
    a = [(i, str(i)) for i in range(20_000)]
    b = list(a)
    b[19_999] = (19_999, "zzz")
    assert c.rows_match(a, a, True)[0]
    assert not c.rows_match(a, b, True)[0]


def test_order_by_detection():
    assert c.has_top_level_order_by("select a from t order by a")
    assert not c.has_top_level_order_by("select a from (select a from t order by a limit 3) x")
    assert not c.has_top_level_order_by("select a from t where s = 'order by'")
    assert not c.has_top_level_order_by("select a from t")


def test_answer_match():
    assert c.match_answer(D("53758257134.8700"), "53758257134.8700")
    assert c.match_answer(D("37734107.00"), "37734107")
    assert not c.match_answer(D("37734107.30"), "37734107")
    assert c.match_answer(D("25.522006"), "25.522005853257337")
    assert c.match_answer(25.522005853257337, "25.522005853257337")
    assert c.match_answer(1478493, "1478493")
    assert not c.match_answer(1478494, "1478493")


def test_roundtrip_sidecar(tmp_path):
    rows = {"q": [(1, D("1.50"), 2.5, "s", dt.date(2020, 1, 2), None)]}
    p = tmp_path / "r.gz"
    c.save_rows(p, rows)
    back = c.load_rows(p)["q"][0]
    assert back[1] == D("1.50")
    assert back[2] == 2.5
    assert back[5] is None
    assert back[4] == "2020-01-02"


def test_sort_keys_with_implicit_aliases_and_terminator():
    # TPC-DS q71's shape: aliases without AS, ORDER BY a base column the alias hides, a final `;`.
    q71 = (
        "SELECT i_brand_id brand_id, i_brand brand, t_hour, t_minute, sum(ext_price) ext_price FROM t "
        "GROUP BY 1, 2, 3, 4 ORDER BY ext_price DESC NULLS FIRST, i_brand_id NULLS FIRST, t_hour NULLS FIRST;\n"
    )
    assert c.derive_sort_keys(q71) == [4, 0, 2]
    assert c.derive_sort_keys("SELECT a, b FROM t ORDER BY b;") == [1]
    assert c.derive_sort_keys("SELECT a AS x, count(*) n FROM t GROUP BY a ORDER BY n DESC, x") == [1, 0]


def test_implicit_alias_is_not_taken_from_expression_words():
    assert c._implicit_alias("x IS NULL") is None
    assert c._implicit_alias("CASE WHEN a THEN 1 ELSE 0 END") is None
    assert c._implicit_alias("INTERVAL 1 day") is None  # a number before the word: not an alias
    assert c._implicit_alias("a + b") is None
    assert c._implicit_alias("sum(x) total") == ("total", "sum(x)")
    assert c._implicit_alias('t.col "My Col"') == ("mycol", "t.col")
    # A doubtful item gets no name, so a key naming it is not derived (whole-row comparison).
    assert c.derive_sort_keys("SELECT INTERVAL 1 day FROM t ORDER BY day") is None


def test_implicit_alias_is_not_taken_after_an_operator_word_or_from_a_unit():
    assert c._implicit_alias("a or b") is None
    assert c._implicit_alias("not flag") is None
    assert c._implicit_alias("a like b") is None
    assert c._implicit_alias("a rlike b") is None
    assert c._implicit_alias("x in y") is None
    assert c._implicit_alias("d + interval '30' day") is None
    assert c._implicit_alias("interval '1' hours") is None
    assert c._implicit_alias("count(*) cnt") == ("cnt", "count(*)")


def test_order_by_names_an_output_alias_before_an_expression():
    # Spark: ORDER BY x is the output column x (column 0), not the base column x selected as z.
    assert c.derive_sort_keys("SELECT y x, x z FROM t ORDER BY x") == [0]
    assert c.derive_sort_keys("SELECT y AS x, x AS z FROM t ORDER BY x DESC, z") == [0, 1]
    assert c.derive_sort_keys("SELECT a x, b x FROM t ORDER BY x") is None  # ambiguous output name
    assert c.derive_sort_keys("SELECT a + 1 AS s FROM t ORDER BY a + 1") == [0]  # no such name: the expression
