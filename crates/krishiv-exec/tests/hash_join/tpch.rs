// SPDX-License-Identifier: Apache-2.0
//! TPC-H join shapes (Q4, Q9, Q13, Q16, Q21, Q22) wired by hand over the
//! hash join, until SQL runs end to end (P2-T074 planner, P2-T080a scan).
//!
//! The data comes from a small deterministic generator that follows dbgen's
//! key relationships and value domains (customers with `custkey % 3 == 0`
//! place no orders, four suppliers per part, order status from line status,
//! the comment patterns the queries search for). Filters and aggregations
//! run in the test; the joins run in the engine with the query's join
//! types, keys and residual conditions. Expected answers come from DuckDB
//! running the official query text over the same data: regenerate with
//! `KRISHIV_TPCH_DUMP=<dir> cargo nextest run -p krishiv-exec --test hash_join --run-ignored only dump_tpch`,
//! then `uv run --project bench python crates/krishiv-exec/tests/fixtures/tpch/expected.py <dir>`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Decimal128Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use krishiv_exec::{
    ExecConfig, ExecNode, HashJoin, HashJoinNode, JoinSpec, JoinType, Predicate, QueryContext,
    Result,
};
use krishiv_memory::SpillScope;

use super::util::*;

const SF: f64 = 0.01;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi`.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + i64::try_from(self.next() % u64::try_from(hi - lo + 1).unwrap()).unwrap()
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        let last = i64::try_from(items.len()).unwrap() - 1;
        items[usize::try_from(self.range(0, last)).unwrap()]
    }

    fn day(&mut self, base: i32, lo: i64, hi: i64) -> i32 {
        base + i32::try_from(self.range(lo, hi)).unwrap()
    }
}

/// Days since 1970-01-01 of a civil date.
fn day(y: i64, m: i64, d: i64) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    i32::try_from(era * 146_097 + doe - 719_468).unwrap()
}

fn year(days: i32) -> i64 {
    (1992..2000).rev().find(|&y| days >= day(y, 1, 1)).unwrap()
}

fn civil(days: i32) -> String {
    let y = year(days);
    let m = (1..=12).rev().find(|&m| days >= day(y, m, 1)).unwrap();
    format!("{y}-{m:02}-{:02}", days - day(y, m, 1) + 1)
}

const NATIONS: [&str; 25] = [
    "ALGERIA",
    "ARGENTINA",
    "BRAZIL",
    "CANADA",
    "EGYPT",
    "ETHIOPIA",
    "FRANCE",
    "GERMANY",
    "INDIA",
    "INDONESIA",
    "IRAN",
    "IRAQ",
    "JAPAN",
    "JORDAN",
    "KENYA",
    "MOROCCO",
    "MOZAMBIQUE",
    "PERU",
    "CHINA",
    "ROMANIA",
    "SAUDI ARABIA",
    "VIETNAM",
    "RUSSIA",
    "UNITED KINGDOM",
    "UNITED STATES",
];
const COLORS: [&str; 16] = [
    "almond", "antique", "azure", "blush", "chiffon", "forest", "green", "honeydew", "ivory",
    "khaki", "lavender", "lime", "navy", "olive", "salmon", "thistle",
];
const PRIORITIES: [&str; 5] = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"];

fn int(v: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(v))
}

fn text(v: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from(v))
}

fn money(v: Vec<i64>) -> ArrayRef {
    let v = v.into_iter().map(i128::from);
    Arc::new(
        Decimal128Array::from_iter_values(v)
            .with_precision_and_scale(15, 2)
            .unwrap(),
    )
}

fn table(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), false))
        .collect();
    let arrays = cols.into_iter().map(|c| c.1).collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

/// The generated tables.
struct Tpch {
    nation: RecordBatch,
    supplier: RecordBatch,
    part: RecordBatch,
    partsupp: RecordBatch,
    customer: RecordBatch,
    orders: RecordBatch,
    lineitem: RecordBatch,
}

#[derive(Default)]
struct Lines {
    order: Vec<i64>,
    part: Vec<i64>,
    supp: Vec<i64>,
    qty: Vec<i64>,
    price: Vec<i64>,
    disc: Vec<i64>,
    ship: Vec<i32>,
    commit: Vec<i32>,
    receipt: Vec<i32>,
}

fn generate() -> Tpch {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "row counts far below i64::MAX"
    )]
    let scale = |n: f64| (n * SF).round() as i64;
    let mut r = Rng(19_920_101);
    let (ns, np, nc, no) = (scale(1e4), scale(2e5), scale(1.5e5), scale(1.5e6));
    let regions = [
        0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2, 4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1,
    ];
    let nation = table(vec![
        ("n_nationkey", int((0..25).collect())),
        (
            "n_name",
            text(NATIONS.iter().map(|&n| n.to_owned()).collect()),
        ),
        ("n_regionkey", int(regions.to_vec())),
    ]);
    let (mut sn, mut snk, mut sc) = (vec![], vec![], vec![]);
    for k in 1..=ns {
        sn.push(format!("Supplier#{k:09}"));
        snk.push(r.range(0, 24));
        sc.push(match r.range(0, 99) {
            0..=2 => "slyly Customer final pinto Complaints",
            3..=4 => "blithely Customer ironic Recommends",
            _ => "carefully regular deposits",
        });
    }
    let supplier = table(vec![
        ("s_suppkey", int((1..=ns).collect())),
        ("s_name", text(sn)),
        ("s_nationkey", int(snk)),
        (
            "s_comment",
            text(sc.into_iter().map(str::to_owned).collect()),
        ),
    ]);
    let supplier_of = |p: i64, i: i64| (p + i * (ns / 4 + (p - 1) / ns)) % ns + 1;
    let (mut names, mut pb, mut pt, mut psz, mut pr) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut psk, mut pss, mut psc) = (vec![], vec![], vec![]);
    for k in 1..=np {
        names.push(
            (0..5)
                .map(|_| r.pick(&COLORS))
                .collect::<Vec<_>>()
                .join(" "),
        );
        pb.push(format!("Brand#{}{}", r.range(1, 5), r.range(1, 5)));
        let ty = [
            r.pick(&["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]),
            r.pick(&["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]),
            r.pick(&["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]),
        ];
        pt.push(ty.join(" "));
        psz.push(i32::try_from(r.range(1, 50)).unwrap());
        pr.push(90_000 + (k / 10) % 20_001 + 100 * (k % 1000));
        for i in 0..4 {
            psk.push(k);
            pss.push(supplier_of(k, i));
            psc.push(r.range(100, 100_000));
        }
    }
    let retail = pr.clone();
    let part = table(vec![
        ("p_partkey", int((1..=np).collect())),
        ("p_name", text(names)),
        ("p_brand", text(pb)),
        ("p_type", text(pt)),
        ("p_size", Arc::new(Int32Array::from(psz))),
        ("p_retailprice", money(pr)),
    ]);
    let partsupp = table(vec![
        ("ps_partkey", int(psk)),
        ("ps_suppkey", int(pss)),
        ("ps_supplycost", money(psc)),
    ]);
    let (mut cn, mut cp, mut ca) = (vec![], vec![], vec![]);
    for _ in 1..=nc {
        let n = r.range(0, 24);
        cn.push(n);
        let (a, b, c) = (r.range(100, 999), r.range(100, 999), r.range(1000, 9999));
        cp.push(format!("{}-{a}-{b}-{c}", n + 10));
        ca.push(r.range(-99_999, 999_999));
    }
    let customer = table(vec![
        ("c_custkey", int((1..=nc).collect())),
        ("c_nationkey", int(cn)),
        ("c_phone", text(cp)),
        ("c_acctbal", money(ca)),
    ]);
    let (start, cutoff) = (day(1992, 1, 1), day(1995, 6, 17));
    let span = i64::from(day(1998, 8, 2) - 151 - start);
    let (mut oc, mut os, mut od, mut op, mut ocm) = (vec![], vec![], vec![], vec![], vec![]);
    let mut l = Lines::default();
    for k in 1..=no {
        let cust = std::iter::repeat_with(|| r.range(1, nc))
            .find(|c| c % 3 != 0)
            .unwrap();
        let date = r.day(start, 0, span);
        let (mut shipped, mut open) = (0, 0);
        for _ in 0..r.range(1, 7) {
            let p = r.range(1, np);
            let qty = r.range(1, 50);
            let ship = r.day(date, 1, 121);
            l.order.push(k);
            l.part.push(p);
            l.supp.push(supplier_of(p, r.range(0, 3)));
            l.qty.push(qty * 100);
            l.price.push(qty * retail[usize::try_from(p - 1).unwrap()]);
            l.disc.push(r.range(0, 10));
            l.ship.push(ship);
            l.commit.push(r.day(date, 30, 90));
            l.receipt.push(r.day(ship, 1, 30));
            if ship <= cutoff {
                shipped += 1;
            } else {
                open += 1;
            }
        }
        oc.push(cust);
        os.push(match (shipped, open) {
            (_, 0) => "F",
            (0, _) => "O",
            _ => "P",
        });
        od.push(date);
        op.push(r.pick(&PRIORITIES));
        ocm.push(if r.range(0, 49) == 0 {
            "quickly special packages wake requests"
        } else {
            "furiously regular accounts"
        });
    }
    let owned = |v: Vec<&str>| text(v.into_iter().map(str::to_owned).collect());
    let orders = table(vec![
        ("o_orderkey", int((1..=no).collect())),
        ("o_custkey", int(oc)),
        ("o_orderstatus", owned(os)),
        ("o_orderdate", Arc::new(Date32Array::from(od))),
        ("o_orderpriority", owned(op)),
        ("o_comment", owned(ocm)),
    ]);
    let lineitem = table(vec![
        ("l_orderkey", int(l.order)),
        ("l_partkey", int(l.part)),
        ("l_suppkey", int(l.supp)),
        ("l_quantity", money(l.qty)),
        ("l_extendedprice", money(l.price)),
        ("l_discount", money(l.disc)),
        ("l_shipdate", Arc::new(Date32Array::from(l.ship))),
        ("l_commitdate", Arc::new(Date32Array::from(l.commit))),
        ("l_receiptdate", Arc::new(Date32Array::from(l.receipt))),
    ]);
    Tpch {
        nation,
        supplier,
        part,
        partsupp,
        customer,
        orders,
        lineitem,
    }
}

fn col<'a>(b: &'a RecordBatch, name: &str) -> &'a ArrayRef {
    b.column(b.schema().index_of(name).unwrap())
}

fn ints<'a>(b: &'a RecordBatch, name: &str) -> &'a Int64Array {
    col(b, name).as_primitive::<Int64Type>()
}

fn strs<'a>(b: &'a RecordBatch, name: &str) -> &'a StringArray {
    col(b, name).as_string::<i32>()
}

fn dates<'a>(b: &'a RecordBatch, name: &str) -> &'a Date32Array {
    col(b, name).as_primitive::<Date32Type>()
}

fn cents(b: &RecordBatch, name: &str, i: usize) -> i128 {
    col(b, name).as_primitive::<Decimal128Type>().value(i)
}

/// The rows of `b` where `keep`, with the columns `cols`.
fn select(b: &RecordBatch, cols: &[&str], keep: impl Fn(usize) -> bool) -> RecordBatch {
    let mask = BooleanArray::from((0..b.num_rows()).map(keep).collect::<Vec<_>>());
    let idx: Vec<usize> = cols
        .iter()
        .map(|c| b.schema().index_of(c).unwrap())
        .collect();
    arrow_select::filter::filter_record_batch(&b.project(&idx).unwrap(), &mask).unwrap()
}

/// A plan with its output schema.
struct Node(Arc<dyn ExecNode>, Schema);

/// A scan of `b` in 1,000-row batches.
fn scan_of(b: &RecordBatch) -> Node {
    Node(
        scan(&b.schema(), batches(b, 1000)),
        b.schema().as_ref().clone(),
    )
}

/// `probe` joined with a scan of `build` on the named keys.
fn hj(
    join_type: JoinType,
    probe: Node,
    probe_keys: &[&str],
    build: &RecordBatch,
    build_keys: &[&str],
    filter: Option<fn(&RecordBatch) -> BooleanArray>,
    spill: &SpillScope,
) -> Node {
    let keys = |s: &Schema, k: &[&str]| k.iter().map(|c| s.index_of(c).unwrap()).collect();
    let spec = JoinSpec {
        join_type,
        probe_schema: Arc::new(probe.1.clone()),
        build_schema: build.schema(),
        probe_keys: keys(&probe.1, probe_keys),
        build_keys: keys(&build.schema(), build_keys),
        null_safe: vec![false; probe_keys.len()],
        filter: filter.map(|f| Arc::new(Cond(f)) as Arc<dyn Predicate>),
        spill: Some(spill.clone()),
    };
    let join = HashJoin::try_new(spec).unwrap();
    let schema = join.schema().as_ref().clone();
    let node = HashJoinNode {
        join,
        build: scan_of(build).0,
        probe: probe.0,
    };
    Node(Arc::new(node), schema)
}

/// A residual condition from a function of the candidate batch.
struct Cond(fn(&RecordBatch) -> BooleanArray);

impl std::fmt::Debug for Cond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Cond")
    }
}

impl Predicate for Cond {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        Ok((self.0)(batch))
    }
}

/// `l1.l_suppkey <> l2.l_suppkey`: columns 1 and 6 of the Q21 candidates
/// (l1 ⋈ supplier has five columns, then l2's `l_orderkey`, `l_suppkey`).
fn other_supplier(b: &RecordBatch) -> BooleanArray {
    let (x, y) = (
        b.column(1).as_primitive::<Int64Type>(),
        b.column(6).as_primitive::<Int64Type>(),
    );
    x.iter()
        .zip(y.iter())
        .map(|(x, y)| Some(x? != y?))
        .collect()
}

/// `s LIKE '%a%b%'`.
fn like(s: &str, a: &str, b: &str) -> bool {
    s.find(a).is_some_and(|i| s[i + a.len()..].contains(b))
}

fn decimal(v: i128, scale: u32) -> String {
    let p = 10i128.pow(scale);
    let sign = if v < 0 { "-" } else { "" };
    let w = scale as usize;
    format!("{sign}{}.{:0w$}", v.abs() / p, v.abs() % p)
}

/// Runs a plan under a `limit`-byte budget; its rows as one batch.
async fn exec(plan: Node, limit: usize) -> (RecordBatch, Arc<QueryContext>) {
    let root = root(limit);
    let q = query(&root, 1, ExecConfig::default());
    let out = run(&runner(4), plan.0, &q).await.unwrap();
    assert_released(&root);
    let data: Vec<RecordBatch> = out.iter().map(|b| b.data().clone()).collect();
    let all = arrow_select::concat::concat_batches(&Arc::new(plan.1), &data).unwrap();
    (all, q)
}

fn expected(q: &str) -> Vec<String> {
    let path = format!("{}/tests/fixtures/tpch/{q}.txt", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

async fn q4(t: &Tpch, s: &SpillScope) -> Vec<String> {
    let (lo, hi) = (day(1993, 7, 1), day(1993, 10, 1));
    let od = dates(&t.orders, "o_orderdate");
    let orders = select(&t.orders, &["o_orderkey", "o_orderpriority"], |i| {
        (lo..hi).contains(&od.value(i))
    });
    let (c, r) = (
        dates(&t.lineitem, "l_commitdate"),
        dates(&t.lineitem, "l_receiptdate"),
    );
    let late = select(&t.lineitem, &["l_orderkey"], |i| c.value(i) < r.value(i));
    let plan = hj(
        JoinType::LeftSemi,
        scan_of(&orders),
        &["o_orderkey"],
        &late,
        &["l_orderkey"],
        None,
        s,
    );
    let out = exec(plan, 256 * MIB).await.0;
    let mut counts = BTreeMap::<String, u64>::new();
    for p in strs(&out, "o_orderpriority").iter().flatten() {
        *counts.entry(p.to_owned()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(p, n)| format!("{p}|{n}"))
        .collect()
}

fn not_special(b: &RecordBatch) -> BooleanArray {
    let c = b.column(3).as_string::<i32>();
    c.iter()
        .map(|c| Some(!like(c?, "special", "requests")))
        .collect()
}

async fn q13(t: &Tpch, s: &SpillScope) -> Vec<String> {
    let cust = select(&t.customer, &["c_custkey"], |_| true);
    let orders = select(&t.orders, &["o_orderkey", "o_custkey", "o_comment"], |_| {
        true
    });
    let plan = hj(
        JoinType::LeftOuter,
        scan_of(&cust),
        &["c_custkey"],
        &orders,
        &["o_custkey"],
        Some(not_special),
        s,
    );
    let out = exec(plan, 256 * MIB).await.0;
    let mut per = HashMap::<i64, u64>::new();
    let (ck, ok) = (ints(&out, "c_custkey"), ints(&out, "o_orderkey"));
    for i in 0..out.num_rows() {
        *per.entry(ck.value(i)).or_default() += u64::from(ok.is_valid(i));
    }
    let mut dist = BTreeMap::<u64, u64>::new();
    for n in per.into_values() {
        *dist.entry(n).or_default() += 1;
    }
    let mut rows: Vec<(u64, u64)> = dist.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    rows.into_iter().map(|(c, n)| format!("{c}|{n}")).collect()
}

async fn q16(t: &Tpch, s: &SpillScope) -> Vec<String> {
    let (b, ty) = (strs(&t.part, "p_brand"), strs(&t.part, "p_type"));
    let sz = col(&t.part, "p_size").as_primitive::<Int32Type>();
    let part = select(
        &t.part,
        &["p_partkey", "p_brand", "p_type", "p_size"],
        |i| {
            b.value(i) != "Brand#45"
                && !ty.value(i).starts_with("MEDIUM POLISHED")
                && [49, 14, 23, 45, 19, 3, 36, 9].contains(&sz.value(i))
        },
    );
    let sc = strs(&t.supplier, "s_comment");
    let bad = select(&t.supplier, &["s_suppkey"], |i| {
        like(sc.value(i), "Customer", "Complaints")
    });
    let ps = select(&t.partsupp, &["ps_partkey", "ps_suppkey"], |_| true);
    let j1 = hj(
        JoinType::Inner,
        scan_of(&ps),
        &["ps_partkey"],
        &part,
        &["p_partkey"],
        None,
        s,
    );
    let plan = hj(
        JoinType::NullAwareAnti,
        j1,
        &["ps_suppkey"],
        &bad,
        &["s_suppkey"],
        None,
        s,
    );
    let out = exec(plan, 256 * MIB).await.0;
    let mut groups = BTreeMap::<(String, String, i32), BTreeSet<i64>>::new();
    let (sk, br, tp) = (
        ints(&out, "ps_suppkey"),
        strs(&out, "p_brand"),
        strs(&out, "p_type"),
    );
    let si = col(&out, "p_size").as_primitive::<Int32Type>();
    for i in 0..out.num_rows() {
        let key = (br.value(i).to_owned(), tp.value(i).to_owned(), si.value(i));
        groups.entry(key).or_default().insert(sk.value(i));
    }
    let mut rows: Vec<_> = groups.into_iter().map(|(k, v)| (k, v.len())).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    rows.into_iter()
        .map(|((b, t, s), n)| format!("{b}|{t}|{s}|{n}"))
        .collect()
}

async fn q21(t: &Tpch, s: &SpillScope) -> Vec<String> {
    let (c, r) = (
        dates(&t.lineitem, "l_commitdate"),
        dates(&t.lineitem, "l_receiptdate"),
    );
    let late = select(&t.lineitem, &["l_orderkey", "l_suppkey"], |i| {
        r.value(i) > c.value(i)
    });
    let all = select(&t.lineitem, &["l_orderkey", "l_suppkey"], |_| true);
    let supp = select(&t.supplier, &["s_suppkey", "s_name", "s_nationkey"], |_| {
        true
    });
    let nn = strs(&t.nation, "n_name");
    let saudi = select(&t.nation, &["n_nationkey"], |i| {
        nn.value(i) == "SAUDI ARABIA"
    });
    let st = strs(&t.orders, "o_orderstatus");
    let fin = select(&t.orders, &["o_orderkey"], |i| st.value(i) == "F");
    let j1 = hj(
        JoinType::Inner,
        scan_of(&late),
        &["l_suppkey"],
        &supp,
        &["s_suppkey"],
        None,
        s,
    );
    let j2 = hj(
        JoinType::LeftSemi,
        j1,
        &["s_nationkey"],
        &saudi,
        &["n_nationkey"],
        None,
        s,
    );
    let j3 = hj(
        JoinType::LeftSemi,
        j2,
        &["l_orderkey"],
        &fin,
        &["o_orderkey"],
        None,
        s,
    );
    let j4 = hj(
        JoinType::LeftSemi,
        j3,
        &["l_orderkey"],
        &all,
        &["l_orderkey"],
        Some(other_supplier),
        s,
    );
    let plan = hj(
        JoinType::LeftAnti,
        j4,
        &["l_orderkey"],
        &late,
        &["l_orderkey"],
        Some(other_supplier),
        s,
    );
    let out = exec(plan, 256 * MIB).await.0;
    let mut per = BTreeMap::<String, u64>::new();
    for n in strs(&out, "s_name").iter().flatten() {
        *per.entry(n.to_owned()).or_default() += 1;
    }
    let mut rows: Vec<_> = per.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    rows.into_iter()
        .take(100)
        .map(|(n, c)| format!("{n}|{c}"))
        .collect()
}

async fn q22(t: &Tpch, s: &SpillScope) -> Vec<String> {
    const CODES: [&str; 7] = ["13", "31", "23", "29", "30", "18", "17"];
    let ph = strs(&t.customer, "c_phone");
    let code = |i: usize| CODES.contains(&&ph.value(i)[..2]);
    let bal = |i: usize| cents(&t.customer, "c_acctbal", i);
    let positive: Vec<i128> = (0..t.customer.num_rows())
        .filter(|&i| code(i) && bal(i) > 0)
        .map(bal)
        .collect();
    let (sum, n) = (
        positive.iter().sum::<i128>(),
        i128::try_from(positive.len()).unwrap(),
    );
    let rich = select(&t.customer, &["c_custkey", "c_phone", "c_acctbal"], |i| {
        code(i) && bal(i) * n > sum
    });
    let orders = select(&t.orders, &["o_custkey"], |_| true);
    let plan = hj(
        JoinType::LeftAnti,
        scan_of(&rich),
        &["c_custkey"],
        &orders,
        &["o_custkey"],
        None,
        s,
    );
    let out = exec(plan, 256 * MIB).await.0;
    let mut per = BTreeMap::<String, (u64, i128)>::new();
    for i in 0..out.num_rows() {
        let e = per
            .entry(strs(&out, "c_phone").value(i)[..2].to_owned())
            .or_default();
        e.0 += 1;
        e.1 += cents(&out, "c_acctbal", i);
    }
    per.into_iter()
        .map(|(c, (n, s))| format!("{c}|{n}|{}", decimal(s, 2)))
        .collect()
}

/// Q9: a six-way join, one on a two-column key.
async fn q9(t: &Tpch, s: &SpillScope, limit: usize) -> (Vec<String>, Arc<QueryContext>) {
    let names = strs(&t.part, "p_name");
    let green = select(&t.part, &["p_partkey"], |i| {
        names.value(i).contains("green")
    });
    let cols = [
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
    ];
    let li = select(&t.lineitem, &cols, |_| true);
    let supp = select(&t.supplier, &["s_suppkey", "s_nationkey"], |_| true);
    let ps = select(
        &t.partsupp,
        &["ps_partkey", "ps_suppkey", "ps_supplycost"],
        |_| true,
    );
    let ord = select(&t.orders, &["o_orderkey", "o_orderdate"], |_| true);
    let nat = select(&t.nation, &["n_nationkey", "n_name"], |_| true);
    let j1 = hj(
        JoinType::Inner,
        scan_of(&li),
        &["l_partkey"],
        &green,
        &["p_partkey"],
        None,
        s,
    );
    let j2 = hj(
        JoinType::Inner,
        j1,
        &["l_suppkey"],
        &supp,
        &["s_suppkey"],
        None,
        s,
    );
    let pk = ["ps_partkey", "ps_suppkey"];
    let j3 = hj(
        JoinType::Inner,
        j2,
        &["l_partkey", "l_suppkey"],
        &ps,
        &pk,
        None,
        s,
    );
    let j4 = hj(
        JoinType::Inner,
        j3,
        &["l_orderkey"],
        &ord,
        &["o_orderkey"],
        None,
        s,
    );
    let plan = hj(
        JoinType::Inner,
        j4,
        &["s_nationkey"],
        &nat,
        &["n_nationkey"],
        None,
        s,
    );
    let (out, q) = exec(plan, limit).await;
    let mut per = BTreeMap::<(String, std::cmp::Reverse<i64>), i128>::new();
    for i in 0..out.num_rows() {
        let amount = cents(&out, "l_extendedprice", i) * (100 - cents(&out, "l_discount", i))
            - cents(&out, "ps_supplycost", i) * cents(&out, "l_quantity", i);
        let y = year(dates(&out, "o_orderdate").value(i));
        let key = (
            strs(&out, "n_name").value(i).to_owned(),
            std::cmp::Reverse(y),
        );
        *per.entry(key).or_default() += amount;
    }
    let rows = per
        .into_iter()
        .map(|((n, y), a)| format!("{n}|{}|{}", y.0, decimal(a, 4)));
    (rows.collect(), q)
}

/// Q4, Q13, Q16, Q21 and Q22 match DuckDB on the same data.
#[tokio::test]
async fn tpch_join_shapes_match_duckdb() {
    let t = generate();
    let (_m, s) = spill();
    assert_eq!(q4(&t, &s).await, expected("q04"));
    assert_eq!(q13(&t, &s).await, expected("q13"));
    assert_eq!(q16(&t, &s).await, expected("q16"));
    assert_eq!(q21(&t, &s).await, expected("q21"));
    assert_eq!(q22(&t, &s).await, expected("q22"));
}

/// Q9 matches DuckDB in memory and under a budget that forces the builds
/// to spill; the `EXPLAIN ANALYZE` metrics record the spill.
#[tokio::test]
async fn tpch_q9_completes_under_a_tiny_budget_with_spill_recorded() {
    let t = generate();
    let (_m, s) = spill();
    let (rows, _) = q9(&t, &s, 256 * MIB).await;
    assert_eq!(rows, expected("q09"));
    let (rows, q) = q9(&t, &s, 3 * MIB).await;
    assert_eq!(rows, expected("q09"));
    assert!(metric(&q, "HashJoinBuild", "spilled_partitions") > 0);
    assert!(metric(&q, "HashJoinBuild", "spill_file_bytes") > 0);
    assert!(metric(&q, "HashJoinProbe", "grace_partitions") > 0);
    assert_eq!(spill_files(&s), 0);
}

/// Writes the generated tables as `|`-separated files for DuckDB.
#[test]
#[ignore = "regenerates the DuckDB input; see the module docs"]
fn dump_tpch() {
    let dir = std::path::PathBuf::from(std::env::var("KRISHIV_TPCH_DUMP").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let t = generate();
    let tables = [
        ("nation", &t.nation),
        ("supplier", &t.supplier),
        ("part", &t.part),
        ("partsupp", &t.partsupp),
        ("customer", &t.customer),
        ("orders", &t.orders),
        ("lineitem", &t.lineitem),
    ];
    for (name, b) in tables {
        let mut out = String::new();
        for i in 0..b.num_rows() {
            let row: Vec<String> = b
                .columns()
                .iter()
                .map(|c| match c.data_type() {
                    DataType::Date32 => civil(c.as_primitive::<Date32Type>().value(i)),
                    _ => cell(c, i),
                })
                .collect();
            writeln!(out, "{}", row.join("|")).unwrap();
        }
        std::fs::write(dir.join(format!("{name}.tbl")), out).unwrap();
    }
}
