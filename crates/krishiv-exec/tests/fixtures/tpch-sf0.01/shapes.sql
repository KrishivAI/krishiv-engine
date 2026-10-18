-- SPDX-License-Identifier: Apache-2.0
-- TPC-H SF 0.01 (DuckDB `dbgen`, exported to Parquet under /w/data) run on
-- Spark 4.2.0 (`spark-sql -f`): writes each aggregation's input (the plan
-- below the aggregate) and the expected results. The inputs were converted
-- to Arrow IPC files (zstd, 4,096-row batches) for tests/aggregate_tpch.rs.
create temporary view lineitem using parquet options (path '/w/data/lineitem.parquet');
create temporary view orders using parquet options (path '/w/data/orders.parquet');
create temporary view customer using parquet options (path '/w/data/customer.parquet');
create temporary view part using parquet options (path '/w/data/part.parquet');
create temporary view partsupp using parquet options (path '/w/data/partsupp.parquet');
create temporary view supplier using parquet options (path '/w/data/supplier.parquet');
-- Q1: the aggregate's input (the scan, filter and projections under it).
insert overwrite directory '/w/out/q1_input' using parquet
select l_returnflag, l_linestatus, l_quantity, l_extendedprice,
  l_extendedprice * (1 - l_discount) as disc_price,
  l_extendedprice * (1 - l_discount) * (1 + l_tax) as charge, l_discount
from lineitem where l_shipdate <= date '1998-12-01' - interval '90' day;
insert overwrite directory '/w/out/q1_expected' using csv
select l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice),
  sum(l_extendedprice * (1 - l_discount)), sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)),
  avg(l_quantity), avg(l_extendedprice), avg(l_discount), count(*)
from lineitem where l_shipdate <= date '1998-12-01' - interval '90' day
group by l_returnflag, l_linestatus;
-- Q13: the inner aggregate's input (left outer join).
insert overwrite directory '/w/out/q13_input' using parquet
select c_custkey, o_orderkey from customer left outer join orders
  on c_custkey = o_custkey and o_comment not like '%special%requests%';
insert overwrite directory '/w/out/q13_expected' using csv
select c_count, count(*) as custdist from (
  select c_custkey, count(o_orderkey) as c_count from customer left outer join orders
    on c_custkey = o_custkey and o_comment not like '%special%requests%'
  group by c_custkey) as c_orders
group by c_count;
-- Q18: the IN-subquery aggregate's input, and the outer aggregate's input.
insert overwrite directory '/w/out/q18_lineitem' using parquet
select l_orderkey, l_quantity from lineitem;
insert overwrite directory '/w/out/q18_input' using parquet
select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, l_quantity
from customer, orders, lineitem
where o_orderkey in (select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300)
  and c_custkey = o_custkey and o_orderkey = l_orderkey;
insert overwrite directory '/w/out/q18_big_orders' using csv
select l_orderkey, sum(l_quantity) from lineitem group by l_orderkey having sum(l_quantity) > 300;
insert overwrite directory '/w/out/q18_expected' using csv
select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
from customer, orders, lineitem
where o_orderkey in (select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300)
  and c_custkey = o_custkey and o_orderkey = l_orderkey
group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice;
-- Q16: the aggregate's input.
insert overwrite directory '/w/out/q16_input' using parquet
select p_brand, p_type, p_size, ps_suppkey from partsupp, part
where p_partkey = ps_partkey and p_brand <> 'Brand#45' and p_type not like 'MEDIUM POLISHED%'
  and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
  and ps_suppkey not in (select s_suppkey from supplier where s_comment like '%Customer%Complaints%');
insert overwrite directory '/w/out/q16_expected' using csv
select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt from partsupp, part
where p_partkey = ps_partkey and p_brand <> 'Brand#45' and p_type not like 'MEDIUM POLISHED%'
  and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
  and ps_suppkey not in (select s_suppkey from supplier where s_comment like '%Customer%Complaints%')
group by p_brand, p_type, p_size;
