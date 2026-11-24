// SPDX-License-Identifier: Apache-2.0
//! Throwaway MODES-2 probe (not committed).
#![cfg(test)]
use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};
use std::time::Instant;

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "probe"]
fn probe() {
    let data = std::env::var("TPCDS").unwrap();
    let qdir = std::env::var("QDIR").unwrap();
    let qs: Vec<String> = std::env::var("QS")
        .unwrap()
        .split(',')
        .map(str::to_owned)
        .collect();
    let s = Session::new().unwrap();
    for e in std::fs::read_dir(&data).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_stem().unwrap().to_str().unwrap().to_owned();
        block_on(async {
            s.sql(&format!(
                "CREATE TEMPORARY VIEW {name} USING parquet OPTIONS (path '{}')",
                p.display()
            ))
            .await?
            .collect()
            .await
        })
        .unwrap();
    }
    println!("q\tchars\tparse_ms\tsql_ms\tplan_ms\tplanning_ns_ms\twall_ms");
    for q in qs {
        let text = std::fs::read_to_string(format!("{qdir}/{q}.sql")).unwrap();
        let text = text.trim().trim_end_matches(';').to_owned();
        let (mut p, mut a, mut x) = (vec![], vec![], vec![]);
        let n: usize = std::env::var("N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7);
        for i in 0..n {
            let t = Instant::now();
            krishiv_sql::parse_statement(&text, &Default::default()).unwrap();
            let tp = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let df = block_on(s.sql(&text)).unwrap();
            let ta = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            block_on(df.explain(ExplainMode::Simple)).unwrap();
            let tx = t.elapsed().as_secs_f64() * 1e3;
            if i > 1 {
                p.push(tp);
                a.push(ta);
                x.push(tx);
            }
        }
        // One real run, read back from system.queries.
        block_on(async { s.sql(&text).await?.collect().await }).unwrap();
        let b = block_on(async { s.sql("SELECT CAST(planning_ns AS DOUBLE)/1e6, CAST(wall_ns AS DOUBLE)/1e6 FROM system.queries WHERE planning_ns IS NOT NULL AND query_text NOT LIKE '%system.queries%' ORDER BY submitted_at DESC LIMIT 1").await?.collect().await }).unwrap();
        use arrow_array::cast::AsArray;
        let pl = b[0]
            .column(0)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0);
        let wl = b[0]
            .column(1)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0);
        println!(
            "{q}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
            text.len(),
            med(p),
            med(a),
            med(x),
            pl,
            wl
        );
    }
}
