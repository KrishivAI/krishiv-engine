//! E04 host: Rust worker threads call Python UDFs over Arrow batches under four runtime options.
mod bridge;
mod runners;

use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use runners::{Runner, Spec};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub struct Batch {
    pub col: ArrayRef,
    pub expected: i64,
}

fn build_pool(body: &str, poison: bool, batch: usize, n: usize) -> Vec<Batch> {
    let mut v = Vec::new();
    for p in 0..n {
        let base = (p * batch) as i64;
        if body == "expensive" {
            let s: StringArray = (0..batch as i64).map(|i| Some(format!("id-{}-abc{}-z", base + i, (base + i) * 7))).collect();
            v.push(Batch { col: std::sync::Arc::new(s), expected: 2 * batch as i64 });
        } else {
            let mut vals: Vec<i64> = (0..batch as i64).map(|i| base + i).collect();
            if poison && p == 0 {
                vals[0] = 0; // pool batches are never poisoned; poison built separately
            }
            let exp: i64 = vals.iter().map(|x| x + 1).sum();
            v.push(Batch { col: std::sync::Arc::new(Int64Array::from(vals)), expected: exp });
        }
    }
    v
}

fn check(res: &ArrayRef, b: &Batch) -> Result<(), String> {
    let a = res.as_any().downcast_ref::<Int64Array>().ok_or("result not Int64")?;
    if a.len() != b.col.len() {
        return Err("length mismatch".into());
    }
    let s: i64 = a.values().iter().sum();
    if s != b.expected {
        return Err(format!("checksum {} != {}", s, b.expected));
    }
    Ok(())
}

fn rss_kb(pid: u32, key: &str) -> u64 {
    let s = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|x| x.parse().ok()).unwrap_or(0)
}
fn cpu_s(pid: u32) -> f64 {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let rest = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let t = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (t(11) + t(12)) / 100.0
}

fn run_iteration(runners: &mut [Box<dyn Runner>], pool: &[Batch], nb: usize) -> Result<(f64, Vec<f64>), String> {
    let ctr = AtomicUsize::new(0);
    let t0 = Instant::now();
    let res: Vec<Result<Vec<f64>, String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = runners
            .iter_mut()
            .map(|r| {
                let ctr = &ctr;
                sc.spawn(move || {
                    let mut lat = Vec::new();
                    let _pin = r.pin_thread();
                    loop {
                        let i = ctr.fetch_add(1, Ordering::Relaxed);
                        if i >= nb {
                            return Ok(lat);
                        }
                        let b = &pool[i % pool.len()];
                        let t = Instant::now();
                        let out = r.run(&b.col)?;
                        check(&out, b)?;
                        lat.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("thread panic".into()))).collect()
    });
    let wall = t0.elapsed().as_secs_f64();
    let mut lat = Vec::new();
    for r in res {
        lat.extend(r?);
    }
    Ok((wall, lat))
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() { 0.0 } else { v[((v.len() - 1) as f64 * p) as usize] }
}

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or(d.into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spec = Spec {
        option: arg(&args, "--option", "A"),
        kind: arg(&args, "--kind", "arrow"),
        body: arg(&args, "--body", "cheap"),
        python: arg(&args, "--python", ""),
        worker: arg(&args, "--worker", ""),
    };
    let threads: usize = arg(&args, "--threads", "1").parse().unwrap_or(1);
    let rows: usize = arg(&args, "--rows", "1000000").parse().unwrap_or(1_000_000);
    let iters: usize = arg(&args, "--iters", "3").parse().unwrap_or(3);
    let warm: usize = arg(&args, "--warmups", "2").parse().unwrap_or(2);
    let batch: usize = arg(&args, "--batch", "8192").parse().unwrap_or(8192);
    if args.iter().any(|a| a == "--no-pin") { runners::NO_PIN.store(true, Ordering::Relaxed); }
    let pool = build_pool(&spec.body, false, batch, 32);

    if args.iter().any(|a| a == "--selftest") {
        return runners::selftest(&spec, &pool);
    }
    if args.iter().any(|a| a == "--crashtest") {
        return crashtest(&spec, batch);
    }

    let nb = rows / batch;
    let mut rs: Vec<Box<dyn Runner>> = Vec::new();
    for _ in 0..threads {
        match runners::make(&spec) {
            Ok(r) => rs.push(r),
            Err(e) => {
                println!("{}", serde_json::json!({"status":"unsupported","option":spec.option,"kind":spec.kind,"body":spec.body,"threads":threads,"reason":e}));
                return;
            }
        }
    }
    let pids: Vec<u32> = rs.iter().flat_map(|r| r.pids()).collect();
    let host = std::process::id();
    for _ in 0..warm {
        if let Err(e) = run_iteration(&mut rs, &pool, nb) {
            println!("{}", serde_json::json!({"status":"error","reason":e,"option":spec.option,"kind":spec.kind,"body":spec.body,"threads":threads}));
            return;
        }
    }
    let mut walls = Vec::new();
    let mut lat_all = Vec::new();
    let cpu0 = cpu_s(host) + pids.iter().map(|p| cpu_s(*p)).sum::<f64>();
    for _ in 0..iters {
        match run_iteration(&mut rs, &pool, nb) {
            Ok((w, l)) => {
                walls.push(w);
                lat_all.extend(l);
            }
            Err(e) => {
                println!("{}", serde_json::json!({"status":"error","reason":e,"option":spec.option,"kind":spec.kind,"body":spec.body,"threads":threads}));
                return;
            }
        }
    }
    let cpu1 = cpu_s(host) + pids.iter().map(|p| cpu_s(*p)).sum::<f64>();
    let total_rows = (nb * batch) as f64;
    let rps: Vec<f64> = walls.iter().map(|w| total_rows / w).collect();
    let mut sorted = rps.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let wall_sum: f64 = walls.iter().sum();
    let host_hwm = rss_kb(host, "VmHWM:");
    let worker_hwm: u64 = pids.iter().map(|p| rss_kb(*p, "VmHWM:")).sum();
    let mut l = lat_all;
    println!(
        "{}",
        serde_json::json!({
            "status":"ok","option":spec.option,"kind":spec.kind,"body":spec.body,"threads":threads,
            "rows_per_iter":total_rows,"batch_rows":batch,"iters":iters,"warmups":warm,
            "rows_per_s_median":median,"rows_per_s_iters":rps,"wall_s_iters":walls,
            "batch_latency_ms_p50":pct(&mut l,0.5),"batch_latency_ms_p99":pct(&mut l,0.99),
            "cpu_s_timed":cpu1-cpu0,"cpu_cores_avg":(cpu1-cpu0)/wall_sum,
            "peak_rss_mb_host":host_hwm as f64/1024.0,"peak_rss_mb_workers_sum":worker_hwm as f64/1024.0,
            "peak_rss_mb_total":(host_hwm+worker_hwm) as f64/1024.0,"n_worker_procs":pids.len(),
        })
    );
    for r in rs.iter_mut() {
        r.shutdown();
    }
}

fn crashtest(spec: &Spec, batch: usize) {
    let mut s = spec.clone();
    s.kind = "crash".into();
    s.body = "cheap".into();
    let pool = build_pool("cheap", false, batch, 32);
    let mut r = match runners::make(&s) {
        Ok(r) => r,
        Err(e) => {
            println!("UNSUPPORTED {e}");
            return;
        }
    };
    let say = |m: &str| {
        println!("{m}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    };
    let run_good = |r: &mut Box<dyn Runner>, n: usize| -> usize {
        let mut ok = 0;
        for i in 0..n {
            let b = &pool[i % pool.len()];
            if let Ok(o) = r.run(&b.col) {
                if check(&o, b).is_ok() {
                    ok += 1;
                }
            }
        }
        ok
    };
    say(&format!("PHASE1 good_batches_ok={}", run_good(&mut r, 50)));
    let poison: ArrayRef = std::sync::Arc::new(Int64Array::from(vec![-1i64; batch]));
    say("PHASE2 sending poison batch (UDF will segfault)");
    match r.run(&poison) {
        Ok(_) => say("PHASE2 poison batch returned OK (no crash?!)"),
        Err(e) => say(&format!("PHASE2 host survived; error={e}")),
    }
    let t = Instant::now();
    match r.recover() {
        Ok(()) => say(&format!("RECOVER ok respawn_ms={:.1}", t.elapsed().as_secs_f64() * 1e3)),
        Err(e) => say(&format!("RECOVER failed {e}")),
    }
    say(&format!("PHASE3 good_batches_ok={}", run_good(&mut r, 50)));
    r.shutdown();
}
