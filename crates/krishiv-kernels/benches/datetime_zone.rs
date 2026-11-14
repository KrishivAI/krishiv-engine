// SPDX-License-Identifier: Apache-2.0
//! Session-zone datetime benchmarks (64K rows, one thread; CodSpeed in CI):
//! `date_trunc('hour', <timestamp_ntz>)` as the SQL layer runs it, the
//! `timestamp_ntz` cast to `timestamp` and then the truncation, in a region
//! zone with clock changes and in `UTC` (KN-4).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::cast::AsArray;
use arrow_array::types::TimestampMicrosecondType;
use arrow_array::{StringArray, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_kernels::cast::{CastMode, CastOptions, TimeZoneRules, cast};
use krishiv_kernels::datetime::{resolve_zone, trunc_timestamp};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: i64 = 1 << 16;
/// 2024-01-01T00:00:00 as local microseconds.
const JAN_2024: i64 = 1_704_067_200_000_000;

fn zones(c: &mut Criterion) {
    // Trip times spread over the year (a clock change in March and October).
    let ntz = TimestampMicrosecondArray::from_iter_values(
        (0..ROWS).map(|i| JAN_2024 + (i * 481_211_011) % (366 * 86_400_000_000)),
    );
    let hours = StringArray::from_iter_values((0..ROWS).map(|_| "hour"));
    let ltz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    for name in ["Europe/Stockholm", "UTC"] {
        let zone = resolve_zone(name).unwrap();
        let zone: &dyn TimeZoneRules = &zone;
        let opts = CastOptions {
            mode: CastMode::Ansi,
            time_zone: zone,
        };
        c.bench_function(&format!("ntz_to_timestamp/{name}"), |b| {
            b.iter(|| cast(black_box(&ntz), &ltz, &opts).unwrap());
        });
        c.bench_function(&format!("date_trunc_hour_ntz/{name}"), |b| {
            b.iter(|| {
                let instants = cast(black_box(&ntz), &ltz, &opts).unwrap();
                let instants = instants.as_primitive::<TimestampMicrosecondType>();
                trunc_timestamp(&hours, instants, zone, true).unwrap()
            });
        });
    }
}

criterion_group!(benches, zones);
criterion_main!(benches);
