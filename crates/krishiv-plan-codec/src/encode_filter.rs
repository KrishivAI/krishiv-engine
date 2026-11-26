// SPDX-License-Identifier: Apache-2.0
//! Runtime join, exchange and scan filters to their wire form (kept out of `encode.rs`, which is at the file-size cap).

use std::cell::Cell;

use krishiv_exec::physical::{ExchangeFilterPlan, JoinFilterPlan, ScanRuntimeFilter};
use krishiv_io::KeyFilterOptions;

use crate::encode::{arrow_bytes, column_path};
use crate::proto as pb;

pub(crate) fn join_filter(f: &JoinFilterPlan) -> pb::JoinFilter {
    let KeyFilterOptions {
        max_values,
        exact_max_values,
        bloom,
        scan_waits,
        bloom_max_items,
        bloom_max_bytes,
        bloom_fpp,
        bloom_max_fpp,
        bloom_bytes,
        bloom_spill_bytes,
        side,
    } = f.options;
    pb::JoinFilter {
        id: f.id.0,
        key: f.key as u64,
        bound_column: f.column.map(|c| c as u64),
        options: Some(pb::KeyFilterOptions {
            max_values: max_values as u64,
            exact_max_values: exact_max_values as u64,
            bloom,
            scan_waits,
            bloom_max_items,
            bloom_max_bytes: bloom_max_bytes as u64,
            bloom_fpp,
            bloom_max_fpp,
            bloom_bytes,
            bloom_spill_bytes,
            side: match side {
                krishiv_io::RangeSide::Both => pb::RangeSide::Both,
                krishiv_io::RangeSide::Upper => pb::RangeSide::Upper,
                krishiv_io::RangeSide::Lower => pb::RangeSide::Lower,
            } as i32,
        }),
    }
}

pub(crate) fn exchange_filter(f: &ExchangeFilterPlan, nested: &Cell<bool>) -> pb::ExchangeFilter {
    pb::ExchangeFilter {
        id: f.id.0,
        column: u64::try_from(f.column).unwrap_or(u64::MAX),
        key_type: arrow_bytes(&f.key_type, nested),
        options: join_filter(&JoinFilterPlan {
            id: f.id,
            key: 0,
            options: f.options,
            column: None,
        })
        .options,
        null_safe: f.null_safe,
    }
}

pub(crate) fn scan_filter(f: &ScanRuntimeFilter, nested: &Cell<bool>) -> pb::ScanRuntimeFilter {
    pb::ScanRuntimeFilter {
        id: f.id.0,
        column: Some(column_path(&f.column)),
        key_type: arrow_bytes(&f.key_type, nested),
    }
}
