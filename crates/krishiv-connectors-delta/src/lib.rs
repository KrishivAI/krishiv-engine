// SPDX-License-Identifier: Apache-2.0
//! Delta Lake connector (layer L5): Krishiv's own `Engine` for `delta_kernel` 0.30.1 over Arrow 60
//! (design `P3-T021-delta.md`, T021a). The kernel reads the log; Krishiv reads the data.

#![forbid(unsafe_code)]

pub mod error;
