// SPDX-License-Identifier: Apache-2.0
//! Public expression types stay unwind safe (`cargo-semver-checks` `auto_trait_impl_removed` against 0.2.0): a
//! caller may hold a `Column` or a `WindowSpec` across `catch_unwind`.

use std::panic::{RefUnwindSafe, UnwindSafe};

use krishiv_session::{Column, WindowSpec};

fn unwind_safe<T: UnwindSafe + RefUnwindSafe>() {}

#[test]
fn column_and_window_spec_are_unwind_safe() {
    unwind_safe::<Column>();
    unwind_safe::<WindowSpec>();
}
