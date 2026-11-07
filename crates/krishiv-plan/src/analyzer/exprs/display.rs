// SPDX-License-Identifier: Apache-2.0
//! The column name Spark gives a call that is `RuntimeReplaceable`, or wraps another call: the name of
//! what it is replaced by (`make_timestamp_ltz(..)` shows as `make_timestamp(..)`), or the wrapper over
//! the call inside it (`try_to_time(x)` shows as `try_to_time(to_time(x))`). Only the shown form changes;
//! the bound call is the one that runs.

use std::sync::Arc;

use crate::{Expr, Origin};

/// The shown form of the call `name(args)` when Spark names it differently, `None` when it shows as written.
/// `args` are the shown arguments, in the order written.
pub(super) fn spark_display(name: &str, args: &[Arc<Expr>], origin: Origin) -> Option<Arc<Expr>> {
    let owned = |a: &[Arc<Expr>]| a.iter().map(|x| Expr::clone(x)).collect::<Vec<Expr>>();
    let call = |n: &str, a: Vec<Expr>| Expr::function(n, a).with_origin(origin);
    let is = |n: &str| name.eq_ignore_ascii_case(n);
    // The numeric forms of `make_timestamp_ltz` (6 and 7 arguments) show as written.
    let shown = if (is("make_timestamp_ltz") || is("try_make_timestamp_ltz")) && args.len() <= 3 {
        call("make_timestamp", owned(args))
    } else if is("try_make_timestamp_ntz") {
        call(
            "try_make_timestamp_ntz",
            vec![call("make_timestamp_ntz", owned(args))],
        )
    } else if is("try_to_time") {
        call("try_to_time", vec![call("to_time", owned(args))])
    } else if is("convert_timezone") && args.len() == 2 {
        // The source zone is written into the call: the session zone.
        let mut full = vec![call("current_timezone", Vec::new())];
        full.extend(owned(args));
        call("convert_timezone", full)
    } else {
        return None;
    };
    Some(Arc::new(shown))
}
