// SPDX-License-Identifier: Apache-2.0
//! The serialize-through mode (`krishiv.exec.debug.planRoundTrip`, P2-T173,
//! design P2-T171 section 8.1): with it on, every query's physical plan is
//! encoded to `krishiv.plan.v1` and decoded before it runs, as a worker
//! would receive it.

use krishiv_common::config::{Config, ConfigValue, EXEC_DEBUG_PLAN_ROUND_TRIP};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::physical::PhysicalPlan;

/// `plan` as a worker receives it in the serialize-through mode: encoded
/// and decoded again, or `None` when the mode is off.
///
/// # Errors
/// `INTERNAL_ERROR` when the format cannot carry the plan or it decodes to
/// a different plan.
pub(crate) fn round_trip(
    plan: &PhysicalPlan,
    config: &Config,
) -> Result<Option<PhysicalPlan>, KrishivError> {
    let on = config
        .get(&EXEC_DEBUG_PLAN_ROUND_TRIP)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(false);
    if !on {
        return Ok(None);
    }
    let fail = |what: String| -> KrishivError {
        InternalError {
            message: format!("plan round trip: {what}"),
        }
        .into()
    };
    let bytes = krishiv_plan_codec::encode(plan).map_err(|e| fail(e.to_string()))?;
    let back = krishiv_plan_codec::decode(&bytes).map_err(|e| fail(e.to_string()))?;
    if back != *plan {
        return Err(fail(
            "the decoded plan differs from the encoded one".to_owned(),
        ));
    }
    Ok(Some(back))
}
