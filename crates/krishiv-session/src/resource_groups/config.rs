// SPDX-License-Identifier: Apache-2.0
//! The resource group file: Trino's `resource-groups.json` shape, restricted to the subset
//! `design/P3-T003-workload-management.md` §3 lists. A key outside the subset is refused at
//! load, naming the key, so a later version that supports it cannot change an old file.

use krishiv_common::error::{KrishivError, KrishivResourceGroupConfigInvalid};
use krishiv_exec::WorkloadClass;
use krishiv_sql::StatementClass;
use regex_lite::Regex;
use serde_json::{Map, Value};

/// How a parent picks the subgroup to dispatch from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Policy {
    WeightedFair,
    QueryPriority,
}

/// A scan quota: bytes per fixed window.
#[derive(Debug, Clone, Copy)]
pub(super) struct ScanQuota {
    pub(super) bytes: u64,
    pub(super) period_ms: u64,
}

/// One group of the tree, with its subgroups.
#[derive(Debug)]
pub(super) struct GroupCfg {
    pub(super) name: String,
    pub(super) hard_concurrency: usize,
    pub(super) max_queued: usize,
    pub(super) policy: Policy,
    pub(super) weight: u64,
    pub(super) soft_memory: Option<usize>,
    pub(super) per_query_memory: Option<usize>,
    pub(super) priority: Option<u8>,
    pub(super) class: Option<WorkloadClass>,
    pub(super) scan_quota: Option<ScanQuota>,
    pub(super) children: Vec<GroupCfg>,
}

/// One selector; every field that is set must match.
#[derive(Debug)]
pub(super) struct SelectorCfg {
    pub(super) user: Option<Regex>,
    pub(super) user_group: Option<Regex>,
    pub(super) source: Option<Regex>,
    pub(super) client_tags: Vec<String>,
    pub(super) query_type: Option<StatementClass>,
    pub(super) group: String,
}

/// The parsed file.
#[derive(Debug)]
pub(super) struct Config {
    pub(super) roots: Vec<GroupCfg>,
    pub(super) selectors: Vec<SelectorCfg>,
    pub(super) default_group: String,
}

fn invalid(path: &str, reason: impl Into<String>) -> KrishivError {
    KrishivResourceGroupConfigInvalid {
        path: path.to_owned(),
        reason: reason.into(),
    }
    .into()
}

/// Parses the group file; `total_memory` is what a `"NN%"` memory limit is a share of.
pub(super) fn parse(json: &str, total_memory: usize) -> Result<Config, KrishivError> {
    let doc: Value = serde_json::from_str(json).map_err(|e| invalid("$", e.to_string()))?;
    let obj = doc
        .as_object()
        .ok_or_else(|| invalid("$", "expected an object"))?;
    only_keys(obj, &["rootGroups", "selectors", "defaultGroup"], &[], "$")?;
    let roots_json = obj
        .get("rootGroups")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("rootGroups", "required, an array of groups"))?;
    if roots_json.is_empty() {
        return Err(invalid("rootGroups", "at least one group is required"));
    }
    let mut roots = Vec::new();
    for (i, g) in roots_json.iter().enumerate() {
        roots.push(group(g, &format!("rootGroups[{i}]"), total_memory)?);
    }
    unique_names(roots.iter().map(|g| g.name.as_str()), "rootGroups")?;
    let default_group = obj
        .get("defaultGroup")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            invalid(
                "defaultGroup",
                "required: the leaf a query no selector matches uses",
            )
        })?
        .to_owned();
    let mut selectors = Vec::new();
    if let Some(list) = obj.get("selectors") {
        let list = list
            .as_array()
            .ok_or_else(|| invalid("selectors", "expected an array"))?;
        for (i, s) in list.iter().enumerate() {
            selectors.push(selector(s, &format!("selectors[{i}]"))?);
        }
    }
    let cfg = Config {
        roots,
        selectors,
        default_group,
    };
    check_leaf(&cfg, &cfg.default_group, "defaultGroup")?;
    for (i, s) in cfg.selectors.iter().enumerate() {
        check_leaf(&cfg, &s.group, &format!("selectors[{i}].group"))?;
    }
    Ok(cfg)
}

/// `path` must name a leaf group (dotted names, `root.etl.nightly`).
fn check_leaf(cfg: &Config, id: &str, at: &str) -> Result<(), KrishivError> {
    let mut parts = id.split('.');
    let first = parts.next().unwrap_or_default();
    let mut cur = cfg
        .roots
        .iter()
        .find(|g| g.name == first)
        .ok_or_else(|| invalid(at, format!("group {id} does not exist")))?;
    for part in parts {
        cur = cur
            .children
            .iter()
            .find(|g| g.name == part)
            .ok_or_else(|| invalid(at, format!("group {id} does not exist")))?;
    }
    if cur.children.is_empty() {
        Ok(())
    } else {
        Err(invalid(
            at,
            format!("group {id} has subgroups; a selector or default must name a leaf"),
        ))
    }
}

fn unique_names<'a>(names: impl Iterator<Item = &'a str>, at: &str) -> Result<(), KrishivError> {
    let mut seen = std::collections::HashSet::new();
    for n in names {
        if !seen.insert(n) {
            return Err(invalid(
                at,
                format!("the name {n} is used twice among siblings"),
            ));
        }
    }
    Ok(())
}

/// Refuses a key that is not in `allowed`; `ignored` keys (Trino-only knobs) pass with a warning.
fn only_keys(
    obj: &Map<String, Value>,
    allowed: &[&str],
    ignored: &[&str],
    at: &str,
) -> Result<(), KrishivError> {
    for k in obj.keys() {
        if allowed.contains(&k.as_str()) {
            continue;
        }
        if ignored.contains(&k.as_str()) {
            tracing::warn!(path = at, key = %k, "resource group key ignored: it has no effect in Krishiv");
            continue;
        }
        let reason = match k.as_str() {
            "softCpuLimit" | "hardCpuLimit" | "cpuQuotaPeriod" => {
                "CPU quotas are not supported in v1 (cluster workload management adds them)"
            }
            _ => "unknown key",
        };
        let key_path = if at == "$" {
            k.clone()
        } else {
            format!("{at}.{k}")
        };
        return Err(invalid(&key_path, reason));
    }
    Ok(())
}

fn group(v: &Value, at: &str, total_memory: usize) -> Result<GroupCfg, KrishivError> {
    let obj = v
        .as_object()
        .ok_or_else(|| invalid(at, "expected an object"))?;
    only_keys(
        obj,
        &[
            "name",
            "hardConcurrencyLimit",
            "maxQueued",
            "schedulingPolicy",
            "schedulingWeight",
            "softMemoryLimit",
            "perQueryMemoryLimit",
            "priority",
            "workloadClass",
            "scanQuota",
            "scanQuotaPeriod",
            "subGroups",
        ],
        &["jmxExport", "softConcurrencyLimit"],
        at,
    )?;
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(&format!("{at}.name"), "required, a string"))?;
    if name.contains("${") {
        return Err(invalid(
            &format!("{at}.name"),
            "name templates (${USER}, ${SOURCE}) are not supported in v1",
        ));
    }
    if name.is_empty() || name.contains('.') {
        return Err(invalid(
            &format!("{at}.name"),
            "a name is not empty and has no '.'",
        ));
    }
    let hard_concurrency = uint(obj, "hardConcurrencyLimit", at)?
        .ok_or_else(|| invalid(&format!("{at}.hardConcurrencyLimit"), "required"))?;
    let max_queued = uint(obj, "maxQueued", at)?
        .ok_or_else(|| invalid(&format!("{at}.maxQueued"), "required"))?;
    if hard_concurrency == 0 {
        return Err(invalid(
            &format!("{at}.hardConcurrencyLimit"),
            "must be at least 1",
        ));
    }
    let policy = match obj.get("schedulingPolicy").map(Value::as_str) {
        None | Some(Some("weighted_fair")) => Policy::WeightedFair,
        Some(Some("query_priority")) => Policy::QueryPriority,
        Some(_) => {
            return Err(invalid(
                &format!("{at}.schedulingPolicy"),
                "expected weighted_fair or query_priority",
            ));
        }
    };
    let weight = uint(obj, "schedulingWeight", at)?.unwrap_or(1);
    if weight == 0 {
        return Err(invalid(
            &format!("{at}.schedulingWeight"),
            "must be at least 1",
        ));
    }
    let soft_memory = match obj.get("softMemoryLimit") {
        None => None,
        Some(v) => Some(size(v, total_memory, &format!("{at}.softMemoryLimit"))?),
    };
    let per_query_memory = match obj.get("perQueryMemoryLimit") {
        None => None,
        Some(v) => Some(size(v, total_memory, &format!("{at}.perQueryMemoryLimit"))?),
    };
    let priority = match uint(obj, "priority", at)? {
        None => None,
        Some(p) if p <= 100 => u8::try_from(p).ok(),
        Some(_) => return Err(invalid(&format!("{at}.priority"), "expected 0 to 100")),
    };
    let class = match obj.get("workloadClass").map(Value::as_str) {
        None => None,
        Some(Some("interactive")) => Some(WorkloadClass::Interactive),
        Some(Some("batch")) => Some(WorkloadClass::Batch),
        Some(Some("background")) => Some(WorkloadClass::Background),
        Some(Some("agent")) => Some(WorkloadClass::Agent),
        Some(_) => {
            return Err(invalid(
                &format!("{at}.workloadClass"),
                "expected interactive, batch, background or agent",
            ));
        }
    };
    let scan_quota = match (obj.get("scanQuota"), obj.get("scanQuotaPeriod")) {
        (None, None) => None,
        (Some(_), _) => {
            let bytes = uint(obj, "scanQuota", at)?.unwrap_or(0);
            let period_ms = match obj.get("scanQuotaPeriod") {
                None => 3_600_000,
                Some(p) => duration_ms(p, &format!("{at}.scanQuotaPeriod"))?,
            };
            Some(ScanQuota { bytes, period_ms })
        }
        (None, Some(_)) => {
            return Err(invalid(
                &format!("{at}.scanQuotaPeriod"),
                "needs scanQuota in the same group",
            ));
        }
    };
    let mut children = Vec::new();
    if let Some(subs) = obj.get("subGroups") {
        let subs = subs
            .as_array()
            .ok_or_else(|| invalid(&format!("{at}.subGroups"), "expected an array"))?;
        for (i, s) in subs.iter().enumerate() {
            children.push(group(s, &format!("{at}.subGroups[{i}]"), total_memory)?);
        }
        unique_names(
            children.iter().map(|g| g.name.as_str()),
            &format!("{at}.subGroups"),
        )?;
    }
    Ok(GroupCfg {
        name: name.to_owned(),
        hard_concurrency: usize::try_from(hard_concurrency).unwrap_or(usize::MAX),
        max_queued: usize::try_from(max_queued).unwrap_or(usize::MAX),
        policy,
        weight,
        soft_memory,
        per_query_memory,
        priority,
        class,
        scan_quota,
        children,
    })
}

fn uint(obj: &Map<String, Value>, key: &str, at: &str) -> Result<Option<u64>, KrishivError> {
    match obj.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| invalid(&format!("{at}.{key}"), "expected a non-negative integer")),
    }
}

/// Bytes as a number, or `"NN%"` of `total`.
fn size(v: &Value, total: usize, at: &str) -> Result<usize, KrishivError> {
    if let Some(n) = v.as_u64() {
        return Ok(usize::try_from(n).unwrap_or(usize::MAX));
    }
    let pct = v
        .as_str()
        .and_then(|s| s.strip_suffix('%'))
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|p| *p <= 100)
        .ok_or_else(|| invalid(at, "expected bytes or a percentage such as \"60%\""))?;
    let share = u128::from(pct).saturating_mul(total as u128) / 100;
    Ok(usize::try_from(share).unwrap_or(usize::MAX))
}

/// `"30s"`, `"15m"`, `"1h"` or `"1d"` in milliseconds.
fn duration_ms(v: &Value, at: &str) -> Result<u64, KrishivError> {
    let bad = || {
        invalid(
            at,
            "expected a duration such as \"30s\", \"15m\", \"1h\" or \"1d\"",
        )
    };
    let s = v.as_str().ok_or_else(bad)?;
    let unit = s.chars().last().ok_or_else(bad)?;
    let n: u64 = s[..s.len() - unit.len_utf8()].parse().map_err(|_| bad())?;
    let mult = match unit {
        's' => 1_000,
        'm' => 60_000,
        'h' => 3_600_000,
        'd' => 86_400_000,
        _ => return Err(bad()),
    };
    match n.checked_mul(mult) {
        Some(ms) if ms > 0 => Ok(ms),
        _ => Err(bad()),
    }
}

fn selector(v: &Value, at: &str) -> Result<SelectorCfg, KrishivError> {
    let obj = v
        .as_object()
        .ok_or_else(|| invalid(at, "expected an object"))?;
    only_keys(
        obj,
        &[
            "user",
            "userGroup",
            "source",
            "clientTags",
            "queryType",
            "group",
        ],
        &[],
        at,
    )?;
    let re = |key: &str| -> Result<Option<Regex>, KrishivError> {
        match obj.get(key) {
            None => Ok(None),
            Some(p) => {
                let p = p
                    .as_str()
                    .ok_or_else(|| invalid(&format!("{at}.{key}"), "expected a regex string"))?;
                Regex::new(&format!("^(?:{p})$"))
                    .map(Some)
                    .map_err(|e| invalid(&format!("{at}.{key}"), e.to_string()))
            }
        }
    };
    let client_tags = match obj.get("clientTags") {
        None => Vec::new(),
        Some(t) => t
            .as_array()
            .and_then(|a| a.iter().map(|x| x.as_str().map(str::to_owned)).collect())
            .ok_or_else(|| invalid(&format!("{at}.clientTags"), "expected an array of strings"))?,
    };
    let query_type = match obj.get("queryType").map(Value::as_str) {
        None => None,
        Some(Some("READ")) => Some(StatementClass::Read),
        Some(Some("WRITE")) => Some(StatementClass::Write),
        Some(Some("DDL")) => Some(StatementClass::Ddl),
        Some(Some("SESSION")) => Some(StatementClass::Session),
        Some(Some("ADMIN")) => Some(StatementClass::Admin),
        Some(_) => {
            return Err(invalid(
                &format!("{at}.queryType"),
                "expected READ, WRITE, DDL, SESSION or ADMIN",
            ));
        }
    };
    let group = obj
        .get("group")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            invalid(
                &format!("{at}.group"),
                "required, the dotted name of a leaf",
            )
        })?
        .to_owned();
    Ok(SelectorCfg {
        user: re("user")?,
        user_group: re("userGroup")?,
        source: re("source")?,
        client_tags,
        query_type,
        group,
    })
}
