// SPDX-License-Identifier: Apache-2.0
//! Session configuration: Spark's `spark.conf`.

use std::sync::{Arc, PoisonError, RwLock};

use krishiv_common::config::{Config, SESSION_TIME_ZONE, Scope, ValueType, lookup};
use krishiv_common::error::{
    CannotModifyStaticConfig, InvalidConfValueRequirement, InvalidConfValueTimeZone, KrishivError,
};
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::datetime::{resolve_zone, zone_id};

/// Prefixes of entries read once, when the session is built (memory pool,
/// spill directory, I/O scheduler): setting them later would not apply,
/// so it fails like a static config.
const BUILD_TIME: &[&str] = &["krishiv.memory.", "krishiv.spill.", "krishiv.io."];

/// The configuration of a session, shared by its clones. Queries read an
/// immutable snapshot ([`Self::snapshot`]).
#[derive(Debug)]
pub(crate) struct ConfStore {
    /// The configuration and its version, bumped on every change: analysis
    /// results are reused only within one version. One lock holds both, so
    /// a snapshot never pairs a configuration with another one's version.
    current: RwLock<(Arc<Config>, u64)>,
    /// The configuration the session was built with: `RESET` restores it,
    /// so values from the builder survive resets as Spark's startup values
    /// do (static entries never change at runtime either way).
    initial: Config,
    /// The default of `spark.sql.session.timeZone`, read once when the
    /// session is built (see [`local_zone`]).
    local_zone: Arc<str>,
}

impl ConfStore {
    pub(crate) fn new(config: Config) -> Self {
        Self::with_local_zone(config, &local_zone())
    }

    fn with_local_zone(config: Config, local_zone: &str) -> Self {
        Self {
            current: RwLock::new((Arc::new(config.clone()), 0)),
            initial: config,
            local_zone: Arc::from(local_zone),
        }
    }

    /// The zone an unset `spark.sql.session.timeZone` means.
    pub(crate) fn local_zone(&self) -> &str {
        &self.local_zone
    }

    /// The configuration one query reads, and its version: an unset session
    /// time zone is the local zone here, so plans, casts and functions
    /// never see it unset. The stored configuration keeps it unset, so `SET`
    /// does not list it (as in Spark).
    pub(crate) fn query_snapshot(&self) -> (Arc<Config>, u64) {
        let (config, version) = self.versioned_snapshot();
        if config.get(&SESSION_TIME_ZONE).is_some() {
            return (config, version);
        }
        let mut effective = Config::clone(&config);
        match effective.set(SESSION_TIME_ZONE.key, &self.local_zone) {
            Ok(_) => (Arc::new(effective), version),
            // Not reached: `local_zone` only returns zones the entry accepts.
            Err(_) => (config, version),
        }
    }

    pub(crate) fn snapshot(&self) -> Arc<Config> {
        self.versioned_snapshot().0
    }

    /// The configuration and its version, read together.
    pub(crate) fn versioned_snapshot(&self) -> (Arc<Config>, u64) {
        let current = self.current.read().unwrap_or_else(PoisonError::into_inner);
        (Arc::clone(&current.0), current.1)
    }

    fn update(
        &self,
        f: impl FnOnce(&mut Config) -> Result<(), KrishivError>,
    ) -> Result<(), KrishivError> {
        let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
        let mut next = Config::clone(&current.0);
        f(&mut next)?;
        *current = (Arc::new(next), current.1.wrapping_add(1));
        Ok(())
    }

    /// The value of `key` as `spark.conf.get` and `SET key` show it: an
    /// unset session time zone shows the local zone.
    pub(crate) fn value(&self, key: &str) -> Option<String> {
        get(&self.snapshot(), key).or_else(|| {
            lookup(key)
                .is_some_and(|entry| entry.key == SESSION_TIME_ZONE.key)
                .then(|| self.local_zone.to_string())
        })
    }

    /// Restores the configuration the session was built with, bumping the
    /// version so running analyses are redone (`RESET` with no key).
    pub(crate) fn reset(&self) {
        let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
        *current = (Arc::new(self.initial.clone()), current.1.wrapping_add(1));
    }

    /// Restores `key` to the value the session was built with: the
    /// builder's value, or the default when neither sets it (`RESET key`).
    pub(crate) fn reset_key(&self, key: &str) {
        let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
        let mut next = Config::clone(&current.0);
        next.reset_key(key, &self.initial);
        *current = (Arc::new(next), current.1.wrapping_add(1));
    }
}

/// The zone Java's `TimeZone.getDefault` gives on this host, which is the
/// default of `spark.sql.session.timeZone`.
pub(crate) fn local_zone() -> String {
    let tz = std::env::var("TZ").ok();
    let link = std::fs::read_link("/etc/localtime").ok();
    let file = std::fs::read_to_string("/etc/timezone").ok();
    local_zone_from(
        tz.as_deref(),
        link.as_deref().and_then(std::path::Path::to_str),
        file.as_deref(),
    )
}

/// [`local_zone`] with its inputs as arguments. Checked against Spark 4.2.0
/// runs: a non-empty `TZ` decides alone (without a leading `:` or `posix/`)
/// and is `GMT` when Java does not know it; else the zone `/etc/localtime`
/// links to (without `posix/`). Java compares a non-link `/etc/localtime` with the zoneinfo
/// files; this reads `/etc/timezone` instead, else `UTC`. Java asks libc for
/// the offset of a `TZ` it does not know, which reads POSIX strings
/// ([`posix_zone_id`]).
fn local_zone_from(tz: Option<&str>, link: Option<&str>, file: Option<&str>) -> String {
    if let Some(tz) = tz.filter(|tz| !tz.is_empty()) {
        let name = tz.strip_prefix(':').unwrap_or(tz);
        let name = name.strip_prefix("posix/").unwrap_or(name);
        return java_zone_id(name)
            .or_else(|| posix_zone_id(name))
            .unwrap_or_else(|| "GMT".to_owned());
    }
    link.and_then(|link| link.rsplit_once("zoneinfo/"))
        .map(|(_, zone)| zone.strip_prefix("posix/").unwrap_or(zone))
        .or_else(|| file.map(str::trim))
        .and_then(java_zone_id)
        .unwrap_or_else(|| "UTC".to_owned())
}

/// The id Java's `TimeZone` gives `name`, when it knows it: a region or
/// short id of the database as written (case matters), or `GMT` with an
/// offset. Java reads no other offset spelling, unlike
/// `spark.sql.session.timeZone`.
fn java_zone_id(name: &str) -> Option<String> {
    let accepted = SESSION_TIME_ZONE.parse(SESSION_TIME_ZONE.key, name).is_ok();
    if !accepted || resolve_zone(name).is_err() {
        return None;
    }
    let is_offset =
        name == "Z" || (FixedOffset::parse(name).is_some() && name.contains(['+', '-']));
    if !is_offset {
        return Some(name.to_owned());
    }
    name.starts_with("GMT").then(|| zone_id(name))
}

/// The id Java gives a POSIX `TZ` string it does not know as a name: `GMT` with
/// the offset of the string's head, whose sign POSIX writes the other way
/// round (`UTC+5` is `GMT-05:00`, `CET-1CEST,M3.5.0,M10.5.0/3` is `GMT+01:00`).
/// Java reads the offset in force now and this one the standard offset, so
/// they differ during daylight saving.
fn posix_zone_id(tz: &str) -> Option<String> {
    let rest = match tz.strip_prefix('<') {
        Some(quoted) => quoted.split_once('>')?.1,
        None => {
            let letters = tz.bytes().take_while(u8::is_ascii_alphabetic).count();
            if letters < 3 {
                return None;
            }
            &tz[letters..]
        }
    };
    let (behind, rest) = match rest.strip_prefix('-') {
        Some(rest) => (false, rest),
        None => (true, rest.strip_prefix('+').unwrap_or(rest)),
    };
    let (hours, rest) = leading_number(rest)?;
    let minutes = match rest.strip_prefix(':') {
        Some(rest) => leading_number(rest)?.0,
        None => 0,
    };
    if hours > 23 || minutes > 59 {
        return None;
    }
    if hours == 0 && minutes == 0 {
        return Some("GMT".to_owned());
    }
    let sign = if behind { '-' } else { '+' };
    Some(format!("GMT{sign}{hours:02}:{minutes:02}"))
}

/// The number `text` starts with, and what follows it.
fn leading_number(text: &str) -> Option<(u32, &str)> {
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    Some((text[..digits].parse().ok()?, &text[digits..]))
}

/// The value of `key` in `config` as text: a registered entry (by its
/// `krishiv.*` key or Spark alias) with its default, else an unregistered
/// key as set.
pub(crate) fn get(config: &Config, key: &str) -> Option<String> {
    match lookup(key) {
        Some(entry) => config.get(entry).map(ToString::to_string),
        None => config.get_unregistered(key).map(str::to_owned),
    }
}

/// Whether `key` can change after the session is built.
pub(crate) fn is_modifiable(key: &str) -> bool {
    match lookup(key) {
        Some(entry) => {
            entry.scope == Scope::Session && !BUILD_TIME.iter().any(|p| entry.key.starts_with(p))
        }
        None => true,
    }
}

/// The runtime configuration of a session (Spark's `spark.conf`). Changes
/// apply to queries that start afterwards.
///
/// ```
/// let session = krishiv_session::Session::new()?;
/// session.conf().set("spark.sql.caseSensitive", "true")?;
/// assert_eq!(session.conf().get("spark.sql.caseSensitive").as_deref(), Some("true"));
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RuntimeConf {
    store: Arc<ConfStore>,
}

impl RuntimeConf {
    pub(crate) fn new(store: Arc<ConfStore>) -> Self {
        Self { store }
    }

    /// The value of `key`: a set value or a registered entry's default;
    /// `None` for an unset key without a default.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert_eq!(session.conf().get("spark.sql.ansi.enabled").as_deref(), Some("true"));
    /// assert_eq!(session.conf().get("my.app.flag"), None);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        self.store.value(key)
    }

    /// Sets `key` to `value` for queries that start afterwards.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.conf().set("krishiv.exec.batchSize", "1024")?;
    /// let e = session.conf().set("krishiv.exec.batchSize", "many").unwrap_err();
    /// assert_eq!(e.error_class().name(), "KRISHIV_INVALID_CONFIG_VALUE");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `CANNOT_MODIFY_STATIC_CONFIG` for an entry fixed when the session
    /// was built ([`Self::is_modifiable`]); `KRISHIV_UNKNOWN_CONFIG` for an
    /// unknown `krishiv.*` key; `KRISHIV_INVALID_CONFIG_VALUE` for a value
    /// of the wrong type. Unknown keys of other prefixes are kept.
    /// `INVALID_CONF_VALUE.TIME_ZONE` for a session time zone (`spark.sql.session.timeZone`
    /// or `krishiv.sql.session.timeZone`) no zone database entry or offset names.
    pub fn set(&self, key: &str, value: &str) -> Result<(), KrishivError> {
        if !is_modifiable(key) {
            return Err(CannotModifyStaticConfig {
                key: format!("\"{key}\""),
            }
            .into());
        }
        // A time zone setting (under its canonical key or the Spark alias)
        // is resolved against the time zone database, which `krishiv-common`
        // has not: the text as given, as Spark's `ZoneId.of` takes it.
        if lookup(key).is_some_and(|e| e.value_type == ValueType::TimeZone)
            && resolve_zone(value).is_err()
        {
            return Err(InvalidConfValueTimeZone {
                conf_name: key.to_owned(),
                conf_value: value.to_owned(),
            }
            .into());
        }
        if lookup(key).is_some_and(|e| std::ptr::eq(e, &krishiv_common::config::SQL_DEFAULT_PATH))
            && krishiv_common::sql_path::parse_conf(value).is_none()
        {
            return Err(InvalidConfValueRequirement {
                conf_name: key.to_owned(),
                conf_value: value.to_owned(),
                conf_requirement: "The value must be empty or a comma-separated SET PATH \
                    element list (same grammar as SET PATH, except PATH is not allowed)."
                    .to_owned(),
            }
            .into());
        }
        self.store.update(|c| c.set(key, value).map(drop))
    }

    /// Resets `key` to the value the session was built with: the
    /// builder's value, or the default when neither sets it.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.conf().set("krishiv.exec.batchSize", "1024")?;
    /// session.conf().unset("krishiv.exec.batchSize")?;
    /// assert_eq!(session.conf().get("krishiv.exec.batchSize").as_deref(), Some("4096"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `CANNOT_MODIFY_STATIC_CONFIG` as for [`Self::set`].
    pub fn unset(&self, key: &str) -> Result<(), KrishivError> {
        if !is_modifiable(key) {
            return Err(CannotModifyStaticConfig {
                key: format!("\"{key}\""),
            }
            .into());
        }
        self.store.reset_key(key);
        Ok(())
    }

    /// Whether `key` can be set at runtime.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert!(session.conf().is_modifiable("spark.sql.ansi.enabled"));
    /// assert!(!session.conf().is_modifiable("krishiv.memory.limit"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn is_modifiable(&self, key: &str) -> bool {
        is_modifiable(key)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_common::config::Config;

    use super::{ConfStore, get, local_zone_from};

    #[test]
    fn a_snapshot_pairs_a_configuration_with_its_own_version() {
        // Each change sets `test.n` to the version it creates, so a snapshot
        // whose value and version differ read them across a change.
        let store = Arc::new(ConfStore::new(Config::default()));
        let writer = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                for n in 1..=2_000_u64 {
                    store
                        .update(|c| c.set("test.n", &n.to_string()).map(drop))
                        .unwrap();
                }
            })
        };
        while !writer.is_finished() {
            let (config, version) = store.versioned_snapshot();
            let value = get(&config, "test.n").map_or(0, |v| v.parse().unwrap());
            assert_eq!(value, version);
        }
        writer.join().unwrap();
        assert_eq!(store.versioned_snapshot().1, 2_000);
    }

    /// Every expected value comes from a Spark 4.2.0 run with that `TZ`.
    #[test]
    fn the_local_zone_is_the_one_java_picks() {
        let tz = |tz| local_zone_from(Some(tz), Some("/x/zoneinfo/Asia/Dubai"), None);
        for (input, expected) in [
            ("Asia/Kolkata", "Asia/Kolkata"),
            (":Asia/Tokyo", "Asia/Tokyo"),
            ("posix/Asia/Tokyo", "Asia/Tokyo"),
            ("UTC", "UTC"),
            ("Etc/UTC", "Etc/UTC"),
            ("PST", "PST"),
            ("EST5EDT", "EST5EDT"),
            ("GMT+5", "GMT+05:00"),
            // Unknown to Java: `GMT`, never the system zone.
            ("Mars/Base", "GMT"),
            ("+05:30", "GMT"),
            ("Z", "GMT"),
            // POSIX strings go to libc.
            ("UTC+5", "GMT-05:00"),
            ("CET-1CEST,M3.5.0,M10.5.0/3", "GMT+01:00"),
            ("EST5:30", "GMT-05:30"),
            ("<+03>-3", "GMT+03:00"),
            ("UTC0", "GMT"),
            ("utc", "GMT"),
            ("Asia/kolkata", "GMT"),
            // An empty `TZ` is no `TZ`.
            ("", "Asia/Dubai"),
        ] {
            assert_eq!(tz(input), expected, "TZ={input:?}");
        }
    }

    #[test]
    fn without_tz_the_local_zone_comes_from_the_system() {
        let link = "/usr/share/zoneinfo/Europe/Berlin";
        assert_eq!(local_zone_from(None, Some(link), None), "Europe/Berlin");
        assert_eq!(
            local_zone_from(None, Some("/usr/share/zoneinfo/posix/Asia/Tokyo"), None),
            "Asia/Tokyo"
        );
        assert_eq!(
            local_zone_from(None, Some("../usr/share/zoneinfo/Etc/UTC"), None),
            "Etc/UTC"
        );
        assert_eq!(
            local_zone_from(None, None, Some("Asia/Tokyo\n")),
            "Asia/Tokyo"
        );
        assert_eq!(local_zone_from(None, Some("/opt/oddity"), None), "UTC");
        assert_eq!(local_zone_from(None, None, Some("Mars/Base\n")), "UTC");
        assert_eq!(local_zone_from(None, None, None), "UTC");
    }

    #[test]
    fn an_unset_zone_is_local_for_queries_and_stays_unset_in_the_store() {
        let store = ConfStore::with_local_zone(Config::default(), "Asia/Kolkata");
        let key = "spark.sql.session.timeZone";
        assert_eq!(store.value(key).as_deref(), Some("Asia/Kolkata"));
        assert_eq!(
            store.value("krishiv.sql.session.timeZone").as_deref(),
            Some("Asia/Kolkata")
        );
        assert_eq!(get(&store.snapshot(), key), None);
        assert_eq!(
            get(&store.query_snapshot().0, key).as_deref(),
            Some("Asia/Kolkata")
        );
        // A set zone wins, and `RESET` goes back to local.
        store.update(|c| c.set(key, "+05:30").map(drop)).unwrap();
        assert_eq!(store.value(key).as_deref(), Some("+05:30"));
        assert_eq!(
            get(&store.query_snapshot().0, key).as_deref(),
            Some("+05:30")
        );
        store.reset_key(key);
        assert_eq!(store.value(key).as_deref(), Some("Asia/Kolkata"));
    }
}
