// SPDX-License-Identifier: Apache-2.0
//! Loader for the Spark `hash()` golden fixtures (`tests/fixtures/spark-4.2.0/hash.json`).
//!
//! The fixture is generated against a real Spark by `tests/scripts/gen_hash_goldens.py`.
//! `hash` is Murmur3 x86_32 (seed 42), `xxhash64` is XXH64 (seed 42); both signed, as Spark returns them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Type categories the fixture must cover (see P0-T20).
pub const REQUIRED_CATEGORIES: &[&str] = &[
    "boolean",
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "double",
    "decimal_le18",
    "decimal_gt18",
    "string",
    "binary",
    "date",
    "timestamp",
    "timestamp_ntz",
    "null",
    "array",
    "map",
    "struct",
    "nested_array_struct",
    "multi_arg",
];

/// Errors loading or validating a fixture.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    /// Reading the file failed.
    #[error("reading {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// The JSON is malformed or has the wrong shape.
    #[error("invalid fixture: {0}")]
    Invalid(String),
}

/// One golden case: `SELECT hash(<expr>)` evaluated by Spark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashCase {
    /// Stable id, `<category>_<n>`.
    pub id: String,
    /// Type category, one of [`REQUIRED_CATEGORIES`].
    pub category: String,
    /// Spark SQL text of the argument list of `hash(...)`.
    pub expr: String,
    /// Spark type of the arguments, as `typeof(struct(expr))` (`struct<col1:T,...>`).
    pub spark_type: String,
    /// Expected `hash(expr)`.
    pub hash: i32,
    /// Expected `xxhash64(expr)`.
    pub xxhash64: i64,
    /// Session configuration the case was generated under (for example the
    /// legacy map-hash switch); empty for the defaults.
    pub conf: BTreeMap<String, String>,
}

/// An expression Spark rejects under the default configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultConfError {
    /// Spark SQL text of the argument list of `hash(...)`.
    pub expr: String,
    /// Type category of the case it mirrors.
    pub category: String,
    /// The error class Spark raised at analysis.
    pub error_class: String,
}

/// The parsed fixture.
#[derive(Debug, Clone)]
pub struct HashFixture {
    /// Spark version the goldens were generated against.
    pub spark_version: String,
    /// All golden cases.
    pub cases: Vec<HashCase>,
    /// Expressions Spark rejects with the default configuration.
    pub default_conf_errors: Vec<DefaultConfError>,
}

impl HashFixture {
    /// Default fixture location.
    #[must_use]
    pub fn default_path() -> PathBuf {
        crate::tests_dir().join("fixtures/spark-4.2.0/hash.json")
    }

    /// Load the default fixture.
    pub fn load_default() -> Result<Self, FixtureError> {
        Self::load(Self::default_path())
    }

    /// Load a fixture from `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FixtureError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| FixtureError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Parse fixture JSON text.
    pub fn parse(text: &str) -> Result<Self, FixtureError> {
        let root: Value =
            serde_json::from_str(text).map_err(|e| FixtureError::Invalid(e.to_string()))?;
        let spark_version = str_field(&root, "spark_version")?;
        let raw = root
            .get("cases")
            .and_then(Value::as_array)
            .ok_or_else(|| FixtureError::Invalid("missing `cases` array".into()))?;
        let mut cases = Vec::with_capacity(raw.len());
        for c in raw {
            let hash = c
                .get("hash")
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| FixtureError::Invalid("case `hash` missing or not an i32".into()))?;
            let xxhash64 = c.get("xxhash64").and_then(Value::as_i64).ok_or_else(|| {
                FixtureError::Invalid("case `xxhash64` missing or not an i64".into())
            })?;
            cases.push(HashCase {
                id: str_field(c, "id")?,
                category: str_field(c, "category")?,
                expr: str_field(c, "expr")?,
                spark_type: str_field(c, "spark_type")?,
                hash,
                xxhash64,
                conf: conf_field(c)?,
            });
        }
        let default_conf_errors = root
            .get("default_conf_errors")
            .map(|v| {
                v.as_array()
                    .ok_or_else(|| {
                        FixtureError::Invalid("`default_conf_errors` not an array".into())
                    })?
                    .iter()
                    .map(|e| {
                        Ok(DefaultConfError {
                            expr: str_field(e, "expr")?,
                            category: str_field(e, "category")?,
                            error_class: str_field(e, "error_class")?,
                        })
                    })
                    .collect::<Result<Vec<_>, FixtureError>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            spark_version,
            cases,
            default_conf_errors,
        })
    }

    /// Categories from [`REQUIRED_CATEGORIES`] with no case.
    #[must_use]
    pub fn missing_categories(&self) -> Vec<&'static str> {
        REQUIRED_CATEGORIES
            .iter()
            .copied()
            .filter(|cat| !self.cases.iter().any(|c| c.category == *cat))
            .collect()
    }
}

/// A case's `conf` object (absent means the defaults).
fn conf_field(case: &Value) -> Result<BTreeMap<String, String>, FixtureError> {
    let Some(conf) = case.get("conf") else {
        return Ok(BTreeMap::new());
    };
    conf.as_object()
        .ok_or_else(|| FixtureError::Invalid("case `conf` is not an object".into()))?
        .iter()
        .map(|(k, v)| {
            v.as_str()
                .map(|v| (k.clone(), v.to_owned()))
                .ok_or_else(|| FixtureError::Invalid(format!("conf `{k}` is not a string")))
        })
        .collect()
}

fn str_field(v: &Value, key: &str) -> Result<String, FixtureError> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| FixtureError::Invalid(format!("missing string field `{key}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn fixture_parses_and_covers_every_type() -> R {
        let fx = HashFixture::load_default()?;
        assert_eq!(fx.spark_version, "4.2.0");
        assert_eq!(fx.missing_categories(), Vec::<&str>::new());
        // Spot checks against values produced by the real Spark 4.2.0 container.
        let get = |id: &str| fx.cases.iter().find(|c| c.id == id).ok_or("case missing");
        assert_eq!(get("int_0")?.hash, 933_211_791); // hash(0)
        assert_eq!(get("null_0")?.hash, 42); // hash(NULL) is the seed
        assert_eq!(get("double_0")?.hash, get("double_1")?.hash); // 0.0 and -0.0 hash equal
        Ok(())
    }

    #[test]
    fn malformed_fixture_is_rejected() {
        assert!(HashFixture::parse("{}").is_err());
        assert!(HashFixture::parse(r#"{"spark_version":"x","cases":[{"id":"a"}]}"#).is_err());
    }

    #[test]
    fn per_case_conf_and_default_conf_errors_are_kept() -> R {
        let fx = HashFixture::load_default()?;
        let legacy = "spark.sql.legacy.allowHashOnMapType";
        let maps: Vec<_> = fx.cases.iter().filter(|c| c.category == "map").collect();
        assert_eq!(maps.len(), 3);
        assert!(
            maps.iter()
                .all(|c| c.conf.get(legacy).map(String::as_str) == Some("true"))
        );
        assert!(
            fx.cases
                .iter()
                .filter(|c| c.category != "map")
                .all(|c| c.conf.is_empty())
        );
        let [err] = fx.default_conf_errors.as_slice() else {
            return Err("expected one default-conf error".into());
        };
        assert_eq!(err.category, "map");
        assert_eq!(err.error_class, "DATATYPE_MISMATCH.HASH_MAP_TYPE");
        Ok(())
    }

    #[test]
    fn malformed_conf_is_rejected() {
        let case = |conf: &str| {
            format!(
                r#"{{"spark_version":"x","cases":[{{"id":"a","category":"int","expr":"1","spark_type":"t","hash":1,"xxhash64":1,"conf":{conf}}}]}}"#
            )
        };
        assert!(HashFixture::parse(&case("{}")).is_ok());
        assert!(HashFixture::parse(&case("[]")).is_err());
        assert!(HashFixture::parse(&case(r#"{"k":1}"#)).is_err());
        let bad = r#"{"spark_version":"x","cases":[],"default_conf_errors":{}}"#;
        assert!(HashFixture::parse(bad).is_err());
        let missing = r#"{"spark_version":"x","cases":[],"default_conf_errors":[{"expr":"e"}]}"#;
        assert!(HashFixture::parse(missing).is_err());
    }
}
