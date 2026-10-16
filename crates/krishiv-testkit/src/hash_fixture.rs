//! Loader for the Spark `hash()` golden fixtures (`tests/fixtures/spark-4.2.0/hash.json`).
//!
//! The fixture is generated against a real Spark by `tests/scripts/gen_hash_goldens.py`.
//! `hash` is Murmur3 x86_32 (seed 42), `xxhash64` is XXH64 (seed 42); both signed, as Spark returns them.

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
}

/// The parsed fixture.
#[derive(Debug, Clone)]
pub struct HashFixture {
    /// Spark version the goldens were generated against.
    pub spark_version: String,
    /// All golden cases.
    pub cases: Vec<HashCase>,
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
            });
        }
        Ok(Self {
            spark_version,
            cases,
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

    /// Fails until `hash()` exists (P2): compares Krishiv's Murmur3 against every golden.
    #[test]
    #[ignore = "P2: krishiv hash() not implemented"]
    fn murmur3_spark_compat() -> R {
        let fx = HashFixture::load_default()?;
        Err(format!(
            "krishiv hash() not implemented; {} golden cases await comparison",
            fx.cases.len()
        )
        .into())
    }
}
