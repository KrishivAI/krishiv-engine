// SPDX-License-Identifier: Apache-2.0
//! DataFrameReader (`spark.read`): the read builder of files, with an optional user schema.

use super::*;

/// Reads files ([`Session::read`]; PySpark's `DataFrameReader`).
#[derive(Debug, Clone)]
#[must_use]
pub struct DataFrameReader {
    session: Session,
    format: String,
    options: BTreeMap<String, OptionValue>,
    schema: Option<krishiv_types::StructType>,
}

impl DataFrameReader {
    pub(crate) fn new(session: Session) -> Self {
        Self {
            session,
            format: "parquet".to_owned(),
            options: BTreeMap::new(),
            schema: None,
        }
    }

    /// The user's schema (PySpark's `DataFrameReader.schema`): the files are
    /// read as its columns, never inferred. As in Spark, the data columns are
    /// its columns in its order, the partition columns found under the paths
    /// come last (one it names takes its type), and every column reads as
    /// nullable, whatever the schema says.
    ///
    /// ```
    /// use krishiv_types::{DataType, StructField, StructType};
    /// # let s = krishiv_session::Session::new()?;
    /// let schema = StructType::new([StructField::new("a", DataType::Integer, true)]);
    /// let df = s.read().schema(schema).format("csv").load(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn schema(mut self, schema: krishiv_types::StructType) -> Self {
        self.schema = Some(schema);
        self
    }

    /// [`Self::schema`] from a DDL string (`"a INT, b STRING"`), as Spark's
    /// `schema(String)` parses it (`StructType.fromDDL`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().schema_ddl("a INT, b STRING")?.format("json").load(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// The parser's errors (`PARSE_SYNTAX_ERROR`, ...) for a string that is
    /// no table schema, and the type errors of its columns.
    pub fn schema_ddl(self, ddl: &str) -> Result<Self, KrishivError> {
        let schema = parse_schema_ddl(ddl, &self.session.parser_options())?;
        Ok(self.schema(schema))
    }

    /// The file format (default `parquet`, as Spark's
    /// `spark.sql.sources.default`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().format("parquet").load(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn format(mut self, format: &str) -> Self {
        self.format = format.to_owned();
        self
    }

    /// A read option; values are typed as secrets where Spark redacts them.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().option("mergeSchema", "false").parquet(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options
            .insert(key.to_owned(), OptionValue::from_user(key, value));
        self
    }

    /// Reads `paths` in the format.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().load(["/data/t", "/data/u"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn load<S: AsRef<str>>(self, paths: impl IntoIterator<Item = S>) -> DataFrame {
        let mut plan = read_files(&self.format, typed_paths(paths), self.options);
        if let PlanNode::Read(ReadSource::Files { schema, .. }) = &mut plan.node {
            *schema = self.schema;
        }
        DataFrame::new(self.session, Arc::new(plan))
    }

    /// Reads Parquet files (PySpark's `read.parquet`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().parquet(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn parquet<S: AsRef<str>>(self, paths: impl IntoIterator<Item = S>) -> DataFrame {
        self.format("parquet").load(paths)
    }
}

/// A table schema string (`"a INT, b STRING NOT NULL"`) as a struct type,
/// as Spark's `StructType.fromDDL` reads it.
///
/// # Errors
/// The parser's errors and the type errors of the columns.
pub(crate) fn parse_schema_ddl(
    ddl: &str,
    options: &krishiv_sql::ParserOptions,
) -> Result<krishiv_types::StructType, KrishivError> {
    let columns = krishiv_sql::parse_table_schema(ddl, options)?;
    let spec = krishiv_sql::ast::DataTypeSpec {
        kind: krishiv_sql::ast::DataTypeKind::Struct(Some(columns)),
        collation: None,
        span: krishiv_sql::ast::Span::new(0, ddl.len()),
    };
    match spec.to_data_type()? {
        krishiv_types::DataType::Struct(s) => Ok(s),
        other => Err(krishiv_common::error::InternalError {
            message: format!("a table schema parsed as {}", other.simple_string()),
        }
        .into()),
    }
}
