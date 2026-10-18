// SPDX-License-Identifier: Apache-2.0
//! Test doubles for analyzer tests and benchmarks (feature `testing`): a
//! function registry of the common SQL functions with Spark's signatures
//! and result types, a one-catalog resolver, a context builder and a
//! single-threaded `block_on`.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use krishiv_common::error::KrishivError;
use krishiv_types::{DataType, DecimalType};

use super::{
    AllowAll, AnalysisContext, ArgType, Catalog, CatalogResolver, FunctionEntry, FunctionLookup,
    FunctionRegistry, InputType, Inputs, MemoryCatalog, Principal, PrincipalKind, ResolutionConf,
    SparkProfile,
};
use crate::{FunctionId, FunctionKind};

/// A registry of common SQL functions, typed as Spark 4.2 types them.
#[derive(Debug, Clone, Copy, Default)]
pub struct TestFunctions;

type Sig = fn(&[ArgType]) -> Result<ArgType, KrishivError>;

fn any_null(a: &[ArgType]) -> bool {
    a.iter().any(|t| t.nullable)
}

fn of(t: DataType, nullable: bool) -> ArgType {
    ArgType {
        data_type: Some(t),
        nullable,
    }
}

fn first(a: &[ArgType]) -> Option<DataType> {
    a.first().and_then(|t| t.data_type.clone())
}

fn string(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(of(DataType::String, any_null(a)))
}

/// `coalesce`: the arguments share one type after their casts.
fn coalesce(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: a.iter().find_map(|t| t.data_type.clone()),
        nullable: a.iter().all(|t| t.nullable),
    })
}

/// `sum`: `decimal(min(38, p + 10), s)`, `bigint` or `double`.
fn sum(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    let t = match first(a) {
        Some(DataType::Decimal(d)) => DataType::Decimal(DecimalType::try_new(
            (u32::from(d.precision()) + 10).min(38),
            d.scale().into(),
        )?),
        Some(DataType::Double | DataType::Float) | None => DataType::Double,
        Some(_) => DataType::Long,
    };
    Ok(of(t, true))
}

/// `avg`: `decimal(p + 4, s + 4)` bounded to 38, else `double`.
fn avg(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    let t = match first(a) {
        Some(DataType::Decimal(d)) => DataType::Decimal(DecimalType::try_new(
            (u32::from(d.precision()) + 4).min(38),
            (u32::from(d.scale()) + 4).min(38),
        )?),
        _ => DataType::Double,
    };
    Ok(of(t, true))
}

fn count(_: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(of(DataType::Long, false))
}

fn min_max(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: first(a),
        nullable: true,
    })
}

fn extract(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(of(DataType::Integer, a.get(1).is_some_and(|t| t.nullable)))
}

/// How each function's arguments are cast.
#[derive(Clone, Copy)]
enum In {
    Any,
    Field,
    Numeric,
    Substring,
    String,
    Common,
}

impl In {
    fn inputs(self) -> Inputs {
        let exact = |t| InputType::Exact(t);
        match self {
            Self::Any => Inputs::Any,
            Self::Field => Inputs::Expects(vec![exact(DataType::String), InputType::Any]),
            Self::Numeric => Inputs::Expects(vec![InputType::Numeric { intervals: true }]),
            Self::Substring => {
                Inputs::Expects(vec![exact(DataType::String), exact(DataType::Integer)])
            }
            Self::String => Inputs::Expects(vec![exact(DataType::String)]),
            Self::Common => Inputs::Common,
        }
    }
}

const REGISTRY: [(&str, FunctionKind, In, Sig); 10] = [
    ("sum", FunctionKind::Aggregate, In::Numeric, sum),
    ("avg", FunctionKind::Aggregate, In::Numeric, avg),
    ("count", FunctionKind::Aggregate, In::Any, count),
    ("min", FunctionKind::Aggregate, In::Any, min_max),
    ("max", FunctionKind::Aggregate, In::Any, min_max),
    ("any_value", FunctionKind::Aggregate, In::Any, min_max),
    ("extract", FunctionKind::Scalar, In::Field, extract),
    ("substring", FunctionKind::Scalar, In::Substring, string),
    ("upper", FunctionKind::Scalar, In::String, string),
    ("coalesce", FunctionKind::Scalar, In::Common, coalesce),
];

impl FunctionRegistry for TestFunctions {
    fn version(&self) -> u64 {
        0
    }

    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        let [name] = name else {
            return FunctionLookup::NotFound;
        };
        let lower = name.to_lowercase();
        if let Some((i, (n, kind, inputs, result))) =
            REGISTRY.iter().enumerate().find(|(_, (n, ..))| *n == lower)
        {
            return FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Registry {
                    name: (*n).to_owned(),
                    id: u32::try_from(i).unwrap_or(u32::MAX),
                },
                kind: *kind,
                inputs: inputs.inputs(),
                result: Arc::new(*result),
            });
        }
        // `randstr` stands for a function Spark added in 4.0 and Krishiv
        // lacks: unknown to a 3.5 session, unsupported in a 4.x one.
        match (lower.as_str(), profile) {
            ("randstr", SparkProfile::V3_5) => FunctionLookup::NotFound,
            ("randstr" | "sha2", _) => FunctionLookup::KnownUnsupported,
            _ => FunctionLookup::NotFound,
        }
    }
}

/// One catalog, the current one, and the current namespace.
pub struct OneCatalog(pub Arc<dyn Catalog>, pub Vec<String>);

impl std::fmt::Debug for OneCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OneCatalog({}, {:?})", self.0.name(), self.1)
    }
}

impl CatalogResolver for OneCatalog {
    fn current_catalog(&self) -> &str {
        self.0.name()
    }

    fn current_namespace(&self) -> &[String] {
        &self.1
    }

    fn epoch(&self) -> u64 {
        0
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>> {
        (name.eq_ignore_ascii_case(self.0.name())).then(|| Arc::clone(&self.0))
    }
}

/// A context over `catalog` (as `spark_catalog.default`), allowing all.
#[must_use]
pub fn context(catalog: MemoryCatalog, conf: ResolutionConf) -> AnalysisContext {
    AnalysisContext {
        profile: SparkProfile::default(),
        conf,
        principal: Arc::new(Principal::new("tester", PrincipalKind::User)),
        policy: Arc::new(AllowAll),
        catalogs: Arc::new(OneCatalog(Arc::new(catalog), vec!["default".to_owned()])),
        functions: Arc::new(TestFunctions),
        sql: None,
    }
}

/// Runs a future that never waits on I/O to completion on this thread.
pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
}
