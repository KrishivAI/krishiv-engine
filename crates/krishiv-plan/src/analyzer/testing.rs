// SPDX-License-Identifier: Apache-2.0
//! Test doubles for analyzer tests and benchmarks (feature `testing`): a
//! function registry with simplified result types (Spark's implicit casts
//! and decimal rules are a later pass's), a one-catalog resolver, a
//! context builder and a single-threaded `block_on`.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use krishiv_common::error::KrishivError;
use krishiv_types::{DataType, DecimalType};

use super::{
    AllowAll, AnalysisContext, ArgType, Catalog, CatalogResolver, FunctionEntry, FunctionLookup,
    FunctionRegistry, MemoryCatalog, Principal, PrincipalKind, ResolutionConf, SparkProfile,
};
use crate::{Builtin, FunctionId, FunctionKind};

/// A registry of the canonical operators and the common SQL functions.
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

fn boolean(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(of(DataType::Boolean, any_null(a)))
}

fn wider(a: &[ArgType]) -> Option<DataType> {
    let mut out: Option<DataType> = None;
    for t in a.iter().filter_map(|t| t.data_type.clone()) {
        out = Some(match (out, t) {
            (None, t) => t,
            (Some(DataType::Double), _) | (_, DataType::Double) => DataType::Double,
            (Some(DataType::Decimal(x)), DataType::Decimal(y)) => {
                DataType::Decimal(if x.precision() >= y.precision() { x } else { y })
            }
            (Some(DataType::Decimal(x)), _) | (Some(_), DataType::Decimal(x)) => {
                DataType::Decimal(x)
            }
            (Some(DataType::Long), _) | (_, DataType::Long) => DataType::Long,
            (Some(x), _) => x,
        });
    }
    out
}

fn arith(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: wider(a),
        nullable: any_null(a),
    })
}

fn division(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: wider(a),
        nullable: true,
    })
}

fn same_as_first(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: a.first().and_then(|t| t.data_type.clone()),
        nullable: any_null(a),
    })
}

fn not_null(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    let _ = a;
    Ok(of(DataType::Boolean, false))
}

fn when(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    let values: Vec<ArgType> = a
        .chunks(2)
        .filter_map(|p| match p {
            [_, v] | [v] => Some(v.clone()),
            _ => None,
        })
        .collect();
    Ok(ArgType {
        data_type: wider(&values),
        nullable: any_null(&values) || a.len().is_multiple_of(2),
    })
}

fn string(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(of(DataType::String, any_null(a)))
}

fn coalesce(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: wider(a),
        nullable: a.iter().all(|t| t.nullable),
    })
}

fn sum(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    let t = match a.first().and_then(|t| t.data_type.clone()) {
        Some(DataType::Decimal(d)) => DataType::Decimal(DecimalType::try_new(
            (d.precision() + 10).min(38).into(),
            d.scale().into(),
        )?),
        Some(DataType::Double | DataType::Float) => DataType::Double,
        _ => DataType::Long,
    };
    Ok(of(t, true))
}

fn avg(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    let t = match a.first().and_then(|t| t.data_type.clone()) {
        Some(DataType::Decimal(d)) => DataType::Decimal(DecimalType::try_new(
            (d.precision() + 4).min(38).into(),
            (d.scale() + 4).min(38).into(),
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
        data_type: a.first().and_then(|t| t.data_type.clone()),
        nullable: true,
    })
}

fn extract(a: &[ArgType]) -> Result<ArgType, KrishivError> {
    Ok(of(DataType::Integer, a.get(1).is_some_and(|t| t.nullable)))
}

const REGISTRY: [(&str, FunctionKind, Sig); 10] = [
    ("sum", FunctionKind::Aggregate, sum),
    ("avg", FunctionKind::Aggregate, avg),
    ("count", FunctionKind::Aggregate, count),
    ("min", FunctionKind::Aggregate, min_max),
    ("max", FunctionKind::Aggregate, min_max),
    ("any_value", FunctionKind::Aggregate, min_max),
    ("extract", FunctionKind::Scalar, extract),
    ("substring", FunctionKind::Scalar, string),
    ("upper", FunctionKind::Scalar, string),
    ("coalesce", FunctionKind::Scalar, coalesce),
];

fn builtin(b: Builtin) -> Sig {
    match b {
        Builtin::EqNullSafe | Builtin::IsNull | Builtin::IsNotNull => not_null,
        Builtin::Add | Builtin::Subtract | Builtin::Multiply => arith,
        Builtin::Divide | Builtin::Remainder => division,
        Builtin::Negative => same_as_first,
        Builtin::When => when,
        Builtin::Concat => string,
        _ => boolean,
    }
}

impl FunctionRegistry for TestFunctions {
    fn lookup(&self, name: &str, profile: SparkProfile) -> FunctionLookup {
        if let Some(b) = Builtin::from_name(name) {
            return FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Builtin(b),
                kind: FunctionKind::Scalar,
                result: builtin(b),
            });
        }
        let lower = name.to_lowercase();
        if let Some((i, (n, kind, result))) = REGISTRY
            .iter()
            .enumerate()
            .find(|(_, (n, _, _))| *n == lower)
        {
            return FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Registry {
                    name: (*n).to_owned(),
                    id: u32::try_from(i).unwrap_or(u32::MAX),
                },
                kind: *kind,
                result: *result,
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
        principal: Arc::new(Principal {
            name: "tester".to_owned(),
            groups: Vec::new(),
            tenant: None,
            kind: PrincipalKind::User,
        }),
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
