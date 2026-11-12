// SPDX-License-Identifier: Apache-2.0
//! The behaviour checks of the file-format suite: memory, cancellation, I/O, options, writer, take and dynamic-filter pruning.

use super::*;

/// Check 8.
///
/// # Errors
/// Memory held at the end, or an error other than a memory error.
pub fn memory_accounting(fx: &dyn FormatFixture) -> SuiteResult {
    let schema = pruning_schema();
    let input = data::batch(&schema, 3000, 41)?;
    for limit in [None, Some(16 << 10), Some(1 << 20)] {
        let env = Env::new(limit)?;
        // The data is written before the limit matters (setup pool only).
        let file = env.write(
            fx,
            "mem/f",
            &schema,
            std::slice::from_ref(&input),
            &fx.unit_rows(300),
        )?;
        let req = ScanRequest::all(Arc::clone(&schema));
        match env.scan(
            fx,
            &file,
            &req,
            None,
            ScanOptions {
                prefetch_bytes: 1 << 20,
                ..ScanOptions::default()
            },
        ) {
            Ok(b) => ensure!(rows(&b) == 3000, "rows lost under a memory limit"),
            Err(e) => ensure!(
                matches!(
                    e.error_class(),
                    ErrorClass::UnableToAcquireMemory | ErrorClass::KrishivQueryAbortedOutOfMemory
                ),
                "limit {limit:?}: not a memory error: {e}"
            ),
        }
        ensure!(
            env.root.reserved() == 0,
            "limit {limit:?}: {} bytes held at the end",
            env.root.reserved()
        );
    }
    Ok(())
}

/// Polls `f` `n` times (time advancing between polls) and drops it.
pub(super) fn poll_n<F: Future>(env: &Env, f: F, n: usize) -> bool {
    env.block_on(async move {
        let mut f = Box::pin(f);
        let mut cx = Context::from_waker(Waker::noop());
        for _ in 0..n {
            if f.as_mut().poll(&mut cx).is_ready() {
                return true;
            }
            tokio::time::advance(Duration::from_millis(3)).await;
        }
        // Dropped here, inside the runtime (timers need it).
        drop(f);
        false
    })
}

/// Check 9.
///
/// # Errors
/// Memory held or a multipart upload left open after a cancel.
pub fn cancellation(fx: &dyn FormatFixture) -> SuiteResult {
    let schema = pruning_schema();
    let input = data::batch(&schema, 2000, 51)?;
    for n in 0..40 {
        let env = Env::new(None)?;
        let file = env.write(
            fx,
            "cancel/f",
            &schema,
            std::slice::from_ref(&input),
            &fx.unit_rows(200),
        )?;
        env.store.set_latency(Duration::from_millis(5));
        let o = env.options(fx, &[])?;
        let req = ScanRequest::all(Arc::clone(&schema));
        let driver = fx.driver();
        let scan = driver.scan(
            &env,
            fx.format(),
            file.clone(),
            req,
            ByteRange::whole(file.size),
            o,
            ScanOptions::default(),
        );
        let finished = poll_n(&env, scan, n);
        // Let tasks the scan left behind (executor drivers) end.
        env.block_on(async { tokio::time::sleep(Duration::from_secs(1)).await });
        ensure!(
            env.root.reserved() == 0,
            "scan dropped after {n} polls holds memory"
        );
        if finished {
            break;
        }
    }
    // Writes: a multipart upload cancelled at every await is aborted.
    let big_schema = Arc::new(Schema::new(vec![Field::new(
        "b",
        DataType::BinaryView,
        true,
    )]));
    // Incompressible bytes, so a compressing format still writes 6 MiB.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let blob: Vec<Option<Vec<u8>>> = (0..24)
        .map(|_| {
            Some(
                (0..256 << 10)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state.to_le_bytes()[0]
                    })
                    .collect(),
            )
        })
        .collect();
    let big = RecordBatch::try_new(
        Arc::clone(&big_schema),
        vec![Arc::new(arrow_array::BinaryViewArray::from(
            blob.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
        ))],
    )
    .map_err(|e| e.to_string())?;
    let mut aborted = 0;
    for n in [1, 4, 8, 16] {
        // The smallest part size, so a 6 MiB file takes a multipart upload
        // whatever the format's overhead.
        let env = Env::with_config(
            None,
            IoConfig {
                multipart_part_bytes: krishiv_io::io::upload::MIN_PART_BYTES,
                ..IoConfig::default()
            },
        )?;
        env.store.set_latency(Duration::from_millis(5));
        let o = env.options(fx, &fx.unit_rows(8))?;
        let driver = fx.driver();
        let write = driver.write(
            &env,
            fx.format(),
            Arc::clone(&big_schema),
            o,
            WriteOptions {
                store: StoreId::new("memory://suite"),
                dir: Path::from("cancelw"),
                task: 0,
                job_uuid: "j".into(),
                max_records_per_file: 0,
                target_file_bytes: 0,
                partition_by: Vec::new(),
                partition_names: Vec::new(),
                zone: Some(0),
                zone_name: None,
            },
            vec![big.clone()],
            false,
        );
        poll_n(&env, write, n);
        // Let background aborts run.
        env.block_on(async { tokio::time::sleep(Duration::from_secs(1)).await });
        let st = env.store.stats();
        ensure!(
            st.uploads_started == st.uploads_completed + st.uploads_aborted,
            "dangling multipart upload after cancelling at poll {n}: {st:?}"
        );
        ensure!(
            env.root.reserved() == 0,
            "write dropped at poll {n} holds memory"
        );
        aborted += st.uploads_aborted;
    }
    ensure!(aborted > 0, "no cancellation hit an open multipart upload");
    Ok(())
}

/// Check 10.
///
/// # Errors
/// A request outside the scheduler, a footer read twice, or counters that
/// do not match the store.
pub fn io_behaviour(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 1500, 61)?;
    let file = env.write(
        fx,
        "io/f",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(500),
    )?;
    let before = env.store.stats();
    env.store
        .fail_next_gets(2, Some(Duration::from_millis(200)));
    let req = ScanRequest::all(Arc::clone(&schema));
    let got = env
        .scan(fx, &file, &req, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    ensure!(rows(&got) == 1500, "rows lost with retried requests");
    let c = env.io.counters();
    ensure!(
        IoCounters::get(&c.retries) >= 2,
        "transient failures were not retried"
    );
    let after = env.store.stats();
    ensure!(
        fx.footerless() || after.tail_gets - before.tail_gets == 1,
        "footer took {} tail reads",
        after.tail_gets - before.tail_gets
    );
    let gets = after.gets - before.gets;
    let reads = IoCounters::get(&c.requests) - (before.puts + before.uploads_started);
    ensure!(
        gets <= reads,
        "store saw {gets} GETs, the scheduler counted {reads} requests"
    );
    ensure!(
        after.unscheduled == 0,
        "{} requests bypassed the scheduler",
        after.unscheduled
    );
    // The second scan finds the footer in the cache.
    env.scan(fx, &file, &req, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    ensure!(
        fx.footerless() || env.store.stats().tail_gets == after.tail_gets,
        "cached footer was read again"
    );
    ensure!(
        fx.footerless() || IoCounters::get(&c.cache_hits) >= 1,
        "no cache hit"
    );
    // Bytes counted are bytes served.
    ensure!(
        IoCounters::get(&c.bytes_read) == env.store.stats().bytes - before.bytes,
        "byte counters differ"
    );
    Ok(())
}

/// Check 11.
///
/// # Errors
/// Wrong option handling or a leaked secret.
pub fn options(fx: &dyn FormatFixture) -> SuiteResult {
    let mut env = Env::new(None)?;
    let format = fx.format();
    let schema = format.options();
    let Some(first) = schema.options.iter().find(|o| o.kind != OptionKind::Secret) else {
        return Ok(());
    };
    // A misspelling: accepted with a did-you-mean warning, listed as ignored.
    let typo = format!("{}x", first.name);
    let user = BTreeMap::from([(typo.clone(), OptionValue::from_user(&typo, "1"))]);
    let (o, warnings) = schema
        .parse(&user, &Config::new())
        .map_err(|e| e.to_string())?;
    ensure!(
        warnings
            == vec![OptionWarning::Unknown {
                key: typo.clone(),
                suggestion: Some(first.name)
            }],
        "warning {warnings:?}"
    );
    ensure!(
        o.ignored() == [typo.clone()],
        "ignored keys {:?}",
        o.ignored()
    );
    // Framework options are not reported; keys are case-insensitive.
    let user = BTreeMap::from([
        (
            "pathGlobFilter".to_owned(),
            OptionValue::from_user("pathGlobFilter", "*.x"),
        ),
        (
            first.name.to_uppercase(),
            OptionValue::from_user(first.name, valid_text(first)),
        ),
    ]);
    let (_, warnings) = schema
        .parse(&user, &Config::new())
        .map_err(|e| e.to_string())?;
    ensure!(warnings.is_empty(), "known options warned: {warnings:?}");
    // Strict mode.
    let (strict, _) =
        Config::from_pairs([("krishiv.io.strictOptions", "true")]).map_err(|e| e.to_string())?;
    let user = BTreeMap::from([(typo.clone(), OptionValue::from_user(&typo, "1"))]);
    let e = schema
        .parse(&user, &strict)
        .err()
        .ok_or("strict mode accepted an unknown option")?;
    ensure!(
        e.error_class() == ErrorClass::KrishivUnknownFormatOption,
        "strict class {}",
        class(&e)
    );
    ensure!(e.to_string().contains(first.name), "no suggestion in {e}");
    // Defaults are typed.
    let d = FormatOptions::defaults(schema);
    for spec in schema.options {
        let has =
            d.bool(spec.name).is_some() || d.int(spec.name).is_some() || d.str(spec.name).is_some();
        ensure!(
            has == (spec.default != krishiv_io::OptionDefault::None),
            "default of {}",
            spec.name
        );
    }
    // A wrongly typed value.
    if let Some(b) = schema
        .options
        .iter()
        .find(|o| matches!(o.kind, OptionKind::Bool | OptionKind::Int { .. }))
    {
        let user = BTreeMap::from([(
            b.name.to_owned(),
            OptionValue::from_user(b.name, "not-a-value"),
        )]);
        let e = schema
            .parse(&user, &Config::new())
            .err()
            .ok_or("bad value accepted")?;
        ensure!(
            e.error_class() == ErrorClass::KrishivInvalidFormatOptionValue,
            "value class {}",
            class(&e)
        );
    }
    // A session-conf fallback.
    if let Some(spec) = schema
        .options
        .iter()
        .find(|o| o.conf_fallback.is_some() && o.kind == OptionKind::Bool)
    {
        let key = spec.conf_fallback.unwrap_or_default();
        let flipped = !matches!(spec.default, krishiv_io::OptionDefault::Bool(true));
        env.session_mut()
            .set(key, &flipped.to_string())
            .map_err(|e| e.to_string())?;
        let o = env.options(fx, &[])?;
        ensure!(
            o.bool(spec.name) == Some(flipped),
            "conf fallback {key} ignored"
        );
    }
    // Secrets: never printed or in cache keys; another key never hits.
    if let Some(key_opt) = fx.metadata_key_option() {
        let canary = "canary-7f3a9b";
        let opts = vec![s(key_opt, &canary)];
        let o = env.options(fx, &opts)?;
        ensure!(
            !format!("{o:?}").contains(canary),
            "secret in Debug of options"
        );
        let pschema = pruning_schema();
        let file = env.write(
            fx,
            "opt/secret",
            &pschema,
            &[data::batch(&pschema, 50, 1)?],
            &opts,
        )?;
        let fp = env
            .block_on(o.metadata_fingerprint(env.io.principal(), env.io.secrets()))
            .map_err(|e| e.to_string())?;
        let key = CacheKey::new(format.name(), &file, fp);
        ensure!(
            !format!("{key:?}").contains(canary),
            "secret in a cache key"
        );
        env.block_on(
            env.io
                .read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o),
        )
        .map_err(|e| e.to_string())?;
        let wrong = env.options(fx, &[s(key_opt, &"another-key")])?;
        let e = env
            .block_on(env.io.read_metadata(
                format.as_ref(),
                &file,
                MetadataNeeds::default(),
                &wrong,
            ))
            .err()
            .ok_or("metadata decoded with the wrong key (cache hit across keys?)")?;
        ensure!(!e.to_string().contains(canary), "secret in an error");
        ensure!(
            e.error_class() == ErrorClass::FailedReadFileCannotReadFileFooter,
            "wrong-key class {}",
            class(&e)
        );
    }
    Ok(())
}

/// A value `spec` accepts.
pub(super) fn valid_text(spec: &krishiv_io::OptionSpec) -> String {
    match spec.kind {
        OptionKind::Bool => "true".to_owned(),
        OptionKind::Int { min, .. } => min.to_string(),
        OptionKind::Enum(values) => values.first().map_or_else(String::new, |v| (*v).to_owned()),
        OptionKind::String | OptionKind::Secret => "x".to_owned(),
    }
}

/// Check 12.
///
/// # Errors
/// Wrong files, statistics or abort behaviour.
pub fn writer(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 2500, 71)?;
    let format = fx.format();
    let driver = fx.driver();
    let write = |dir: &str, max_records: u64, target: u64, batches: Vec<RecordBatch>, fail| {
        let o = env.options(fx, &fx.unit_rows(300))?;
        env.block_on(driver.write(
            &env,
            Arc::clone(&format),
            Arc::clone(&schema),
            o,
            WriteOptions {
                store: StoreId::new("memory://suite"),
                dir: Path::from(dir),
                task: 7,
                job_uuid: "8c1e".into(),
                max_records_per_file: max_records,
                target_file_bytes: target,
                partition_by: Vec::new(),
                partition_names: Vec::new(),
                zone: Some(0),
                zone_name: None,
            },
            batches,
            fail,
        ))
        .map_err(|e| e.to_string())
    };
    // maxRecordsPerFile.
    let files = write(
        "w1",
        1000,
        0,
        vec![input.slice(0, 1700), input.slice(1700, 800)],
        false,
    )?;
    let names: Vec<String> = files
        .iter()
        .map(|f| f.path.filename().unwrap_or_default().to_owned())
        .collect();
    let suffix = format.file_suffix(&env.options(fx, &[])?);
    let want: Vec<String> = (0..3)
        .map(|i| krishiv_io::spark_file_name(7, "8c1e", i, &suffix))
        .collect();
    ensure!(names == want, "file names {names:?}");
    ensure!(
        names[0] == format!("part-00007-8c1e-c000{suffix}"),
        "Spark naming"
    );
    ensure!(
        files.iter().map(|f| f.rows).collect::<Vec<_>>() == [1000, 1000, 500],
        "rows per file"
    );
    let mut back = Vec::new();
    let o = env.options(fx, &[])?;
    for f in &files {
        let file = env.block_on(env.file(&f.path))?;
        ensure!(file.size == f.size, "reported size differs");
        let md = env
            .block_on(
                env.io
                    .read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o),
            )
            .map_err(|e| e.to_string())?;
        ensure!(
            md.statistics() == f.statistics,
            "writer statistics differ from reader statistics"
        );
        back.extend(
            env.scan(
                fx,
                &file,
                &ScanRequest::all(Arc::clone(&schema)),
                None,
                ScanOptions::default(),
            )
            .map_err(|e| e.to_string())?,
        );
    }
    let out = nullable(&schema);
    ensure!(
        concat(&out, &back)? == input.clone().with_schema(out).map_err(|e| e.to_string())?,
        "written data differs"
    );
    // Target-size rollover.
    let files = write("w2", 0, 1, vec![input.clone()], false)?;
    ensure!(files.len() > 1, "no rollover at the target size");
    ensure!(
        files.iter().map(|f| f.rows).sum::<u64>() == 2500,
        "rows lost on rollover"
    );
    // A failed write leaves nothing, not even the files it rolled over.
    ensure!(
        write("w3", 1000, 0, vec![input], true).is_err(),
        "the failing write succeeded"
    );
    let left = env.block_on(env.inner.list(Some(&Path::from("w3"))).collect::<Vec<_>>());
    ensure!(left.is_empty(), "abort left {} objects", left.len());
    ensure!(env.root.reserved() == 0, "writer memory held");
    Ok(())
}

/// Check 14.
///
/// # Errors
/// Rows differing from the scan, a changed file read, or bytes read for
/// nothing.
pub fn take(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 1000, 21)?;
    let path = "take/f";
    let file = env.write(
        fx,
        path,
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(128),
    )?;
    let o = env.options(fx, &[])?;
    let format = fx.format();
    if format.capabilities(&o).random_access.is_none() {
        return Ok(());
    }
    let projection: Vec<ColumnPath> = schema
        .fields()
        .iter()
        .map(|f| ColumnPath::new(f.name()))
        .collect();
    let read = |file: &FileRef, rows: Vec<u64>| -> Result<(Vec<RecordBatch>, u64), KrishivError> {
        let req = krishiv_io::TakeRequest::new(
            file.clone(),
            Arc::clone(&schema),
            projection.clone(),
            krishiv_io::FieldMapping::default(),
            rows,
        );
        env.block_on(async {
            let mut s = krishiv_io::take(
                format.as_ref(),
                &env.io,
                req,
                &o,
                env.reservation("take"),
                ScanOptions::default(),
            )
            .await?;
            let mut out = Vec::new();
            while let Some(b) = s.next_batch().await? {
                out.push(b);
            }
            Ok((out, s.stream().map_or(0, |s| s.metrics().units)))
        })
    };
    // Out of order, with repeats, in a few units of many.
    let rows = vec![999, 3, 500, 3, 0, 501, 127, 128];
    let (got, units) = read(&file, rows).map_err(|e| e.to_string())?;
    let mut seen = Vec::new();
    for b in &got {
        let idx = b.column_by_name(ROW_INDEX_COLUMN).ok_or("no row index")?;
        let idx =
            arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(idx.as_ref());
        for (k, &ri) in idx.values().iter().enumerate() {
            let want = input.slice(usize::try_from(ri).map_err(|e| e.to_string())?, 1);
            let have = b.slice(k, 1);
            for c in 0..schema.fields().len() {
                ensure!(
                    have.column(c).to_data() == want.column(c).to_data(),
                    "row {ri} holds other values"
                );
            }
            seen.push(ri);
        }
    }
    ensure!(
        seen == [0, 3, 127, 128, 500, 501, 999],
        "rows read: {seen:?}"
    );
    ensure!(units < 8, "{units} units read for rows in at most 6");
    // A position past the end is an error, never a short result.
    ensure!(
        read(&file, vec![1000]).is_err(),
        "a missing row read as nothing"
    );
    // Nothing to read: no request at all, even for a file that is gone.
    let gone = FileRef {
        path: Path::from("take/gone"),
        ..file.clone()
    };
    let (empty, _) = read(&gone, Vec::new()).map_err(|e| e.to_string())?;
    ensure!(empty.is_empty(), "an empty take returned rows");
    // A file replaced after the row ids were made.
    let replaced = Path::from("take/changing");
    let bytes = env.get_raw(&file.path)?.to_vec();
    let listed = env.put_raw(&replaced, bytes.clone())?;
    let mut changed = bytes;
    changed[4] ^= 0xFF;
    env.put_raw(&replaced, changed)?;
    let e = read(&listed, vec![1]).err().ok_or("changed file read")?;
    ensure!(
        e.error_class() == ErrorClass::KrishivFileChangedDuringRead,
        "changed class {}",
        class(&e)
    );
    Ok(())
}

/// Check 13: dynamic-filter pruning.
///
/// A `Final` membership filter on `i` plus a range filter on `i` prune only
/// soundly: every row passing both filters is still read. A scan with only
/// the dynamic filter isolates its path: it returns the same passing rows
/// and, where the format keeps statistics, prunes units. Formats without
/// statistics (CSV, JSON lines: [`FormatFixture::footerless`]) cannot prune
/// by statistics, so only the soundness halves apply to them.
///
/// # Errors
/// A dropped matching row.
pub fn dynamic_filter_pruning(fx: &dyn FormatFixture) -> SuiteResult {
    use arrow_array::types::Int32Type;
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = sorted_batch(&schema, 3000, 5)?;
    let file = env.write(
        fx,
        "dyn/f",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(100),
    )?;
    // Keys clustered at the file's start (sorted on `i`): most units hold
    // no key, so statistics prune them where the format keeps any.
    let ints = arrow_array::cast::AsArray::as_primitive::<Int32Type>(input.column(0).as_ref());
    let mut keys: Vec<i32> = Vec::new();
    for v in ints.iter().flatten().take(40) {
        if !keys.contains(&v) {
            keys.push(v);
        }
        if keys.len() == 8 {
            break;
        }
    }
    ensure!(!keys.is_empty(), "no int keys in the input");
    let (lo, hi) = (
        *keys.iter().min().unwrap_or(&0),
        *keys.iter().max().unwrap_or(&0),
    );
    let mut builder = KeyFilterBuilder::new(DataType::Int32, KeyFilterOptions::default())
        .ok_or("int32 has no runtime filter")?;
    builder
        .add(&arrow_array::Int32Array::from(keys.clone()), None)
        .map_err(|e| e.to_string())?;
    let published = Arc::new(builder.finish());
    let filter = Arc::new(RuntimeFilter::new(
        FilterId(1),
        ColumnPath::new("i"),
        DataType::Int32,
    ));
    filter.publish(FilterSnapshot::Final(Arc::clone(&published)));
    let key_set: std::collections::HashSet<i32> = keys.into_iter().collect();
    // The range filter the membership implies: units outside `[lo, hi]`
    // prove empty by bounds alone.
    let range = PushdownExpr::And(vec![
        PushdownExpr::cmp(ColumnPath::new("i"), CmpOp::GtEq, ScalarValue::Int32(lo)),
        PushdownExpr::cmp(ColumnPath::new("i"), CmpOp::LtEq, ScalarValue::Int32(hi)),
    ]);
    let mut base = ScanRequest::all(Arc::clone(&schema));
    base.metadata_cols.row_index = true;
    let mut full = base.clone();
    full.predicate = Some(range.clone());
    full.dynamic
        .push(ScanFilter::new(Arc::clone(&filter), &schema).ok_or("no column i")?);
    let unfiltered = env
        .scan(fx, &file, &base, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    let filtered = env
        .scan(fx, &file, &full, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    let indexes = |batches: &[RecordBatch]| -> SuiteResult<std::collections::HashSet<i64>> {
        let mut out = std::collections::HashSet::new();
        for b in batches {
            let idx = b.column_by_name(ROW_INDEX_COLUMN).ok_or("no row index")?;
            let idx = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
                idx.as_ref(),
            );
            out.extend(idx.values().iter().copied());
        }
        Ok(out)
    };
    let got = indexes(&filtered)?;
    // The rows passing both filters, by the oracle.
    let mut want = std::collections::HashSet::new();
    for row in 0..input.num_rows() {
        let ri = i64::try_from(row).map_err(|e| e.to_string())?;
        if data::eval(&range, &input, row) != Some(true) {
            continue;
        }
        if ints.is_null(row) || !key_set.contains(&ints.value(row)) {
            continue;
        }
        want.insert(ri);
    }
    ensure!(!want.is_empty(), "the filters match no row");
    for ri in &want {
        ensure!(
            got.contains(ri),
            "dynamic pruning dropped matching row {ri}"
        );
    }
    let all = indexes(&unfiltered)?;
    // The oracle itself is checked: the unfiltered scan holds every
    // passing row.
    let unfiltered_restricted: std::collections::HashSet<i64> =
        all.intersection(&want).copied().collect();
    ensure!(
        unfiltered_restricted == want,
        "the unfiltered scan missed a row"
    );
    // Where the format keeps statistics, the filters prune something.
    let o = env.options(fx, &[])?;
    if !fx.footerless() && fx.format().capabilities(&o).statistics {
        ensure!(
            rows(&filtered) < rows(&unfiltered),
            "no unit pruned by the dynamic or range filter"
        );
    }
    // The dynamic filter in isolation: no range predicate, so only the
    // membership filter can prune. A format that discards
    // `ScanRequest::dynamic` still passes the combined scan above via the
    // range predicate alone.
    let mut dynamic_only = base;
    dynamic_only
        .dynamic
        .push(ScanFilter::new(Arc::clone(&filter), &schema).ok_or("no column i")?);
    let dynamic_scanned = env
        .scan(fx, &file, &dynamic_only, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    let got_dynamic = indexes(&dynamic_scanned)?;
    let mut want_dynamic = std::collections::HashSet::new();
    for row in 0..input.num_rows() {
        if ints.is_null(row) || !key_set.contains(&ints.value(row)) {
            continue;
        }
        want_dynamic.insert(i64::try_from(row).map_err(|e| e.to_string())?);
    }
    ensure!(
        !want_dynamic.is_empty(),
        "the dynamic filter matches no row"
    );
    for ri in &want_dynamic {
        ensure!(
            got_dynamic.contains(ri),
            "dynamic pruning dropped matching row {ri}"
        );
    }
    if !fx.footerless() && fx.format().capabilities(&o).statistics {
        ensure!(
            rows(&dynamic_scanned) < rows(&unfiltered),
            "no unit pruned by the dynamic filter alone"
        );
    }
    ensure!(env.root.reserved() == 0, "memory held after scans");
    Ok(())
}

/// Instantiates the conformance suite for a fixture:
/// `file_format_conformance!(module_name, fixture_expression);`
#[macro_export]
macro_rules! file_format_conformance {
    ($name:ident, $fixture:expr) => {
        mod $name {
            #[allow(
                unused_imports,
                reason = "the fixture expression may name items of the caller"
            )]
            use super::*;
            use $crate::file_format as suite;
            fn run(check: fn(&dyn suite::FormatFixture) -> suite::SuiteResult) {
                let fx = $fixture;
                if let Err(e) = check(&fx) {
                    panic!("{e}");
                }
            }
            #[test]
            fn round_trip() {
                run(suite::round_trip)
            }
            #[test]
            fn projection() {
                run(suite::projection)
            }
            #[test]
            fn pruning_soundness() {
                run(suite::pruning_soundness)
            }
            #[test]
            fn selection_and_row_index() {
                run(suite::selection_and_row_index)
            }
            #[test]
            fn split_invariance() {
                run(suite::split_invariance)
            }
            #[test]
            fn schema_evolution() {
                run(suite::schema_evolution)
            }
            #[test]
            fn corrupt_inputs() {
                run(suite::corrupt_inputs)
            }
            #[test]
            fn memory_accounting() {
                run(suite::memory_accounting)
            }
            #[test]
            fn cancellation() {
                run(suite::cancellation)
            }
            #[test]
            fn io_behaviour() {
                run(suite::io_behaviour)
            }
            #[test]
            fn options() {
                run(suite::options)
            }
            #[test]
            fn writer() {
                run(suite::writer)
            }
            #[test]
            fn dynamic_filter_pruning() {
                run(suite::dynamic_filter_pruning)
            }
            #[test]
            fn take() {
                run(suite::take)
            }
        }
    };
}
