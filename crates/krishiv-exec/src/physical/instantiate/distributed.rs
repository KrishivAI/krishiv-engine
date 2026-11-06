// SPDX-License-Identifier: Apache-2.0
//! The parts of instantiation that only a task of a cut plan uses (design
//! P2-T176, P2-T177): a map task's exchange write (with the runtime filters
//! it collects), a reduce task's exchange reads, the values earlier stages
//! computed, and broadcasts, shared per worker when the worker has
//! [`BroadcastTables`].

use super::*;

impl Instantiation<'_> {
    /// Publishes the values of the task's cross-stage filters (computed by
    /// earlier stages) into the cells its scans declared.
    pub(super) fn take_filters(&mut self) -> Result<()> {
        for (id, bytes) in &self.task.filters {
            if self.values.filter(*id).is_err() {
                // No scan of this fragment reads it.
                continue;
            }
            let keys = KeyFilter::decode(bytes)
                .map_err(|e| internal(format!("runtime filter {id} of the task: {e}")))?;
            self.values.publish(Partial {
                value: ValueId::Filter(*id),
                version: Version::BATCH,
                producer: Producer::default(),
                payload: Value::Filter(FilterSnapshot::Final(Arc::new(keys))),
            })?;
        }
        Ok(())
    }

    /// The exchange `e` written as this map task's output, over
    /// `partitions` reduce partitions, collecting its filters.
    pub(super) fn exchange_write(&mut self, e: &ExchangePlan, partitions: u32) -> Result<Built> {
        let exchange = self.exchange()?;
        for f in &e.filters {
            // Declared here, with no scan: the task's value service keeps
            // the partial for whoever runs the task.
            if self.values.filter(f.id).is_err() {
                self.values.declare_filter(
                    f.id,
                    ColumnPath::new(format!("exchange-{}", e.id.0)),
                    f.key_type.clone(),
                )?;
            }
        }
        let input = self.node(&e.input)?;
        let sink = Arc::new(
            ExchangeSink::new(
                exchange,
                MapOutputId {
                    exchange: e.id,
                    stage_attempt: self.task.stage_attempt,
                    map: self.task.partition,
                    attempt: self.task.attempt,
                },
                e.partitioning.clone(),
                partitions,
                Arc::clone(&input.schema),
            )?
            .with_filters(
                e.filters.clone(),
                Arc::clone(&self.values),
                self.worker.producer,
            ),
        );
        Ok(Built {
            schema: input.schema,
            node: Arc::new(ExchangeWrite {
                input: input.node,
                sink,
            }),
        })
    }

    /// The rows this task reads from the exchange `id` of another stage
    /// (its [`TaskSpec::exchanges`] entry). A gather is read by one driver
    /// in map order; a broadcast is published as this producer's partial,
    /// as on one worker.
    pub(super) fn exchange_source(
        &mut self,
        id: crate::physical::ExchangeId,
        partitioning: &Partitioning,
        schema: &SchemaRef,
    ) -> Result<Built> {
        let exchange = self.exchange()?;
        let input = self.task.exchanges.get(&id).ok_or_else(|| {
            internal(format!(
                "exchange {} is read by a task that names no input of it",
                id.0
            ))
        })?;
        let ordered = *partitioning == Partitioning::Single;
        let source = Arc::new(ExchangeRead::new(
            exchange,
            id,
            input.stage_attempt,
            &input.reduce,
            input.maps,
            Arc::clone(schema),
            ordered,
        ));
        let node: Arc<dyn ExecNode> = Arc::new(SourceNode(source));
        let built = Built {
            schema: Arc::clone(schema),
            node: if ordered {
                Arc::new(OrderedNode { inner: node })
            } else {
                node
            },
        };
        if *partitioning == Partitioning::Broadcast {
            return self.publish_broadcast(id, built);
        }
        Ok(built)
    }

    /// The build of a split nested-loop join's half (design P2-T052b,
    /// 2026-11-06): its cut broadcast exchange read by one driver in
    /// exchange order (map partition, frame, row), so every task numbers
    /// the build rows alike. Not published as a broadcast value: nothing
    /// shares it.
    pub(super) fn ordered_broadcast(&mut self, build: &PhysNode) -> Result<Built> {
        let PhysNode::ExchangeSource {
            id,
            partitioning: Partitioning::Broadcast,
            schema,
        } = build
        else {
            return Err(internal(
                "a split nested-loop join whose build is not a cut broadcast exchange".to_owned(),
            ));
        };
        self.ordered_read(*id, schema)
    }

    /// Every reduce partition this task reads of the exchange `id`, by one
    /// driver in `(reduce, map)` order.
    fn ordered_read(&self, id: crate::physical::ExchangeId, schema: &SchemaRef) -> Result<Built> {
        let exchange = self.exchange()?;
        let input = self.task.exchanges.get(&id).ok_or_else(|| {
            internal(format!(
                "exchange {} is read by a task that names no input of it",
                id.0
            ))
        })?;
        let source = Arc::new(ExchangeRead::new(
            exchange,
            id,
            input.stage_attempt,
            &input.reduce,
            input.maps,
            Arc::clone(schema),
            true,
        ));
        Ok(Built {
            schema: Arc::clone(schema),
            node: Arc::new(OrderedNode {
                inner: Arc::new(SourceNode(source)),
            }),
        })
    }

    /// The build rows of a split nested-loop join (its second half): the
    /// broadcast read again in exchange order, chosen by the bitmap the
    /// task brings ([`TaskSpec::marks`]).
    pub(super) fn unmatched_build(&mut self, u: &UnmatchedBuildPlan) -> Result<Built> {
        let bytes = self.task.marks.get(&u.marks).ok_or_else(|| {
            internal(format!(
                "nested-loop marks #{} are read by a task that brings none",
                u.marks.0
            ))
        })?;
        let marks = BuildMarks::decode(bytes)?;
        let input = self.ordered_read(u.exchange, &u.build_schema)?;
        let join = NestedLoopJoin::try_new(NestedLoopSpec {
            join_type: u.join_type,
            probe_schema: Arc::clone(&u.probe_schema),
            build_schema: Arc::clone(&u.build_schema),
            filter: None,
            spill: None,
            publish: None,
        })?;
        let op = join.unmatched_build(marks)?;
        Ok(Built {
            schema: join.schema(),
            node: Arc::new(OperatorNode {
                input: input.node,
                op,
            }),
        })
    }

    pub(super) fn exchange(&self) -> Result<Arc<dyn Exchange>> {
        self.worker
            .exchange
            .clone()
            .ok_or_else(|| internal("a task with exchanges on a worker without one".to_owned()))
    }

    /// The input of the broadcast exchange `id`, published as this
    /// producer's partial of its value: on one worker the join over it
    /// shares one build among its drivers.
    pub(super) fn broadcast(
        &mut self,
        id: crate::physical::ExchangeId,
        input: &PhysNode,
    ) -> Result<Built> {
        let input = self.node(input)?;
        self.schemas.insert(id, Arc::clone(&input.schema));
        self.publish_broadcast(id, input)
    }

    /// `input` published as this producer's partial of the broadcast
    /// exchange `id`.
    pub(super) fn publish_broadcast(
        &mut self,
        id: crate::physical::ExchangeId,
        input: Built,
    ) -> Result<Built> {
        let value = ValueId::Broadcast(id);
        let published = self.values.publish(Partial {
            value,
            version: Version::BATCH,
            producer: self.worker.producer,
            payload: Value::Broadcast(BroadcastBuild::new(&input.node)),
        })?;
        if published != Publication::Complete {
            // In one process every producer partition is this one: another
            // partial (`Waiting`) or a second exchange of this id
            // (`Ignored`) cannot be lowered.
            return Err(internal(format!(
                "{value} is {published:?} in this instantiation"
            )));
        }
        // The value names this input without owning it (see
        // `BroadcastBuild`): the plan holds it.
        Ok(input)
    }

    /// `j` over the worker's shared build of its broadcast input, when it
    /// can share one (design P2-T177 unit 3, [`BroadcastTables`]): a
    /// broadcast join reading a cut broadcast exchange, on a worker with
    /// the resource, not building its preserved side. `None` builds as
    /// usual.
    pub(super) fn shared_broadcast_join(&mut self, j: &HashJoinPlan) -> Result<Option<Built>> {
        let Some(tables) = self.worker.broadcasts.clone() else {
            return Ok(None);
        };
        let PhysNode::ExchangeSource {
            id,
            partitioning: Partitioning::Broadcast,
            schema,
        } = &*j.build
        else {
            return Ok(None);
        };
        if j.distribution != crate::physical::JoinDistribution::Broadcast
            || j.join_type.builds_preserved()
        {
            return Ok(None);
        }
        let exchange = self.exchange()?;
        let input = self.task.exchanges.get(id).ok_or_else(|| {
            internal(format!(
                "exchange {} is read by a task that names no input of it",
                id.0
            ))
        })?;
        let key = BroadcastKey {
            exchange: *id,
            stage_attempt: input.stage_attempt,
        };
        let rows: Arc<dyn ExecNode> = Arc::new(SourceNode(Arc::new(ExchangeRead::new(
            exchange,
            *id,
            input.stage_attempt,
            &input.reduce,
            input.maps,
            Arc::clone(schema),
            false,
        ))));
        // The shared build's filters publish into a service of its own; the
        // task republishes them (see `BroadcastTables`).
        let own = LocalValues::new();
        for f in &j.runtime_filters {
            let cell = self.values.filter(f.id)?;
            own.declare_filter(f.id, cell.target().clone(), cell.key_type().clone())?;
        }
        let own: Arc<dyn ValueService> = Arc::new(own);
        let probe = self.node(&j.probe)?;
        let shared = hash_join(
            &probe.schema,
            schema,
            j,
            &self.env,
            None,
            &own,
            Producer::default(),
        )?;
        let join = hash_join(
            &probe.schema,
            schema,
            j,
            &self.env,
            self.spill.clone(),
            &self.values,
            self.worker.producer,
        )?;
        let sink = Arc::new(PrebuiltSink::new(
            &join,
            tables,
            key,
            BuildJob::new(shared, rows),
        ));
        Ok(Some(Built {
            schema: join.schema(),
            node: Arc::new(SharedBuildJoinNode {
                join,
                sink,
                build_schema: Arc::clone(schema),
                probe: probe.node,
            }),
        }))
    }
}
