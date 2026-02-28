# Ray Data Runner Design and Plan

## Scope

### In scope

- Add a `RayDataRunner` implementation of `RunnerInterface` that executes `CuratorStage` pipelines using Ray Data
  instead of Cosmos-Xenna.
- Both runners coexist — Xenna remains the default, Ray Data is opt-in.
- All existing stages run unmodified on either runner.
- Pixi multi-environment support works identically on both runners.

### Out of scope

- Removing Cosmos-Xenna or deprecating `XennaRunner`.
- Changing the `CuratorStage` or `PipelineTask` interfaces.
- Adopting PyArrow tables as the inter-stage data format (see rationale below).
- Ray Data Streaming (serve/online inference mode).

---

## Why This Runner

Cosmos-Xenna implements a custom streaming execution engine — actor pools, queues, auto-scaling, backpressure, work
stealing — on top of Ray Core primitives. This engine works but is proprietary, hard to debug externally, and does not
benefit from upstream Ray improvements.

Ray Data provides the same capabilities (streaming execution, actor pools, backpressure, auto-scaling, fault tolerance)
as a first-party Ray library. Moving to Ray Data means:

- **Reduced maintenance**: Upstream Ray team maintains the execution engine.
- **Better observability**: Native Ray Dashboard integration for Data pipelines.
- **Dataset checkpointing**: Materialize intermediate results between phases (aligns with the Composable Pipeline
  design).
- **Ecosystem compatibility**: Ray Data datasets can feed directly into Ray Train, Ray Serve, etc.

---

## Architecture

### What stays the same

| Component                                           | Change?                                                                                        |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|
| `CuratorStage` interface                            | No change. `process_data(tasks) -> tasks` is the stable contract.                              |
| `CuratorStageSpec`                                  | No change. `RayDataRunner` reads resource/env fields, ignores Xenna-specific scheduling knobs. |
| `PipelineTask` / `SplitPipeTask` / `Video` / `Clip` | No change. Data model stays as attrs objects.                                                  |
| `run_pipeline()` in `pipeline_interface.py`         | No change. Already accepts `runner: RunnerInterface`.                                          |
| `_build_pipeline_stage_specs()`                     | No change. Normalization, profiling wrappers, and default-filling apply to both runners.       |
| `PixiRuntimeEnv`                                    | No change. Already a `ray.runtime_env.RuntimeEnv` subclass, works with Ray Data directly.      |
| Model download                                      | No change. `_prepare_to_run_pipeline()` / `download_models()` are runner-independent.          |

### What changes

**New file**: `cosmos_curate/core/interfaces/ray_data_runner.py`

Contains `RayDataRunner(RunnerInterface)` — the primary production code change. Minimal CLI integration
(wiring a `--runner` flag into `splitting_pipeline.split()` and `run_pipeline()` callers) is also required.

**New file**: `cosmos_curate/pipelines/examples/hello_ray_data_runner_pipeline.py`

Demonstrates running the existing hello world stages through the Ray Data runner, proving the adapter works end-to-end.

---

## Data Format: Why Not PyArrow Tables

The inter-stage data stays as Python objects (attrs-based `PipelineTask` instances), not PyArrow tables. Rationale:

1. **The data model is a deeply nested object graph**, not a flat table. `SplitPipeTask` → `Video` (list) → `Clip` (
   list) → `Window` (list), each with heterogeneous fields (`bytes`, `ndarray`, nested dicts). Arrow's type system can
   represent this via nested structs and list columns, but every stage that reads `task.video.clips[i].encoded_data`
   would need to be rewritten.

2. **Ray Data supports Python-object batches natively.** `map_batches` with `batch_format="numpy"` passes
   `dict[str, np.ndarray]`. Since our `"task"` column contains Python objects, NumPy wraps them in an object-dtype
   array, which the wrapper indexes into. We use this mode — stages receive the same `list[PipelineTask]` they receive
   today.

3. **Arrow is already used at the right boundary.** `ClipWriterStage` converts to `pa.Table` for Lance writes. Input
   manifests can be read as Arrow. The storage boundary is where columnar format pays off, not in-flight between GPU
   inference stages.

4. **No rewrite tax.** 50+ existing stages implement `process_data(tasks: list[PipelineTask])`. Keeping Python objects
   means zero stage changes.

---

## RayDataRunner Design

### The adapter pattern

Each `CuratorStage` is wrapped in a callable class that Ray Data's `map_batches` can use with `ActorPoolStrategy`:

```python
class _StageActorWrapper:
    """Wraps a CuratorStage as a Ray Data map_batches callable."""

    def __init__(self, stage: CuratorStage, num_setup_attempts: int = 3):
        self._stage = stage
        self._destroyed = False
        for attempt in range(num_setup_attempts):
            try:
                self._stage.stage_setup()
                break
            except Exception:
                if attempt == num_setup_attempts - 1:
                    raise
                logger.warning("stage_setup() failed (attempt %d/%d), retrying...", attempt + 1, num_setup_attempts, exc_info=True)

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        tasks = batch["task"]
        results = self._stage.process_data(tasks)
        if results is None:
            return {"task": []}
        return {"task": results}

    def __ray_shutdown__(self):
        """Ray actor lifecycle hook — called on graceful actor shutdown."""
        if not self._destroyed:
            self._stage.destroy()
            self._destroyed = True

    def __del__(self):
        # Fallback cleanup if __ray_shutdown__ did not fire (e.g., forced kill).
        if not self._destroyed:
            self._stage.destroy()
            self._destroyed = True
```

The wrapper is intentionally thin — it delegates everything to the existing `CuratorStage` methods. Cleanup uses Ray's
`__ray_shutdown__` lifecycle hook for deterministic resource release (GPU memory, file handles), with `__del__` as a
fallback for force-kill scenarios. The `_destroyed` flag ensures `destroy()` is idempotent. The retry loop in `__init__`
handles transient `stage_setup()` failures (see `num_setup_attempts_python` mapping below).

### Mapping CuratorStageSpec to Ray Data

| CuratorStageSpec field      | Ray Data equivalent                                     |
|-----------------------------|---------------------------------------------------------|
| `stage.resources.gpus`      | `num_gpus` parameter on `map_batches`                   |
| `stage.resources.cpus`      | `num_cpus` parameter on `map_batches`                   |
| `stage.conda_env_name`      | `runtime_env=PixiRuntimeEnv(name)`                      |
| `stage.stage_batch_size`    | `batch_size` parameter (see batch size discussion below)                                         |
| `num_workers_per_node`      | `compute=ActorPoolStrategy(min_size=..., max_size=...)` |
| `over_provision_factor`     | Ignored (Ray Data has its own auto-scaling)             |
| `worker_max_lifetime_m`     | Ignored (Ray Data manages actor lifecycle)              |
| `worker_restart_interval_m` | Ignored                                                 |
| `num_run_attempts_python`   | `max_restarts_per_actor` on `ActorPoolStrategy`         |
| `num_setup_attempts_python` | Retry loop around `stage_setup()` in `_StageActorWrapper.__init__` |

### Batch size design

All existing `CuratorStage` implementations use `stage_batch_size = 1` (the default). The runner passes this directly
to `map_batches(batch_size=stage.stage_batch_size)`. This means Ray Data creates single-item batches, which has higher
per-item task/serialization overhead compared to `batch_size=None` (block-level passing).

This trade-off is intentional:

1. **Correctness first.** `batch_size=1` matches the current Xenna behavior exactly. Every stage's `process_data()`
   receives a one-element list, which is how all 50+ stages are tested today.
2. **Overhead is negligible for our workloads.** Stages perform heavy GPU inference (captioning, embedding) or I/O
   (video decode, transcode). The per-batch dispatch overhead is dwarfed by actual processing time.
3. **Future batching is opt-in.** When a stage overrides `stage_batch_size` to a value > 1, the runner will
   automatically pass larger batches to `map_batches`. No runner changes are needed — the stage just needs to handle
   multi-item input in `process_data()`.

### Node-level setup

`CuratorStage` has a two-phase setup: `stage_setup_on_node()` (once per node, e.g. copy weights to local SSD) and
`stage_setup()` (once per actor). Ray Data does not have a native "once per node" hook.

Options:

1. **Run `stage_setup_on_node()` inside `__init__` with a node-local lock.** Use a file lock keyed by
   `(stage_name, ray.get_runtime_context().get_node_id())` so the first actor on each node runs it, others wait.
2. **Run node setup as a pre-step.** Before building the Ray Data pipeline, schedule one Ray task per node per stage
   that needs node setup. This mirrors what Xenna does internally.

Option 2 is simpler and more predictable. The `RayDataRunner.run()` method will handle this before constructing the
dataset pipeline.

### Pipeline construction

```python
class RayDataRunner(RunnerInterface):
    def run(self, input_tasks, stage_specs, model_weights_prefix):
        # 1. Download models (reuse existing logic).
        # _prepare_to_run_pipeline() returns ExecutionMode (STREAMING vs BATCH based on
        # GPU availability). Ray Data ignores this — it always streams and handles
        # insufficient GPUs via queuing rather than falling back to batch mode.
        _prepare_to_run_pipeline(stage_specs, model_weights_prefix)

        try:
            # 2. Run per-node setup for stages that need it
            self._run_node_setup(stage_specs)

            # 3. Build Ray Data pipeline
            ds = ray.data.from_items([{"task": t} for t in input_tasks])

            for spec in stage_specs:
                stage = spec.stage
                ds = ds.map_batches(
                    _StageActorWrapper,
                    fn_constructor_args=(stage,),
                    batch_size=stage.stage_batch_size,
                    batch_format="numpy",
                    compute=self._actor_pool_strategy(spec),
                    num_gpus=stage.resources.gpus if stage.resources.gpus > 0 else 0,
                    num_cpus=stage.resources.cpus,
                    runtime_env=self._build_runtime_env(stage),
                )

            # 4. Materialize and extract results
            results = ds.materialize()
            return [row["task"] for row in results.iter_rows()]
        finally:
            if ray.is_initialized():
                shutdown_cluster()

    def _run_node_setup(self, stage_specs):
        """Schedule one Ray task per node for stages with non-trivial stage_setup_on_node().

        Detection: a stage needs node setup if its class overrides stage_setup_on_node()
        (i.e., type(stage).stage_setup_on_node is not CuratorStage.stage_setup_on_node).

        Execution: for each (stage, node) pair, schedule a Ray task pinned to that node
        via ray.remote(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).
        Collect all futures and call ray.get() to block until all complete — actors must not
        start before node setup finishes.

        Error handling: if any node setup task fails, re-raise immediately and abort pipeline
        construction. Retries can be added later if transient failures are observed in practice.
        """
        ...

    def _actor_pool_strategy(self, spec) -> ActorPoolStrategy:
        """Map CuratorStageSpec worker counts to Ray Data ActorPoolStrategy.

        Includes max_restarts_per_actor from spec.num_run_attempts_python.
        """
        ...

    def _build_runtime_env(self, stage) -> RuntimeEnv | None:
        """Build a PixiRuntimeEnv from the stage's conda_env_name, or None for default."""
        ...
```

### Handling stage output cardinality

`CuratorStage.process_data()` can return a different number of tasks than it receives (dynamic chunking — e.g.,
splitting a video into clips produces more tasks). It can also return `None` to drop tasks (filtering).

Ray Data's `map_batches` supports this: the output batch can have a different size than the input batch. Returning a
batch with zero rows (e.g., `{"task": []}`) drops those tasks. This maps directly to the wrapper's `__call__` behavior.

### Handling fractional GPUs

Xenna supports fractional GPUs (e.g., `gpus=0.25`) for stages that share a GPU. Ray Data's `map_batches` also supports
fractional `num_gpus`. No special handling needed.

---

## What We Lose (and Whether It Matters)

| Xenna feature                                       | Ray Data equivalent                               | Gap                                                                                         |
|-----------------------------------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------|
| Custom auto-scaler (3-min window, speed estimation) | Ray Data auto-scaling (backpressure-driven)       | Different heuristics, likely fine. May need tuning.                                         |
| Work stealing between actor pools                   | Not available in Ray Data                         | Loss. Impact unclear — Xenna disables it anyway (`enable_work_stealing=False`).             |
| `slots_per_actor=2` (prefetch next batch)           | Ray Data has internal prefetching                 | Equivalent.                                                                                 |
| Per-stage monitoring verbosity                      | Ray Dashboard                                     | Different UX, more standard.                                                                |
| Batch execution fallback (not enough GPUs)          | Ray Data handles resource constraints via queuing | Different approach — Ray Data queues instead of serializing stages. May actually be better. |

The biggest behavioral difference: Xenna's streaming executor runs all stages concurrently with explicit backpressure
control (`max_queued_multiplier=1.5`). Ray Data also streams with backpressure but uses its own heuristics. Pipeline
throughput characteristics may differ and will need benchmarking.

---

## CLI Integration

A new flag selects the runner:

```text
--runner {xenna,ray-data}    Execution backend (default: xenna)
```

This flag is added to `run_pipeline()` callers (e.g., `splitting_pipeline.split()`). It constructs the appropriate
runner:

```python
runner = RayDataRunner() if args.runner == "ray-data" else XennaRunner()
run_pipeline(input_tasks, stages, runner=runner, ...)
```

No changes to stage definitions, argparse stage flags, or pipeline assembly logic.

---

## Testing Strategy

### Unit tests

- `test_stage_actor_wrapper.py`: Verify the wrapper correctly forwards `process_data`, handles `None` returns (
  filtering), handles output cardinality changes (chunking).
- `test_ray_data_runner.py`: Verify pipeline construction, resource mapping, runtime env mapping.
- `test_node_setup.py`: Verify node setup runs exactly once per node per stage (not per actor), completes before actors
  start processing (use synchronization primitives), and that failures abort pipeline construction.

### Integration tests

- Run `hello_world_pipeline` through both runners, assert identical outputs.
- Run a minimal split pipeline (download → remux → fixed-stride → transcode → write) through both runners on a single
  node with synthetic input. Compare output clips.

### Benchmark

- Run the full splitting pipeline on a representative dataset with both runners. Compare:
    - Wall-clock time
    - GPU utilization over time
    - Peak memory usage
    - Throughput (clips/sec)

This benchmark is the gate for promoting Ray Data runner from experimental to default.

---

## Relationship to Composable Pipeline Design

The Composable Pipeline extension design introduces `CurationPhase` and `PipelineBuilder`. That
design is **orthogonal to this one** — phases produce `list[CuratorStage | CuratorStageSpec]`, which both runners
consume identically.

One synergy: Ray Data's `Dataset.materialize()` enables **phase-boundary checkpointing**. If the `PipelineBuilder`
constructs the pipeline phase-by-phase, the `RayDataRunner` could optionally materialize between phases, enabling:

- Resume from a failed phase without re-running earlier phases
- Phase-level profiling and monitoring
- Writing intermediate Lance checkpoints at phase boundaries

This is a future enhancement, not part of the initial implementation.

---

## Implementation Plan

Merge requests, ordered by dependency:

- [ ] Add `RayDataRunner` with stage wrapper
- [ ] Add node-setup pre-step
- [ ] Add `--runner` CLI flag
- [ ] Add hello world example
- [ ] Integration test: minimal split pipeline
- [ ] Benchmark on representative workload

MRs 1-4 are pure additive code with no changes to existing files (except the CLI flag wiring in MR 3). MR 5 validates
correctness. MR 6 gates promotion to default.

---

## Open Questions

- **Ray Data version pinning**: Which minimum Ray version do we target? `ActorPoolStrategy` API has changed across Ray
  2.x releases. The current Ray version in `pixi.toml` should be checked.
- **Profiling wrapper compatibility**: The existing `profiling_wrapper()` monkey-patches `process_data()` on
  `CuratorStage`. This should work transparently with the Ray Data wrapper since it patches before the wrapper captures
  the stage, but needs verification.
- **`stage_save_wrapper` compatibility**: Same question — the replay/save wrapper patches `process_data()`. Should work
  but needs a test.
- **Multi-node data locality**: Xenna's queue system tracks which node produced each ObjectRef and tries to schedule
  downstream work on the same node. Ray Data has its own locality-aware scheduling but the heuristics differ. This may
  affect performance for large payloads (encoded video bytes).
