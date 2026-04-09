# Ray Data Runner Design and Plan

## Summary

This document defines the architecture and execution semantics for adding a `RayDataRunner` backend to Cosmos Curate.
The goal is to let existing `CuratorStage` pipelines run on Ray Data without changing stage or task interfaces, while
keeping Xenna as the default backend and Ray Data as an opt-in alternative. It also defines the first implementation
slice: CLI runner selection and a runner-backed hello-world example.

## Scope

### In scope

- Add a `RayDataRunner` implementation of `RunnerInterface` that executes `CuratorStage` pipelines using Ray Data
  instead of Cosmos-Xenna.
- Both runners coexist — Xenna remains the default, Ray Data is opt-in.
- All existing stages run unmodified on either runner.
- Pixi multi-environment support is a target behavior on both runners.

### Out of scope

- Removing Cosmos-Xenna or deprecating `XennaRunner`.
- Changing the `CuratorStage` or `PipelineTask` interfaces.
- Adopting PyArrow tables as the inter-stage data format (see rationale below).
- Ray Data Streaming (serve/online inference mode).

---

## Why Ray Data Runner

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

## Architecture Invariants

### What stays the same

| Component                                           | Change?                                                                                        |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|
| `CuratorStage` interface                            | No change. `process_data(tasks) -> tasks \| None` is the stable contract.                      |
| `CuratorStageSpec`                                  | No change. `RayDataRunner` reads resource/env fields, ignores Xenna-specific scheduling knobs. |
| `PipelineTask` / `SplitPipeTask` / `Video` / `Clip` | No change. Data model stays as attrs objects.                                                  |
| `run_pipeline()` in `pipeline_interface.py`         | No change. Already accepts `runner: RunnerInterface`.                                          |
| `_build_pipeline_stage_specs()`                     | No change. Normalization, profiling wrappers, and default-filling apply to both runners.       |
| `PixiRuntimeEnv`                                    | No change. Already a `ray.runtime_env.RuntimeEnv` subclass, works with Ray Data directly.      |
| Model download                                      | No change. `_prepare_to_run_pipeline()` / `download_models()` are runner-independent.          |

The stage builder functions (e.g. `build_ingest_stages`, `build_transcode_stages`) continue to produce
`list[CuratorStage | CuratorStageSpec]`, which both runners consume identically. They remain a construction-time
convenience and do not participate in data flow at runtime.

Runner selection is orthogonal to the Ray Data executor design above. Callers choose which `RunnerInterface`
implementation to construct and pass into `run_pipeline()`; this document's First Implementation Slice defines the
initial `--runner` wiring for the split pipeline.

One future synergy is group-boundary checkpointing. Ray Data's `Dataset.materialize()` could support materializing
between logical groups of stages, enabling resume points, group-level monitoring, and intermediate Lance checkpoints.
This is a future enhancement, not part of the initial runner design.

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

## Runner Design

### `_StageActorWrapper` adapter pattern

Each `CuratorStage` is wrapped in a callable class that Ray Data's `map_batches` can use with `ActorPoolStrategy`:

```python
class _StageActorWrapper:
    """Wraps a CuratorStage as a Ray Data map_batches callable."""

    def __init__(self, stage: CuratorStage, num_setup_attempts: int):
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

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        tasks = batch["task"].tolist()
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

### `CuratorStageSpec` to Ray Data mapping

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

### Batch size semantics

`CuratorStage.stage_batch_size` defaults to `1`, which matches the behavior of most existing stages. Some stages
intentionally override it, and the runner should pass `stage.stage_batch_size` directly to
`map_batches(batch_size=stage.stage_batch_size)`. When the effective batch size is `1`, Ray Data creates single-item
batches, which has higher per-item task/serialization overhead compared to `batch_size=None` (block-level passing).

This trade-off is intentional:

1. **Correctness first.** The runner should pass `stage.stage_batch_size` through directly, matching Xenna's
   per-stage batching behavior. For stages that keep the default batch size of `1`, `process_data()` receives
   one-element task lists.
2. **Overhead is often negligible for our workloads.** Many stages perform heavy GPU inference (captioning, embedding)
   or I/O (video decode, transcode), where the per-batch dispatch overhead is dwarfed by actual processing time.
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
        runner_owns_ray_session = False

        # 1. Download models (reuse existing logic).
        # _prepare_to_run_pipeline() returns ExecutionMode (STREAMING vs BATCH based on
        # GPU availability). Ray Data ignores this — it always streams and handles
        # insufficient GPUs via queuing rather than falling back to batch mode.
        _prepare_to_run_pipeline(stage_specs, model_weights_prefix)

        try:
            # 2. Connect to Ray if needed and record lifecycle ownership.
            runner_owns_ray_session = self._ensure_ray_session()

            # 3. Run per-node setup for stages that need it
            self._run_node_setup(stage_specs)

            # 4. Build Ray Data pipeline
            ds = ray.data.from_items([{"task": t} for t in input_tasks])

            for spec in stage_specs:
                stage = spec.stage
                ds = ds.map_batches(
                    _StageActorWrapper,
                    fn_constructor_args=(stage, spec.num_setup_attempts_python or 3),
                    batch_size=stage.stage_batch_size,
                    batch_format="numpy",
                    compute=self._actor_pool_strategy(spec),
                    num_gpus=stage.resources.gpus if stage.resources.gpus > 0 else 0,
                    num_cpus=stage.resources.cpus,
                    runtime_env=self._build_runtime_env(stage),
                )

            # 5. Materialize and extract results
            results = ds.materialize()
            return [row["task"] for row in results.iter_rows()]
        finally:
            if runner_owns_ray_session and ray.is_initialized():
                shutdown_cluster()

    def _ensure_ray_session(self) -> bool:
        """Connect to Ray if needed and return whether this runner owns the session."""
        ...

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

### Output cardinality and filtering behavior

`CuratorStage.process_data()` can return a different number of tasks than it receives (dynamic chunking — e.g.,
splitting a video into clips produces more tasks). It can also return `None` for filtering behavior.

Ray Data's `map_batches` supports this: the output batch can have a different size than the input batch. Returning a
batch with zero rows (e.g., `{"task": []}`) represents task dropping for that batch. The runner should preserve the
intended filtering behavior of stages that return `None`.

### Fractional GPU handling

Xenna supports fractional GPUs (e.g., `gpus=0.25`) for stages that share a GPU. Ray Data's `map_batches` also supports
fractional `num_gpus`. No special handling needed.

---

## Xenna vs Ray Data Behavioral Differences

| Xenna feature                                       | Ray Data equivalent                               | Gap                                                                                         |
|-----------------------------------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------|
| Custom auto-scaler (3-min window, speed estimation) | Ray Data auto-scaling (backpressure-driven)       | Different heuristics, likely fine. May need tuning.                                         |
| Work stealing between actor pools                   | Not available in Ray Data                         | Loss. Impact unclear — Xenna disables it anyway (`enable_work_stealing=False`).             |
| `slots_per_actor=2` (prefetch next batch)           | Ray Data has internal prefetching                 | Equivalent.                                                                                 |
| Per-stage monitoring verbosity                      | Ray Dashboard                                     | Different UX, more standard.                                                                |
| Batch execution fallback (not enough GPUs)          | Ray Data handles resource constraints via queuing | Different approach — Ray Data queues instead of serializing stages. May actually be better. |

The biggest behavioral difference: Xenna's streaming executor runs all stages concurrently with explicit backpressure
control (`max_queued_multiplier=1.5`). Ray Data also streams with backpressure but uses its own heuristics. Pipeline
throughput characteristics may differ. Benchmarking is deferred to a separate follow-on effort.

Xenna uses `_prepare_to_run_pipeline()` to choose between `STREAMING` and `BATCH` based on available GPUs. The Ray
Data runner still reuses `_prepare_to_run_pipeline()` for model download and cluster preparation, but it does not adopt
Xenna's execution-mode fallback semantics. Ray Data remains a streaming dataset pipeline and relies on resource queuing
when GPUs are constrained.

---

## Must-Resolve Before Implementation

- **Cluster lifecycle ownership**: `RayDataRunner` must track whether it initialized or connected to Ray and must only
  shut down cluster state that it owns. Unconditional shutdown at the end of `run()` is not acceptable.
- **Profiling / stage-save compatibility**: The first slice routes through `_build_pipeline_stage_specs()`, so
  profiling and stage-save compatibility cannot be left implicit. Profiling compatibility must be verified before the
  first slice is declared complete, either through code inspection or testing. Stage-save must be either verified for
  the first slice or explicitly declared unsupported.
- **Hello-world example contract**: The runner-backed hello-world example must call `run_pipeline()` directly with an
  explicit `RayDataRunner` instance, reuse the same stages as `hello_world_pipeline.py`, and remain script-driven
  rather than CLI-driven.

---

## Open Questions

- **`num_workers_per_node` under Ray Data**: The current design assumes a global actor-pool size derived from per-node
  intent, but it does not yet define how strictly per-node worker semantics are preserved. This is follow-on design
  work under `CVC-799`.
- **Ray Data version pinning**: Which minimum Ray version do we target? `ActorPoolStrategy` API has changed across Ray
  2.x releases. The current Ray version in `pixi.toml` should be checked.
- **Multi-node data locality**: Xenna's queue system tracks which node produced each ObjectRef and tries to schedule
  downstream work on the same node. Ray Data has its own locality-aware scheduling but the heuristics differ. This may
  affect performance for large payloads (encoded video bytes).

---

## First Implementation Slice

The first implementation slice exposes Ray Data as an opt-in backend through an existing pipeline entrypoint and
proves the runner-backed path works end-to-end with the hello-world example. Broader runner validation and
benchmarking remain deferred.

This initial slice is intended to prove the single-node runner path and the first end-to-end integration path;
multi-node worker-placement and node-setup semantics remain follow-on design work under `CVC-799`.

### In-scope

- `--runner {xenna,ray-data}` wiring
- a runner-backed hello-world example

This slice does not attempt to validate every `CuratorStage` behavior or claim full parity with Xenna across all
pipelines. It establishes the first end-to-end integration path for selecting and exercising the Ray Data runner.

### CLI integration

Add an explicit runner-selection flag to `cosmos_curate.pipelines.video.splitting_pipeline.split()` for this slice:

```text
--runner {xenna,ray-data}    Execution backend (default: xenna)
```

Callers construct the appropriate runner and pass it into `run_pipeline(...)`. Stage definitions, stage assembly logic,
and argparse stage-specific flags remain unchanged.

### What changes in this slice

This slice introduces the minimum implementation needed to exercise the Ray Data backend through the selected split
pipeline entrypoint:

- wire `--runner` through `cosmos_curate.pipelines.video.splitting_pipeline.split()`
- construct `RayDataRunner()` when `args.runner == "ray-data"`
- keep `XennaRunner()` as the default path
- add `cosmos_curate/pipelines/examples/hello_ray_data_runner_pipeline.py` as a runner-backed hello-world example that
  uses existing hello-world stages through the `RunnerInterface` path

The old raw Ray Data proof-of-concept example in
`cosmos_curate/pipelines/examples/hello_ray_data_pipeline.py` is not the target architecture for this slice and should
not be used as the reference implementation pattern.

### Acceptance criteria

This slice is integration-first by design.

Acceptance requires:

- the selected CLI entrypoint accepts `--runner {xenna,ray-data}`
- omitting `--runner` preserves the current Xenna default
- selecting `--runner ray-data` routes execution through `RayDataRunner`
- the runner-backed hello-world example executes successfully through the Ray Data path
- the runner-backed hello-world example uses the existing stage model rather than a raw Ray Data table/row rewrite

#### Hello-world equivalence

Hello-world equivalence for this slice means behavioral parity at the example level, not byte-for-byte identical model
output.

The acceptance check is:

- both backends run the same ordered hello-world stages
- both backends complete successfully on the same input prompts
- both backends perform the same prompt-lowercasing and prompt-printing behavior
- both backends produce non-empty GPT-2 output and attach it to the task/output path expected by the example

Exact string equality for generated GPT-2 text is not the acceptance gate for this slice.

### Explicit non-goals for this slice

This slice does not validate:

- node-level setup behavior
- Pixi multi-environment switching or parity with Xenna
- broader stage compatibility across video/AV pipelines
- throughput, resource efficiency, or benchmark parity

### Deferred items

The following remain outside this first implementation slice:

- `_StageActorWrapper` unit tests
- broader `RayDataRunner` unit tests
- node setup validation
- Pixi multi-environment validation
- broader integration coverage beyond the hello-world path
- benchmark planning and benchmark execution
