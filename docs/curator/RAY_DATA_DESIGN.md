# Ray Data Design

## Summary

This document defines the architecture for building Cosmos Curate pipelines on Ray Data as an alternative execution
engine alongside Cosmos-Xenna. Ray Data pipelines are **implemented separately** from Xenna pipelines — no shared
Arrow functions, no adapter layers, no bridge code. Each Ray Data pipeline is written idiomatically using
column expressions and `map_batches`, while Xenna pipelines continue to run unchanged. The two engines may coexist
long-term, with each used where it fits best.

## Why Ray Data

Cosmos-Xenna implements a custom streaming execution engine — actor pools, queues, auto-scaling, backpressure, work
stealing — on top of Ray Core primitives. This engine works but is proprietary, hard to debug externally, and does not
benefit from upstream Ray improvements.

Ray Data provides the same capabilities (streaming execution, actor pools, backpressure, auto-scaling, fault tolerance)
as a first-party Ray library. Building pipelines on Ray Data offers:

- **Reduced maintenance**: Upstream Ray team maintains the execution engine.
- **Better observability**: Native Ray Dashboard integration for Data pipelines.
- **Arrow-native data model**: Ray Data's internal block format is Arrow. Stages that operate on `pa.Table` have
  zero serialization overhead between stages, and intermediate results can be checkpointed to Lance/Parquet with
  no conversion. This also enables a columnar inter-stage contract that makes stages composable and portable
  across pipelines (see [Arrow as the inter-stage contract](#arrow-as-the-inter-stage-contract)).
- **Ecosystem compatibility**: Ray Data datasets feed directly into Ray Train, Ray Serve, etc.
- **Operator fusion**: Ray Data automatically fuses adjacent stateless stages (e.g. lowercasing + printing become a
  single operator), which is a free optimization that Xenna does not provide.

---

## Architecture

### Core principle: separate implementations, shared models

Ray Data pipelines are written independently from Xenna pipelines. Each is idiomatic to its own execution engine.
Shared code is limited to model/inference logic, common utilities, and runtime environment support — pipeline
orchestration (data flow, batching, stage wiring) is implemented separately.

```
              ┌───────────────────┐    ┌────────────────────────┐
              │ Xenna pipeline    │    │ Ray Data pipeline      │
              │ CuratorStage +    │    │ expressions +          │
              │ PipelineTask      │    │ map_batches            │
              └────────┬──────────┘    └───────────┬────────────┘
                       │                           │
                       └──────────┬────────────────┘
                                  │
                       ┌──────────▼─────────┐
                       │  Model / inference │
                       │  code (shared)     │
                       └────────────────────┘
```

- **Lightweight transforms** (text normalization, filters) use column expressions (`ds.with_column`).
- **Stateful stages** (GPU inference) are classes with `__init__` for model loading and `__call__` for inference,
  passed to `map_batches` which manages the actor lifecycle.
- **Xenna pipelines** continue to use `CuratorStage`, `PipelineTask`, and `run_pipeline()` unchanged.

### No runner abstraction for Ray Data

Ray Data *is* the execution engine. A Ray Data pipeline is a chain of Dataset transforms — column expressions
for simple operations, `map_batches` for heavy computation:

```python
ds = ray.data.from_items([{"prompt": p} for p in prompts])
ds = ds.with_column("prompt", col("prompt").str.lower())
ds = ds.map_batches(GPT2Predictor, batch_size=1, batch_format="pyarrow",
                    num_gpus=0.8, compute=ActorPoolStrategy(size=1),
                    runtime_env=PixiRuntimeEnv("transformers"))
ds.show()
```

Resource declarations (`num_gpus`, `num_cpus`), batch size, actor pool sizing, and runtime environments are
`map_batches` kwargs — not stage properties. The processing function doesn't need to know it's running in a pipeline.

### Arrow as the inter-stage contract

In Ray Data pipelines, the inter-stage data format is `pa.Table`. Arrow tables are the contract between stages — a
stage declares what it needs by reading columns and what it produces by adding or transforming them. The schema is
the interface.

**Composability.** Because the contract is "columns in, columns out," stages compose freely. A captioning stage
that reads `"frame"` and writes `"caption"` can slot into any pipeline that provides a `"frame"` column —
video splitting, image curation, or a new pipeline that doesn't exist yet.

**Portability between pipelines.** A deduplication stage that reads `"embedding"` and filters rows works
identically in a video pipeline and an image pipeline — the schema, not a class hierarchy, defines compatibility.

**Composability validation.** Pipeline composition can be validated at construction time by checking that each
stage's input columns exist in the schema produced by prior stages. Arrow schemas provide this validation with
real types.

**No serialization boundary.** Arrow is Ray Data's native block format, so data stays columnar end-to-end with
no pickle overhead. Passing Python objects through Ray Data requires pickle serialization at every stage boundary;
Arrow tables flow through natively.

**Natural checkpointing.** Any stage boundary can be materialized to Lance or Parquet with zero conversion. This
enables resume points, debugging (inspect the table between stages), and incremental pipeline execution.

**Vectorized operations.** Arrow compute functions (e.g. `pc.utf8_lower`, `pc.match_substring`) operate on
entire columns without Python loops, which is significantly faster for text and numeric transformations.

### Relationship to existing components

| Component                   | Role                                                               |
|-----------------------------|--------------------------------------------------------------------|
| `CuratorStage` interface    | Used by Xenna pipelines only. Not used by Ray Data pipelines.      |
| `CuratorStageSpec`          | Used by Xenna pipelines only.                                      |
| `PipelineTask` / data model | Used by Xenna pipelines only. Ray Data pipelines use Arrow tables. |
| `run_pipeline()`            | Xenna entry point. Ray Data pipelines have their own entry points. |
| `PixiRuntimeEnv`            | Shared. Passed as `runtime_env` kwarg to `map_batches`.            |
| Model download              | Shared. `download_models()` is backend-independent.                |
| Model / inference classes   | Shared. The main code reused across both engines.                  |

---

## Building Ray Data Pipelines

New Ray Data pipelines are built one at a time, starting with simple examples and progressing to production workloads.
For pipelines that already have a Xenna implementation, the Ray Data version can be validated against it. Where Ray
Data proves to be the better fit, it may replace the Xenna version; where Xenna works better, it stays.

### Code sharing

Model/inference code (tokenizers, GPU kernels, model weight loading) and common utilities (Arrow helpers, runtime
environments) are shared between both engines. Pipeline orchestration (how data is read, batched, passed between
stages, and written) is implemented separately — the orchestration is what differs between Xenna and Ray Data,
and forcing it through a shared interface constrains both sides without meaningful benefit.

### Example: column expression

```python
ds = ds.with_column("prompt", col("prompt").str.lower())
```

### Example: stateful stage (GPU inference)

For stages that need model lifecycle management, a class with `__init__` (model loading) and `__call__`
(inference) is passed to `map_batches`:

```python
class GPT2Predictor:
    def __init__(self):
        self._model = GPT2()
        self._model.setup()

    def __call__(self, batch: pa.Table) -> pa.Table:
        outputs = [self._model.generate(p) for p in batch["prompt"].to_pylist()]
        return with_column(batch, "output", pa.array(outputs))


ds = ds.map_batches(GPT2Predictor, batch_size=1, batch_format="pyarrow",
                    num_gpus=0.8, compute=ActorPoolStrategy(size=1))
```

### Long-term outlook

If Ray Data proves to be the better engine for all pipelines, the Xenna dependency could eventually be removed.
But both engines may coexist long-term — the separate implementation approach supports either outcome without
upfront commitment.

---

## Pixi Multi-Environment Support

Stages that require specific Pixi environments (e.g. `transformers`, `unified`) pass `runtime_env=PixiRuntimeEnv(name)`
as a `map_batches` kwarg. This is the same `PixiRuntimeEnv` already used by the Xenna path — it is a
`ray.runtime_env.RuntimeEnv` subclass that configures `pixi run` as the Python executable.

---

## Open Questions

- **Multi-node model download**: The Xenna path uses `_prepare_to_run_pipeline()` / `download_models()` to distribute
  model weights across nodes before pipeline execution. Ray Data pipelines need an equivalent pre-step. For
  single-node, model download happens in the actor `__init__`. Multi-node download coordination is deferred.
- **Multi-node data locality**: Xenna tracks which node produced each ObjectRef and tries to co-locate downstream work.
  Ray Data has its own locality-aware scheduling but the heuristics differ. Performance comparison for large
  payloads (encoded video bytes) is needed.

---

## Task List

- [x] `hello_ray_data_pipeline.py`: Standalone Ray Data hello-world pipeline using expressions and `map_batches`
- [ ] Ray Data versions of production pipelines (video splitting, captioning, embedding, filtering)
- [ ] Refactor existing pipeline utilities/helpers to work for both engines where applicable
- [ ] Multi-node model download for Ray Data pipelines
- [ ] Performance comparison between Xenna and Ray Data for the same workloads
