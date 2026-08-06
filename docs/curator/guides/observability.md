# Cosmos Curator - Observability Guide

- [Cosmos Curator - Observability Guide](#cosmos-curator---observability-guide)
  - [Profiling and Instrumentation](#profiling-and-instrumentation)
    - [CLI Flags](#cli-flags)
    - [CPU Profiling (pyinstrument)](#cpu-profiling-pyinstrument)
    - [Memory Profiling (memray)](#memory-profiling-memray)
    - [GPU Profiling (torch.profiler)](#gpu-profiling-torchprofiler)
    - [Distributed Tracing (OpenTelemetry)](#distributed-tracing-opentelemetry)
      - [Instrumenting Custom Stage Code](#instrumenting-custom-stage-code)
    - [Output and Post-processing](#output-and-post-processing)
  - [Structured Logging](#structured-logging)
    - [Enabling](#enabling)
    - [Field schema](#field-schema)
    - [Elasticsearch / log-shipper recommendations](#elasticsearch--log-shipper-recommendations)
    - [Worker log forwarding (`log_to_driver`)](#worker-log-forwarding-log_to_driver)
    - [Ray backend (C++) logs](#ray-backend-c-logs)
  - [Performance Metrics](#performance-metrics)
  - [Grafana Dashboard](#grafana-dashboard)
  - [Deployment](#deployment)
    - [K8s-based Platforms](#k8s-based-platforms)
    - [Slurm Environment](#slurm-environment)

This guide walks through profiling, instrumentation, metrics, monitoring dashboard, and deployment methods.

## Profiling and Instrumentation

Cosmos Curator provides automatic, framework-level profiling at three levels:

1. **Per-stage** -- injected transparently by `_build_pipeline_stage_specs()` in
   `pipeline_interface.py`.  Every stage's lifecycle methods (`stage_setup_on_node`,
   `stage_setup`, `process_data`) are automatically instrumented.
2. **Root / driver process** -- the `profiling_scope()` context manager in each
   pipeline's `run_pipeline.py` entry point wraps the entire pipeline execution,
   capturing work done outside of Ray actors (argument parsing, task construction,
   orchestration).
3. **Distributed tracing** -- OpenTelemetry spans via Ray's tracing hook, capturing
   cross-actor communication, scheduling delays, and the end-to-end distributed
   timeline across all processes.

A useful way to think about these tools:

- **Per-stage profiling** (pyinstrument, memray) = **microscope** -- deep view into
  one process (CPU call stacks, memory allocations within a single actor).
- **Distributed tracing** (OpenTelemetry) = **satellite view** -- shallow but wide
  perspective across all processes (which actor waited for which, where scheduling
  delays happened, end-to-end flow).

No changes to individual stage code are needed.

### CLI Flags

All profiling flags use purpose-based naming (`--profile-<what>`), not
implementation-based.

| Flag | Purpose | Backend | Status |
|---|---|---|---|
| `--perf-profile` | Lightweight per-task stats (wall-clock, idle, RSS, timestamps) | StageTimer | Available |
| `--profile-cpu` | CPU flame-tree profiling | pyinstrument | Available |
| `--profile-memory` | Heap memory allocation profiling | memray | Available |
| `--profile-gpu` | GPU kernel/operator profiling (CUDA) | torch.profiler | Available |
| `--profile-cpu-exclude <names>` | Comma-separated scopes to exclude from CPU profiling | pyinstrument | Available |
| `--profile-memory-exclude <names>` | Comma-separated scopes to exclude from memory profiling | memray | Available |
| `--profile-gpu-exclude <names>` | Comma-separated scopes to exclude from GPU profiling (default: `_root`) | torch.profiler | Available |
| `--profile-tracing` | Distributed cross-actor tracing (spans as NDJSON) | OpenTelemetry / Ray | Available |

All `--profile-*` flags imply `--perf-profile`.

Scope names used in `--profile-*-exclude` match stage class names
(e.g. `VideoDownloader`, `ClipWriterStage`) and the special name `_root` for the
driver process.  By default, `_root` is excluded from both GPU profiling
(the driver typically has no CUDA context) and memory profiling (memray
conflicts with pyinstrument's `sys.setprofile` hook on long-lived
processes).

### CPU Profiling (pyinstrument)

Wraps each stage's `stage_setup_on_node()`, `stage_setup()`, and
`process_data()` with a pyinstrument `Profiler` session.  The driver
process (`_root` scope) is also profiled when `--profile-cpu` is enabled.
Produces:

- **Per-call text summary** dumped to stdout for immediate visibility.
- **Per-call HTML flame-tree** and `.pyisession` file saved to `<output-path>/profile/cpu/`.

Example:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path /config/test_data/raw_videos/ \
    --output-clip-path /config/test_data/output_clips/ \
    --profile-cpu --verbose
```

### Memory Profiling (memray)

Wraps each stage's `stage_setup_on_node()`, `stage_setup()`, and
`process_data()` with a memray `Tracker`.
Produces:

- **Per-call stats summary** dumped to stdout (total allocations, peak memory, top allocation sites).
- **Per-call `.bin` capture** and **HTML flamegraph** saved to `<output-path>/profile/memory/`.

Example:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path /config/test_data/raw_videos/ \
    --output-clip-path /config/test_data/output_clips/ \
    --profile-memory --verbose
```

Both `--profile-cpu` and `--profile-memory` can be enabled simultaneously.
The driver process (`_root`) is excluded from memory profiling by default
to avoid the memray/pyinstrument `sys.setprofile` conflict:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path /config/test_data/raw_videos/ \
    --output-clip-path /config/test_data/output_clips/ \
    --profile-cpu --profile-memory --verbose
```

Memray native-trace report generation can also ask debuginfod for
missing native debug symbols when `DEBUGINFOD_URLS` is set.  This can
make runs appear stuck while memray builds statistics or
flamegraphs, so the test suite clears `DEBUGINFOD_URLS` globally.  For
manual memray CLI debugging, use `DEBUGINFOD_URLS=` if execution
appears stuck in native symbol lookup:

```bash
DEBUGINFOD_URLS= memray stats path/to/profile.bin
```

### GPU Profiling (torch.profiler)

Wraps each stage's `stage_setup_on_node()`, `stage_setup()`, and
`process_data()` with `torch.profiler.profile`.
Captures CUDA kernel launches, operator breakdown, GPU memory allocations,
and CPU-GPU synchronization.  Silently disabled on CPU-only workers or
when `torch` is not installed.
Produces:

- **Per-call key_averages table** dumped to stdout for immediate visibility.
- **Per-call Chrome Trace JSON** saved to `<output-path>/profile/gpu/`.

Example:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path /config/test_data/raw_videos/ \
    --output-clip-path /config/test_data/output_clips/ \
    --profile-gpu --verbose
```

### Distributed Tracing (OpenTelemetry)

Captures distributed spans across the entire pipeline via Ray's built-in
OpenTelemetry integration **plus application-level spans** from
`StageTimer` and `_ProfiledStage`.  Enable with `--profile-tracing`.

Three levels of spans are produced:

| Level | Source | What it captures |
|-------|--------|-----------------|
| **Ray spans** (automatic) | Ray OTel hook | Actor creation, method invocations, task scheduling |
| **Lifecycle spans** | `_ProfiledStage` | `setup_on_node`, `setup`, `process_data`, `destroy` |
| **Sample spans** | `StageTimer.time_process()` | Per-sample processing inside `process_data` |

Produces per-worker NDJSON span files under `<staging-dir>/traces/`
(collected post-pipeline by `ArtifactDelivery`) and optional live export
to an OTLP collector (Jaeger, Grafana Tempo, etc.).

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path /config/test_data/raw_videos/ \
    --output-clip-path /config/test_data/output_clips/ \
    --profile-tracing --verbose
```

#### Instrumenting Custom Stage Code

The public tracing API lives in
`cosmos_curator/core/utils/infra/tracing.py`.  All functions are
**safe to call unconditionally** -- when tracing is not enabled
(`--profile-tracing` not passed), they become zero-cost no-ops.
No `if tracing_enabled` guards are needed.

**You do NOT need to call any setup function.**  Each pipeline's
`main()` wraps its execution in `profiling_scope(args)` -- the
top-level context manager that reads CLI flags and configures
all profiling and tracing infrastructure automatically:

```python
# run_pipeline.py (pipeline entry point)
def main() -> None:
    args = parse_args()
    with profiling_scope(args):   # <-- outermost wrapper, sets up everything
        args.func(args)
```

Stage code just imports and uses the public API -- no setup needed:

- **`--profile-tracing` passed**: spans are recorded and exported.
- **`--profile-tracing` NOT passed**: all API calls become
  zero-cost no-ops.  No overhead, no errors.

**Choosing the right API:**

| API | Creates new span? | When to use |
|-----|-------------------|-------------|
| `@traced(name)` | Yes (whole function) | Entire function body is one logical span; attributes known at decoration time |
| `traced_span(name)` | Yes (scoped block) | Span around a **subset** of a function, or attributes set **dynamically** |
| `TracedSpan.current()` | No (wraps existing) | Annotate a span created higher up (e.g. by `_ProfiledStage`) |

```
Decision tree:

    Whole function is one span?
          |               |
          v (yes)         v (no / partial)
    @traced(name)    traced_span(name)
                          |
    Need to annotate a parent span too?
          |
          v (yes)
    TracedSpan.current()
```

**Quick examples:**

Create spans at the **batch** or **phase** level, not per-item.
At scale (100 M+ items), per-item spans generate excessive trace
data.  Record per-item details as aggregated attributes instead.

```python
from cosmos_curator.core.utils.infra.tracing import (
    TracedSpan, traced, traced_span, StatusCode,
)


class MyStage(CuratorStage):

    # 1. @traced -- whole method is one span (static attributes)
    @traced("MyStage.load_model", attributes={"model.name": "resnet50"})
    def _load_model(self):
        self._model = load_weights("resnet50")

    def process_data(self, tasks):
        # 2. traced_span -- batch-level span, dynamic attributes
        with traced_span("MyStage.infer",
                         attributes={"stage.input_count": len(tasks)}) as span:
            results = [self._model(t.data) for t in tasks]
            span.set_attribute("stage.output_count", len(results))

        # 3. Error recording on caught exceptions (phase-level span)
        with traced_span("MyStage.download") as span:
            try:
                data = bulk_download([t.url for t in tasks])
                span.set_attribute("io.download_bytes", len(data))
            except TimeoutError as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, "Timeout")
                data = fallback_download(tasks)

        # 4. TracedSpan.current() -- annotate the parent span
        #    (created by _ProfiledStage, NOT by this code)
        TracedSpan.current().set_attribute("stage.total", len(tasks))
        return tasks
```

For the full deep-dive -- public span API (`TracedSpan`, `traced_span`,
`@traced`), Ray tracing hook internals, trace context propagation,
library auto-instrumentation, environment variables, and Jaeger setup --
see the **[Distributed Tracing Guide](../reference/distributed-tracing.md)**.

### Output and Post-processing

All profiling artifacts are written to `<output-path>/profile/` with
subdirectories `cpu/`, `memory/`, `gpu/`, and `traces/`.  In multi-node
clusters, `ArtifactDelivery` collects artifacts from all nodes
post-pipeline and uploads them to the output path (local, S3, or Azure).

Quick reference for viewing results:

- **CPU**: Open `.html` in browser; merge with `benchmarks/merge_cpu_profiles.py`
- **Memory**: Open `.html` flamegraph; summarize with `benchmarks/merge_memory_profiles.py`
- **GPU**: Open `.json` in [Perfetto UI](https://ui.perfetto.dev/)
- **Traces**: Open `.jsonl` in [Perfetto UI](https://ui.perfetto.dev/)

For the full deep-dive -- backend internals, dynamic subclass architecture,
LIFO nesting, `profiling_scope` driver setup, file naming convention,
artifact delivery flow, profile merging scripts, and error handling --
see the **[Profiling Guide](profiling.md)**.

## Structured Logging

Cosmos Curator, cosmos-xenna, and Ray share a single logging toggle,
`PYTHON_LOG_FORMAT`, with two values:

- `text` (default): human-readable logs, unchanged from prior behavior.
- `json`: structured JSON logs, integrated with Ray's structured logging.

`PYTHON_LOG` still controls levels (RUST_LOG-style, e.g.
`PYTHON_LOG=debug,myapp.db=warning`), independent of the format.

### Enabling

On **k8s / NVCF** (Helm), set it in the chart — the value is written to the curator
ConfigMap and reaches every pod through the statefulset's `envFrom`, so head and
worker pods stay consistent:

```yaml
logging:
  format: json
```

For **local Docker / Slurm**, export the variable before launching; it is forwarded
into the container automatically:

```bash
export PYTHON_LOG_FORMAT=json
```

See the [Helm chart README](../../../charts/cosmos-curator/README.md#structured-logging)
for the chart values.

### Field schema

In JSON mode every source — the Ray driver/workers, the pre-`ray.init` fallback,
and the launcher scripts (`onto_nvcf.py` / `onto_slurm.py`) — emits the **same flat
schema**, so a single log-shipper mapping parses them all. Ray lines additionally
carry Ray's job/worker context:

| Field | Source | Meaning |
| --- | --- | --- |
| `timestamp_ns` | emit time | Integer epoch nanoseconds (primary sort key). Derived from `time.time()`, so effective precision is OS/float-bounded (typically ~µs); pair with `seq` for tiebreaking. |
| `asctime` | emit time | Human-readable emit time |
| `levelname`, `message`, `name` | stdlib/Ray | Level, message, logger name |
| `funcName`, `lineno`, `process` | stdlib/Ray | Call site + process id |
| `pod` | `POD_NAME` → `SLURMD_NODENAME` → hostname | Emitting node/instance (k8s pod name; SLURM node name or hostname off-k8s) |
| `replica` | trailing `-N` of `POD_NAME` | k8s/NVCF StatefulSet replica ordinal; `""` off-k8s (SLURM/local) |
| `pid` | `os.getpid()` | Process id |
| `run_id` | `CURATOR_RUN_ID` env | Run/request id (NVCF request id on NVCF) |
| `seq` | per-process counter | Gap-free monotonic tiebreaker |
| `trace_id` | active OTel span | 32-char lowercase-hex id of the enclosing trace. Only when tracing is enabled **and** a span is active; otherwise the key is absent |
| `span_id` | active OTel span | 16-char lowercase-hex id of the span that emitted the line (same conditions as `trace_id`) |
| `trace_sampled` | active OTel span | Boolean: whether the trace was sampled, i.e. whether the backend actually retained it |
| `job_id`, `worker_id`, `node_id`, `task_*` | Ray | Ray context (Ray lines only) |

All identity/order fields (`pod`, `replica`, `pid`, `run_id`, `seq`) are top-level in
every source, and Ray's `JSONFormatter` surfaces them alongside its own context. The
logger `name` is requested from Ray via `additional_log_standard_attrs=["name"]` so
Ray lines match the fallback/launcher schema (older Ray without that parameter simply
omits `name`).

### Span correlation fields

Running with `--profile-tracing` on top of `PYTHON_LOG_FORMAT=json` stamps `trace_id`,
`span_id`, and `trace_sampled` onto every record created while a span is active — no
extra toggle. Records emitted outside a span carry none of the three keys, so a log UI
never renders a link to a trace that does not exist.

Both halves are settable purely from the environment, so correlation can be turned on
for a whole deployment rather than per invocation: `PYTHON_LOG_FORMAT=json` plus
`COSMOS_CURATOR_PROFILE_TRACING=1`, both forwarded into the container by the local and
Slurm launchers. Set `COSMOS_CURATOR_PROFILE_TRACING=0` for a single run to opt out of a
cluster-wide default.

Turning `trace_id` into a clickable link to the trace is a function of your log backend
and is not configured from this repo — the mechanism differs enough between stacks that
there is no portable recipe. See the
**[Distributed Tracing reference](../reference/distributed-tracing.md)** for how to
enable tracing and export spans to a collector.

### Elasticsearch / log-shipper recommendations

- Remap the index `@timestamp` from `timestamp_ns` (emit time) rather than the
  ingest time — it is consistent across launcher, driver, and worker logs.
- Use `seq` as a tiebreaker when sorting records whose timestamps collide within a
  process (it is monotonic and gap-free per process, reset per process).

### Worker log forwarding (`log_to_driver`)

By default JSON mode leaves `ray.init(..., log_to_driver=True)` unchanged, so
per-worker/actor logs are forwarded to the driver's stdout just like in text mode —
you see the full pipeline output in one place. Note that Ray prefixes forwarded worker
lines with an `(ActorClass pid=…, ip=…)` tag, so those particular lines are *not* pure
JSON (the JSON payload follows the prefix).

For a **stdout-scraping k8s deployment** that wants the driver stream to stay pure,
single-schema JSON, set `PYTHON_LOG_TO_DRIVER=false`: worker logs are then written only
as structured lines to the **per-node Ray log files** (`/tmp/ray/session_*/logs/…`),
which the log pipeline must ship separately. Do **not** use this for local Docker /
NVCF runs where stdout is the only log sink — worker logs would otherwise be invisible.

Ray's own root JSON threshold can be tuned with `PYTHON_LOG_RAY_LEVEL` (defaults to the
`PYTHON_LOG` level; TRACE is clamped to DEBUG).

### Ray backend (C++) logs

`log_to_driver` and `PYTHON_LOG_TO_DRIVER` only govern *routing* of Python worker
logs to the driver; they do not change any log's format. Ray's own C++ backend/system
components (raylet, GCS, object store, dashboard agent) are a separate stream whose
format is controlled by `RAY_BACKEND_LOG_JSON`. This is **opt-in and independent of
`PYTHON_LOG_FORMAT`**: enabling JSON application logs does not automatically switch the
backend logs. To emit backend logs as JSON, set `logging.rayBackendJson: true` in the
chart (writes `RAY_BACKEND_LOG_JSON=1` to the curator ConfigMap) or export
`RAY_BACKEND_LOG_JSON=1` yourself. Ray reads this env var natively; the curator/xenna
code never sets or overrides it.

## Performance Metrics

[Prometheus](https://prometheus.io/)-compatible metrics are exported at port `localhost:9002/metrics`.

A list of useful [PromQL](https://prometheus.io/docs/prometheus/latest/querying/basics/) queries for performance debugging are summarized below:

```bash
# Measured stage speed, i.e. process time per task per actor for each stage
sum by (stage) (ray_pipeline_actor_process_time)

# Number of busy vs. idle workers per stage
sum by (stage, state) (ray_pipeline_actor_count{state!="target", state!="pending"})

# Input & output queue sizes per stage
sum by (stage) (ray_pipeline_input_queue_size)
sum by (stage) (ray_pipeline_output_queue_size)

# Cross-stage object size
(sum by (stage) (ray_pipeline_stage_deserialize_size_total))
/ on (stage) group_left ()
(sum by (stage) (ray_pipeline_stage_deserialize_count_total))

# Communication time / process time; i.e. are we able to hide cross-stage data movement
(sum by (stage) (ray_pipeline_stage_deserialize_time_total))
/ on (stage) group_left ()
(sum by (stage) (ray_pipeline_stage_process_time_total))

# GPU utilization averaged by stage
avg by (stage) (
    ray_pipeline_stage_gpu_alloc * on (SessionName, NodeAddress, GpuIndex) group_left
    label_replace(ray_node_gpus_utilization, "NodeAddress","$1","ip", "(.+)")
)

# GPU memory usage averaged by stage
avg by (stage) (
    ray_pipeline_stage_gpu_alloc * on (SessionName, NodeAddress, GpuIndex) group_left
    label_replace(ray_node_gram_used, "NodeAddress","$1","ip", "(.+)")
)

# CPU usage aggregated per stage
sum by (stage) (ray_pipeline_actor_resource_usage{stage!="", resource="cpu"}) / 100

# Average CPU usage per actor for each stage
(sum by (stage) (ray_pipeline_actor_resource_usage{stage!="", resource="cpu"}))
/ on (stage)
(sum by (stage) (ray_pipeline_actor_count{state="running"})) / 100

# System memory usage aggregated per stage
sum by (stage) (ray_pipeline_actor_resource_usage{stage!="", resource="memory"})
```

## Grafana Dashboard

An awesome monitoring dashboard is provided at [cosmos-curator-oss.json](../../../examples/observability/grafana/cosmos-curator-oss.json).

The panels are organized in the following rows:
- `Pipeline`:
  - per-stage progress, average process time, actor count
- `GPU`
  - per-stage GPU allocation, utilization, and memory usage
  - overall GPU stage-worker utilization (whether auto-scaling is able to keep GPU stages always busy)
  - overall GPU utilization
- `CPU & System Memory`
  - per-stage aggregated CPU allocation & usage (identify CPU bottlenecks)
  - per-stage average per-actor CPU usage (whether the resource request is appropriate)
  - per-node CPU utilization
  - per-stage memory usage (identify which stage is the root cause for e.g. system OOM)
  - per-node memory usage
- `Actor Status`
  - per-stage idle-red / busy-green actors (whether GPU stages are starved)
- `Actor Pool Queuing`
  - per-stage input/output queue sizes, number of used/empty slots (identify which stage is the bottleneck)
- `Xenna Internals`
  - cross-stage data movement size (this goes through front-end CPU network and hence should not be too large)
  - timing of main loop of streaming executor (whether the main orchestration thread is the bottleneck)

![Monitoring Dashboard](../../assets/cosmos-curator-dashboard.png)

## Deployment

### K8s-based Platforms

On K8s-based platforms, including [NVCF](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html),
the [Helm chart](../../../charts/cosmos-curator/README.md) provided includes a [Prometheus Agent](https://prometheus.io/blog/2021/11/16/agent/)
which can scrape the metrics endpoint and [remote-write](https://prometheus.io/docs/specs/prw/remote_write_spec/)
to a [Thanos-like](https://thanos.io/) endpoint.

The relevant configurable entries in the chart can be found in [values.yaml](../../../charts/cosmos-curator/values.yaml):

```yaml
metrics:
  enabled: true
  remoteWrite:
    endpoint: ...
    certPath: ...
    keyPath: ...
```

Do note that current version of the Helm chart will need some tweaks to work on vanilla Kubernetes clusters.

### Slurm Environment

On Slurm, there is an option `--prometheus-service-discovery-path` to the `cosmos-curator slurm submit` command.

If you provide a valid path, which needs to be accessible from the compute nodes,
a service-discovery file named `prometheus_service_discovery_{slurm_job_id}.json` will be created under that path.
The file can tell [Prometheus](https://prometheus.io/) where to find the metrics endpoints and what external labels to attach.

```json
[
  {
    "labels": {
      "job": "ray",
      "slurm_job_user": "haowang",
      "slurm_job_id": "80305",
      "slurm_job_name": "hao-test"
    },
    "targets": [
      "pool0-0218:9002",
      "pool0-0219:9002",
      "pool0-0220:9002",
      "pool0-0221:9002"
    ]
  }
]
```

Then you can configure the cluster's Prometheus with the [file-based service discovery approach](https://prometheus.io/docs/guides/file-sd/), like

```yaml
scrape_configs:
- job_name: 'cosmos-curator'
  file_sd_configs:
  - files:
    - '<the same path you passed in>/prometheus_service_discovery_*.json'
```
