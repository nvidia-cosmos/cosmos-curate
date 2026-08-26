# Data Integrity Pipeline

## Summary

`data-integrity` is a [Curator Next](curator-next.md) recipe that runs the existing
data-integrity metrics across many autonomous-vehicle sessions with Ray Data, and persists
the resulting measurements and evaluations into one store root. The first target is a large
multi-session autonomous-vehicle recording bucket.

Where `di-check` measures one video and `di-session` measures one session, this recipe
takes an input that identifies many sessions, fans out to one Ray task per session, and
commits everything a single invocation measured under one `run_id`.

It adds no metric, no modality, no evaluation policy and no schema. The metric kernel
(`cosmos_curator/core/sensors/data_integrity/`) and the Lance store
(`cosmos_curator/next/recipes/data_integrity/store.py`) are consumed as they are.

---

## Motivation

[Data Integrity Design](data-integrity-design.md) draws its boundary explicitly: the kernel
"measures and judges", and "everything needed to *run* it at recording scale is
deliberately not designed here". Of the six open problems that section lists, `di-check` and
`di-session` answered one — an end-to-end entry point — at single-input and single-session
granularity. This recipe carries that same problem to many sessions, and resolves none of the
other five: full-fidelity sensor input, composing metrics into checks, declaring checks as
data, generalizing one-read-many-metrics past the camera path, and multi-sensor metrics.

That same section also places three constraints on any solution, and all three are met
without touching the kernel:

- **An undefined measurement must not be evaluated.** The engine still decides this per
  metric, the store still records `is_defined` in three states, and a skipped metric's
  verdict row still carries a null `margin` and `threshold`.
- **The input is a stream of full-fidelity windows, larger than memory, and one read should
  drive many metrics.** A session is the unit of distributed work, but its streams are still
  measured one at a time by the existing engine, which reads each once and folds it into
  every instrument. Nothing buffers a recording.
- **Measurement stays separate from evaluation policy.** Both go through the same builders
  the CLIs use, into the same separate tables.

## Constraints

- Build directly on Ray Data, not the Xenna stage model in `core/interfaces` that
  `pipelines/video/` is built on. (`pipelines/ray_data/` is already Ray-Data-native.)
- Every session processed by one invocation lands in one store root.
- Preserve the store's append, commit, identity, measurement, evaluation and
  re-evaluation compatibility semantics.
- A completed and persisted run exits 0 regardless of data-quality findings. A nonzero exit
  means the pipeline could not meet its operational contract — invalid config, Ray failure,
  or a failed write.
- Data-quality failures and unreadable inputs are persisted as findings; the run continues.
- Consume the recipe's library APIs, not the internals of its CLIs.
- The recipe package must not import Ray at import time, so
  `python -m cosmos_curator.next.recipes.data_integrity.cli --help` stays fast and
  Ray-free.

---

## Current Behavior

### The APIs this recipe consumes

| Symbol | Location | Purpose |
| --- | --- | --- |
| `run_checks` | `sources.py:90` | measure one stream URI, returning check results, video info and the resolved config |
| `discover_streams` | `discovery.py:48` | expand one session path into sorted stream URIs |
| `new_run_id` | `store.py:100` | `uuid4().hex`, shared by every row in one run |
| `content_identity` | `store.py:130` | provenance `HEAD` / `stat` for one source; best-effort, never raises |
| `stream_key` | `store.py:171` | the dedup key one stream's rows agree on |
| `INSTRUMENTS` / `instrument` | `core/sensors/data_integrity/instruments.py:330,341` | the five instrument specs, and the name → spec lookup both row builders take |
| `append_rows` | `store.py:312` | create-or-append rows to one Lance dataset |
| `commit_run` | `store.py:329` | the commit point; must be the last dataset write |
| `write_run` | `store.py:385` | the CLIs' one-session path: build rows, append, commit, write the manifest |
| `write_manifest` | `store.py:533` | overwrite `manifest.json` to describe the most recent run |
| `stream_result` | `core/sensors/data_integrity/results.py:286` | package metrics and video info into a `StreamResult`, default selector included |
| `StreamResult` | `core/sensors/data_integrity/results.py` | per-stream result; `error` carries an unreadable input |

All paths are under `cosmos_curator/next/recipes/data_integrity/` unless noted. This recipe
imports nothing from `cli.py` or `session_cli.py`.

`discover_streams` expands exactly one session path — local directories via a recursive
`rglob`, cloud prefixes via `_cli_cloud.list_cloud_objects` — filtered to
`VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv")` and returned sorted. Its `limit` caps
streams per session, where 0 means no cap.

### What a session is

The session layout is already settled, and already exercised against the target bucket.
`session_cli.py` defines a session as "one recording -- typically a `clips/<uuid>/`
directory (or cloud prefix) holding one video per camera", and its usage block gives
`--session-path s3://<bucket>/clips/<uuid>/` as a worked example.

### Nothing enumerates sessions

`_cli_cloud.list_cloud_objects` (`cosmos_curator/core/sensors/scripts/_cli_cloud.py:374`)
lists **objects**, with no `Delimiter` / `CommonPrefixes` support, so asking which sessions
exist under a dataset root means paging every object beneath it. Session enumeration is the
one capability this recipe needs that the DI code does not have — but it is not unwritten:
`multimodal-split` has the same listing, and the [Input Contract](#input-contract) hoists and
shares it rather than adding a second one.

### `write_run` is per-session by construction

`write_run` takes `session_path` as a single scalar, stamps it onto every stream,
measurement and evaluation row, then commits — "Last, and deliberately so: until this lands,
nothing written above is visible to a reader" (`store.py:506`) — and writes the manifest.
Because the commit lives inside it, `write_run` cannot be called once per session and still
produce one run. Everything it composes is public, though — see [Persistence](#persistence)
— except two row builders, `_stream_row` and `_measurement_row`, which this recipe promotes.

### How config-backed recipes run today

A recipe registers a `PipelineKind` (`cosmos_curator/next/core/pipeline_kind.py`) by adding
it to the `BUILTIN_PIPELINE_KINDS` registry in
[builtin_pipeline_kinds.py](../../../cosmos_curator/client/pipeline_cli/builtin_pipeline_kinds.py)
— one import and one tuple entry, in one file. `PipelineKindRegistry` indexes kinds by name
and rejects duplicates, and both CLI entry points resolve a kind through it.

`PipelineKind` is a frozen dataclass of a name plus seven callables, all of them reachable.
`pipeline template` / `validate` / `render` / `schema` call the first five, `pipeline presets
list` calls `list_presets` on every registered kind, and `prepare_run` is the run path:
`pipeline_runtime.main` reads the config's `kind`, fetches that kind, calls
`prepare_run(config, set_overrides=...)` for a deferred `PreparedPipelineRun`, and invokes it
to get a `PipelineRunOutput` carrying a JSON payload and a one-line message.

The `kind` string is matched exactly — `load_pipeline_kind_name` returns it verbatim and the
registry looks it up unchanged — so hyphen and underscore are different kinds rather than two
spellings of one. That is deliberate: `video-split` selects the Curator Next recipe while
`video_split` selects the legacy one. Next recipes are hyphenated, so this one is
`data-integrity` everywhere: in `kind:`, as the CLI's kind argument, and as the registered
name. `test_robot_action_split_underscore_spelling_is_not_supported` pins the rule.

That runtime already implements the exit-status behavior this recipe needs. An `OSError`,
`TypeError`, `ValueError` or pydantic `ValidationError` from the lookup or from `prepare_run`
becomes `typer.Exit(2)` — and because `prepare_run` resolves the config before it returns its
callable, a bad config fails there rather than after Ray has started. An exception from the
run itself propagates (or becomes `_fail("runtime", ...)` under `--json`), nonzero either
way. A completed run echoes its message or payload and exits 0. `caption_judge` is the
precedent for findings — it returns `passed: False` in its payload and still exits 0.
Execution is `pixi run run-pipeline`; there is deliberately no `cosmos-curator pipeline run`
subcommand, and `test_pipeline_run_is_not_host_cli_command` pins that.

---

## Recipe Design

### New Kind: `data-integrity`

A distinct kind, structured like `next/recipes/video_split`: a Pydantic config module, a
`pipeline_kind.py` exposing the config and runtime surface, and a `pipeline.py` driver that
owns the Ray Data work. Unlike the other Next recipes, there is no media output — the only
output is store rows.

The recipe lives beside the DI code it drives, under
`cosmos_curator/next/recipes/data_integrity/`. `video_split` is the closest precedent and a
complete one: it is a Ray Data pipeline under `next/`, its `pipeline_kind.py` defers every
import so the CLI stays Ray-free, its `run_config(resolved) -> dict` is the shape
`prepare_run` wraps, and it refuses an input selection that realizes nothing rather than
publishing an empty result — the same rule this recipe applies to an empty session
expansion.

The Ray conventions are already in `next/core/ray_runtime.py`: `ensure_ray_initialized`,
`configure_ray_data_progress` and `configure_ray_data_stability`, called in that order at the
top of `video_split`'s driver. This recipe imports the same three. The helpers that would
size the stage from the live cluster — `capped_slots_for_items` and `live_ray_node_count`,
with `DEFAULT_IO_SLOTS_PER_NODE` — have no public home: they are private to
`pipelines/ray_data/_runtime.py` and `pipelines/ray_data/constants.py:19`, which
`next/AGENTS.md` forbids importing from `next/`. `session_concurrency` therefore takes a
plain default.

### Config Contract

```yaml
schema_version: 1
kind: data-integrity

input:
  sessions: []              # explicit session paths
  session_list_uri: null    # file of session paths: one per line, or a JSON array
  session_roots: []         # dataset roots to expand into session paths
  session_depth: 1          # prefix levels below each root that name a session
  limit: null               # max streams per session; null maps to discover_streams limit=0

checks:
  expected_hz: null         # null = per-stream header rate, or unavailable (see below)
  batch_size: 0             # timestamps per metric update; 0 feeds the whole array at once
  thresholds:               # shown at their kernel defaults; omit any key to keep it
    max_strict_violations: 0
    max_rate_deviation_percent: 5.0
    max_gaps: 0
    max_jitter_percent: 10.0
    allow_frame_reordering: false

output:
  store_root: s3://example-bucket/di_store/

execution:                  # all fields optional; shown with defaults
  s3_profile_name: null
  azure_profile_name: default
  endpoint_url: null
  session_concurrency: 8      # concurrent session-measurement tasks
  stream_attempts: 3          # attempts per stream when the failure looks transient
  append_batch_size: 512      # sessions per driver-side append
  progress: false             # Ray Data progress bars
```

`checks:` carries only the knobs `run_checks` already takes. It is not the check-declaration
format that [Data Integrity Design](data-integrity-design.md) leaves open, and taking the
name should not be read as a position on it.

Leaving `expected_hz` null is the right default across a heterogeneous dataset, since
`resolve_expected_hz` then takes each stream's own nominal `avg_frame_rate`
(`ExpectedHzSource.HEADER`). Where a container reports no nominal rate it resolves to
`UNAVAILABLE`, and `run_metrics` builds no rate, gap or jitter metric at all — three of the
five come back undefined for that stream rather than wrong. That is the `is_defined` null row
in the exit-status table.

`execution` names the three credential fields explicitly rather than taking the single
`storage_profile` that `robot-action-split` and `multimodal-split` take, because
`run_checks`, `discover_streams` and `content_identity` already take them individually and
this recipe passes them through. The two are not the same field under different names —
`storage_profile` selects a Curator profile while `s3_profile_name` selects an AWS one — and
[Root expansion](#root-expansion-shares-one-lister-with-multimodal-split) is where that
difference has to be handled.

`output.store_root` must be a local directory or an `s3://` prefix.
`get_lance_storage_options` raises on an `az://` root rather than silently writing an
unauthenticated store, and `store_cli.validate_store_path` rejects one at parse time for the
two CLIs — because the store is written last, so a typo should not cost a whole session. This
config should validate it the same way. Reading *sessions* from `az://` is unaffected.

### Input Contract

`input` accepts any combination of three forms:

- **`sessions`** — explicit session paths. The primitive case, and what the tests use.
- **`session_list_uri`** — a newline-delimited or JSON file of session paths, for large
  runs where the list is produced elsewhere.
- **`session_roots`** with **`session_depth`** — expand a dataset root by listing the
  prefixes below it. This is what makes a whole recording bucket usable directly.
  `session_depth` is validated at 1, the shared lister's native scope and the layout
  `di-session` documents; a deeper root is rejected rather than silently flattened, and
  supporting one is additive (see [Open Questions](#open-questions)).

At least one must be non-empty, and an expansion that yields zero sessions is a
configuration error rather than a successful empty run — the same rule `video_split` applies
to an input selection that realizes nothing.

#### Root expansion shares one lister with `multimodal-split`

Root expansion needs a delimiter listing, `list_objects_v2(Delimiter="/")`, and nothing
shared offered one: `S3Client.list_recursive_directory` in
`core/utils/storage/s3_client.py` pages every object beneath a prefix, and `_cli_cloud` does
the same. [MR 1104][mr1104] added exactly the listing this recipe needs, for
`multimodal-split`, but private to that recipe: `_list_child_session_ids` in
`next/recipes/multimodal_split/discovery.py`, S3 through a paginated delimited list and local
through a `scandir` of immediate children. The primitive is storage-level rather than
recipe-level, so it is hoisted once and consumed twice rather than written a second time
here:

- `list_child_prefixes(s3_client, *, bucket, prefix)` in `core/utils/storage/s3_client.py`,
  holding the paginated delimited loop and its `FileNotFoundError` for a prefix that holds
  nothing. A thin `S3Client.list_child_prefixes(uri)` method delegates to it, so
  `multimodal-split` keeps the ergonomics it had.
- `list_child_directories(path)` in `core/utils/storage/storage_utils.py`, the `scandir`
  half, symlink semantics unchanged.

`multimodal_split._list_child_session_ids` keeps its own scheme dispatch and now calls both;
its tests pass untouched, which is what makes the hoist observably behavior-preserving.

**The hoisted S3 helper takes a client rather than building one.** `S3Client` gets
credentials from `get_s3_client_config`, which reads Curator's own creds file at
`S3_PROFILE_PATH` and falls back to the NVCF secret store, raising when neither exists. Every
other cloud read in this recipe goes through
`_cli_cloud.make_s3_client(source, s3_profile_name, endpoint_url)` — an AWS named profile
plus an explicit endpoint override, with boto3's default credential chain behind it. Those
are two different credential sources, so a helper that constructed its own client would let a
run measure a bucket it cannot enumerate. Sharing the paging loop and leaving client
construction to the caller avoids that and costs nothing in reuse.

`session_roots` on `az://` is rejected rather than expanded: neither shared lister offers a
delimited listing there. Sessions themselves may still live on `az://`; only expanding a root
into them cannot.

Consuming it settles four smaller things:

- **The unit of reuse is the lister, not `discover_candidate_sessions`.** That entry point is
  typed to `MultimodalSplitInputConfig` and returns a `pa.Table` of
  `CANDIDATE_SESSION_SCHEMA` (`source_session_id`, `session_uri`). This recipe keeps its own
  three input forms above the lister rather than adopting that config.
- **No store column is renamed.** Their `session_uri` is the string this doc calls
  `session_path`, and their `source_session_id` is the child directory name — which is *not*
  the store's `session_id`. That stays `identity.session_id(session_path)`, a
  backend-namespaced digest. The child name belongs in logs, not in a row.
- **`limit` means two different things.** Theirs caps sessions, applied after dedup and
  sort; this config's caps streams per session, and keeps that meaning. A session cap is
  worth having here too, but it has to arrive as `session_limit` rather than widening
  `limit` into a field that means whichever the reader assumes.
- **Local sessions stay absolute paths, not `file://` URIs.** MR 1104 resolves them with
  `os.path.abspath`, which is what `identity.normalize_source` does to a scheme-less source.
  A `file://` URI would keep its scheme verbatim there and hash to a different `stream_id`
  than the same file named by path.

---

## Work Granularity

**One Ray Data record per session, not per stream.** A task lists its session and then
measures every stream in it, in sequence.

The session is the unit because a session is what a check is about. Every metric today
judges one stream on its own, which is why the store's rows are per-stream, but the checks
this store is heading for are cross-sensor: does the camera timeline agree with the IMU's,
do the sensors of one recording cover the same interval. Those have no per-stream task to
run in. Giving one worker the whole session is what makes them expressible without another
change to the execution model.

Throughput is a separate question from what the measurement is about, and it does not argue
the other way in practice. A rig carries the same sensor suite from session to session, so
sessions hold similar stream counts rather than skewed ones, and Ray Data hands the next
session to whichever worker frees up first. If a session ever does run long enough to matter,
the lever is inside the task — `run_session` already takes `max_workers` for exactly this —
not a different record.

The Lance schema stays per-stream, so nothing about the store anticipates those checks yet —
a session-grain verdict has no row to live on today. Aligning the unit of work now is what
keeps that a schema change rather than a rewrite.

Within a session the kernel's requirement is unchanged: each stream is read once, through
`run_checks`, which takes exactly one URI, and folded into every instrument.

### Ray Dataset Record Structure

Two record shapes — what the driver feeds in and what the stage emits — each a flat dict so
every record shares one Arrow schema:

```python
# driver-built input: one record per session
{"session_path": str}

# after map(check_session): one record per session
{
    "session_path": str,
    "streams": int,         # streams the session held
    "unreadable": int,      # of those, how many could not be read
    "failed_metrics": int,
    "rows": bytes,          # serialized {dataset_name: [store row, ...]}, the whole session
}
```

The counters are what the run summary is built from: the driver cannot count streams before
the run, since listing now happens inside the tasks, so it sums them as batches arrive.

Workers build the store rows rather than handing the `StreamResult` back for the driver to
convert, for two reasons. The first is the per-stream `content_identity` HEAD: `write_run`
avoids issuing those serially only because the session CLI hands it `cloud_stats` gathered
for its progress display, and a Ray driver has no such collection, so building rows
centrally would mean one blocking HEAD per stream inside that loop. Doing it in the worker
puts the request where the stream is already being opened. The second is that rows arrive
already stamped with their own session, which leaves the driver a plain appender with
nothing to regroup.

`rows` is opaque because the store keeps one Lance dataset per metric, each with its own
schema (`store_schema.MEASUREMENT_SCHEMAS`), so a session's rows do not share a single
Arrow type and cannot travel as typed columns of a single Ray record. The flat sibling
columns carry everything logging and the run summary need, so nothing reads `rows` except
the append step.

All Lance writes stay on the driver. Concurrent `append_rows` from many workers would hit
the create-or-append race that `robot_action_split/lance_sink.py` flags as an open TODO,
and there is no reason to inherit it.

`stream_idx` is not in the record: it selects which video stream to open *inside* a
container, and stays at the `run_checks` default of 0, as `di-session` does — so a container
holding more than one video stream has only its first measured. `check_session` packages each
successful run through the shared `results.stream_result` helper and takes the default
selector, exactly as the session runner does, because `stream_key` hashes the selector
alongside the source: hand-building a `StreamResult` with a different selector would fork
stream identity from what the CLIs write for the same file. `run_id` and `created_at` are
computed once on the driver and passed into `check_session`, so every row in one run shares
them.

## Pipeline Execution

```text
pipeline.run(config)
  │
  ├─ expand_sessions(config.input)          → list[str] of session paths
  │      explicit list + list file + root expansion; distinct, and none nested
  │      inside another; an empty result is a config error
  │
  ├─ run_id = store.new_run_id()            → one id for the whole invocation
  │
  ├─ ray.data.from_items([{"session_path": p} for p in sessions])
  │      .map(check_session)                → one record per session, carrying its rows
  │           lists the session, then measures each stream in sequence
  │           a failed listing raises, for Ray's map retry
  │           a failed stream becomes a stream row carrying `error`
  │      .iter_batches()                    → finished sessions arrive on the driver
  │
  ├─ per batch, on the driver:
  │      sum the counters, then
  │      append_rows(...) → stream.lance, measurements/<metric>.lance (x5), evaluation.lance
  │
  ├─ zero streams across every session      → config error, before the commit
  ├─ commit_run(root, run_id=run_id, ...)   → run.lance; everything above becomes visible
  ├─ write_manifest(root, run_id=run_id, ...)  → manifest.json
  └─ return summary dict                    → PipelineRunOutput; runtime echoes it; exit 0
```

Items are wrapped into single-key dicts because `from_items` on bare strings names the
column `item`; `next/recipes/video_split/pipeline.py:103` wraps the same way. Seeding from
`from_items` also spreads the sessions across blocks on its own. Were this recipe to seed
from `multimodal-split`'s discovery table instead, it would have to repartition first —
`ray.data.from_arrow` maps one table to one block however the table is chunked, so the whole
run would collapse into a single task.

One stage means the driver never holds the run: nothing is materialized between listing and
measurement, and a batch of finished sessions is the most that crosses it. It also means
there is one concurrency knob rather than two. Listing is IO-bound and measurement is
decode-bound, which would argue for sizing them separately, but a task now does both, so
`session_concurrency` sizes the pair; each task takes Ray's default single CPU, since the
engine decodes one stream at a time.

The cost of one stage is that the run's stream count is not known until every session has
been listed. Two things follow. The summary counts are summed from the batch counters as they
arrive rather than known up front, and the "no streams anywhere" error can only be raised
after the append loop — which leaves rows written but uncommitted, and therefore invisible,
exactly as any other run that does not reach its commit.

De-duplicating sources is no longer possible mid-run, and does not need to be: since no
session is nested inside another (see [One source reached twice in one
run](#one-source-reached-twice-in-one-run)), no source is reachable from two of them.

Measurement retries happen inside the worker, not around it.
`configure_ray_data_stability` (`next/core/ray_runtime.py`) sets
`DataContext.max_map_retries = 3` with `retried_map_errors` limited to transient transport
failures — `OSError`, `ConnectionError`, `TimeoutError`, `EndpointConnectionError`,
`ReadTimeoutError`, `IncompleteRead` — which is the class of error a long S3 `GET` hits. That
machinery only sees exceptions that *escape* the map function, and the measurement half of
`check_session` never lets one out, so for that half it would never fire — the helper's own
docstring says as much of any recipe that turns IO errors into per-item rows. Rather than give
up the never-raise contract to reach it, the retry lives where the exception is already
caught: `run_one_stream` takes `max_attempts`, and `check_session` passes
`execution.stream_attempts`. Per-stream isolation is unchanged — a stream that fails every
attempt is still one row rather than a failed session — and a read timeout stops being
recorded as a data-quality finding manufactured by the network. `max_attempts` defaults to 1,
so `di-session` retries nothing and behaves exactly as before.

Calling `configure_ray_data_stability` still earns its place, because the listing half does
raise. A transport error there escapes and the map-level retry covers it, and retrying costs
nothing: listing is the first thing the task does, so no measurement is repeated.

Which failures count as transient is decided by a short list of exception class names beside
`run_one_stream`, matched against the raised exception's MRO — a third copy of Ray's own list
in principle, but not in practice: the two existing copies (`_MAP_RETRY_PATTERNS` in
`next/core/ray_runtime.py` and its twin in the legacy `_runtime.py`) are private, and
`next/AGENTS.md` forbids `next/` importing the legacy tree at all. Matching the hierarchy
rather than a formatted string is also what lets this list be shorter: bare `OSError` is
deliberately absent, because here it would drag in `FileNotFoundError` and `PermissionError`
and spend the whole budget re-confirming that a path is still missing.

---

## Persistence

One `run_id` for the whole invocation, and one commit as the last dataset write, so a run
that dies partway through leaves rows that are present but never believed by a reader. See
[Data-Integrity Store — Lance Schema](data-integrity-store-schema.md), section "One run,
several tables — so a run needs a commit point". This recipe does not change that rule,
only how many sessions precede it.

The driver's whole sequence is already public API, and not a novel use of it: `reevaluate`
already writes this way, calling `append_rows` and then `commit_run` directly rather than going
through `write_run`. `storage_options` is built once with
`get_lance_storage_options(root, ...)` — the `_cli_cloud` helper that `store.py` and every
`read_*` already use — and passed to the appends and the commit below; `write_manifest` takes
the profile and endpoint themselves instead:

1. `append_rows(rows, join(root, dataset), schema, storage_options)` per dataset, as
   batches of finished sessions arrive — `stream.lance`, five `measurements/<metric>.lance`
   (`store_schema.metric_dataset_path`, one per entry in `INSTRUMENTS`), and
   `evaluation.lance`.
2. `commit_run(...)` once, appending the single `run.lance` row that makes everything above
   visible. Its `num_streams` is every stream the invocation attempted across all sessions,
   errored ones included, which is how `write_run` counts them and what makes that row
   describe the invocation; `reevaluate` is the only caller passing null there, because a
   re-judge touches no stream.
3. `write_manifest(...)`, overwriting the store's single `manifest.json` to describe the
   most recent run.

This sequence has been run end to end on local Ray — four streams across two sessions, one
unreadable, plus a third session path dropped for being nested inside one of them — and
produces one committed run, each session's provenance on its own rows, no verdict rows for
the unreadable stream, and a store `reevaluate` then re-judges. What remains unproven is
scale and the cloud paths, not the store contract.

Nothing is grouped by session on the way in. Every row already carries its own
`session_path`, stamped by the worker that produced it, so the driver appends in arrival
order and per-session provenance survives without a groupby. Row payloads cross the driver a
batch at a time rather than a run at a time, and for the whole run it holds only the session
list and the counters.

### What this needs from `store.py`

Two renames: `_stream_row` and `_measurement_row` become `stream_row` and
`measurement_row`, joining the already-public `build_evaluation_row`, with `write_run`
updated to the new names. `write_run` gathers a whole session before writing anything, which
a driver measuring thousands of streams across many sessions cannot do, so the builders it
uses are the public seam a driver needs. Workers then build every row through public API —
`content_identity` for provenance, `stream_key(result)` for the dedup key, `instrument` for
each result's spec, and the three builders. `content_identity` has to be handed the run's
`s3_profile_name` / `azure_profile_name` / `endpoint_url`, exactly as `write_run` hands them:
it never raises, so an unprofiled `HEAD` records empty provenance instead of failing.

No new function, no schema field, no dataset name, no dedup key and no commit rule changes.
`write_run` keeps its one production call site, `store_cli.persist_run`, which is what both
CLIs go through, so neither CLI changes.

### The run row's `session_path` is null

`commit_run` and `write_manifest` both take a scalar `session_path` describing the run as a
whole. A multi-session run passes `None`, which the schema permits
(`store_schema.py:157`) and which `di-check` already writes today for a single-stream run.

This is not the rejected alternative below. That one passes `None` *down into every row*,
erasing `session_id`, `session_path` and `relative_key` per stream. Here only the
run-level summary is null and the data rows keep their own session. Nothing reads a run row
by session either: `completed_runs` selects the `run_id` column alone, and `read_runs`
sorts by `(committed_at, run_id)` without filtering.

### `session_rollup` answers a wider question now

`session_rollup(root)` takes no session argument: it reads every committed stream and verdict
under the root and reduces them to one status. On a store `di-session` wrote, that is a session
verdict; on a store this recipe wrote, it is a whole-dataset verdict under the same name. No
stored row changes, and nothing in the pipeline calls it — `session_id` and `session_path` are
on every row, so a per-session rollup is a group-by away. Adding that argument is follow-up
work rather than part of this recipe.

### One source reached twice in one run

`stream.lance` and the metric tables de-duplicate on `stream_id` alone, ordered by
`created_at` DESC then `run_id` DESC. `stream_id` hashes the absolute
`(source, selector_type, selector_value)` triple, deliberately independent of the session it
was discovered under, so `di-check` and `di-session` produce one row for the same bytes
rather than two (store schema, "Dedup keys" and "Stream identity").

Every row in one invocation shares one `run_id` and one `created_at`. So if a source is
reached through two session paths — overlapping `session_roots`, a repeated entry, or a root
nested inside another — its two rows collide on the key with nothing left to break the tie.
`_latest_per_key` sorts on `(created_at, run_id)` and keeps the first row per key, so the tie
falls through to Lance's physical row order: stable for a given dataset, but decided by which
Ray task finished first rather than by anything meaningful. One row wins, the other's
`session_path` goes with it, and nothing reports that it happened. Writing one run this way
confirms it: two sessions sharing a source leave a single stream row, carrying whichever
session was appended first. This is the one way the design can lose the per-session
provenance it exists to preserve.

Expansion is therefore the only place this is prevented, since each session is measured on
its own and no later stage sees two of them together. `expand_sessions` returns distinct
paths, and then drops any session that lies under another — matching whole path segments, so
`clips/a` swallows `clips/a/b` but not `clips/ab`. The enclosing session is the one kept,
because listing recurses, so it already covers the nested one's streams; dropping it would
lose coverage where dropping the nested one loses nothing. Sorting first is what makes the
choice deterministic rather than dependent on which input form named which path, and both
drops are logged.

Two spellings of one path and two paths for one source are all this catches. Two genuinely
distinct sessions that both point at the same bytes — through a symlink, or a bucket alias —
would still collide, and are out of scope: nothing in a listing distinguishes them.

`tool` is `"data-integrity"`, joining the existing `di-check` / `di-session` /
`di-reevaluate` values; the comment in `store_schema` that enumerates them needs the new
entry.

One invocation applies one threshold policy. `commit_run` takes a single `thresholds` and
stamps `policy_id(thresholds)` on the run row and on every evaluation row, so the `checks:`
block configures the run rather than individual sessions.

## Failure Isolation and Exit Status

No stream failure propagates out of `check_session`. A failed open or decode becomes
`StreamResult(error=str(exc), metrics=[])`, which the store already persists as a stream row
with no measurement or evaluation rows, and the session's other streams are measured
regardless. That is not merely the same *representation* `session_runner` produces today — it
is the same logic, so `check_session` calls it rather than restating it: `_run_one_stream` is
promoted to `run_one_stream`, unchanged for its existing caller apart from the new
`max_attempts`. Data-quality FAILs are ordinary evaluation rows. Neither reaches the exit
status. Transient transport failures are retried inside the worker before they are recorded
that way, as [Pipeline Execution](#pipeline-execution) describes.

A failed *listing* is the one failure that does escape, and Ray's map retry covers it. Should
it exhaust the retries, the run fails rather than silently covering fewer sessions than it
was asked to — which is the right outcome for an error that means a whole session went
unmeasured.

| Situation | Persisted as | Exit |
| --- | --- | --- |
| Metric FAIL against the thresholds | evaluation row | 0 |
| Unreadable or undecodable input | stream row with `error` | 0 |
| Metric ran but was undefined on too little data | measurement row with `is_defined` false, plus a `SKIPPED` evaluation row | 0 |
| Metric never ran, with no usable rate | measurement row with `is_defined` null, plus a `SKIPPED` evaluation row | 0 |
| Invalid config, or an input set expanding to zero sessions | nothing | nonzero |
| Ray execution failure | rows may exist, uncommitted and unread | nonzero |
| Store write or commit failure | nothing visible to a reader | nonzero |
| Interrupted with Ctrl-C | rows may exist, uncommitted and unread | nonzero |

`run()` returns `{"run_id", "sessions", "streams", "unreadable", "failed_metrics",
"store_root"}`, which the kind's `prepare_run` wraps in a `PipelineRunOutput` — the dict as
`json_payload`, plus a one-line `message` naming the run id, the stream count and the
findings. Findings are counted there, never encoded in the exit status. Per-session progress
is logged at INFO.

---

## Module Inventory

New, under `cosmos_curator/next/recipes/data_integrity/`:

| File | Contents |
| --- | --- |
| `config.py` | Pydantic models (frozen, strict, `extra="forbid"`), `kind: data-integrity`, `load_config` / `resolve_config` with dotted overrides |
| `sessions.py` | `expand_sessions(input_config, *, execution)` for the three input forms, over the shared child-prefix listers |
| `processing.py` | `check_session(record, *, config, run_id, created_at) -> dict`, the Ray UDF, over `discover_session(session_path, *, config) -> list[str]`; only the listing can raise |
| `pipeline.py` | driver `run_config(config) -> dict` plus an argparse `main()`; the only module that imports Ray |
| `pipeline_kind.py` | `DATA_INTEGRITY_KIND = PipelineKind(name="data-integrity", ...)`, all seven callables, `list_presets` returning `[]`; every import deferred into its callable (`noqa: PLC0415`) and `_prepare_run` resolving the config before returning the closure, mirroring `video_split/pipeline_kind.py` |

Modified: `store.py` (the two renames above), `session_runner.py` (`_run_one_stream` promoted
to `run_one_stream`, plus `max_attempts`), `store_schema.py` (the `tool` comment at `:155`,
one word), `core/utils/storage/s3_client.py` and `storage_utils.py` (the two hoisted listers),
`next/recipes/multimodal_split/discovery.py` (repointed at them), and
`client/pipeline_cli/builtin_pipeline_kinds.py` (one import, one tuple entry). The two
composition-root tests that pin the registered names —
`tests/cosmos_curator/client/pipeline_cli/test_builtin_pipeline_kinds.py` and
`test_pipeline_app.py`, which spells the same list into an error message — change with it.

## Relationship to `di-check` and `di-session`

| Concern | `di-check` | `di-session` | `data-integrity` |
| --- | --- | --- | --- |
| Unit of input | one video | one session | many sessions |
| Session enumeration | n/a | caller supplies the path | expanded from config |
| Concurrency | none | `ThreadPoolExecutor` | Ray Data |
| Entry point | argparse CLI | argparse CLI | config file + `pipeline_runtime` |
| Metric kernel | shared | shared | shared |
| Store write | `write_run` | `write_run` | `append_rows` then `commit_run` |
| Exit status | `PASS` / `FAIL` / `ERROR` as 0 / 1 / 2 | same | 0 on completion, nonzero only on operational failure |

Exit status is the only row where this recipe deliberately behaves differently rather than
simply doing more. A nonzero code on FAIL is useful in an interactive CLI; a pipeline over
thousands of sessions must not report a data-quality finding as a job failure.

---

## Alternatives Considered

**One run per session, N runs per invocation.** Call `write_run` once per session and skip
the two renames entirely. Rejected because it writes N rows to `run.lance` and overwrites
`manifest.json` N times for one invocation, so "what did this invocation do" stops being
answerable from the store and no row's `num_streams` describes the invocation as a whole —
which is the run-level provenance this recipe exists to add.

**Aggregate every session into one `write_run(session_path=None)` call.** Also needs no
store change. Rejected because a null `session_path` there propagates into every stream,
measurement and evaluation row, nulling `session_id`, `session_path` and `relative_key`
per row — the provenance that has to survive. Passing null for the run row alone, as
[Persistence](#persistence) does, is a different thing.

**Stream-granularity Ray records, listed in a stage of their own.** Buys one thing today: a
global listing pass can de-duplicate sources, where the single stage relies on the nesting
guard instead. Rejected because a cross-sensor check has no per-stream task to run in, so
this would need the decision reversed later rather than extended — see [Work
Granularity](#work-granularity).

**Session-grain store rows to match.** A `session.lance` table, and verdicts that hang off a
session rather than a stream, is what a cross-sensor check ultimately needs. Deferred rather
than rejected: nothing measured today produces a session-grain verdict, so the table would
have no rows to hold. Aligning the unit of work first is what keeps it an additive schema
change.

**Reuse `session_runner.run_session` inside a Ray task.** Attractive because it is the
existing single-session orchestrator. Rejected for what surrounds the loop rather than the
loop itself: it re-runs discovery, assembles a `SessionReport` this recipe does not persist,
and owns the progress hooks a CLI needs. `check_session` calls `run_one_stream` directly
instead, which is the part that is shared. Its `max_workers` pool is the shape an
intra-session concurrency knob would take if one is ever wanted — deferred, not rejected,
and until then there is only one knob because the pool's default is a single worker.

**A bespoke argparse entry point reusing `di-session`'s exit codes.** Rejected: that returns
1 on a data-quality FAIL. `pipeline.py` keeps an argparse `main()` for local iteration, but
it mirrors the runtime's contract.

**A recipe-local delimiter lister.** Rejected because `multimodal-split` has the same listing
([MR 1104][mr1104]): `curator-next.md`'s rule is that reuse waits until something else needs
it, and something else now does.

---

## Delivery Plan

1. **Prep** — the two store renames with `write_run` following, `_run_one_stream` promoted
   and given `max_attempts`, and the two listers hoisted out of `multimodal-split`. Each is
   small enough to review on sight, and together they unblock everything else.
2. **Config, input contract, processing, CLI registration** — everything except Ray. This
   establishes the exit-status and summary contract early, as `robot_action_split` did.
3. **Ray Data wiring** — the per-session `map` stage, the concurrency knob, and the
   local-Ray tests.
4. **Example config** — plus a run against a multi-session cloud subset, which is the one
   claim on this page that a local run cannot make.

## Testing Strategy

Under `tests/cosmos_curator/next/recipes/data_integrity/`:

| Test | Asserts |
| --- | --- |
| `test_config.py` | resolution, dotted overrides, template completeness, and the rejections: an empty input block, an `az://` or bucketless store root, a deeper `session_depth`, the underscored kind |
| `test_sessions.py` | all three input forms, S3 roots against a fake lister; empty expansion raises; overlapping roots and repeated entries collapse to distinct sessions; a nested session is dropped while a session merely sharing a name prefix is not |
| `test_processing.py` | a session yields a row per stream it holds, plus five measurement and five evaluation rows for each readable one; an unreadable stream yields a stream row carrying `error` without costing its siblings; an empty session yields counters and no rows; a failed listing escapes for Ray's retry while a failed measurement never does |
| `test_pipeline.py` | local Ray over two sessions with one corrupt file: exit 0, exactly one committed `run_id`, each session's `session_path` present on its own rows, findings persisted; a third session path nested inside one of them is measured once, under the session enclosing it; sessions holding no streams fail without committing; an unwritable store root fails the run |
| `test_pipeline_kind.py` | the kind's seven callables, and a subprocess guard that registering it imports neither Ray nor Lance |
| `test_store.py` (extend) | a caller can assemble a run from the public builders, `append_rows` and `commit_run` |
| `test_session_runner.py` (extend) | `max_attempts` defaults to one attempt, retries a transport error, and never retries a malformed file or a missing path |

Plus the registered-name tuples in `test_builtin_pipeline_kinds.py` and `test_pipeline_app.py`,
which spell the same list into an error message. Upstream,
`test_importing_composition_root_defers_config_and_runtime_modules` already pins that
registering a kind does not drag its config and runtime modules into the CLI's import graph,
which is what keeps the Ray-free guard true from the other side.

The boundary is this recipe's own seams. `run_checks`, the instruments and the evaluators
are already covered by the kernel and recipe suites and are not re-tested here.

---

## Open Questions

**How should `session_depth` reach a lister that only returns immediate children?** The
shared lister stops at one level, which is all `multimodal-split` needs, since a session there
is always a direct child of the prefix. Depth 1 is therefore what ships, and a deeper root is
rejected. Supporting one later means either walking level by level from the driver — one
delimited list per prefix, which fans out quickly — or giving the shared lister a depth
argument and letting it walk for both recipes. The same question decides whether a glob is
worth supporting instead of a fixed depth: more general, but it invites recursive listing
costs on a dataset this size. Either way it is additive, and nothing above needs rework for
it.

Three earlier questions are settled, and the sections above describe the outcome rather than
the choice:

- The two store row builders are public, so a driver can compose a run from one place instead
  of per-call `noqa`s or one run per session.
- `check_session` reuses `run_one_stream` rather than restating its wide catch, and the
  bounded retry rides on that helper's new `max_attempts` — default 1, so `di-session` is
  unchanged.
- `session_concurrency` takes a plain default. `next/AGENTS.md` forbids `next/` importing
  the legacy `pipelines/ray_data` tree, which is where the helpers that would size it from
  the live cluster live, so promoting them is that migration's work rather than this
  recipe's.

---

## Acceptance Criteria

The first implementation is complete when:

- one invocation over multiple sessions writes exactly one `run_id` into one store root,
  and every row carries the `session_path` and `session_id` of the session it came from
- a run whose inputs include both data-quality failures and unreadable files exits 0,
  persisting the failures as evaluation rows and each unreadable input as a stream row
  carrying `error`
- invalid config, a Ray execution failure, and a failed store write each exit nonzero, and
  no data-quality finding produces a nonzero exit
- an input set that expands to zero sessions is rejected as a configuration error
- a session path nested inside another configured session is measured once, under the session
  enclosing it, so no two rows in a run collide on `stream_id`
- a store written by `di-session` is readable and appendable by the pipeline, and a store
  written by the pipeline is re-judgeable by `reevaluate.reevaluate()` (which stamps
  `tool="di-reevaluate"` and keys off `measurement_run_id`), with no schema change
- `di-check` and `di-session` keep their current output and exit codes
- `cosmos-curator pipeline template` and `pipeline schema` accept `data-integrity` as their
  kind argument, `pipeline validate` / `pipeline render` accept a `kind: data-integrity`
  config, and the underscored spelling is not accepted anywhere
- `pixi run run-pipeline` executes the recipe from a config file
- `python -m cosmos_curator.next.recipes.data_integrity.cli --help` still does not import
  Ray
- the pipeline runs end to end over a multi-session subset of the target bucket and
  persists a committed run

## Non-Goals

- new metrics or new modalities
- distributed re-evaluation from stored measurements
- restarts after failure, and idempotent re-runs
- any change to `_cli_cloud.py`, to `di-check` / `di-session` behavior, or to the Lance
  schema itself (the `tool` comment aside)

[mr1104]: https://gitlab-master.nvidia.com/aidot/cosmos-curator-public/cosmos-curator/-/merge_requests/1104
