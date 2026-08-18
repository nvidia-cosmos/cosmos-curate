# Curator Next: Video Split

## Summary

`video-split` turns source videos into fixed-stride MP4 clips. Ray Data distributes CPU transcoding, S3 stores the clip
media, and Lance stores clip metadata and source outcomes. The recipe runs on either a local Ray cluster or a
[managed Ray cluster on Slurm](curator-next-slurm-ray.md). Each source is processed as one unit: download it once,
derive its spans, and transcode every clip from the worker-local copy.

Config kind dispatch is exact: `video-split` selects this recipe, while `video_split` still selects the deprecated Ray
Data pipeline. The similar names are temporary and one will remain after `cosmos_curator.pipelines.ray_data` is removed.
The Xenna and legacy Ray Data implementations are behavioral references for this port.

This recipe also establishes the execution, publication, and failure-handling pattern for later Curator Next recipes.
`robot-action-split` is separate and is expected to reuse that pattern where it applies.

## Scope

The recipe:

- accepts MP4 sources from S3
- generates fixed-duration spans at a fixed stride, with a configurable minimum tail duration
- transcodes the first video stream with a CPU H.264 encoder
- copies the first audio stream when present
- writes each clip as a standalone MP4 to S3
- writes one Lance dataset for canonical clip metadata
- writes one Lance dataset with a success or failure outcome for every source video

Scene detection, GPU transcoding, filtering, captioning, and other curation steps are outside this recipe.

## Inputs

A run uses exactly one input form:

- `uris`: an explicit set of S3 MP4 object URIs
- `root_uri`: an S3 prefix whose MP4 descendants are discovered recursively

Before Ray starts, the recipe normalizes, deduplicates, and sorts the source URIs. A root is resolved once per
invocation. S3 object keys are used verbatim, without percent-decoding, and a normalized URI is assumed to identify
immutable content. Replacing an object at the same URI is outside the contract.

An empty selection fails before publication. This protects existing snapshots from a mistyped or not-yet-populated
input prefix.

## Outputs

Media and tabular metadata remain separate:

- clip MP4s are written beneath a configured media root
- `clips.lance` contains one row per successfully written clip
- `sources.lance` contains one row per realized source video

Each clip row contains its source, requested span, deterministic identity and URI, and source and output media
properties. Each source row contains its status, clip counts, media properties, and any failure diagnostic. Source media
properties are null only when probing failed; this still gives failed and valid zero-clip sources useful records.

Clip identities and paths depend only on the source URI, requested span, and media contract. They do not depend on run
order or worker placement. Clips live in one flat, identity-addressed prefix; consumers use `source_id` in Lance rather
than navigating a per-source directory tree.

## Target Scale

A representative run turns about 30,000 sources into 1,000,000 clips, or roughly 30 clips per source. The defaults are
tuned around that shape:

- 30,000 sources provide ample source-level parallelism without splitting one source across workers
- each source is downloaded once and its worker-local file is reused for every clip
- 1,048,576 clips per publication batch matches Lance's native row limit, producing one roughly 400–550 MB metadata
  fragment per million clips; Lance also retains its 90 GiB soft file-size limit

Clip rows and bytes never reach the driver. Only per-source summaries do, so source count—not clip count—is the main
driver-side scaling limit.

## Dataflow

```text
S3 source selection
        |
        v
for each source:
  restore a completed result, or
  download once, probe, split, transcode, and upload
        |
        v
per batch: write clip Lance fragments and extract source outcomes
        |
        +-------------------------------+
        |                               |
        v                               v
commit clips.lance              order and commit sources.lance
```

Source processing writes the S3 object to a temporary worker-local file, probes it, derives the selected spans, and
transcodes those spans from the same file. After processing every planned clip, the worker emits one complete source
outcome plus the successful clip rows. Failed clip rows do not need to cross the worker boundary: their count and first
diagnostic are already captured by the source outcome. Failed and valid zero-clip sources therefore produce an outcome
without requiring a synthetic plan marker.

Publication repartitions those terminal records independently of source processing. Each publication batch writes
Lance fragments and returns any complete source outcomes it contains. The driver receives only fragment metadata and
one row per source; it validates and restores source-selection order without reconstructing information from clip rows.

## Media Contract

Spans have an inclusive start and exclusive end. A short final span is kept only when it meets the configured minimum
duration.

Transcoding preserves source dimensions and nominal frame rate, encodes video with `libopenh264`, and copies the first
audio stream when present. FFmpeg chooses the exact video boundaries. Copied audio follows encoded packet boundaries and
may differ slightly from the requested span.

Published media properties are measured rather than inferred. In particular, frame count is null when the container
does not report it; the recipe does not estimate it from duration and frame rate.

Clip paths identify a logical media contract, not a byte hash. Replay may overwrite the same path, and historical Lance
versions do not preserve earlier MP4 bytes independently. A fixed transcoder or toolchain behavior change that alters
the logical output therefore requires a media-contract version bump and produces different clip identities.

## Source Outcomes and Publication

A source succeeds when probing succeeds and every planned clip is written. A valid source with no retained spans
succeeds with zero clips. If probing or any clip fails, the source fails, but successful sibling clips remain in
`clips.lance`; failed clips do not get clip rows. Planned, published, and failed counts distinguish partial output from
complete failure.

Each run publishes complete snapshots with `Overwrite`. Concurrent runs targeting the same dataset URIs are unsupported.
The clip snapshot is committed first. The source snapshot follows and records the exact clip version it describes. The
two commits are not atomic: if source publication fails, the run fails and the new clip snapshot may remain. A rerun
overwrites both.

`sources.lance` is the logical metadata-publication commit point. Consumers that need a coherent pair open the current
source snapshot first, then read the exact clip URI and version recorded there; they do not independently pair both
datasets' latest versions. Until the source commit succeeds, the prior source snapshot continues to name the prior valid
clip metadata version.

Workers write clip rows as uncommitted Lance fragments and return their metadata. The driver commits all fragments in
one transaction, reassigning fragment IDs that collide across workers. An empty fragment list is a valid empty snapshot.

Media and storage failures attributable to one item become failure outcomes for the current snapshot. They are retried
with bounded exponential backoff first. Unsupported media is not retried after FFprobe has identified it, while FFmpeg
failures use the full retry budget because FFmpeg does not distinguish permanent corruption from a transient transport
failure. Source-read failures can describe one selected source and therefore become source outcomes. Media-destination
failures describe shared run configuration and fail the task or run rather than producing thousands of clip failures.
Other configuration errors, missing tools, and programming bugs likewise fail the task or run. A run is operationally
successful only after both snapshots are published.

## Elastic Ray Execution

The pipeline does not fix its worker count at startup. New workers can pick up pending work; losing workers only reduces
throughput, and having no workers pauses progress. Source processing requests CPUs, so Slurm lanes may provide either
CPU-only nodes or CPU capacity on accelerator nodes.

While the head survives, Ray handles node loss by replaying tasks and reconstructing data from lineage. Worker-local
files are temporary, and deterministic S3 paths make repeated media writes safe. Ray Data may put several source rows in
one block task, but each fully successful source already has its own durable result. Replaying the block restores those
sources and repeats only incomplete or failed ones. Losing a publication task may leave an unused Lance fragment. This
is safe at-least-once execution, not byte-level exactly once.

## Recovery Across Ray Runs

The head is the boundary of Ray's recovery. A replacement cluster has no access to the old lineage or driver state, so
cross-run recovery belongs to the recipe. The design follows the legacy
`cosmos_curator/pipelines/video/splitting_pipeline.py`: durable output-side records, rather than Ray state, identify
completed work.

Recovery is automatic; there is no run or attempt ID. A fully successful source writes one atomic `result.arrow` under
the media root in a namespace derived from the split and transcode contract. The result contains the complete source
outcome and all successful clip rows needed to rebuild both Lance snapshots. Worker count, retry limits, and other
execution tuning are deliberately excluded from the namespace, so those settings can change between invocations without
invalidating completed work. A changed split or transcode contract gets a different namespace.

Every invocation resolves the current source selection. A source with a saved result is restored without accessing its
media. A source without one is processed from the beginning. This includes sources that failed to probe or transcode on
an earlier invocation: failures are not checkpointed, so a transient failure does not become permanent. A valid
zero-clip source is successful and does write a result.

There is no separate saved plan and no per-clip receipt log. Those intermediate checkpoints would reduce replay work,
but they are not needed for correctness when the source is the retry unit. An MP4 by itself is not a checkpoint: a crash
after one or more uploads but before `result.arrow` safely repeats the source and overwrites the same deterministic clip
paths.

The current source selection, restored outcomes, and new outcomes are always assembled into complete `Overwrite`
snapshots. `Append` is deliberately avoided for the canonical datasets: retries could duplicate rows, sources removed
from the input could remain visible, and contract changes could mix incompatible data.

The relevant crash windows are therefore:

| Last durable event | Recovery behavior |
| --- | --- |
| No saved result | Process the complete source; deterministic media writes may be repeated. |
| `result.arrow` written | Restore the source and clips without accessing the source media. |
| All source results written, no Lance commit | Rebuild metadata fragments and commit the complete snapshots. |
| Clip snapshot committed, source snapshot absent | Restore source results and overwrite both snapshots again. |

Checkpoint files are a cache, not part of either published dataset. Removing them only gives up reuse on the next run;
it does not damage an existing snapshot. Cache reads and writes use bounded retries and are best-effort: an unavailable,
corrupt, or incompatible checkpoint is treated as a miss and the source is processed normally. Old contract namespaces
and checkpoints for sources no longer selected can be expired with an S3 lifecycle policy. Lance version and
unreferenced-fragment cleanup remains a separate storage concern.

A retained checkpoint assumes that all clip objects it names remain durable. Media cleanup must therefore remove the
corresponding recovery result first; otherwise a later run could restore metadata for media that no longer exists.

## Validation

The current implementation should demonstrate that:

- dispatch remains exact: `video-split` selects this recipe and `video_split` keeps its legacy behavior
- input selection, spans, and clip identities are deterministic across worker counts and partitioning
- empty input fails without changing existing snapshots
- every source-processing call downloads its source once and reuses the worker-local copy for all clips
- workers write clip fragments and the driver commits them once, including empty output and colliding fragment IDs
- every source worker emits one complete outcome, including failed and valid zero-clip sources
- successful sibling clips remain published when one clip fails, and the source counts describe the partial result
- item errors become outcomes while configuration and programming errors fail the run
- worker replay is safe through deterministic media paths
- a replacement Ray cluster restores fully successful source results without an attempt ID
- execution-only config changes reuse checkpoints, while split or transcode changes do not
- failed sources are retried from the beginning and restored results produce duplicate-free complete snapshots
- the source snapshot names the exact clip version, and failure between the two commits fails the run
