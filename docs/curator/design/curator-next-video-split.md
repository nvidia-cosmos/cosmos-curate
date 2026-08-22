# Curator Next: Video Split

## Summary

`video-split` turns source videos into fixed-stride MP4 clips. Ray Data overlaps source downloads, CPU transcoding, and
clip uploads as independent streaming operators. S3 stores clip media, Lance stores the canonical clip metadata, and a
JSON report records only source or clip failures.

This is intentionally a happy-path implementation. It does not currently recover work across Ray runs, write source
outcomes, or promise atomicity between media objects and the final Lance commit.

Config kind dispatch is exact: `video-split` selects this recipe, while `video_split` selects the deprecated Ray Data
pipeline. The similar names are temporary while the older package is being retired.

## Scope

The recipe:

- accepts MP4 sources from S3
- generates fixed-duration spans at a fixed stride, with a configurable minimum tail duration
- transcodes the first video stream with a CPU H.264 encoder
- copies the first audio stream when present
- writes each clip as a standalone MP4 to S3
- publishes one complete Lance snapshot containing successful clip metadata
- replaces one JSON error report after each successful clip commit

Scene detection, GPU transcoding, recovery, filtering, captioning, and embedding are outside this increment.

## Inputs

A run uses exactly one input form:

- `uris`: an explicit set of S3 MP4 object URIs
- `root_uri`: an S3 prefix whose MP4 descendants are discovered recursively

Before Ray starts, the recipe normalizes and deduplicates the source selection. Recursive discovery schedules larger
objects first, with URI order breaking size ties; explicit selections remain URI-ordered. Scheduling order does not
affect clip identity.

A normalized URI is assumed to identify immutable content. Replacing an object at the same URI is outside the current
contract. An empty selection fails before publication so a mistyped root cannot erase a prior snapshot.

## Outputs

Media and metadata remain separate:

- clip MP4s live at `<media_root>/clips/<clip_id>.mp4`
- `<media_root>/lance` contains one row per successfully uploaded clip
- `errors.json` contains an array of source and clip errors; a clean run writes `[]`

Each clip row contains the source URI and ID, requested span, deterministic clip identity and URI, source media
properties, and output media properties. Clip identity depends only on source URI, requested span, and the media
contract. It does not depend on run order, worker placement, or execution tuning.

The error report is operational rather than a canonical downstream dataset. Source errors have `scope: "source"` and
null clip fields. Errors after planning have `scope: "clip"` and identify the affected clip and span. Successful sources
do not emit a terminal source record; valid sources that plan zero clips are silent.

## Dataflow

```text
S3 source selection
        |
        v
download + probe + in-memory plan       0.25 CPU + one curator_io slot
        |
        v
source flat_map generator               configured transcode CPUs
  FFmpeg batch -> clip file -> bytes -> yield one clip/error at a time
        |
        v
clip upload map                         0.25 CPU + one curator_io slot
  upload bytes -> drop bytes -> emit clip metadata/error
        |
        v
streaming metadata repartition
        |
        +---------------------------+
        |                           |
        v                           v
worker-written Lance fragments      streamed JSON error rows
        |                           |
        v                           |
atomic Lance snapshot overwrite     |
        |                           |
        +-------------> replace errors.json
```

The download stage hands one source byte payload to one source-level transcode task. That task writes one temporary
source file and reuses it for all planned spans. Spans are processed in bounded multi-output FFmpeg batches. A failed
batch is retried clip-by-clip so one damaged span does not discard successful siblings.

The transcode function is a generator. It reads and yields one completed clip at a time rather than building a list for
the source. Ray's byte-sized output blocks and streaming executor backpressure bound the clip payloads in the object
store. Independent upload tasks consume those blocks while other source tasks continue transcoding. Upload removes the
payload column before metadata reaches publication.

Publication coalesces metadata into bounded row batches. Workers write uncommitted Lance fragments and return only
fragment descriptors plus individual JSON error records. The driver consumes those results incrementally, retaining
fragment descriptors in memory while streaming errors into a temporary file. It commits the complete clip snapshot and
then uploads the completed error file, including an empty array for a clean run.

## Failure Semantics

Expected source and media failures become data records:

- `source-read`: a source could not be downloaded after retries
- `source-probe`: FFprobe could not describe the source
- `transcode`: FFmpeg could not produce one planned clip
- `clip-probe`: FFprobe could not validate one produced clip

Unexpected exceptions, missing worker tools, invalid internal records, exhausted clip-upload retries, Lance failures,
and error-report upload failures fail the task or run. This prevents a broken environment or shared destination from
being reported as a large collection of bad media.

Successful clip uploads use deterministic destinations and tolerate Ray task replay by replacement. If a run fails
before the Lance commit, uploaded MP4s or uncommitted Lance data may remain invisible to canonical readers. A later fresh
run safely overwrites deterministic clip paths and replaces the complete Lance snapshot.

The Lance overwrite is the canonical reader-visibility boundary. `errors.json` is written afterward; if that write
fails, the run fails even though the clip version is already visible. A rerun repairs the report.

## Configuration

```yaml
schema_version: 1
kind: video-split
input:
  root_uri: s3://example-bucket/raw
output:
  media_root: s3://example-bucket/curated/video-split
```

`clips_lance_uri` defaults to `<media_root>/lance`, and `errors_uri` defaults to
`<media_root>/errors.json`. The Lance destination may be a driver-local path for local runs; media and errors
remain on S3.

Execution tuning controls CPU reservations, FFmpeg batch size and threads, retry/timeout behavior, publication batch
size, and progress output. These settings do not participate in clip identity.

## Deferred Recovery

Recovery will be added without changing the source-to-clips happy path. After each successful upload, a pass-through
operator can write clip receipts containing full clip metadata plus the source plan count/digest. Recovery can then:

1. discover the current source set;
2. group and deduplicate clip receipts by source;
3. remove sources whose receipt set proves the complete plan;
4. rerun every other source through this same pipeline; and
5. union reused clip metadata with fresh clip metadata before publication.

No separate durable plan manifest or final source reconciliation is required. A zero-clip completion receipt may be
added later to avoid harmlessly replanning valid short sources.

## Validation

The current increment should demonstrate:

- deterministic identities and canonical clip schema
- one source download reused by all of its clip transcodes
- generator fan-out before a long source finishes
- separate download, transcode, and upload Ray Data operators
- bounded publication and error-report collection
- successful sibling clips surviving an isolated media failure
- a complete clip overwrite and a replaced JSON report on each successful run
- an empty source selection leaving prior outputs untouched

Cross-run head-failure recovery is not claimed until durable clip receipts and recovery selection are implemented and
qualified separately.
