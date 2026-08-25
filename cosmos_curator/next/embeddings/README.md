# `embeddings/` — modality feature embedders (Curator Next)

A generic capability: turn a structured input dataset (one keyed row per item)
into one independent, key-aligned **feature column group per modality**, added
**directly onto the source `clips.lance`** with native Ray Data. There is no
separate embedding table — each modality owns a private `embedding_<modality>_*`
namespace on the same row. This is the **embedding stage only** — no clustering,
curation, or reporting.

```text
                        +--> modality A --> + embedding_a_* columns --+
one keyed source  ------+--> modality B --> + embedding_b_* columns --+--> same table,
table (clips.lance)     +--> modality . --> + embedding_._* columns --+    one row per item
```

**Reuse:** a new modality is added by defining its `EmbeddingColumnGroup` in
`schemas.py` and building a `ModalityFill` for it in the recipe's `modalities.py`
— the schema-evolution and column-write machinery is modality-agnostic, and a
modality is *data* (group, source columns, embedder, actor shape) rather than a
subclass. Its embedder must be **key-blind** and **cardinality- and
order-preserving**: the batch it is handed carries `clip_id` (the fill joins on it)
but the embedder must not read it, and it returns exactly one row per input row, so
a clip it cannot embed becomes an all-NULL row rather than a dropped one. The recipe attaches the key positionally, so a batch whose row count changes
raises `FillContractError`, as does one whose columns do not cast to the stored
schema — the contract in full is
[design doc](../../../docs/curator/design/curator-next-embeddings.md) §4.1, §4.2,
§7, §11.3.

**Current implementation (egocentric robotics).** The shipped realization embeds
`robot_action_split` clips into three modalities — text, image, action (+ a
fitted action-PCA basis), all written back onto the one `clips.lance`:

```text
clips.lance --+--> text  (SentenceTransformer) --> + embedding_text_*   (2 x 384-d, unit-norm)
  (one row     +--> image (HF vision backbone)   --> + embedding_image_*  (384-d, unit-norm)
   per clip)   +--> action (wrist + PCA)         --> + embedding_action_* (97-d, raw PCA coords)
                                                     clips.lance__action_pca/<fingerprint>.npz
```

The default checkpoints are BGE-small (text) and DINOv2-small (image); the code
names the embedders by role (`SentenceTransformerTextEmbedder`,
`HfVisionImageEmbedder`) and carries the concrete checkpoint in a `ModelSpec`
(`DEFAULT_TEXT_MODEL` / `DEFAULT_IMAGE_MODEL`), so swapping the checkpoint does
not rename any type.

**Why three vectors:** a later curation leg needs a *distance* between clips to
balance and de-duplicate. Two clips can match on task text yet look different and
move differently, so text / image / action are three independent redundancy
axes. Full rationale, schemas, and the action-descriptor geometry live in the
design doc: **[docs/curator/design/curator-next-embeddings.md](../../../docs/curator/design/curator-next-embeddings.md)**.

## Why three embeddings, not one

Each modality answers a different question about the **same** clip, and each one
*intentionally ignores* what the others capture — so they complement rather than
duplicate one another:

```text
Robot clip
    +-- What is happening?      --> Text   (task / subtask semantics)
    +-- What does it look like? --> Image  (scene / object appearance)
    +-- How was it performed?   --> Action (wrist motion / dynamics)
```

| Modality | Captures | Ignores | Compared by |
|---|---|---|---|
| **Text** | task / subtask meaning, paraphrases | appearance, motion, environment | cosine (unit-norm) |
| **Image** | scene / object appearance, viewpoint, lighting | task meaning, motion over time | cosine (unit-norm) |
| **Action** | hand / wrist trajectories, manipulation style, dynamics | appearance, scene, language | Euclidean (raw PCA) |

Any single modality alone would group or split clips incorrectly:

- **Same task, different object** — `"pick up the mug"` (white vs blue mug): text
  and action match, only **image** keeps the object variety.
- **Same scene, different task** — same kitchen, `"press button"` vs `"open
  drawer"`: image matches, only **text/action** separate the tasks.
- **Same motion, different wording** — `"pick up bottle"` vs `"grab bottle"`:
  text strings differ, **action** (and paraphrase-robust text) recover the match.

The three vectors are stored as **co-located column groups on the same
`clips.lance` row**, so a clip's modalities are already aligned — no
modality-to-modality join is needed. A consumer can read one group or several; a
later curation leg is expected to fuse them into one distance for clustering,
de-duplication, and balanced sampling:

```text
Text . Image . Action --> fusion --> similarity --> clustering --> dedup --> balanced set
```

**This leg produces only the modality-specific embeddings** — it does *not*
implement fusion, clustering, or curation. See the design doc §1 for the full
relationship model, correlations, and worked examples.

## Task vs subtask: two text vectors

The text leg stores **two** 384-d vectors per clip, not one. Every clip is one
**subtask** that belongs to a larger **task**:

```text
Task: "Make coffee"
  +-- Pick up cup
  +-- Insert capsule
  +-- Press button
  +-- Pour coffee       <- each leaf is one clip's subtask
```

- `embedding_text_subtask` — the fine-grained instruction ("Insert capsule"); the
  dominant signal for de-duplication and fine clustering.
- `embedding_text_task` — the parent activity ("Make coffee"); groups a clip with
  its siblings for task-level balancing and analytics.

Why not one blended string like `"Make coffee. Insert capsule."`? A blend is
irreversible — you could no longer measure task- and subtask-similarity
separately, weight them (`0.7*subtask + 0.3*task`), search by one level, or
cluster on one level. Storing them apart lets a downstream leg pick a strategy
**without re-embedding**. Both vectors share one row, come from the same model in
a single pass, and sit in the same `embedding_text_*` group; they are kept in
separate columns only because they represent different semantic levels. Full
rationale + examples: design doc §4.3.

## How data flows through embeddings

The embed leg starts from the clips Lance table that `robot_action_split`
already wrote, and reads **nothing else** from the original dataset — no episode
video, no LeRobot `data/*.parquet`, no `meta/`. It reads one row per
`(span, view)` clip, embeds it, and writes the vectors back onto that same row.

### The source row (one `(span, view)` clip)

Each modality reads only its own source columns. `clip_id` is both the row's
identity and the key each fill joins on:

| Field | Where it came from (set by `robot_action_split`) | Feeds |
|---|---|---|
| `clip_id` | SHA-256 of `(media_contract_version, span_group_id, video_bitrate[, view_name])` | the write join key (no embedder reads it) |
| `task_name` | `meta/tasks.parquet` label, materialized into the row | text |
| `subtask_name` | `meta/subtasks.parquet` label, materialized into the row | text |
| `clip_uri` | `.../video/<view>/<clip_id>.mp4` — the **generated clip**, not the raw recording | image |
| `action_data_uri` | `.../action/<action_id>.bin` — one artifact **per span** | action |

A **span** is one contiguous subtask cut of one episode; a multi-camera source
emits one **view** (one clip) per camera. Every view of a span shares one action
artifact, so several `clip_id`s can legitimately point at the same
`action_data_uri`.

### The three flows

```text
                    clips.lance row
     +-------------------+---------------+--------------------+
     | task/subtask text |   clip_uri    |   action_data_uri  |
     v                   v               v
 collapse whitespace  open MP4 stream   read ACT2 / decode
     |                first frame        wrist trajectories
   BGE-small          DINOv2-small       600-d descriptor
     |                   |                  |  PCA (97-d)
     v                   v                  v
 embedding_text_*     embedding_image_*  embedding_action_*   + __action_pca/<fp>.npz
 (2 x 384 unit)       (384 unit)         (97 raw PCA)
```

- **Text** — reads the two strings (no video, no parquet); emits two 384-d
  unit-norm vectors (subtask + task). Every scanned row fills (never fails).
- **Image** — applicable where `clip_uri` is present; decodes the **first
  displayable** frame and runs the vision backbone -> one 384-d unit-norm vector.
  An unreadable clip fills as an all-NULL image group (retried next run),
  preserving one output per scanned row.
- **Action** — applicable where `action_data_uri` is present (Mecka-only, purely
  structural — no `source_dataset` gate); builds a dual-wrist 600-d descriptor and
  projects it through a content-addressed PCA basis -> one 97-d **raw** PCA vector.
  A malformed / non-mecka artifact fills as an all-NULL action group.

### Output: `embedding_*` column groups on `clips.lance`

Each modality owns a group of nullable columns, with widths fixed by
`fixed_size_list` and no field-level metadata: a column's owner is its
`embedding_<modality>_*` prefix, and the authoritative mapping is the
`EMBEDDING_COLUMN_GROUPS` registry in `schemas.py`. A group's columns exist only
**after** its modality has run, so a consumer tests **column presence**, not just
NULL.

| Group | Columns | Scale |
|---|---|---|
| text | `embedding_text_subtask` (fsl<f32,384>), `embedding_text_task` (fsl<f32,384>), `embedding_text_model_id` (string) | unit-norm |
| image | `embedding_image` (fsl<f32,384>), `embedding_image_model_id` (string) | unit-norm |
| action | `embedding_action` (fsl<f32,97>), `embedding_action_descriptor_version` (string), `embedding_action_pca_fingerprint` (string) | raw PCA |
| action PCA | `clips.lance__action_pca/<sha256-fingerprint>.npz` (immutable, content-addressed basis) | — |

**Direct-column persistence — the fragment is the unit of work.** A modality adds
its group in one metadata `add_columns` commit, then fills it: the driver pins a
version and lists the **fragment ids** (a metadata read — no row is scanned or
counted), hands those ids to an actor pool, and each worker re-opens the table at
that version, scans its own fragment's pending rows, embeds them, and writes the
group's columns with `LanceFragment.update_columns` joined on `clip_id`. Only
fragment metadata (never a vector) reaches the driver, which commits every touched
fragment in **one** atomic `LanceOperation.Update`.

Neither step creates a **tombstone**: `add_columns` on an all-nullable schema is
metadata-only, and `update_columns` rebinds field ids to a new column file without
replacing any row, so Lance writes no deletion vector. A fragment with nothing
pending is not rewritten at all, so a re-run over a filled table commits nothing.

**A failed fragment is skipped, not fatal.** The worker warns, the run commits
every fragment that did succeed, and the skipped rows keep their NULL group values
— so the ordinary pending predicate re-selects them and the next run refills them
with no operator action. The one exception is `FillContractError`, whose contract
is "a failure the run must stop for rather than skip past": a violated invariant of
the fill (a broken embedder — wrong row count, or a schema that does not cast — or
a fragment absent at the pinned version), and a total outage (nothing written while
at least one fragment failed). Which of the two occurred, and so whether to re-run
or to fix an input, is in the message. See design doc §7.

## Reuse the action components on their own

The extractor, projector and fused embedder are plain callables over Arrow batches,
so they run without Ray or Lance; `PcaArtifactStore` owns the immutable basis and
its storage. What none of them do is write `clips.lance` — the column write and its
commit belong to the recipe's `fill.py`.

```python
import pyarrow as pa
from cosmos_curator.next.embeddings.action.embedder import (
    DualWristMotionDescriptorExtractor,
    DualWristMotionProjector,
    DualWristMotionReadConfig,
)
from cosmos_curator.next.embeddings.action.pca import PcaArtifactStore, action_pca_root_uri

# Any Arrow batch of the extractor's source columns will do -- no Lance needed.
batch = pa.table({"clip_id": ["a", "b"], "action_data_uri": ["s3://.../1.bin", "s3://.../2.bin"]})

extractor = DualWristMotionDescriptorExtractor(DualWristMotionReadConfig())
descriptors = extractor(batch)  # 600-d descriptors, one row per input row

# Load the immutable basis a group's rows reference by fingerprint (fit separately
# on the first run); PcaArtifactStore loads/saves <root>/<fingerprint>.npz.
store = PcaArtifactStore(action_pca_root_uri("clips.lance"))
pca = store.load("<fingerprint>")
columns = DualWristMotionProjector(pca).project(descriptors)
# `columns` holds exactly the embedding_action_* fields, one row per input row and
# in input order -- no key. That contract is what lets the recipe's fill worker
# attach clip_id positionally and write the group with one update_columns call.
```

`DualWristMotionEmbedder` fuses these two steps and is what a worker actually
constructs; use the pieces separately only when you want the 600-d descriptors
themselves (the PCA fit does exactly that).

## FAQ

- **Why one table with column groups, not three tables?** Co-locating each
  modality on the `clips.lance` row gives independent presence / regeneration with
  no modality-to-modality join — design doc §3.
- **Why two text embeddings, and do they embed the same data?** Two semantic
  levels from two *different* strings — design doc §4.3.
- **Why one image frame / why PCA / why 97-d?** Cheap correlated-frame proxy;
  reduce a redundant 600-d motion descriptor — design doc §4.2, §5.
- **Can I regenerate one modality / re-run safely?** Yes — `modalities` selects
  which run; a re-run fills only still-NULL rows (which is also how a skipped
  fragment retries), and `--reset-group` replaces a whole group — design doc §7,
  §11.
- **Why are some clips NULL / missing in a modality?** A modality leaves the row's
  group all-NULL when it cannot embed it or the row is inapplicable; a modality
  that never ran has no such column at all — design doc §3, §8.
- **What if I delete a PCA `.npz`?** Bases are immutable and content-addressed —
  deleting one that live rows reference makes the next action run fail with a clear
  missing-basis error. `--reset-group action` followed by a run fits a fresh basis
  and recomputes the group with it — design doc §5, §11.
