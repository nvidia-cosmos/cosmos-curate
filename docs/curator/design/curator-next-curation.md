# Curator Next — Curation (clustering, de-duplication, balanced selection)

Curation reads the wide clips table, decides which clips belong in a smaller, less
redundant, task-fair training subset, and writes that verdict back onto **the same rows
of the same table** as two `curate_*` columns.

**Wide table in, wide table out.** There is no side table, no staging tree, and no
sidecar report. `clips.lance` is both the input and the output, which is why the config
field is named `clips_lance_uri` rather than a source.

Curation is a full recomputation. A different selection target means a different run;
there is no persisted ordering to re-threshold and no incremental path.

This document is the authority for what curation means and why it is built this way. The
[embeddings design](curator-next-embeddings.md) explains how the `embedding_*` columns get
onto that table; this document begins there.

---

## The two questions curation answers

Curation is often described as "clustering embeddings", which hides the fact that it
answers two independent questions about the same corpus:

| Question | Depends on | Answered by |
|---|---|---|
| **Is this recording redundant?** | what the clip *is* — instruction, appearance, motion | multimodal fusion + locality clustering + SemDedup |
| **Is this meaning fairly represented?** | what the clip is *labelled* | annotation semantics + balanced selection |

The two use different representations because they are different questions. A pair of
clips can be:

- **redundant but differently labelled** — two recordings of the same physical action
  where an annotator wrote different words;
- **identically labelled but not redundant** — the same instruction performed in a
  different room, or with a visibly different motion.

Any design that answers both questions with one grouping gets one of them wrong. Curation
therefore maintains **two geometries** over the same rows.

```text
              clip similarity geometry                 annotation semantics geometry
                        |                                          |
        +---------------+---------------+                 +--------+--------+
        |               |               |                 |                 |
     subtask          image           action            task             subtask
      text                                              text              text
        |               |               |                 |                 |
        +---------------+---------------+                 +--------+--------+
                        |                                          |
                        v                                          v
                 DEDUP LOCALITY                          REPRESENTATION BALANCING
             (which pairs to compare)                   (which meanings get a turn)
```

Neither geometry refines the other: one locality cluster contains clips from many tasks,
and one task spans many locality clusters. They share **no identifier** — `curate_cluster_id`
is never a fairness group, and a fairness group is never a dedup partition.

The two geometries are **distinct, not independent**: they are computed over different
metrics and are not interchangeable, but they are measurably correlated, for a reason
built into the design. See
[Three geometries, not two](#what-must-stay-separate) for the measurement and its cause.

Both geometries read vectors from the same wide row. The distinction is *which* vectors and
*for what*: the similarity geometry fuses subtask text, image and action into one metric,
while the semantics geometry reads the canonical task label **string** — using
`embedding_text_task` only to decide when two task labels name the same thing — and
partitions `embedding_text_subtask` into a fixed number of cells for its second level. That
second partition is a clustering too, over a different vector at a different `k` and for a
different purpose; see [Three geometries, not two](#what-must-stay-separate).

---

## Data flow

Three reads of the table, three shuffles over the lineage they produce, and one
commit. The reads are the sampled fit, a bounded label gather, and the one
row-scale scan; the write reopens each fragment it touches.

```text
                        clips.lance @ pinned version v
        clip_id │ task_name │ subtask_name │ embedding_text_task
                │ embedding_text_subtask │ embedding_image │ embedding_action
                                     │
  (1) FIT   bounded fragment prefix ─► fused vectors  ─► single-GPU k-means
                                    └─ subtask text   ─► single-GPU k-means
                                     │
                     centroids (k × 865) + (k2 × 384) ──── broadcast ────┐
                                     │                             │
  (2) LABELS one Ray task per fragment: project task_name and
            embedding_text_task, roll up to one row per distinct canonical task
                                     │
            driver merges near-duplicate TASK labels, over LABELS
            (O(L), never O(N)) → label-to-representative map ──── broadcast ──┐
                                     │                                       │
  (3) SCAN  one Ray task per fragment id  ◄──────────────────────────────────┴┘
            scanner pushes down "every weighted vector IS NOT NULL"
            canonicalize the task label, fuse, assign nearest locality centroid,
            assign nearest subtask centroid
              eligible      → __dedup_key = cluster id, reason NULL
              non-finite    → __dedup_key = -1, reason 'invalid_embedding'
              zero-norm     → __dedup_key = -2, reason 'invalid_embedding'
                                     │
  (4) DEDUP groupby(__dedup_key) ─► one whole-GPU task per cluster
            key < 0 passes through untouched — no GPU math, no NaN spread
            the fused vector is dropped here; a distance and a transient
            similarity score survive
            SKIPPED ENTIRELY when dedup_eps is None: a narrow projection
            drops the fused vector in its place, so the schema is identical
                                     │
            apply the label-merge map, then materialize()   ← ~20 GB, no vectors
                                     │
  (5) FAIR  count survivors per merged (task, subtask cluster) → driver quotas
            water-fill, then rank within group and cut to quota
            rows that already carry a reason pass through, outside the cut
                                     │
                              materialize()   ← second barrier: the verdicts are
                                     │          counted for completeness, then written
  (6) WRITE groupby(__frag) ─► update_columns(left_on="clip_id") per fragment
            the driver collects fragment metadata only, never rows
                                     │
                        ONE LanceOperation.Update
                                     ▼
                        clips.lance @ version v+1
              … + curate_selection_reason │ curate_cluster_id
```

Per pass:

| Pass | Input | Output | Scope |
|---|---|---|---|
| Fit | a bounded prefix of fragments | locality basis + level-2 subtask basis | sample only, one GPU task |
| Labels | one fragment, `task_name` and `embedding_text_task` | one row per distinct canonical task, with its count and one vector | bounded by distinct tasks, not rows |
| Scan | one fragment | fused assignment + subtask cell assignment + task canonicalization + invalid verdicts | one fragment at a time |
| Dedup (conditional) | one locality cluster | `duplicate` verdicts + a distance per survivor + a transient similarity score | within one cluster only; skipped entirely at `dedup_eps = None` |
| Fair | survivor counts per merged `(task, subtask cell)` | quota per group, then a verdict per row | driver holds counts; workers rank rows |
| Write | one fragment's verdicts | one fragment's updated metadata | one fragment at a time |

**Why the labels get their own pass.** The merge needs one text vector per
distinct task label, and the main lineage must never carry a 384-wide text vector past
the scan, because everything after the scan shuffles. So the gather is its own
reduction that emits `O(L)` rows while the row-scale scan projects only the fused
blocks. Fusing the two into one pass would push the widest column in the run
through the dedup shuffle to save reading one column. The level-2 key needs no such pass
because it is decided per row inside the scan, against a basis the fit already broadcast.

Both barriers exist for the same reason: a Ray dataset is lazy, and each of these
two lineages is read twice — the merged rows to count groups and then to cut
them, and the verdict rows to check completeness and then to write them.

Every read names one pinned Lance version, captured once before any scan. A concurrent
embedding commit therefore affects the next curation run, not the current one.

**Rows whose required vectors are NULL never enter the lineage at all.** They are not lost:
they read `curate_* = NULL`, and the embedding column itself says why. Pushing that filter
into the scanner is what keeps Curate from reading hundreds of gigabytes of vectors it
cannot use.

### Why the fit runs on a sample

The clustering step exists to bound how many rows one GPU must hold at once, but the fit
itself is the largest single allocation in the system: the clustering runtime holds roughly
four full-width copies of its input at peak. At the supported corpus size that is several
terabytes, well beyond one device, so the centroid basis is fit on a bounded sample and
every row is then assigned by a comparatively cheap matrix product.

The sample is a deterministic prefix of whole **fragments** in manifest order. Whole
fragments rather than a row limit, because a row-level limit is not obviously deterministic
across runs and random sampling would need a seed plus a second pass. The budget is counted
in **eligible** rows — the per-fragment predicate counts preflight already holds — so it
still needs no extra read. Eligible rather than physical, because a leading fragment that
holds no usable vector would otherwise spend the whole budget and leave the prefix with
nothing to fit.

The prefix **crosses** that budget: each fragment is taken before the budget is tested, and
the fit then truncates inside the crossing fragment. Stopping short is the trap, because the
fit does not fail on a sample too small — it clamps `k` down to what it got and reports
success, having partitioned the corpus by a basis nobody asked for. Crossing costs at most
one fragment of extra rows read, and bounds no allocation: each matrix is sized at
`min(its own row budget, the prefix's eligible rows)` — `fit_sample_rows` for the locality
matrix, and a per-centroid budget that never exceeds it for the subtask one ([The two
ceilings on `subtask_clusters`](#the-two-ceilings-on-subtask_clusters)). Eligible in that
second term
because the sample scan reads only predicate-passing rows, so sizing a buffer by what the
prefix *stores* would reserve node RAM for rows that never arrive.

A manifest prefix is deterministic but it is **not a uniform sample** — it is the
earliest-written rows. Because the cap cuts *inside* a fragment, the effective sample is a
prefix of rows in write order, which is a stronger bias than a prefix of fragments.

The producer makes that correlation **certain at dataset granularity** rather than
hypothetical. `source_dataset` is a required scalar on the ingest config, stamped on every
row of the run, and a run's fragments are appended sequentially from a single driver — so the
table is a concatenation of dataset-pure slabs, and a prefix covers the earliest one to few
datasets out of N. What is *not* settled is whether those datasets occupy different regions of
the fused space: the boundary is a fact, its consequence is not. Measuring it needs a
per-fragment `task_name` / `subtask_name` distribution on a real table, since querying
`source_dataset` only re-confirms what the code already proves.

Within a dataset the picture is better than it looks. Whole-fragment sampling is cluster
sampling, whose effective size tracks the number of *clusters* drawn rather than rows, and at
8,000-row fragments the prefix draws `4,000,000 / 8,000` = **500** of them — enough that the
within-dataset design effect stopped being the binding concern. That came from the producer
raising its fragment count for unrelated reasons, not from anything this leg does, and it
leaves `fit_sample_rows` as a lever on cluster count but **not** on dataset coverage: a
longer contiguous prefix draws more clusters from the same earliest datasets.

All of this is acceptable only because a cluster is a computational partition: the same prefix
would be indefensible if `curate_cluster_id` carried meaning.

How much of the corpus the prefix actually covered is therefore reported: the run logs the
prefix's **eligible row count as a fraction of all eligible rows** once, on the driver. That
figure is `min(fit_sample_rows, eligible)`, a ceiling rather than a measurement — the fit has
not run yet, which is why the line reads "at most ~X%". That single number is what separates
the two regimes an operator cares about — a sample that covered most of the corpus cannot be
badly biased whatever the write order, while a sample covering a few percent is fitted on
whatever those fragments happened to hold. It is a count against a count, so it costs nothing
beyond what the run already knows.

`k` is derived on the driver and may only be **clamped downward** inside the fit task, to the
sample's **realized** row count — the rows that actually survived the eligibility predicate,
which only the fit can know, because eligibility is a filter rather than a count. That
realized count, not the driver's ceiling above, is what the fit reports back and what the
centroids artifact records.

The trade is a corpus-scale shuffle — rows must be regrouped by cluster so each cluster's
rows sit together for de-duplication — and the knowledge that a very small, very tight
cluster may not receive its own centroid. Locality clustering tolerates that, because a
cluster is only a computational partition.

---

## What curation persists

Exactly **two columns**, both nullable, added to `clips.lance` in one metadata-only commit:

| Column | Type | Meaning |
|---|---|---|
| `curate_selection_reason` | `string` | why this clip was or was not selected |
| `curate_cluster_id` | `int32` | the **locality** partition it was de-duplicated within (see [What `curate_cluster_id` is, and is not](#what-curate_cluster_id-is-and-is-not)) |

They are atomic siblings: the table carries both or neither, and exactly one present is a
corrupt schema rather than a state to repair. Both being nullable is what makes the
widening metadata-only — every pre-existing row reads NULL without a data file being
rewritten, so adding the columns cannot create a tombstone.

Among the rows a run claimed, `curate_cluster_id` is NULL exactly when the reason is
`invalid_embedding`, because such a clip never entered the clustering at all. A row the run
never claimed reads NULL in *both* columns, so a NULL cluster id on its own does not identify
an unusable vector — pair it with the reason, as [the state model](#the-state-model) does.

### The five reasons

`curate_selection_reason` has exactly five values, written lowercase because they are a
persisted format contract that every downstream export queries directly
(`WHERE curate_selection_reason = 'selected'`):

| Reason | Meaning |
|---|---|
| `selected` | survivor inside its group's quota at this target |
| `duplicate` | near-duplicate of a strictly earlier row in its cluster; never competes for selection budget |
| `below_quota` | survivor in a funded fairness group that lost the within-group ordering |
| `unfunded` | survivor in a fairness group that received no budget at this target |
| `invalid_embedding` | every required vector present, but one is non-finite or zero-norm, so the clip could not be fused |

`below_quota` and `unfunded` distinguish "your group had room and you did not make the cut"
from "your group got nothing" — states a single boolean cannot express, and that mean
different things when tuning a run. The distinction is free to compute: the quota map is
already broadcast to every worker in order to select at all, so an unfunded group is one
comparison against a value the worker already holds.

**There is deliberately no `missing_embedding`.** A row whose required vector is NULL is
never claimed by a run, and `embedding_action IS NULL` already answers "why is this row
uncurated" — so a stored value would be a mirror field restating a queryable predicate.
`invalid_embedding` earns a value for the opposite reason: discovering it means reading the
vector and testing it, which no predicate can do. The two are not two values; they are one
value and one derivation, and that asymmetry is the whole justification.

Whether an invalid vector was non-finite or zero-norm, and which modality it came from,
stays out of the schema. A debugger reads the offending clip's own vector columns, which
name the modality directly; a reader of a training set does not need the distinction.

### The state model

Five states, all readable from one column plus columns that already exist:

| Query | State |
|---|---|
| `curate_selection_reason IS NULL` and every weighted vector present | **not curated** — the run did not cover this row: the columns were just added, or the row was appended after the last run |
| `curate_selection_reason IS NULL` and any weighted vector NULL | **could not be curated** — a required embedding is absent, pending or inapplicable, indistinguishable on a wide table |
| `= 'selected'` | in the curated subset; `curate_cluster_id` non-NULL |
| `IN ('duplicate', 'below_quota', 'unfunded')` | evaluated and excluded; `curate_cluster_id` non-NULL |
| `= 'invalid_embedding'` | evaluated and unusable; `curate_cluster_id` NULL |

There is no `false` anywhere: "not selected" is `curate_selection_reason <> 'selected'` and
"not curated" is `IS NULL`. On a completed run over a quiescent table the **not curated**
count is zero, and the run enforces that **before it commits**: the verdict rows are
counted per reason, and a run whose reasons do not account for every eligible row at the
pinned version raises and writes nothing. It is the single check proving no row was lost or
duplicated across the shuffles, and placing it before the commit is deliberate — a
post-commit assertion would be reporting a table that was already published.

### What is deliberately not persisted

| Not persisted | Why | How to get it |
|---|---|---|
| `distance_to_centroid` | ~1 GB at 250M rows for a value whose only use is re-deriving an ordering whose outcome the reason already records | the run logs the distribution's p50 and p95 once (see below); the per-row value needs a re-run, and is transient by design |
| duplicate similarity score | a marginality diagnostic with no named consumer | the run logs the distribution's p99, p99.9 and max plus the duplicate yield at a ladder of candidate thresholds — counterfactual rungs *plus the run's own `dedup_eps`*, marked, so the operating point is on the ladder (see below); the per-row value needs a re-run, and is transient by design |
| the fairness group a row belonged to | the level-1 half is recoverable from the row's own `task_name`; the level-2 half is a fit-dependent cell index with no column behind it, and the merged-group identity is not stable across appends (see [Merging near-duplicate labels](#merging-near-duplicate-labels)) | recompute the merge and re-run the fit |
| a `selected` boolean | a one-byte projection of `curate_selection_reason = 'selected'` | query the reason |
| run id, the config's field values, merge statistics | per-row provenance for a fact about the whole run; the commit records the config's *digest* instead, so a version can be identified without a column carrying it (see [Run identity on the commit](#run-identity-on-the-commit)) | the digest on the commit names the rules; the canonical config text is archived in the centroids artifact; the run reports the target and the merge statistics once, in [`CurateResult`](#what-a-run-reports-instead-curateresult) |
| the producer identity of each consumed embedding group | it is a property of the source columns, already stored beside them, not of a verdict | preflight logs it and the centroids artifact archives it (see [Producer identity](#producer-identity-is-part-of-the-source-contract)); on the table, read the group's own `provenance_columns` |
| `__frag`, `__dedup_key` | in-flight routing only, never columns on the table | not applicable |

The two transient columns matter enough to name. `__frag` carries the id of the fragment a
row was read from, stamped by the reader that owns that fragment, so the write can regroup
verdicts by their physical home without deriving a fragment from a row address.
`__dedup_key` carries the cluster a row is scored within, or a **negative sentinel** for a
row that must reach the write *without* entering the similarity pass — a non-finite vector
would otherwise propagate NaN across its whole cluster. The sentinel is a routing value and
not a cluster id, which is why `__dedup_key < 0` becomes a **NULL** `curate_cluster_id` on
the way to storage: no sentinel is ever readable as a cluster.

There are **two** negative sentinels, one per bypass cause: a non-finite vector and a
zero-norm one. Both persist the identical verdict — `invalid_embedding` with a NULL
`curate_cluster_id` — so splitting them is **verdict-invariant** and buys diagnosis alone.
It buys it for free **on the scored path**: the de-duplication stage's grouping key is
corpus-global, so a distinct value per cause turns that stage's existing per-group log line
into an exact corpus-wide count per cause in a single aggregate, with no extra pass. Under
`dedup_eps = None` the stage never groups at all, so no such line exists; per-cause
observability there comes from the scan's own warning, which counts non-finite and zero-norm
rows separately but emits **per scan task**, so a reader sums it instead of reading one
total. The exactly-one-line property is the scored path's; the two causes stay
distinguishable either way. The alternative — one sentinel plus a
persisted discriminator — was already rejected on the grounds that the offending row's own
vector columns name the modality directly; a routing value that never reaches storage
re-opens nothing.

Dropping `distance_to_centroid` would leave an operator with no way to see how tight the
locality clusters came out, so the run reports the **cluster radius** instead: the p50 and
p95 of the distance distribution, folded into a fixed 2048-bin histogram over `[0, 2]` and
logged once. The read-out exists because `k` rises with the corpus while `dedup_eps` is a
fixed constant, so the same eps is a different fraction of a cell's extent at every corpus
size, and there is no persisted column to check that against. It is a real cost rather than
a free one — one narrow pass over a float32 column — bought because the alternative is an
un-calibratable threshold. The driver holds the bin counts, not the rows, so the report is
`O(bins)` at any corpus size.

**The duplicate similarity score is reported the same way, and for the same reason.** The
retention pass emits each row's maximum similarity to a strictly earlier row as a transient
column, and the pass that already folds the radius histogram folds a second one over it —
2048 bins over `[0, 1]`. A row carrying **no** score — one that bypassed the similarity pass,
or every row when the stage was skipped — is **excluded from the bins and from the
denominator**, which is why the line reports its total as "over N **scored** row(s)". Bin 0
is the underflow bucket **for scored values only**: row 0 of every group is forced to 0.0 and
an antipodal pair scores negative, so both land there.
The read-out is the distribution's **p99, p99.9 and max**, plus a **counterfactual ladder**:
how many rows *would* have been flagged at each of `eps` in `{0.001, 0.005, 0.01, 0.02,
0.05, 0.10}`. Percentiles alone answer "how close does this corpus come to the threshold";
the ladder answers the question an operator actually has, which is "what would a different
threshold have cost me", and it answers it from bin counts the pass already holds rather
than from a re-run per candidate value.

**The run's own `dedup_eps` is always a rung**, spliced into those static values and marked
with a trailing `*`. Without it a run configured looser than the widest static rung reports a
ladder whose every entry is a counterfactual, so the reader cannot locate the threshold that
actually produced the verdicts among the ones that did not. Because each rung counts from the
first bin whose *lower* edge reaches `1 - eps`, the marked rung is a lower bound accurate to
one bin width rather than an exact restatement of the `duplicate` verdict count.

**An unscored row and a scored row whose value is zero are different populations**, and
keeping them apart is what makes the read-out safe to calibrate from. Binning the unscored
rows at zero would stack the whole bypass population at the bottom of the distribution and
drag every reported percentile down with it, so on a corpus with a large `invalid_embedding`
share a tight threshold would read as much looser than it is — an operator trusting that
reading loosens `eps` when they should tighten it. Excluding them keeps every percentile a
statement about the rows the threshold can actually act on, and bin 0 a statement about rows
that were scored and came out at or below zero.

This is deliberately the **same contract the radius report has** and not a step beyond it:
log-only, never persisted, and absent from [`CurateResult`](#what-a-run-reports-instead-curateresult).
A histogram is `O(bins)`; a per-row score is ~1 GB at 250M rows for a value with no named
consumer, which is why persisting it stays rejected. Emitting it as a transient column is
not the same decision as storing it.

### Which rows a run claims

A row is claimable only if **every block carrying weight has a vector**. The eligibility
predicate is one `IS NOT NULL` clause per non-zero-weight block, ANDed together and pushed
down into the scanner.

Omitting a zero-weight block's clause is the supported escape for a corpus whose labels are
identifiers rather than language: setting that modality's weight to zero removes it from the
fused metric *and* stops it being required of a row, with no code change.

Weight zero does not remove the *column* from the run's requirements for `task` text.
`embedding_text_task` must exist on the table whatever the weights are, because the task
label merge runs on every run and reads it per distinct label. `embedding_text_subtask` is
different: it is required exactly when its block carries weight, so zeroing the subtask
weight drops that block out of the metric, out of the per-row predicate, out of the schema
requirement **and** out of level-2 fairness, which then collapses to one cell per task.

---

## What `curate_cluster_id` is, and is not

**`curate_cluster_id` is a computational locality partition. It is not a semantic category.**

Its entire contract is to make within-cluster all-pairs comparison affordable and to bound
what one GPU must hold at once. The cluster count follows from a memory target:

```text
k = ceil(eligible_rows / target_mean_cluster_rows)
```

so mean cluster size stays fixed as the corpus grows and per-worker memory is
scale-invariant. `k` is **not** an estimate of how many kinds of clip exist.

Two clips sharing a `curate_cluster_id` are close in fused space. That is a useful proxy for
"possibly the same recording" and a poor proxy for "the same task". Consequences a reader
should expect:

- **It is not stable across runs.** Adding rows changes the sample, moves the centroids, and
  changes group sizes. A re-run reproduces a previous selection only if the inputs, weights
  and seed have not changed.
- **A cluster has no name and no meaning.** Do not build reporting, filtering, or dataset
  documentation that presents a `curate_cluster_id` as a category.
- **Do not use it as a balancing group.** Budget would then flow to whichever task happens
  to occupy more clusters, which tracks row counts rather than fairness.
- **Do not use semantic groups as dedup partitions.** That changes which pairs are compared,
  and therefore changes the duplicate decision itself.

Shipping the centroids artifact makes the column *more* inviting to misread, not less, which
is why the guardrail is stated on the column, in the code, and here.

**`k == 1` is a real and benign degenerate case.** The 200,000-row default is tuned for the
250M–500M envelope; on a small corpus `k` resolves to 1. Exhaustive de-duplication in one
cluster has perfect duplicate recall and no cluster-boundary false negatives, so the
*result* is strictly better — but three things go quietly degenerate: dedup runs as a single
task with no parallelism, `curate_cluster_id` is `0` for every row, and the centroids
artifact holds one centroid. The run therefore warns on the driver when `k == 1`, naming the
row count and the knob, so the degeneracy is loud rather than buried in a document.

At `k == 1` every eligible row lands in one cluster, so the per-group device ceiling below
becomes reachable by a corpus that is merely large rather than skewed. The warning
therefore carries a second clause: when the eligible row count exceeds that ceiling —
11,152,540 rows at the fused width of 865 on an 80 GiB device — it says so and names
`target_mean_cluster_rows` as the remedy, because raising `k` above 1 is the only way to
split the work. The clause is **advisory**: the binding refusal still lives at the
allocation site inside the de-duplication task, where the row count and the device are both
facts rather than driver-side estimates. Warning on the driver buys the operator the message
before the run spends its scan; it does not replace the guard.

---

## Fusion

### The vector

Each modality block is L2-normalized, scaled by `sqrt(w)`, and concatenated in a fixed
order:

```text
fused = [ sqrt(0.6)*u_subtask , sqrt(0.2)*u_image , sqrt(0.2)*u_action ]
        384                     384                 97                  = 865 dims
```

Two properties hold because the weights sum to 1:

```text
||fused|| = 1
1 - cos(fused_A, fused_B) = 0.6*(1-cos_subtask) + 0.2*(1-cos_image) + 0.2*(1-cos_action)
```

The second identity is exact, and it is the reason the whole design works: a single cosine
distance over one fused vector **is** the weighted multi-modal distance, so ordinary
spherical k-means and ordinary cosine de-duplication operate on it directly with no
modality-aware code anywhere downstream. The `sqrt(w)` scaling rather than `w` is what makes
it exact, because the dot product of concatenated blocks is additive.

The block order is result-defining, because it fixes which coordinate range belongs to which
modality. A centroids artifact is only readable against the order that produced it.

The block **widths** are therefore checked against the stored columns on every decoded batch,
because the order alone is not self-enforcing. A fixed-width vector column pins that all of
its vectors share *one* width, never *which* width, and concatenation joins mismatched widths
without complaint: an 8-wide text column yields a 489-dim vector that is finite and
unit-norm, so both identities above still hold while every coordinate after the text block
belongs to the wrong modality. The run would then cluster, de-duplicate and commit a
selection against a basis nobody chose, reporting success throughout. A stored width that
differs from the one the run fuses is a hard failure at decode, not a warning.

The weights are an **interface, not a tuning knob**: changing one re-clusters everything and
therefore changes every downstream result. Two runs under different weights are not
comparable. They are also chosen *together with* `dedup_eps` — under the identity above, a
duplicate needs the weighted sum of per-block distances to fall below `eps`, so at
`subtask = 0.6, eps = 0.01` the text distance alone must stay under 0.017. The effective
duplicate rule is a **conjunction**: duplicates describe the same subtask *and* look alike
*and* move alike. The general form of that per-block bound, `cos_m > 1 - eps/w_m`, is worked
through at the default weights in the [runbook](../guides/curate-runbook.md), because it is
what an operator needs while choosing a threshold rather than while reading the design.

### Why each modality is present

| Modality | Weight | What it contributes | What its absence would miss |
|---|---|---|---|
| Subtask text | 0.6 | what the clip is *supposed to be* | clips of unrelated instructions judged redundant because they look alike |
| Image | 0.2 | scene, lighting, viewpoint, objects | same instruction recorded in a different place treated as a duplicate |
| Action | 0.2 | how the motion was executed | same instruction and scene with visibly different execution treated as a duplicate |

Text dominates because two clips of different instructions are rarely redundant however
similar they appear. Image and action exist to prevent the *false positives* that text-only
similarity produces.

Read the "what it contributes" column strictly. Action contributes *execution* similarity
and nothing else — it is measurably blind to which task is being performed. On 131,602 Mecka
clips with repaired labels the action block scores a task-separation ratio of **0.990** and a
same-task-closer **AUC of 0.543**: different-task pairs sit no farther apart than same-task
pairs, and the discrimination is a coin flip. The same measurement run recovered AUC 0.853
for text and 0.879 for image on the same rows and the same grouping, so this is a property
of the modality rather than of the labels or the apparatus. It is also physically expected —
gross dual-wrist kinematics do not distinguish woodworking from `cleaning shoes`, since both
are reach, grasp, and manipulate seen from the head. Action's 0.2 therefore buys redundancy
protection, which is what the table claims, and carries no task information for anything
else to borrow.

More modalities is not automatically better: at a fixed total weight, each added block
dilutes the others. A modality earns inclusion by carrying redundancy signal the others
lack.

The 0.6 text default carries a **precondition**, not a caveat: it assumes `subtask_name`
holds real language. On a corpus whose labels are still identifiers (`subtask_6186`) the
text block contributes identifier-collision distance and then dominates the metric. Setting
`subtask = 0.0` is the documented escape and needs no code change — a zero-weight block
contributes a zero sub-vector, keeps the fused vector unit-norm, and drops out of the
eligibility predicate.

Since the level-2 fairness key became a partition of that same block, `subtask = 0.0`
also switches level 2 off: no level-2 basis is fitted, every row takes the reserved cell
`NO_SUBTASK_CLUSTER`, and fairness reduces to the task level alone. That is the intended
reading of the escape — a corpus whose subtask labels are identifiers has no subtask
semantics to be fair over — and the degeneracy WARNING reports it if the operator did not
mean it.

Action vectors are stored raw, because their magnitude encodes gesture size. Fusion is where
they are L2-normalized into the cosine space — the one deliberate scale conversion in the
system, placed here rather than at write time so the stored magnitude survives for other
consumers.

That unbounded stored magnitude is why the norm is both accumulated **and divided** in
float64, with only the stored result narrowed back to float32. There are two thresholds, and
a row crossing either one normalizes to all zeros: finite, kept, and sitting at the origin —
the one way `||fused|| = 1` could fail without anything raising.

| Threshold | What overflows | Cleared by |
| --- | --- | --- |
| norm ≈ 1.8e19 | the float32 sum of squares | accumulating the reduction at float64 |
| norm ≈ 3.4e38 | the float32 norm itself | dividing by that float64 norm un-narrowed |

The second threshold is far less likely to be reached by real embeddings than the first —
at the 384-wide text block a coordinate would have to average about 1.7e37 — but it is
reachable *with every coordinate finite*, because `sqrt` of a representable sum of squares
can still exceed the float32 ceiling. It is guarded for the same reason as the first: the
failure is silent, and the guard costs nothing. Narrowing the divisor to "keep the division
in float32" would reinstate the first failure one step later.
The division writes into a preallocated float32 buffer (`np.divide(..., out=...)`), which
lets NumPy buffer the promotion in fixed-size chunks rather than materializing a float64
copy of a corpus-scale block — measurably the cheapest of the three forms, so correctness
here costs no memory.

### Why `task_embedding` is not a fused block

The text leg stores two vectors per clip, and only `embedding_text_subtask` enters the fused
vector.

Both are produced from the same annotation text by the same model, differing only in which
field was encoded, so their similarities are strongly correlated. Concatenating both would
count that text twice and push the effective text share of the fused distance above the
configured 0.6 without saying so. It would also add little: task-level agreement is largely
implied by subtask-level agreement in the same annotation.

`embedding_text_task` is not unused, though. It is read by the **fairness** geometry, to
decide which differently-worded task labels mean the same thing — see
[Merging near-duplicate labels](#merging-near-duplicate-labels). It is read per *distinct
label*, never per row, which is what keeps the merge `O(L)`.

`embedding_text_subtask` is read twice, at two granularities, and the two must not be
confused. Fused into the similarity metric it is read **per row** and weighted; clustered
into level-2 cells it is read **per row** again, unweighted and unfused, against its own
384-dimensional basis. The same column serves both because "which recordings are alike" and
"which instruction meanings are alike" are both questions its geometry answers; nothing is
shared between the two reads but the bytes.

### What the fused vector is for

It answers "are these two recordings the same thing". It has exactly two consumers:
**locality clustering**, which bounds the pairs de-duplication compares, and the
**near-duplicate gate** itself. Both are similarity questions.

It is **not** a task identifier, and nothing should treat it as one. Semantic equivalence
across differently-worded annotations is not visible in it: two labels denoting the same
task can sit far apart in fused space. Closing that gap is what the balancing geometry is
*for*, and it closes it over the task label string, the task label vector, and the subtask
text vector's own partition — never over the fused vector.

So the following are **non-goals of the fused vector**, and none of them is a gap waiting to
be filled:

| Not for | Why not | What to use instead |
|---|---|---|
| task retrieval | no block is a task index; action's contribution is measurably task-blind (AUC 0.543) | the canonical `task_name` / `subtask_name` strings |
| task balancing | fairness must be exact over labels, and a locality cluster mixes many tasks | the fairness geometry, which never reads the fused vector |
| task grouping | `curate_cluster_id` is a computational partition, not a category | `GROUP BY task_name, subtask_name` |
| a task proxy for any new consumer | the two useful modalities for task are text and image; action would dilute, not help | read the label columns directly |

The single-modality corollary matters for anyone tempted to reach past the fused vector:
`embedding_action` on its own answers "did these two clips move alike", never "are these two
clips the same task". Anything phrased as a task question must be answered from the label
strings.

---

## De-duplication

De-duplication removes near-identical *recordings*. It is not label de-duplication, and the
two must not be substituted for one another:

```text
semantic grouping        !=   near-duplicate clip removal
"these two annotations         "these two recordings are
 mean the same thing"           the same recording"
```

### Semantics

- **What is compared:** fused 865-d clip vectors, all pairs **within one locality cluster**.
  Never across clusters; never label text.
- **Why partition at all:** all-pairs comparison is quadratic in the group size.
  Partitioning into clusters of a target size makes it affordable and bounds per-GPU memory.
  Duplicate detection is *defined* as within-cluster, so the partition costs nothing against
  its own contract.
- **Ordering:** rows are processed farthest-from-centroid first, with `clip_id` ascending and
  then `fragment_id` ascending as tie-breaks.
- **Rule:** a row is a duplicate if and only if its maximum cosine similarity to any
  strictly *earlier* row exceeds `1 - eps`. The comparison is **strict** and is made in
  float32 against float32 scores, so a row sitting exactly on the boundary survives and
  cannot be judged differently by a reader working in float64. The first row in retention
  order has no earlier row and therefore no score, so a cluster's farthest row is never a
  duplicate.
- **"Earlier" includes rows already flagged as duplicates.** This is deliberate and is
  neither connected components nor greedy maximal-independent-set. On a chain where
  `cos(a,b) = cos(b,c) = 0.995` and `cos(a,c) = 0.90`, the rule drops `c` because of the
  already-dropped `b`. A future implementation may approximate the *completeness* of the
  maximum, but must never restrict its *population* to survivors only — that would silently
  change the algorithm.
- **Survivor selection:** because rows are ordered farthest-from-centroid first, the survivor
  of a near-duplicate pair is the one further from its cluster centre.

A row carrying the negative dedup sentinel — a non-finite or zero-norm vector — passes
through untouched, with no GPU arithmetic at all. That is not an optimization: a
single NaN entering a cosine GEMM would poison every score in the cluster. The retention
stage filters these rows out before ``groupby(__dedup_key)``, projects them through the
same post-dedup schema without grouping, and unions them back with the scored clusters so
invalid rows never form two corpus-wide shuffle partitions.

De-duplication publishes its `duplicate` count through the run's in-process
[result](#what-a-run-reports-instead-curateresult), against an eligible-row denominator in
the same object. It does not publish a duplicate-edge table: a duplicate's nearest earlier
row may itself be a duplicate rather than a final representative, so any transitive reading
of those edges would be wrong. Deeper analysis belongs in an offline tool over a re-run, not
in the pipeline.

### Making the stage optional

`dedup_eps` is `float | None`. `None` **skips the stage entirely**; a float runs it, and
that float still carries a **strict** lower bound rather than admitting zero, because the
retention test is a strict `> 1 - eps` and a byte-identical pair scores exactly `1.0`: at
`eps = 0` the stage would drop nothing while still paying for every GEMM. A run that wants
no de-duplication says so by not running the stage, not by disarming its threshold — the
two are now separate expressions of intent rather than one knob doing double duty, and
`eps = 0` remains a no-op wearing a threshold's clothes.

That floor sits at `2**-24`, not at zero, because the clothes fit a whole band of values and
not just one. The comparison runs in float32 — deliberately, so a row on the boundary cannot
be judged one way here and the other way by a reader working in float64 — and float32 carries
a 24-bit mantissa, so the largest value it can represent under `1.0` is `1 - 2**-24`. An `eps`
far beneath that step rounds the threshold back to exactly `1.0f`, which no cosine exceeds. An
`eps` of `1e-9` is therefore inert for precisely the reason `0` is, while looking like an
unusually strict setting. The floor is that step and not the exact round-off boundary, which
sits an octave lower just above `2**-25` and falls half-way between two float32 neighbours —
a bound there would rest on how a single float64 subtraction rounds, so the step is the honest
one, and every value it turns away is within an ulp of inert. The config refuses the whole band
at parse time with a message pointing at `null`, rather than letting the run schedule a
whole-GPU pass per cluster to mark nothing;
`duplicate_mask` itself stays unguarded, because a raise inside a `map_groups` UDF arrives
with its type erased and only after that scheduling has already happened.

Three properties of the skip path are contract:

- **The schema is identical on both paths.** Skipping does not mean "pass the lineage
  through untouched": an explicit narrow projection takes the stage's place, so the
  865-wide fused vector is shed at exactly the same point. Without it the skip path would
  route ~865 GB at the 250M envelope into the fairness barrier that is affordable
  *because* the vectors are gone.
- **The fit still runs.** `curate_cluster_id` and the within-group distance ordering come
  from the **scan**, not from de-duplication, so a run that de-duplicates nothing still
  clusters, still assigns, and still writes a locality partition. Skipping the fit as well
  would change what the column means and what the default ordering can rank.
- **Skipping changes verdicts, and only when configured.** Rows that would have carried
  `duplicate` become selection candidates instead, so they compete for budget, they change
  every fairness group's survivor count, and they change which rows clear a quota. That is
  the intended effect and it is the only verdict-affecting change in this set: every other
  surface here — the score histogram, the three metrics, the two thresholds, the second
  bypass sentinel — is **verdict-invariant** by construction.

Because the two paths differ in verdicts, comparing across them needs care that the write
itself cannot supply. Within one table there is nothing to worry about: the write is total, so
every row carries the latest run's verdict or NULL, and nothing survives from the run before.
What the table does not record is *which path* produced a verdict, so a `duplicate` row read
from an **older table version** is not comparable with a `selected` row from a skip-path run —
the two verdicts are each current for their own version and still mean different things. See
[Rerun, rebuild, and comparing targets](#rerun-rebuild-and-comparing-targets).

### Device memory, the derived tile, and the cluster ceiling

One cluster is scored on one card, and `k` is chosen from a **mean** cluster size, so the
largest cluster is a property of the data rather than of the configuration. Two things
follow, and both are contract.

**The memory model.** For a cluster of `m` rows at fused width `d`, peak device memory is
the larger of two terms, read off the allocation sites in the kernel:

```text
normalize peak   m * (2 * 4d + 12)          the unit copy alongside its input
loop peak        m * (9 * tile + 4d + 12)   three (tile, m) blocks at once
```

The `9` is per **cell** of one `(tile, m)` block: 4 bytes of float32 similarities, 1 byte of
the boolean strictly-earlier mask, and 4 more for the float32 array the masked select
materializes before the row maximum reduces it. Three blocks is the high-water at two
distinct instants — that masked-select line, and the top of the next iteration, where the
previous block's similarities and mask are still bound while the new product is allocated —
so the loop body has **no room for a fourth block**, and a cast of the already-float32
product measures 13 bytes a cell instead of 9. The `12` is per **row** of two whole-group
index arrays. The doubling in the normalize term is the consequential one: normalization
allocates its result while its input is still referenced, so a cluster's vectors cost two
copies for the duration of that call — which **halves** the ceiling relative to a
single-copy reading of the same code.

The model bounds **device** memory only. A group also costs roughly `2 * 4dm` bytes of node
RAM, for the Arrow-decoded matrix and its retention-order gather, and nothing refuses on
that figure — at the device ceiling it is ~77 GB, so sizing a node for one cluster per GPU
is an operator assumption rather than an enforced one.

**The tile is derived, not configured.** It is a memory knob, so it earns no config field.
Each group derives it from its own row count against the device's **total** memory, less a
10% reserve for the CUDA context and pool fragmentation, quantized **down to a power of
two**. Total rather than free, and quantized, for the same reason: free memory varies
between runs, and the tile decides the GEMM's blocking and therefore the last bit of a
cosine, so a tile derived from free memory would make the documented cross-tile ulp of
drift a per-run property instead of a per-GPU-model one.

**The tile never exceeds 4096**, the value the leg has always used. That cap is
load-bearing rather than cautious: any cluster that fits at 4096 keeps it and is therefore
bitwise unchanged, so the derivation is a pure extension, and the only clusters whose
arithmetic differs at all are ones that previously could not run.

**Above the ceiling the stage refuses.** When even a tile of one row does not fit, the
vectors alone exhaust the card and no tile choice helps, so the group raises with a
self-contained message — the cluster key, its row count, the device total, the feasible
maximum for that device, and the remedies — rather than failing as a CUDA out-of-memory
error. Self-contained because the raise happens inside a Ray UDF, where the exception type
is erased and only the message reaches the operator. Splitting the cluster is **not** among
the remedies: retention is the maximum similarity over all strictly earlier rows *within*
one cluster, so a split changes verdicts. The remedies are a finer partition (lower
`target_mean_cluster_rows`) or a basis that is not concentrating rows into few clusters.

At the fused width of 865 on an 80 GiB device the model resolves to: the tile holds at 4096
up to roughly 1.9M rows per cluster — about 9.6x the 200,000-row default mean — shrinks by
powers of two above that, and the refusal begins at **11,152,540 rows**, roughly 56x the
default mean.

### Interaction with balancing

Duplicates are excluded **before** group sizes are counted, so they never consume selection
budget. Ordering matters: balancing a population that still contained duplicates would spend
fairness budget on redundancy.

---

## Balanced selection

### The requirement

> Given the surviving clips and a target of `M`, every task should receive comparable
> retention opportunity regardless of how many rows or locality clusters it occupies, and
> within a task every subtask should receive coverage before any subtask is over-sampled.

Two properties of that sentence drive the design: fairness is **uniform** (not proportional
to size), and it is **nested** (task first, then subtask within task).

The target's denominator is the **survivor population** — not the corpus and not the
eligible rows. `target_fraction = 0.5` means half of the clips that survived
de-duplication; duplicates, invalid rows, and rows the run never claimed are all outside it.
A count clamps to the population; a fraction rounds half-up and then clamps to at least one
survivor, so a small population cannot round a live target down to nothing. Neither form set
means keep every survivor.

A run in which de-duplication left **no** survivor raises instead of committing. Every
eligible row would carry `duplicate` or `invalid_embedding`, and a committed table saying
"nothing was selected" is indistinguishable at a glance from one nobody curated.

### What "level" means, and why there are exactly two

A *level* is a tier of the grouping hierarchy, not a number the algorithm produces. This
design has two: task is level 1, and subtask-within-task is level 2. There is no level 3
because the annotation model has no third granularity to be fair over; a deeper hierarchy
would need a third group key and a third allocation pass, and no consumer has asked for one.

The two levels no longer derive their keys the same way. Level 1 is a canonical label with a
merge over it; level 2 is a cell of a k-means partition over `embedding_text_subtask`. They
remain one hierarchy because level 2 is still nested inside level 1 — a task's quota is what
its cells divide — but a reader should not expect symmetry between them.

The word turns up again in a different sense under Allocation: the *fill line* `L` inside a
single pass. That `L` is an integer cap applied within one pass, not a tier. The two senses
are easy to conflate because they share a word, so this document says "level" only for the
tier and "fill line `L`" for the cap.

### Grouping: canonical labels, then a similarity merge

Balancing groups come from annotation text, not from the fused geometry:

```text
Level 1:  canonical task label
Level 2:  (canonical task label, subtask cluster id)
```

The level-1 label is read from the clips table's own `task_name` column, which is
authoritative for annotation text. The level-2 key is **not** a label: it is the index of
the k-means cell the row's `embedding_text_subtask` falls in, at a fixed
`subtask_clusters`. `subtask_name` is not read by curation at all.

The substitution exists because `subtask_name` does not survive the corpus target. It is
free-form annotator prose resolved per shard with shard-local indices, so the corpus
vocabulary is the SUM of per-shard vocabularies: measured at **0.816 distinct labels per
row** with near-zero cross-shard collision, which is ~204M labels at 250M rows. That one
quantity was simultaneously the merge's input size `L`, the quota's group count `G`, and a
shuffle key, so it broke three things at once. A cell index bounds all three at
`subtask_clusters` regardless of corpus size.

`subtask_clusters` defaults to **16** and is a bounded integer (`ge=1`), not a
corpus-derived quantity — deliberately, because deriving it from `N` is exactly what would
put the ceiling back in motion. At the measured 2,738 tasks the level-2 group ceiling is
`2738 × 17 ≈ 46,500`, so a target of 100k+ clips can fund every cell more than once.

#### The two ceilings on `subtask_clusters`

Compute cost does not *bind*, but it is not free either, and the difference is worth stating
because it is the sentence an operator will lean on when they decide moving `k` upward costs
nothing. Assignment really is negligible: an `argmax` of one row against `k` centroids of 384
dims, invisible next to the scan at any plausible `k`. The **fit** is not. Its sample is
`min(fit_sample_rows, k × 100,000)` (`pipeline._subtask_sample_rows`), bounded again by the
prefix's eligible rows, so raising `k` from 16 to 40 grows the subtask matrix from ~2.3 GiB to
the `fit_sample_rows` cap, inside the one whole-GPU task that is already holding the 865-wide
locality sample. It saturates there, which is why it never becomes the binding ceiling — the
semantic one stops the field first. The deeper reason it cannot bind is structural: the
subtask sample never exceeds `fit_sample_rows` and its 384 dims are under half of the fused
865, so this matrix stays below 45% of the locality one the same task already holds, at every
`k`. What bounds the value is two unrelated constraints, and **which one binds depends on the
target**:

| Ceiling | Bound | What goes wrong above it |
|---|---|---|
| Arithmetic | `k <= target / merged_tasks` | A task's budget falls below its occupied cell count, so its level-2 fill line is **zero**, its quota is `min(capacity, 0)` for every cell, and the entire budget is handed out by the residual pass. Which cells are funded is then decided by `fairness_residual_seed`, and — because capacity never enters an order — a cell holding 5,000 clips and one holding 12 have exactly the same chance. |
| Semantic | `k` well below the spellings per task, **estimated ~40** | A partition finer than the number of distinct wordings inside one task gives each spelling its own cell, so the key stops *grouping* wordings and starts *enumerating* them — which is the free-form label key this field was introduced to replace, reached by a different route. It also weakens the property that makes a level-2 merge unnecessary: two spellings of one instruction sit at cosine ~0.99 and share a cell of 16, but not necessarily a cell of 100. |

At the design scale the arithmetic ceiling is slack — 250M rows against a multi-million
target leaves room for `k` in the thousands — so **the semantic ceiling is what holds the
default at 16**. Raising the value at all is unsupported: ~40 marks where the argument
definitively breaks, not a licence to reach it, and nothing establishes a benefit above the
default. On a small corpus or at a scarce target the arithmetic ceiling collapses and becomes
the binding one:

| Corpus and target | Merged tasks | Arithmetic ceiling | Binding ceiling |
|---|---:|---:|---|
| 250M rows, target 10M | 2,738 | ~3,650 | semantic |
| 131,589 eligible rows, target 50,000 | 2,287 | ~21 | semantic |
| 10M rows, target 10,000 | ~2,287 | ~4 | **arithmetic** |

"Binding" here means *which ceiling forces `subtask_clusters` below the default of 16*: an
arithmetic ceiling comfortably above 16 leaves the semantic one in charge, even when it is
the tighter of the two in absolute terms.

The two ceilings differ in how well they are established, and the difference decides how
each may be used. The arithmetic one is exact and checkable before a run from a config value
and a log line, so it can gate a value. The **~40 is an estimate, not a measurement**:
unlike the `0.816` labels per row above, no query in this repository produced it, and none
can be run after the fact, because curation no longer reads `subtask_name` — the run cannot
see the vocabulary it replaced. Establishing it means counting distinct `subtask_name`
values per `task_name` directly on the clips table, which is worth doing before anyone
argues for a larger `k`. Until then it supports the *direction* of the ceiling — cells must
stay coarse enough to group wordings — and is not a precise cut-off.

The reported unfunded-group share is the read-out for the arithmetic ceiling, and it is the
only one, since the field is an operator's own choice rather than a corpus property. It is a
**backstop rather than a substitute** for computing the ceiling beforehand, because a `k`
modestly above the ceiling lands *under* the 20% WARNING. Take the third row above and set
`k = 5` rather than 4: level 1 spreads 10,000 over 2,287 tasks as `L = 4` with 852 tasks
drawing a fifth clip, so the 1,435 tasks left at 4 cannot reach their fifth cell. Their
level-2 fill line is zero and every clip they contribute is placed by the seed. Yet taking
each task to hold a survivor in all five cells, the corpus-wide unfunded share is
`1,435 / 11,435 ≈ 12.5%` — under the threshold, so the run says nothing. Nothing reports the semantic
ceiling at all, because the run cannot see the spelling vocabulary it has already replaced.

A row whose `embedding_text_subtask` is absent, non-finite or directionless takes the
reserved cell `NO_SUBTASK_CLUSTER = -1` rather than a NULL — the same reason
`__dedup_key` uses `NO_DEDUP_GROUP`. Not because a NULL key cannot be shuffled: Ray Data
groups NULL as its own group (verified on Ray 2.55.1), and the group count relies on that
for `curate_selection_reason`, which is NULL for exactly the survivors it counts. The
reason is that **one** key type — `fairness.Level2Key`, a `(str, int)` pair — serves the
quota's key, the selection shuffle's key, and the map `select_within_quota` looks itself up
in; a nullable cell would widen that type to `int | None` in three places that have to
agree on it. A row with no usable vector also has no position in the space to be assigned
from. That reserved cell is where the `+ 1` in every group-count ceiling in this document
comes from — and it is slack rather than reachable, for the reason immediately below.

#### The reserved level-2 cell never holds a survivor

The reserved cell and the fitted cells are **mutually exclusive across a run**, which bounds
what its arbitrary id can cost. The two configurations are exhaustive:

- *The subtask block carries weight.* `eligibility_filter` then requires
  `embedding_text_subtask IS NOT NULL`, so absence cannot reach the scan, and the same norm
  floor that sends a row to the reserved cell also fails it in the fused gate — so the row is
  already `invalid_embedding` and `survivor_group_counts` drops it. Every survivor holds a
  fitted cell.
- *It does not, or no basis was fitted.* Then **every** row takes the reserved cell, each task
  holds exactly one level-2 group, and there is no sibling for it to be ordered against.

The invariant rests on one predicate rather than on two floors agreeing.
`vectors._finite_and_directed` is the single site that compares a row norm against
`_MIN_NORM`; `vectors.unit_rows` consumes its verdict to gate the cell and
`vectors._classify` consumes it to gate the fusion, so the two cannot drift apart. A ladder
of norms straddling that floor is pinned in `tests/.../test_vectors.py`, which now pins the
routing: it fails if either surface regains a comparison of its own. The `+ 1` in the
group-count ceilings above is therefore an upper bound that no single run reaches.

Grouping on the NULL works; **reading the result back needs one extra step**, and this is
where the choice above stops being free. Ray Data describes each aggregate output partition
independently, so the blocks of one reduction can disagree about the table they belong to in
two ways, and both reach every run:

- A partition whose rows all carry a NULL `curate_selection_reason` returns that column
  typed `null`, while the partition that saw a `duplicate` returns it typed `string`. This
  hazard grows as the corpus gets *cleaner* — the fewer reasoned rows there are, the more
  certain it is that some partition observed none of them, and a corpus with a single
  duplicate reproduces it reliably.
- A partition holding no rows at all returns **no schema at all** — zero columns, not a
  zero-row copy of its siblings'. Every partition past the number of distinct groups is
  empty, and the default partition count (200) far exceeds the handful of groups either
  reduction produces, so most blocks are these.

So both driver read-backs go through `_concat_aggregate_blocks` in
[`pipeline.py`](../../../cosmos_curator/next/recipes/curation/pipeline.py), which drops the
empty partitions — they carry no row to keep and no type to learn from — and makes the rest
agree by promoting `null` only to a type a sibling actually observed. A populated block with
a different field set, or two naming different concrete types, is left to fail as real
schema drift. This is a framework boundary, not a Curate invariant: Ray also widens the
`int32` cell key to `int64` on the way out, so the aggregate's output types are Ray's to
choose and nothing downstream may assume the in-flight schema survived the reduction.

#### Two ways to have no level-2 basis

The centroids artifact writes `subtask_centroids` as `(0, TEXT_DIM)` whenever no level-2
basis was fitted, and a fitted basis always holds at least one centroid — so zero rows is
unambiguous about *whether* a fit happened and says nothing about *why*. There are two
causes and only one of them is a loss:

| cause | what it means | how the run reports it |
| - | - | - |
| the subtask block carries no weight | the documented escape for a corpus whose labels are identifiers rather than language. No level-2 clustering was requested | **INFO.** A configured state, and warning on one teaches an operator to stop reading warnings |
| the block carries weight but no sampled row had a usable subtask direction | every row takes the reserved cell, so fairness granularity silently falls back to the canonical task label alone — the whole point of the level-2 cell | **WARNING** naming what was lost and pointing at `embedding_text_subtask` on the fit sample's fragment prefix |

The distinction survives only where the weights are still in hand, which is beside the fit
(`_report_subtask_basis`). Everything downstream — the artifact, the scan's cell column,
the quota's group set — records the two identically.

Labels are canonicalized in four steps: Unicode NFC normalization, whitespace collapse, case
folding, then stripping trailing sentence punctuation. So `"Open Folder"`, `"open folder"`
and `"open folder."` are one group.

This is curation's own rule, not the embedding leg's. The embedding leg collapses whitespace
only, because its job is to make the *vector* stable; curation's job is to decide when two
labels name the same task, which is a different question.

One consequence is worth stating plainly, because it looks like an inconsistency and is
not: the four-step fold applies to the **task** label only. At level 2 there is no string
to fold — two clips share a level-2 cell when their subtask text embeddings bundle
together, which is a question about meaning rather than about spelling. A capitalization
difference perturbs the embedding slightly and lands in the same cell for the same reason a
reworded instruction does: the cell is coarse by construction, at `subtask_clusters` cells
for the whole corpus.

Case folding stops short of merging on meaning. Punctuation beyond the trailing position is
left alone, since separators inside a label can carry structure.

### Merging near-duplicate labels

Canonicalization alone is not enough, because post-repair labels are **prose**. Exact string
grouping would put `"open folder"` and `"open the folder"` in two groups, each drawing its
own quota — precisely the fragmentation that dilutes a task's representation. So after
canonicalization, curation **merges near-duplicate labels by embedding similarity**.

- **Only level 1 merges.** Level 1 merges task labels using `embedding_text_task`
  above `merge_theta_task` (default `0.95`). There is no level-2 merge and no
  `merge_theta_subtask`. The merge's job at level 2 was to fold near-duplicate
  subtask prose into one group; a k-means cell over the same embedding does that
  directly, and it does it with a bounded output rather than an input-sized one, so
  the second merge had nothing left to contribute. It was deleted rather than made
  switchable.
- **Merging is by leader label, not by transitive edges.** The highest-count label becomes a
  representative first and absorbs every label within its threshold; the next unabsorbed
  label becomes the next representative. This bounds every member to one reference point.
  Connected components over a threshold graph was **rejected**: merging would be transitive
  where similarity is not, so a chain of near-misses drifts a group arbitrarily far from its
  origin.
- **It runs on the driver, over distinct labels, never over rows.** The same input string
  always yields the same vector, so `embedding_text_task` is a function of `task_name` and
  one representative vector per distinct label suffices. Those vectors arrive from the
  dedicated label pass in the [data flow](#data-flow), which reduces each fragment to one
  row per distinct canonical label; the driver never touches a row-scale text column. State
  is `O(L)`, not `O(N)`, and `L` is now the **task** vocabulary alone: the
  repaired corpus measures **2,738** distinct tasks, which at 384 dims and float32 is
  ~4.2 MB per buffer and a few tens of megabytes at the merge's peak. The subtask
  vocabulary — 107,341 labels, ~165 MB per buffer, the term that dominated this
  paragraph and did not extrapolate to the 250M–500M envelope — is no longer an input
  to anything: level 2 is a bounded cell index. The task vocabulary is bounded by the
  annotation model rather than by the row count, so the `L² · d / 2` driver kernel is
  now bounded too. The
  resulting label-to-representative map is broadcast into the grouping that was happening
  anyway, so no new shuffle is introduced.

**Merging tasks merges their level-2 cell spaces.** If task `A` and its variant `A'`
both hold rows in subtask cell 7, that becomes **one** level-2 group, not two. This is
intended — same semantic task, same region of instruction meaning — but it is a
behavioural consequence of level-1 merging rather than something a reader should have
to discover. The cell space itself is global: cell 7 means the same region of the
subtask embedding space in every task, because one basis is fitted per run.

The two fragmentation distortions the merge removes differ in kind:

| Level | Without merging | Bound |
|---|---|---|
| Task | With capacities `[600, 300, 900]` and target 300, `L = 100` gives `[100, 100, 100]` — a task written two ways draws **200** while its single-labelled peer draws **100**. Merging restores `[900, 900] → [150, 150]`. | **Unbounded**: a task written *N* ways takes roughly *N* times its share until capacity binds. |
| Subtask | The distortion was per-wording fragmentation of a free-form label. A cell index removes it by construction: two wordings of one instruction fall in the same cell, and a task's level-2 group count is at most `subtask_clusters + 1` whatever its annotators wrote. | **Removed**, not merely bounded — there is no per-wording group left to fragment. |

**What the merge moved is reported, in clips rather than in labels.** The label counts
already reported — labels in, representatives out — measure the merge's *size*, not its
*consequence*. A merge that folds 400 labels into 380 is unremarkable if those 20 labels
held a handful of clips each and alarming if one of them held a third of the corpus, and
the label counts cannot tell the two apart. So every run also logs the number of clips
whose fairness group changed identity because their label was absorbed, and that count
reaches [`CurateResult`](#what-a-run-reports-instead-curateresult) as a scalar.

Above **10%** of eligible clips moved, the report becomes a WARNING. Like the unfunded
threshold it is a module constant with no config field, and like it a **small-corpus
signal** — a corpus whose task vocabulary is bounded by the annotation model does not move
a tenth of its clips between groups unless the threshold is absorbing unrelated labels,
which is exactly the degenerate merge the [Limitations](#limitations) name as this design's
one silent failure. It does not detect that failure — nothing on the table can — but it
gives it a loud correlate at the moment it happens, which is the mitigation the Limitations
call operational.

**Merged-group identity is not persisted.** No third column and no label-to-representative
artifact; `curate_*` stays at exactly two columns, which is what keeps the data model
statable in one sentence. Three consequences follow, recorded as a known trade:

- **`below_quota` and `unfunded` are not auditable from stored state.** Both refer to "your
  group", and which group that was is not recoverable from the row: a representative depends
  on the label set, `merge_theta_task`, *and* the clip counts that order representatives,
  while the level-2 half of the key depends on a basis the table does not carry.
- **The merge is reproducible by recompute, not recoverable from storage.** Given the same
  corpus and the same threshold it is deterministic and re-derivable. It is *not*
  re-derivable after rows are appended, because changed counts reorder representatives.
- **A degenerate merge leaves no trace.** See the Limitations: this is the one place in the
  design where a wrong config value can produce a silent, plausible, incorrect result.

### Allocation: integer max-min quotas, applied twice

Group keys are normalized to ascending order once, which is what makes each task's cells a
contiguous slice the nested pass can read without re-sorting. For a target `M`, find the
largest **integer** fill line `L` (a uniform cap for this pass) with

```text
sum over groups of min(size_g, L) <= M
```

give each group `min(size_g, L)`, then give one more to each group whose capacity **exceeds**
`L` — the groups the line cut, never the ones it left whole — until the exact target is
reached, taken in that level's *residual order*: a seeded digest, not the ascending key, for
the reason given below. Applied once across task groups, then once inside
each task across its subtask groups using that task's quota as the target. There is no
minimum-per-group setting: max-min order already gives each group one before any group gets
two.

**The fill line is the whole allocation, so reading it is how you predict a run.** It sorts
every group into one of two populations, and little else about a group matters — only its
rank in the residual order, which breaks the tie for the last row:

- **saturated** (`size_g <= L`) — kept in full, at a selection rate of 100%, and never
  eligible for the residual row since it has no capacity above the line;
- **capped** (`size_g > L`) — kept at `L` rows, whether the group holds 500 or 5,000,000,
  plus the one residual row if its rank is low enough.

That single fact explains how uniform a given run will be, and it makes uniformity a
consequence of the *target*, not a fixed property of the design:

```text
target / survivors -> 0                              target / survivors -> 1
        |                                                          |
  L is small: nearly every group is capped        L is large: nearly every group
  and receives the same L rows. The output        is saturated and passes through
  is flat and carries no trace of the             whole. The output reproduces the
  corpus composition.                             input distribution; no balancing
                                                  is left to observe.
```

Neither end is a malfunction and neither is the "right" setting — the target chooses a point
on that line. The operator-facing form of this, with the regimes tabulated and the no-op
diagnostic, is in the
[runbook](../guides/curate-runbook.md#what-the-target-does-to-the-shape-of-the-selection). Note what this does to selection *rates*: at a low target a 50-row group and a
1,000,000-row group both contribute `L`, so their retention rates differ by four orders of
magnitude even though their counts are identical. Equal counts and equal rates are different
objectives, and this design deliberately serves the first.

**The ordering deliberately ignores group size.** Capacity decides when a group saturates,
but never who is served first. When the budget cannot reach every group, somebody must go
unrepresented, and choosing by size would reinstate exactly the bias this whole mechanism
exists to remove: the largest tasks would be the ones that always get a representative. The
argument is simply the objective carried to its conclusion — if representation must not
depend on size, then the rule that *denies* representation must not either.

This is not a rare boundary case. It applies at both levels, and at level two it fires
whenever a task's quota is smaller than its occupied cell count. A task holding rows in
fifty cells with a quota of twenty funds twenty of them.

**Both levels order their residual by a seeded digest, because neither key is neutral.**
Excluding by size is one bias; excluding by *key order* is another, and each level has its
own correlation that the ascending key would otherwise fold into the selected set. One seed,
`fairness_residual_seed`, drives both and is result-defining. One rather than two, because
the tiers are already decorrelated by their *keys*: the level-2 digest is taken over the
`(task, cell)` pair, so a cell's rank differs per task without a second seed to make it
differ. A separate level-2 seed would add a knob with no independent question behind it, and
a second value to record in the config digest.

*Level 1 hashes the label.* Canonical task labels are verb-initial annotation prose — "Make
coffee", "Pick up capsule" — so their alphabet is ordered by action type, and funding an
alphabetical prefix drops whole families of instructions together and drops the same ones on
every run. On a synthetic corpus of verb-prefixed labels with the target set below the task
count, key order left whole action families with **zero** clips while a digest order did not.

*Level 2 hashes the `(task, cell)` pair.* A fitted cell id is an arbitrary position in an
unsorted basis, which is the reason it looked safe to fund by — but it is a **global** index,
so ascending cell id is the *same* order under every parent. A cell that loses its task's
last place loses it under every task at once, and a contiguous range of the subtask partition
reaches the output with no clips anywhere in the corpus. Hashing the pair rather than the
cell is what breaks that: the rank of a cell differs per task, so the loss still falls
somewhere but never on the same cell everywhere. Hashing the cell alone would rank it
identically under every parent and reproduce the defect exactly.

The reserved cell `NO_SUBTASK_CLUSTER = -1` would sort ahead of every fitted cell in key
order, which under a positional residual would have given the rows with no usable subtask
direction their task's first place. Two independent facts make that id inert. It never
competes with a fitted cell at all — the load-bearing fact, argued under
[the reserved cell](#the-reserved-level-2-cell-never-holds-a-survivor) — and the residual is
no longer taken in key order at either level, so a negative id confers nothing even in a
fixture that does seat the two side by side.

Two things this does **not** buy, both important. It does not change *how many* groups go
unfunded in any way that carries information — within one pass the fill line is a function of
the capacity multiset and the target alone, so the count is identical under any order. Nesting
leaves a residue bounded by the level-1 remainder — a parent that wins an extra row starves one
fewer cell, and parents differ in how many cells they occupy — and that remainder is not small:
on 200 one-cell tasks beside 200 five-cell tasks at a target of 1,320 the reported count moved
across a span of 14 groups over twelve seeds. A signal that drifts by tens of groups under
reseeding is even less able to detect a key-correlated allocation than a bit-identical one
would be. And it does not *guarantee* coverage: the loss becomes
uncorrelated with the key, not bounded, so a small family or a rare cell can still draw
nothing by luck. Coverage of the level-2 partition follows in practice only because the draw
is repeated once per task — with 16 cells and a per-task budget of 4, a given cell is missed
everywhere with probability `0.75 ^ tasks` — which is a consequence of decorrelation, not a
promise. Guaranteeing every family a place would need a coverage-first allocator, which this
is not.

**The unfunded population is therefore reported, unconditionally.** Every run logs the
count of level-2 groups that received no budget and that count as a **share** of the
groups holding a survivor. The share is the load-bearing half: a count is unreadable
without knowing what it is a count of, and the same twenty unfunded groups mean something
different against fifty groups than against fifty thousand. Both are `O(1)` derivations
from the quota map the driver already built, and the count is the one number in this set
that also reaches [`CurateResult`](#what-a-run-reports-instead-curateresult).

Above **20%** unfunded the report becomes a WARNING. That threshold is a module constant
rather than a config field, because it is a statement about when the allocator has stopped
meaning what level-2 fairness is for, not a preference an operator should be able to
silence. It is also, deliberately, a **small-corpus signal**: at the 250M envelope with the
measured 2,738 tasks and the default 16 cells, `G` is bounded near 46,500 against a
multi-million target, so every group is funded, unfunded is zero, and the warning is
structurally silent. It exists for the corpora where the target is comparable to the group
count — which is where the residual order actually decides who is represented.

In that regime the allocator hands out one clip per funded group, so which cells get funded
is decided by `fairness_residual_seed` and not by the corpus: the allocation is still exact,
uniform and spread across the partition, but it has stopped meaning what level-2 fairness is
for. Seeding the order fixed *where* the loss falls, never how large it is, so this share
remains the honest read-out for scarcity.

**Starvation is warned on per cause, because the two causes answer to different knobs.** A
task funded nothing at all means the target is below the task count, and only raising the
target or **lowering** `merge_theta_task` — a label joins a representative only *above* the
cosine, so a lower theta merges more — can help. `subtask_clusters` is irrelevant there,
since even one cell per task cannot be funded, so the level-1 line says so outright. Cells
starved *inside* a funded task are the read-out for `subtask_clusters`. The level-1 line is
unthresholded because the condition is binary: a task is starved only when the level-1 fill
line is zero, which happens exactly when the target is below the task count.

**A starved task suppresses the level-2 line.** Below the task count every funded task holds
a quota of exactly one, which makes the unfunded share a function of `subtask_clusters`
rather than of the corpus — so reporting it would send the reader to lower the very knob the
level-1 line has just said cannot help. Suppressing it is also what keeps one owner for the
zero-quota number: with every task funded, the zero-quota groups *are* the cells starved
inside funded tasks, so the count the driver reports is the count the 20% threshold was
tested against. Both lines are distinct from the merge-collapse warning, which fires on too
**few** groups; these fire on too many, and the three are never merged.

Six properties define the contract:

1. **Exact total.** Quotas sum to `min(target, total survivors)`.
2. **Capacity.** No group is allocated more rows than it has.
3. **Coverage, within one level.** Every non-empty group receives one before any group
   receives two; if the target is smaller than that level's group count, exactly `target` of
   its groups are funded whatever their sizes, chosen by that level's residual order — one
   seeded digest, taken over the task label at level 1 and over the `(task, cell)` pair at
   level 2. Nesting weakens it *across* the two: a task holding a single cell hands that cell
   its entire parent share, so a target equal to the level-2 group count does not promise
   every cell a clip.
4. **Uniformity.** Unsaturated group quotas differ by at most one, with surplus from a
   saturated group redistributed.
5. **No floating-point arithmetic.** The computation is integer throughout, so a quota
   cannot depend on rounding.
6. **Determinism independent of construction order.** Group boundaries are decided by the
   ascending group key and, for each level's residual, by the digest of that level's key under
   `fairness_residual_seed`; within a group the order is the configured within-group policy
   with `clip_id` and then `fragment_id` as the final tie-breaks. No random number generator
   appears anywhere in selection — the digest is a pure function of that key and the seed,
   which is why it uses blake2b rather than Python's `hash()`, salted per interpreter.

**Why uniform rather than proportional.** Proportional-above-floor allocation tracks row
counts, which is the imbalance the requirement asks us to correct. With a 1000-row task and a
100-row task at a target of 200, proportional yields roughly 182/18 while uniform yields
100/100.

**Why nested rather than flat.** Flat grouping over `(task, subtask)` pairs gives a task with
three subtasks three times the weight of a task with one. Nesting allocates across tasks
first, so subtask count cannot buy a task extra share.

### Worked example

Take a target of `M = 200` over two tasks whose row counts differ 10:1:

```text
task A : 1000 rows across 3 subtasks (600 / 300 / 100), spread over 10 locality clusters
task B :  100 rows across 2 subtasks ( 60 /  40), in 1 locality cluster
```

Four candidate rules, and what each hands task A versus task B:

| Allocation rule                                  | task A | task B | note                                        |
| ---------------------------------------------- | -----: | -----: | ----------------------------------------- |
| proportional to row count                        |    182 |     18 | 10:1, the defect the requirement corrects   |
| the same rule applied once per task              |    181 |     19 | floors only; the residual still flows by size |
| flat uniform over the 5 `(task, subtask)` pairs  |    120 |     80 | task A wins by owning 3 of the 5 pairs       |
| **nested uniform max-min (this design)**         | **100**| **100**| equal opportunity, independent of size       |

Only the nested rule equalizes the two tasks. It then repeats one level down, using each
task's own quota as the level-2 target:

```text
                       M = 200
                          │   water-fill across tasks  (level 1)
                          ▼
          ┌───────────────┴───────────────┐
          │ task A = 100  │  task B = 100  │     task B binds at its capacity of 100
          └───────┬───────┴───────┬────────┘
                  │ water-fill     │ water-fill
                  ▼  (level 2)     ▼  (level 2)
            600 / 300 / 100      60 / 40
            L = 33, remainder 1  both under capacity
            → 34 / 33 / 33       → 60 / 40
```

Every level-2 set sums to its level-1 quota, and the whole allocation sums to `M = 200`.

### Applying the quotas

Quotas are computed once, on the driver, from group *sizes* alone — bounded by the number of
distinct observed label groups, never by the row count. They are then broadcast, each group's
rows are ranked locally, rows inside the quota are marked `selected`, and the rest carry the
reason that explains why not.

Because the quota computation sees only counts, execution shape cannot influence it: no
partition count, batch size, or shuffle strategy can move a clip between selected and not.
The driver's state is `O(k) + O(G)` — centroids plus groups — and never `O(N)`. `G` is
now genuinely bounded: it counts merged `(task, subtask cell)` pairs holding a
survivor, so `G <= merged_tasks × (subtask_clusters + 1)`. At the measured 2,738
tasks and the default `k = 16` that is at most ~46,500 groups at any corpus size,
against the 107,341 and rising that the label key gave at 131,602 rows. The `+ 1` is
the reserved cell for a row with no usable subtask vector.

Rows that already carry a reason from an earlier pass — `duplicate` and `invalid_embedding` —
pass through the fairness stage untouched and are excluded from the cut.

### Which rows win inside a funded group

By default, the ones **farthest from their locality centroid**, with `clip_id` and then
`fragment_id` as tie-breaks so the outcome is reproducible.

That is a real choice with published backing rather than an arbitrary one. Distance to
centroid is the same self-supervised *prototypicality* signal the data-pruning literature
uses: rows near a centroid are typical of their neighbourhood, rows far from it are atypical
or hard. The relevant finding is that the better direction depends on how much data you are
keeping — when data is abundant and you retain a large fraction, keeping the hard examples
wins; when data is scarce and you retain a little, keeping the prototypical ones wins.
Curation retains 10-50% of a corpus in the hundreds of millions, which is squarely the
abundant regime.

Be precise about what "far" is made of, because the fused distance is weighted. A row's
distance to centroid is 0.6 instruction-atypicality plus 0.2 appearance-atypicality plus 0.2
motion-atypicality. So `farthest` prefers clips that are unusual *in those three ways* — and,
because the action block carries no task signal (see
[Why each modality is present](#why-each-modality-is-present)), its share prefers unusual
**motion**, not an unusual task. That is a legitimate diversity goal on its own terms:
atypical execution of a common instruction is exactly the kind of sample a redundant corpus
is short of. What `farthest` does **not** do is spread selection across tasks. Task-level
spread comes from the fairness quotas, which run before this ordering and decide how many
rows each group may spend; ordering only decides which rows inside an already-funded group
win. Reading `farthest` as a task-diversity mechanism would be double-counting a job the
quotas already did.

Because that trade-off is regime-dependent rather than settled, the direction is configurable
— `farthest`, `nearest`, or `neutral`, which orders by `clip_id` alone and imposes no
geometric preference at all. The mode is result-defining, so two runs that differ here are
not comparable. The default is `farthest`.

Two objections were considered and dismissed on the numbers. De-duplication also processes
farthest-first and keeps the farther row of each near-duplicate pair, but it only acts above
`1 - eps` similarity, so it is choosing between near-identical rows and barely moves the
distance distribution. And a sampled fit could in principle make "far" mean "in a region the
sample missed" — but at roughly three thousand sample rows per centroid the basis is well
estimated, and an under-modelled region is arguably the novel content worth keeping at high
retention anyway.

A distance **range or threshold filter** is deliberately not offered. The distance
distribution shifts with `k`, with the fit sample and with the corpus, so a fixed threshold
would mean something different on every run; and because filtering would happen before group
sizes are counted, it could silently empty a fairness group and change which tasks are
representable at all.

**There is no persisted retention order.** An earlier design computed a target-independent
total order so that a later target change could be answered by re-thresholding. That reuse
turned out not to be a requirement, and it was the sole justification for a persisted rank,
two group-id columns, compatibility digests, and a public reuse entry point. Selecting at one
target and re-running for another is the simpler contract, and it is the one curation
implements.

---

## Write-back and completion

### The mechanism

The verdicts are narrow rows — a key, a fragment id, a routing key and a reason — and they
are written back to the fragments they came from. **The write is total**: every fragment of
the table is written on every run, and every row of a written fragment gets a value, NULL
where the run did not claim it.

```text
verdict rows ─► groupby(__frag) ─► one task per fragment       fragments no group named
                                     │                                    │
                                     │  reopen at the pinned version      │  reopen, same
                                     │  get_fragment(__frag)              │  guards, empty
                                     │  assert clip_id unique in group    │  verdict set
                                     │  assert clip_id unique, non-null   │       │
                                     │  __dedup_key < 0 → NULL cluster    │       ▼
                                     │  project onto ALL stored rows,     │  → NULL, NULL
                                     │    NULL where unclaimed            │       │
                                     │  update_columns(left_on="clip_id") │       │
                                     ▼                                    ▼       │
                             one JSON metadata string per fragment ◄──────────────┘
                                     │
                                     ▼
              ONE Transaction( LanceOperation.Update(fragments, fields_modified) )
```

Two passes, because the verdict pass is keyed on `__frag`: a fragment holding no eligible row
produces no group, so no shuffle can route work to it. The second pass supplies those
fragments from the pinned manifest instead. It is not a rare path — the table is appended as
per-dataset slabs, so a modality absent for one whole dataset takes entire fragments out of
the eligible set together.

Splitting the write across two passes means totality is a property of how they are composed,
and the verdict pass alone would commit a perfectly valid transaction over a subset. So the
composition is **checked, not assumed**: the driver holds the two passes to their sum and
raises `CurateWriteError` if the blanking pass answered for fewer fragments than the verdict
pass left uncovered. Nothing downstream could otherwise distinguish a total write from one
whose second pass was skipped — the commit succeeds either way, and the unvisited fragments
quietly keep a previous run's verdicts.

That fragment count is necessary but not sufficient, because the uncovered set is *derived
from* the payloads that came back. A verdict payload that never arrives moves its fragment
into the uncovered set, where the blanking pass dutifully answers for it — the fragment
arithmetic balances while that fragment's rows go to NULL. So the driver also holds the rows
the two passes claimed to `eligible_rows` and raises `CurateWriteError` on **any** difference,
not merely a shortfall — a surplus means a stage duplicated rows, which would publish two
verdicts for one clip. The row count is the only axis on which either loss is visible, and
both totals are already in hand, so the check costs two integers.

The two passes therefore report different things on purpose: the verdict pass counts the rows
it *claimed*, while the blanking pass claims nothing and reports zero however many rows it
NULLed. This is load-bearing and easy to break, because both passes reach the fragment through
one shared writer that reports `group.num_rows` — the count of rows handed to it, which is
zero for a blanking call. Switching that one line to the rows `update_columns` actually wrote
would make the blanking pass report the rows it NULLed, which reads as a surplus and aborts
every run over a table holding a fragment with no eligible row. The count is deliberately of
rows claimed rather than rows matched, and the guards above are what make the two equal: a
claimed `clip_id` missing from the fragment is refused before the join, so a claim that would
match nothing cannot reach it.

Three properties make the write cheap and non-destructive to the columns Curate does not own:

- `update_columns` writes a new file holding only the two `curate_*` columns, for the
  fragment's existing rows in their existing offset order. No row is replaced, so no deletion
  vector is written and every row address survives — `num_deletions` stays 0 on every
  fragment.
- The join is **left-outer** on the ordinary persisted `clip_id`. A fragment row absent from
  the update table keeps its previous value, while one present carrying NULL is overwritten
  with NULL. The write depends on the second half, which is why it passes every stored row and
  never relies on omission.
- `fields_modified` scopes the rebinding to Curate's two field ids, so every other column —
  source fields and `embedding_*` alike — stays bound to the file it already had, byte for
  byte.

Totality is what gives NULL a single meaning: **the run that committed last did not claim
this row.** Writing only the eligible subset would be marginally cheaper and would leave a
row that was curated once and is now ineligible holding a verdict computed against a
population that is no longer the one on the table — indistinguishable, to any reader, from a
current one, and enough to make a `selected` count exceed the target the run was given. The
extra cost is bounded: both passes read each fragment's `clip_id` column, which the guards
below already require, and write metadata rather than rows.

Two structural guards are worth naming because their failure modes are silent. A verdict
group must name exactly **one** fragment: a row from another fragment would match no
`clip_id` here, so its verdict would be dropped with no error rather than written to the
wrong place. A `clip_id` must not repeat **within** the verdict group: Lance resolves a
duplicate key by taking one matching row's value, and curation legitimately assigns
*different* verdicts to identical-vector rows — the second is a duplicate of the first — so
the write would be plausible and wrong. That check is free on data already in hand.

The persisted fragment is checked separately before `update_columns`: every stored
`clip_id` must be non-null and unique within the fragment, and every verdict
`clip_id` must already be present in that fragment. A unique verdict row can still
match two persisted rows when the table already holds a duplicate key, match an
ambiguous target when a row's key is NULL, or leave stored rows unchanged when the
verdict names a key the fragment does not hold — any of these would publish one
verdict onto rows this run did not claim, or leave stale verdicts in place. A corpus-wide uniqueness scan (~17 GB of digests) is deliberately not
performed: the hazard is intra-fragment only, and a `clip_id` shared across fragments is
resolved independently and correctly by each fragment's own call.

Because the worker returns a single string column, driver state is `O(fragments)` rather than
`O(rows)`, however wide the verdict rows were.

### All-or-nothing, and why that differs from the embeddings leg

**Any fragment failure aborts the run before the commit.** The embeddings leg skips a failed
fragment and commits the rest; curation deliberately does not.

The reason is that a Curate verdict is **not a per-row computation**. `below_quota` and
`unfunded` are decided against a corpus-global quota, so committing only the fragments that
happened to succeed would publish verdicts derived from a population that was never written —
silently inconsistent, and inconsistent in a way every row still looks plausible under. An
embedding, by contrast, is a function of one row and is equally true whether or not its
neighbours landed.

So a failure propagates out of the worker and the run ends **without publishing a verdict**. A
run that aborts after some workers finished leaves their column files at the target, but they
are unreferenced by any manifest and therefore **inert** — no version can read them. An
operator seeing extra files after a failure is looking at garbage to collect, not at a partial
commit.

"Without publishing a verdict" is deliberately narrower than "with the table untouched",
because a Curate run is **two** commits on a table it is the first to curate: the preflight
widening, then the verdict write. The widening is metadata-only, costs exactly one version,
and happens only on that first run — a later run finds the columns present, adds nothing, and
commits nothing (measured: `v1 -> v2` then `v2 -> v2`). It also stamps no `kind`, so a reader
resolving a basis from the commit history finds no Curate commit and correctly reports the
table as never curated rather than as a run that claimed nothing. A retry re-uses that widened
version instead of adding another.

Folding the two into one commit is possible only for a first run and is not worth its cost:
adding and populating a column in a single transaction needs `merge_columns` + `Merge`, which
is add-only and marked an upstream Internal API, while every subsequent run must use
`update_columns` + `Update` because those APIs cannot update an existing column. That buys a
second write path — the highest-risk code in the leg — exercised once per table lifetime, to
remove a version no reader misreads.

The commit is the only completion signal. No `_SUCCESS` marker, no in-schema `complete` flag,
no second current-pointer file: the Lance manifest already is an atomic pointer, and a second
one could disagree with it.

### Run identity on the commit

**A committed version says which rules produced it.** The transaction carries four
properties and nothing else:

| Property | Value | Why it is on the commit |
|---|---|---|
| `kind` | `curator-next-curation` | Several legs write this table; without a tag a version in its history is unattributable. Every other `next` leg tags its commits the same way. |
| `schema_version` | the resolved config's generation | Covered by the digest as well, but a digest is opaque and the generation is what a reader needs before deciding whether a comparison is even meaningful. |
| `config_digest` | `sha256:…` over the config's result-defining fields **and the code-defined contract** | Two versions holding the same two columns are otherwise indistinguishable. Equal digests mean the two runs applied the same **rules**; unequal digests mean they are not comparable. It is not a build stamp — it covers the declarations that decide a verdict, not the whole code state, so two releases that differ only in, say, a log line still report one identity. |
| `centroids_fingerprint` | `sha256` hex of the centroids artifact's bytes | The basis is content-addressed and so carries no version in its name; this is the only value that says which basis a version's `curate_cluster_id` values were assigned against. It doubles as an integrity check — a reader re-hashes the bytes it fetched. |

All four are **identity**: each names the run or an object the run produced, never a row or
a measurement. `centroids_fingerprint` in particular is a reference, not provenance: it
points at an already-durable object, which is why it can be stamped at commit time at all.

**The input version is deliberately not stamped.** Lance persists the transaction's
`read_version` itself, so `read_transaction(version).read_version` already answers "which
table state was this computed against". A copy in the properties would be a second value that
could disagree with the one actually read.

**What the digest covers is a deny-list**, in
[`config.py`](../../../cosmos_curator/next/recipes/curation/config.py) as
`_NON_RESULT_DEFINING`: every config field joins the identity except the table's address, the
storage profile, and the dedup concurrency — a location, a credential selector, and a
scheduling cap, none of which changes an outcome. The direction is the point. A threshold
added to the config later is covered without anyone remembering to list it, whereas an
allow-list would omit it and two runs applying genuinely different rules would report one
identity — the exact failure a digest exists to prevent. The cost of the safe direction is a
digest that sometimes changes when it need not, which can waste a comparison but cannot
mislead one.

**The digest also covers what no config file names.** A deny-list over config fields reaches
only what an operator writes, but the verdicts equally depend on rules that live in code:

- the order of `FUSED_BLOCKS`, which fixes what each coordinate range of the fused vector
  means, published as a `[weight field, vector column, width]` triple per block — so
  repointing one block at a different embedding column moves the identity just as reordering
  the blocks or rewidthing one does;
- the `CurateReason` vocabulary written into `curate_selection_reason`;
- `TASK_COLUMN` and `TASK_VECTOR_COLUMN` (published as `task_label` and `task_vector`), the
  two columns level-1 fairness reads: the task label the groups are keyed on, and the vector
  the merge compares those labels by. Neither is named by any config field, so repointing
  either moves every fairness group while every config file stays byte-identical;
- the label-canonicalization rule itself, which decides which spellings collapse into one
  fairness group and therefore which rows compete for one quota.

A release that reorders the blocks re-clusters the whole corpus, and one that folds labels
differently regroups it, while every config file stays byte-identical — so without this half
the old and new runs would report one identity. `config.contract_fingerprint()` supplies it
under the `__contract__` key, and it is **derived from those rules rather than a hand-bumped
version number** — for the same reason the config half is a deny-list. An identity someone
has to remember to update is one that eventually is not.

Canonicalization is the one entry that is a function rather than a declaration, so it is
carried as *what the function does*: `fairness.canonicalization_contract()` runs
`canonicalize_label` over a fixed probe set, one probe per folding step, and publishes the
`[probe, folded]` pairs. Running the rule rather than naming it is what makes it impossible
to change a folding step and leave the identity behind. The pairs stay readable text rather
than a nested hash because the digest already hashes them, and an operator diffing two
archived configs can then see *which* fold moved.

A probe covers only what it actually exercises, and one step folds against a *set* rather
than a single character: the trailing marks it strips. A probe spelling one mark by hand
witnesses that mark alone, so adding or removing another leaves the identity unmoved while
labels regroup. The two punctuation probes are therefore built **from**
`_TRAILING_PUNCTUATION`, which keeps them exhaustive as the set changes. The probes are
otherwise deliberately synthetic rather than plausible task labels — nothing reads them as
data, and a reader who takes them for examples looks for a meaning they do not carry.

**The digest is not the rules.** It says *that* two versions differ, never *where*. The
canonical JSON it hashes is archived in the centroids artifact as `resolved_config`, next to
its own `config_digest`, so a reader can recompute the hash and confirm the sidecar belongs to
the version it sits beside — the archive is in a sibling directory anyone can write to, and
the recomputation is what ties the file to the commit.

**Still not persisted, and still deliberately:** no run id, no run record, no merge
statistics, no per-row provenance. The run reports its counts once, in
[`CurateResult`](#what-a-run-reports-instead-curateresult), and those are a summary of one
process rather than a fact about a row.

Preflight separately reads the embedding groups' producer identities, because the fused space
is only one metric if one producer filled each consumed group — see
[Producer identity](#producer-identity-is-part-of-the-source-contract). Those identities are
recorded in the log and the centroids artifact rather than on the commit: they describe the
*source columns*, which already store them, not the verdict.

One consequence still follows from recording no run id:

- **The commit must name the centroids artifact.** The artifact is keyed by the **hash of
  its own bytes** at `{clips_lance_uri}__curate_centroids/{fingerprint}.npz` and published
  **before** the commit, which then records that fingerprint as
  `centroids_fingerprint`. Location therefore derives from the table as it did before —
  "for version *N*, read *N*'s commit, load the object it names" — but the artifact no
  longer has to wait for a version number it can only learn by committing first. This is
  the same content-addressed pattern the action-PCA basis already uses
  ([`pca.py`](../../../cosmos_curator/next/embeddings/action/pca.py) for the store itself,
  [`action_pca.py`](../../../cosmos_curator/next/recipes/embeddings/action_pca.py) for the
  fit-or-reuse policy over it), and it is what keeps
  the commit the run's single visibility boundary: everything the commit references is
  already durable when it lands, so there is no window in which a verdict is readable while
  the basis behind it is still being written.

  A directory listing cannot substitute for the reference. Content-addressed names carry no
  version, so a reader holding only the listing cannot tell which object produced which
  version — and the version-keyed scheme this replaced could not either when an object was
  missing: "newest artifact at or below *N*" silently answered with a *different* run's
  basis, which is a wrong interpretation of a cluster id rather than an absent one.

The artifact holds the raw, **unnormalized** `(k, 865)` centroid array and everything needed
to read a coordinate or re-derive the fit: the block column names, block dimensions and
fused width, the weights, the effective `k`, the **read** version the sample was drawn from,
the sample's row count, the k-means seed, the ids of the sampled fragments, the **producer identity of
each consumed embedding group** (`producer_columns` / `producer_identities`, two arrays
parallel by position), and the run's rules as text (`resolved_config`) beside their
`config_digest` — the same digest the commit stamps, so hashing the archived text is what
confirms this artifact belongs to the version it sits beside. The centroids live in the **fused** space, so the block order and
weights are what make any coordinate mean anything, and the producer identities are what say
whose vectors those coordinates were assembled from — the same basis fitted over another
producer's embeddings of the same width is a different metric that no other archived field
distinguishes. The seed and the fragment ids are what let the fit be reproduced from the
artifact alone.
Each run writes a new artifact and nothing cleans them up, because old table versions stay
readable and must stay interpretable; at roughly 9 MB per run that accumulation is
immaterial.

It also holds the level-2 basis: `subtask_centroids` at `(subtask_k, 384)`, the observed
`subtask_k`, and `subtask_column` naming the vector column that was fitted. A run that
fitted no level-2 basis writes `subtask_centroids` at shape `(0, 384)`, which is
unambiguous rather than merely empty — a basis that *was* fitted always holds at least one
centroid, so a zero row count can only mean "not fitted". The reproducibility inputs
already recorded — the k-means seed and the sampled fragment ids — cover both fits, because
one seed and one sampled prefix feed both.

The archive records the **read** version rather than the committed one, because the read
version is the only one that exists when the basis is published, and it is also the more
accurate fact: the sample was drawn from that state. A reader cross-checks it against
`transaction.read_version` on the commit that names the fingerprint; the two describe one
run, so a disagreement means the object belongs to a different one.

The write is **not best-effort, and cannot half-succeed**. It precedes the commit, so a
failure to publish the basis ends the run with no verdict published — the same outcome as
any other pre-commit failure, needing no special exception type, no repair path, and no
guidance about whether a retry is safe.

Nothing time- or run-derived enters the archive, so one fit always serializes to one name.
That is what makes a content hash a usable name, but it does not make a **retry free**: the
GPU fit is not bitwise reproducible across runs (see the `k-means fit determinism` row under
[What must stay separate](#what-must-stay-separate)), so a retry normally publishes a *new*
object and leaves the previous one unreferenced. A run that dies between the write and the
commit leaves an unreferenced object the same way. Neither is reachable by any reader, and
both cost the same ~9 MB as a referenced one — which is why nothing collects them.

### What a run reports instead: `CurateResult`

Because nothing is persisted, the run's whole report is the object it returns —
[`CurateResult`](../../../cosmos_curator/next/recipes/curation/pipeline.py) — and it lives
only for as long as the caller holds it. The runner logs it and exits; no reader can
recover it afterwards.

Every field is a scalar or a per-level tuple, so the result cannot become a back door
around the `O(k) + O(G)` driver contract — which bounds the driver only while `G` is bounded,
per [Applying the quotas](#applying-the-quotas):

| Field | Meaning |
|---|---|
| `clips_lance_uri` | the table read and widened |
| `read_version` | the version every read was pinned to, captured after the metadata widening |
| `committed_version` | the version this run created — the only handle to this selection |
| `eligible_rows` | rows the predicate claimed; the denominator of everything |
| `written_rows` | verdict rows the write claimed; refused before the commit unless it equals `eligible_rows`, so a short write never reports success |
| `requested_k` / `effective_k` | `k` as derived from `target_mean_cluster_rows`, and the count the fit actually returned after any downward clamp |
| `fit_rows` | usable rows the k-means sample held |
| `fairness_groups` | distinct **merged** `(task, subtask cell)` groups holding at least one survivor — the `G` of the quota arithmetic |
| `unfunded_groups` | how many of those groups received no budget at this target; read as a share of `fairness_groups` |
| `target` | the resolved keep-count the quota was allocated at |
| `reason_counts` | rows per reason; sums to `eligible_rows` |
| `subtask_k` | level-2 centroids the fit returned, so the level-2 group count is reported as observed rather than as requested; `0` when no level-2 basis was fitted (see [Two ways to have no level-2 basis](#two-ways-to-have-no-level-2-basis)) |
| `merge_stats` | the task merge: labels in, representatives out, clips moved between groups, and wall-clock. One entry, not one per level — there is only one merge |
| `centroids_uri` | the basis artifact, named by the hash the commit stamps as `centroids_fingerprint` |

`fairness_groups` is worth reading precisely, because three plausible counts differ. It
is **not** the number of distinct `(task_name, subtask_name)` pairs in the corpus, and
**not** the number of `(task, cell)` pairs the basis allows: it is the number of merged
`(task, subtask cell)` groups that still hold a survivor after de-duplication, so it is
bounded above by `merged_tasks × (subtask_k + 1)` and normally well below it. So it is normally smaller than the distinct pair count,
both because the merge folds wording variants together and because a group whose every row
was a duplicate does not appear. A count far below the number of tasks the corpus actually
contains is the signal that the merge thresholds have collapsed the vocabulary — see the
[Limitations](#limitations).

`merge_stats` is the other diagnostic worth watching: `labels_out` is both the
representative count and the realized inner extent of the merge's `O(L * R)` similarity
work, so it explains the level's cost as well as its outcome, while `clips_moved` is the
same merge measured in the unit that matters to a selection. Neither it nor anything else
here is written anywhere, which is why the merge's collapse warning is emitted to the log
at the time it happens.

**What the result deliberately does not carry is any distribution.** The cluster-radius
histogram, the duplicate-score histogram and its counterfactual ladder, and the fit
sample's coverage fraction are all logged and none of them appears here. The boundary is
not squeamishness about size — a histogram is `O(bins)` — it is that `CurateResult` is the
object a caller may hold and act on, and a distribution is something an operator reads once
while calibrating. Two scalars earned a field because they answer a question about *this*
selection that no other surface answers: how much of the corpus changed fairness group, and
how much of it was left unrepresented.

---

## Rerun, rebuild, and comparing targets

Curation is a full recomputation with an idempotent, all-or-nothing commit. There is no
incremental invalidation, no compatibility fingerprint, no resumption, and no staging.

| Situation | Behaviour |
|---|---|
| First run | one metadata-only `add_columns`, compute, one `Update` |
| Unchanged rerun | recomputes and overwrites both columns; identical output given the same version, weights, thresholds and seed |
| Changed config | recomputes and overwrites; the previous run remains a previous table version, but nothing on the table records which config produced either |
| De-duplication turned off or on between runs | recomputes and overwrites, so no verdict is left stale — but the two runs are **not comparable**, because a row the earlier run called `duplicate` is a selection candidate in the later one. The verdicts are current and still mean different things |
| Narrowed eligibility between runs | the write is total, so a row that was curated and is now ineligible is blanked rather than left holding the earlier verdict. No operator action, no detector query |
| Changed embeddings | a new source version, so recompute. A row whose vector was NULL and is now filled moves from "could not be curated" to a real verdict automatically |
| Appended clips | new fragments read `curate_* = NULL`, i.e. not curated. No special case |
| Concurrent append during a run | the `Update` rebases — it names specific fragments and field ids, so it cannot conflict with an append of *new* fragments, and appended rows keep NULL. The run's own completeness check is unaffected, because it is evaluated against the eligible count at the **pinned** version, which the append cannot change. The appended rows simply read "not curated" until the next run, so an operator counting NULLs against the *committed* version sees them |
| Interrupted fit / scan / dedup / fairness | Ray retries from lineage; nothing durable was written |
| Interrupted write | orphaned column files, no manifest change, `curate_*` unchanged. No partial visibility, ever |
| Rebuild from scratch | re-run, optionally dropping the two columns first |

**Comparing two selection targets: one selection per table, compared by version.** Each run
commits a new table version, and the previous version's `curate_*` column files stay
referenced by the previous manifest — so `lance.dataset(uri, version=N-1)` still reads the
prior selection. The repository runs no version cleanup, so those versions persist by
default. Any prior selection is *also* exactly reproducible by re-running its config, since
curation is a full recomputation with a fixed seed. The caveat is the one above: mapping a
version back to its target depends on the operator's own records.

This is why a side selection table, a `__filtering_*` URI grammar, and a run-named
`selection_label` are all absent rather than preserved — the capability they existed for is
Lance-native.

**Shrinking the eligible set needs no operator care, because the write is total.** Every row
of every fragment is written on every run — NULL where the run did not claim it — so a second
run under weights that require a modality the first did not (raising `subtask` from 0 is
enough) blanks the newly ineligible rows rather than leaving them carrying the first run's
verdicts. That is what lets `curate_selection_reason IS NOT NULL` be read as "the latest run
selected, deduplicated or rejected this row" with no qualification, and it is why a consumer
counting `selected` sees one run's target rather than the union of two runs' selections.

The cost is one pass over the fragments no verdict group named. It is a second pass because
the verdict pass is keyed on `__frag`: a fragment holding no eligible row produces no group,
so no shuffle can route work to it. Not a rare path either — the table is appended as
per-dataset slabs, so a modality absent for one whole dataset takes entire fragments out of
the eligible set together. Both passes write whole fragments, and neither writes rows, so the
totality costs metadata rather than data.

---

## What must stay separate

Concepts that are easy to conflate, and the consequence of conflating them:

| Concept | Determined by | Affects the result? | If conflated |
|---|---|---|---|
| Locality cluster (`curate_cluster_id`) | fused geometry, `k`, the fit sample | yes — which pairs are compared | used as a balancing group, budget tracks row counts |
| Canonical task label | annotation text + the level-1 merge | yes — level-1 fairness | used as a dedup partition, the duplicate rule changes |
| Subtask cluster cell (level-2 key) | `embedding_text_subtask` geometry + `subtask_clusters` + the fit sample | yes — level-2 fairness | confused with the locality cluster, or read as a semantic subtask id |
| Merge threshold | `merge_theta_task` | yes — which task labels are one group | applied to level 2, which has no labels to merge |
| Fragment id (`__frag`) | physical table layout | **no** — routing only | treated as data, the write depends on layout |
| Ray partition / batch size | execution shape | **no** | results depend on how the run was scheduled |
| Dedup task concurrency | free GPUs, `dedup_concurrency` | **no** | scheduling mistaken for cluster identity |
| Whether de-duplication runs | `dedup_eps` is a float or `None` | **yes** — a skipped `duplicate` verdict becomes a selection candidate | two runs across the skip boundary compared as if the target alone changed |
| Bypass sentinel value (`-1` vs `-2`) | which cause made the vector unusable | **no** — both persist `invalid_embedding` with a NULL cluster | a routing value read as a cluster, or as a persisted discriminator |
| Reported distributions and thresholds | the radius and duplicate-score histograms, the three run metrics, the two warning constants | **no** — read-out only | a warning read as a guard, or a histogram expected on `CurateResult` |
| k-means fit determinism | single-GPU cuML + pinned `random_state` | **yes, numerical provenance** | assuming bitwise invariance across GPU or library versions hides near-tie centroid changes |
| Fit sample size | `fit_sample_rows` | **yes** | assuming centroids are a property of the corpus alone |

Ray task layout, batch sizes and dedup concurrency exist only for throughput. Two things sit
on the other side of that line and are therefore **reported** rather than claimed invariant:
the fit configuration, because GPU float reduction can still move a near-tie centroid across
library or hardware versions, and the fit sample size, because a different sample fits
different centroids. Moving any other concept across that line is an architectural regression
even when every test still passes, because it makes output depend on how the run happened to
be scheduled.

**Three geometries, not two.** The similarity geometry clusters the **fused** vector to
bound which pairs de-duplication compares. The semantics geometry reads the canonical
task label. The level-2 key is a third: a clustering of the **subtask text** block
alone. It is not the locality cluster under another name and the recorded decision that
locality clusters stay separate from fairness groups is unchanged by it — the two
clusterings are over different vectors (865-dim fused versus 384-dim subtask text), at
different `k` (derived from the corpus versus operator-fixed), for different purposes
(which pairs to compare versus which populations to fund), and neither is ever read as
the other. What the level-2 key is emphatically not is a semantic subtask
**identifier**: like `curate_cluster_id`, it is a partition index whose numbering is
whatever the fit produced, and it is not persisted.

**Distinct is not the same as statistically independent, and the overlap is measured.**
On the 131,602-row Mecka corpus the locality cluster and the level-2 cell share a
normalized mutual information of **0.50**: about **55%** of the fairness key is already
determined by knowing the locality cluster. The cause is structural rather than incidental
— the fused vector weights subtask text at 0.6, and the level-2 cell is a k-means partition
of that same embedding, so the two partitions are largely reading the same signal at
different weights and widths. Against an action-derived cell the same statistic falls to
**0.15**, which confirms the diagnosis: the overlap is the shared text block, not a
property of clustering.

The measurement has **no operational consequence**, which is why it is recorded here rather
than corrected. Locality clustering balances nothing — it decides which pairs
de-duplication compares and nothing else — and the fairness quota is built from survivor
counts over `(canonical task, subtask cell)` and never reads `curate_cluster_id` at any
point. Every remedy that would reduce the overlap changes the fused-vector weights, and
therefore the persisted centroids and every verdict in the run, in order to correct a
statistic nothing depends on. So the honest statement is the one above and not a stronger
one: the two geometries are **distinct, computed over different metrics, and not
interchangeable** — they are not statistically independent, and this document does not
claim they are.

---

## Configuring and launching a run

Curation is the `curate` **pipeline kind**, so a run is described by one YAML or JSON file and
launched by the generic pipeline CLI. `CurateConfig` carries two required envelope fields ahead
of everything else — `schema_version: 1` and `kind: curate` — and `config.resolve_config` is the
one resolver behind every surface that reads a file: `pipeline validate`, `pipeline render`, and
`run-pipeline`. One file therefore cannot mean different things to the command that checks it and
the command that runs it.

Both envelope fields are **required rather than defaulted**, which is what makes them worth
having: a default would let a file written against a later generation, or one naming a different
pipeline, be read under this generation's field meanings instead of being refused.

**A resolved config is also the run's identity.** `result_defining_digest` hashes every field
that decides an outcome, plus the code-defined contract those fields are interpreted against,
and the commit stamps that digest on the version (see
[Run identity on the commit](#run-identity-on-the-commit)), so a committed table version can
name the rules that produced it without a column carrying them. That is the argument for a
reviewable file over a flag set: several fields are result-defining, two runs differing in one
of them are not comparable, and the digest is only as meaningful as the text it hashes is
readable.

`recipes/curation/pipeline_kind.py` is the adapter. It holds no logic beyond projecting
`CurateResult` onto the CLI's output shape, and it defers every import — Lance, Ray, cuML, and
the pydantic model itself — into the callback that needs it, because the CLI imports it at
startup to render `--help`. `prepare_run` resolves the config eagerly and returns a closure that
holds the run, which is what lets the generic runtime distinguish a config fault from a run fault
by where it arose.

Operational detail — fields, defaults, calibration, and reading the outcome — is in the
[Curate runbook](../guides/curate-runbook.md).

## Execution and data movement

Ray Data owns the corpus-scale scan, the shuffle, the grouping and the vectorized transforms.
Direct Lance owns the bounded schema and version operations and the atomic commit.

The shape is Ray-as-scheduler plus Lance-as-reader: the driver enumerates fragment ids and
feeds them in as work items, and each worker opens its own fragment scanner at the pinned
version. Nothing reads the table through a single dataset-wide reader, so no stage
materializes table-scale state on one node.

**GPU work is two discrete kinds of Ray task, and there is no long-lived pool.**

1. **Fit** is a single whole-GPU task that loads the bounded sample, runs single-GPU cuML
   k-means with effective `k` clamped to the sample it fit on, and returns the basis to the
   driver. The GPU is released when the task returns.
2. **De-duplication** dispatches one whole-GPU task per non-empty cluster, each scoring its
   own cluster with a tiled cosine GEMM whose tile it derives from its own row count and the
   device it landed on. The work is embarrassingly parallel across clusters;
   `dedup_concurrency` caps how many run at once, and `None` fans out to every free GPU.

Everything between them is CPU. In particular **assignment** happens inside the scan and
holds no GPU: the centroids are broadcast and every eligible row is scored on CPU as it is
read. So does the label merge, which runs on the driver over distinct labels.

Both GPU stages name the `cuml` pixi environment explicitly through their runtime
environment, because a task whose imports live outside the driver's environment otherwise
only works when the driver happens to have been launched there. Their heavy imports are
deferred into the task body so the modules still import on a CPU-only host.

This replaces an earlier design that acquired one held GPU pool with a collective handshake
and kept it for the whole run. The task model needs no placement group, no NCCL bootstrap and
no idle-pool reasoning: a step that is not running holds no GPU, so the assignment pass leaves
every GPU free and nothing has to be released in a `finally`.

**Storage, stated honestly: staging did not go to zero, it moved.** The design has no shared
staging tree and therefore no shared-filesystem prerequisite — but the `__dedup_key` shuffle
still moves the fused population, and at the 250M envelope that is roughly 865 GB routed
through the Ray object store with **node-local spill**. On top of that, two `materialize()`
barriers each hold roughly 20 GB cluster-wide, and they are affordable precisely because the
fused vectors are dropped before the first of them: what survives is a narrow verdict row. A
deployment must size node-local scratch, not a shared filesystem.

Both barriers exist for the same reason. A Ray dataset is lazy, so reading an unmaterialized
lineage twice re-executes everything behind it — including the run's GPU stage. The first
barrier lands after de-duplication and the label merge, because the fairness pass reads
those rows twice: once to count groups and once to rank and cut them. The second lands after
the cut, because the verdict rows are read twice as well: once to check that every eligible
row carries a reason, and once to write.

Fragment geometry is logged at preflight, because curation inherits the embeddings leg's
exposure to it: at a pathological thousand-row geometry a 250M-row table is a quarter of a
million work items and as many metadata strings. That is observed rather than asserted.

Preflight validates **data contracts only**, and fails on the driver in seconds rather than
inside a write worker after the whole de-duplication pass. It checks five things: the weighted vector columns exist; `clip_id` and `task_name`
exist and are `string`; `embedding_text_task` exists regardless of weight, because the
task merge always runs; each consumed embedding group names exactly one producer in every one
of its provenance columns, as soon as any vector this run reads from that group is filled;
and at least one row satisfies the eligibility predicate.
`embedding_text_subtask` is required exactly when its block carries weight — the same
predicate the eligibility check uses — so a corpus that never ran the text leg is still
curatable on image and action alone. `subtask_name` is not required by anything. The `clip_id` type check is the load-bearing one —
`update_columns` rejects a key-type mismatch inside the write worker, which is after
everything expensive. The empty-eligible-set check is the second: a corpus whose weighted
modalities were never embedded would otherwise fit, shuffle and commit a selection of
nothing.

### Producer identity is part of the source contract

Shape is not a sufficient contract. A run concatenates the consumed blocks into one vector
and compares every row against every centroid in a single cosine space, so two producers of
the same **width** — two text models, or two action PCA bases — fuse into a geometry with no
shared meaning while every distance it yields stays finite and plausible. Nothing downstream
can catch it: the clusters, the duplicate verdicts and the quotas all come out looking
ordinary. A width check passes such a table, which is why the contract has to reach past
shape.

Preflight therefore reads the `provenance_columns` of every **consumed** group and requires
each to name at most one identity. A group is consumed when its block carries weight, plus
the text group always, because the level-1 merge reads `embedding_text_task` whatever the
weights — the same asymmetry the column contract already encodes for that column. A
de-weighted group's provenance is not read at all: a group that never enters the metric
cannot corrupt it. An **absent** provenance column on a consumed group is refused rather
than skipped: an unrecorded producer is as unusable as two, since neither can be shown to be
one, and tolerating absence would leave a trivial bypass. The remaining requirement is
stated **per column, against the vectors this run actually reads**: every provenance column
of a consumed group must name an identity as soon as any of those vectors is filled. A
column that is entirely NULL is not an error only when none of them is filled either — the
group exists but nothing filled it, which the empty-eligible-set check reports far better.

Both halves of that carry weight. **Per column**, because the action group records its
descriptor version and its PCA basis independently, so a basis refitted over descriptors
that were never re-versioned leaves the vectors attributed to the descriptors alone, and a
per-group check is satisfied by whichever column happens to be filled. **Against the
run-consumed set** rather than the group's `primary_vector`, because for text the two
differ: the fused block is `embedding_text_subtask`, while the vector that makes the group
consumed at all is `embedding_text_task`. Probing the primary vector would therefore both
miss a corpus holding task vectors of unrecorded provenance and refuse one whose subtask
vectors this run never reads. Vectors without a recorded producer are refused even when
eligibility would claim the rows, because the centroids artifact must name whose geometry
was fused.

This is the consuming end of a rule the embeddings leg already enforces at the producing end
(`validate_embedding_group` refuses to *fill* a group that carries two identities). Both are
needed because one table can be filled by two runs whose configs differed, and only a reader
of the whole column afterwards can see that.

On a table the gate accepts, the check costs one bounded distinct-value scan per consumed
provenance column — at most four — pushed down into Lance, which applies both the
distinctness and a limit of two, so the driver receives at most two values per column
however long the corpus is. Those are the only **data** reads preflight makes on such a
table, and they are unavoidable at read time: no fragment or dataset statistic Lance exposes
can prove that a column holds a single value, and these columns carry no index. A bitmap
index would answer it from index metadata alone, but building one is a producer-side action
a read-only contract check must not take.

A consumed group that recorded no identity for some column costs one further read: a
filtered count of the rows where any vector this run consumes from that group is set. That
one is **not** bounded — a fixed-size-list column carries no scalar index and no statistic
answers `IS NOT NULL`, so it reads the column's validity, linearly in rows. Measured on
pylance 9.0.0 over local NVMe it costs roughly 20 ns/row when the column is empty and
110 ns/row when it is full, which puts a 250M-row table at about 5–30 s.

It runs only when a consumed group recorded no identity for some column. That is *usually* a
table about to be refused, but not always, and the exception is worth knowing before it
surprises someone sizing preflight: `eligibility_filter` tests only the **weighted** blocks,
so it never names the task vector. A text group nobody filled, curated at zero subtask
weight, therefore reaches this count, finds nothing, and lets the run proceed — a completing
run can pay the scan once. Every *weighted* group in that state fails the empty-eligible-set
check instead, which is why the case is narrow rather than routine.

The count is kept exact rather than early-terminating because the row figure is what tells an
operator whether a stray partial backfill or the whole group is at fault.

The scan runs before the metadata widening, so a table the gate refuses gains no new version.
It reads the whole column rather than only the eligible rows, so the identities it reports are
corpus-wide; it also assumes no embeddings fill is committing concurrently, which is not a
supported mode. And it establishes two things: that the recorded identities agree, and that
no consumed group holds vectors this run reads while one of its provenance columns records
nothing. What it does not establish is the per-**row** pairing — that each filled vector
sits on a row whose provenance value is filled too stays the embeddings leg's invariant,
since a group's fields are written from one validity mask.

The identities are recorded with the run, in the preflight log line and in the centroids
artifact — not on the commit, which carries the run's own identity rather than its sources'
(see [Run identity on the commit](#run-identity-on-the-commit)). The precision differs by
modality and is worth knowing: for **action** the PCA fingerprint is content-derived, so the
check is exact — two different bases cannot share a fingerprint, and the per-column
requirement above means the fingerprint cannot be absent while action vectors are filled.
For **text** and **image**
the identity is a model id, which pins the weights but not every preprocessing choice around
them, so two runs of the same model id under different preprocessing still read as one
producer. That is a strictly weaker guarantee than the action case, and closing it would
require the embeddings leg to widen what it records.

Environment properties (node topology, free space, GPU availability) are deliberately not
validated: they fail loudly and belong to whoever chose the infrastructure. Two failure
shapes reach the caller from the driver — `FileNotFoundError` for an entirely absent table
and `ValueError` for every contract violation — plus `CurateWriteError` for a write-back
contract violation. A violation detected *inside* a Ray Data task instead arrives as a Ray
error wrapper, because Ray replaces a task's exception with its own; such a failure is
identifiable by its message, not by its type.

---

## Reference scale and evidence

Reference hardware is an allocation of 8x H100-80GB (640 GB aggregate VRAM) with 1 TB host
RAM per node. Single-node and multi-node allocations are both supported. Product target is
250M typical / 500M supported.

**`N` is the eligible row count** — rows where every weighted vector is present — not the
corpus row count. Only the text block covers every clip; image and action cover their
applicable subsets, so a corpus where some clips lack media or action data does strictly less
work than the corpus size implies.

| Quantity at 250M | Value | Evidence |
|---|---:|---|
| `k` at the 200,000-row default | 1,250 | derived |
| fit sample, raw | ~13.8 GB | 4M rows x 865 float32 |
| fit sample, at the cuML peak | ~55 GB of 80 GB on one device | source-derived ~4x; GPU probe must measure |
| fit sample, host matrix in the fit task | ~12.9 GiB | `min(4M, prefix eligible)` rows x 865 float32, one buffer, filled in place |
| single-device fit ceiling | ~6.2M rows | derived from the same ~4x peak; the value the warning names |
| fit rows per centroid | ~3,200 | derived |
| `__dedup_key` shuffle volume | ~865 GB | 865 float32 per eligible row |
| mean-cluster dedup, raw | ~692 MB | 200,000 rows x 865 float32 |
| mean-cluster dedup, at the GEMM peak | ~8.1 GB | the memory model at `m` = 200,000 and tile 4096 |
| largest cluster the tile holds at 4096 | ~1.9M rows | derived from the memory model |
| largest cluster any tile holds | 11,152,540 rows | derived; above it the stage refuses |
| each `materialize()` barrier (two of them) | ~20 GB | narrow verdict rows, no vectors |
| persisted `curate_*` columns | ~1-2 GB | one string + one int32 per row |
| end-to-end wall time | "hours" | **estimated until a reference end-to-end measurement** |

Note which limit binds. GPU memory no longer scales with the corpus, because the fit is
sampled and de-duplication is bounded by the largest single cluster. What scales is the
**shuffle**, and it spills to node-local scratch. A deployment that runs out of anything will
run out of scratch first — so that is the number to size, though curation does not check it:
an environment that cannot hold the run fails visibly, and the operator chose the
environment.

Every measurement records the environment it was taken in, because two toolchains are in
play: the shipped pixi environments resolve Ray 2.57.0, pylance 10.0.0, pyarrow 25.0.0 and
Python 3.13, while the local development environment used for `pytest` / `ruff` / `mypy` has
Ray 2.55.1, pylance 9.0.0, pyarrow 24.0.0 and Python 3.12. The write-back is proven on both
pylance versions. Two narrower gaps: **local Ray behaviour is validated on 2.55.1, not the
2.57.0 that ships**, and the total write's premise — that an explicit NULL in the update table
overwrites a stored value rather than being ignored — is measured on **9.0.0 only**. pylance
documents it neither way, so it is worth re-measuring when the shipped pin moves. Documentation labels estimates as such; only probe or benchmark output may be
called measured.

---

## Alternatives considered

| Alternative | Verdict | Reasoning |
|---|---|---|
| A separate selection table keyed by `clip_id` | rejected | Every consumer then joins to answer any question, and the table needs its own URI grammar, its own schema version, its own run naming, and its own provenance to be interpretable. Two nullable columns on the row the answer is about need none of that, and version time-travel supplies the history the side table's run naming existed for. |
| A `missing_embedding` reason | rejected | `embedding_x IS NULL` already answers it, so the value would be a mirror field. Deleting it also removed a whole scan branch and avoids ~230 GB of reads on partially-embedded rows. |
| Persisting `curate_distance_to_centroid` | rejected | ~1 GB at 250M rows for a value whose only use is re-deriving an ordering whose outcome the reason already records. |
| Persisting a duplicate similarity score | rejected | ~1 GB for a marginality diagnostic with no named consumer. It is not a durable field today either — it lived in staging that no longer exists — so keeping it would be adding, not preserving. |
| Persisting merged-group identity | rejected | A third column that would freeze a value which is not stable across appends, since changed counts reorder representatives. |
| Transaction properties recording the run's config values | rejected in favour of a digest | The properties themselves are kept — every `next` leg tags its commits and the whole config's *digest* rides along (see [Run identity on the commit](#run-identity-on-the-commit)). What is rejected is copying the field values there: a growing key set on every commit, duplicating text the centroids artifact already archives, where the digest answers "same rules or not" in one fixed-width value. |
| A separate Lance run manifest | rejected | A second table needs its own URI grammar, schema version and retention story to answer a question the commit can answer in place. Transaction properties are already per-version readable, so the manifest's only distinct capability is querying across versions without walking them — which no consumer has asked for. |
| Stamping the input version in the properties | rejected | Lance persists the transaction's `read_version` itself. A copy is a second value that can disagree with the one actually read, which is worse than no copy. |
| Skipping a failed fragment and committing the rest | rejected | Correct for the embeddings leg, where a vector is a function of one row. A Curate verdict is decided against a corpus-global quota, so a partial commit publishes verdicts derived from a population that was never written. |
| Balance by locality cluster | rejected | Budget follows how many clusters a task occupies, which tracks row counts. A task with 1000 rows over 10 clusters beats a 100-row task in 1 cluster by roughly 10:1 — the imbalance the requirement asks us to remove. |
| Proportional-above-floor allocation | rejected | The intuitive reading of "balanced", but it distributes residual by size and so preserves the dominant group's share. |
| Flat grouping over `(task, subtask)` pairs | rejected | Simpler — one allocation instead of two — but a task's share becomes proportional to its subtask count. |
| Exact-string label groups with no merge | rejected | Post-repair labels are prose, so exact grouping fragments a quota across wording variants — the failure the task text vector exists to prevent. Task fragmentation is unbounded in the number of spellings. |
| Semantic grouping by threshold graph + connected components | rejected | Merging is transitive over edges even when similarity is not, so a chain of near-misses drifts a group arbitrarily far from its origin. Leader assignment bounds every member to one reference point instead. |
| A canonical subtask **label** as the level-2 key | rejected | Its vocabulary tracks the row count — 0.816 distinct labels per row measured, with near-zero cross-shard collision — so it was simultaneously the merge's `L`, the quota's `G` and a shuffle key, and unbounded in all three. A k-means cell over the same embedding answers the same question with a ceiling that does not move with `N`. |
| A second merge threshold for level 2 | rejected | Superseded rather than tuned: with level 2 a bounded partition there are no subtask labels left to merge, so `merge_theta_subtask` was deleted. The reason it once had to be independent of the task threshold — short task names and long subtask sentences sit at different similarity floors — is now moot at level 2 and unchanged at level 1. |
| A ratio guard that fails the run on a degenerate merge | rejected | Its own thresholds would be unverifiable guesses. Accepted cost: a collapsed merge commits successfully. See the Limitations. |
| `task_embedding` as a fourth fused block | rejected | Correlated with the subtask vector — same text, same model — so it double-counts text and silently raises its effective weight. It earns its place in the fairness merge instead. |
| Text-only similarity for dedup | rejected | Cannot distinguish the same instruction recorded in a different scene or executed differently, so it over-removes. |
| Persisted target-independent retention order | rejected | Would make a later target change a re-threshold instead of a re-run. That reuse is not a requirement, and it was the sole justification for a persisted rank, two group-id columns, compatibility digests and a public reuse entry point. Revisit if target iteration becomes routine and measured re-run cost is prohibitive. |
| One k-means pass over the whole population | rejected at this scale | Strictly simpler, and correct while the corpus fit in aggregate GPU memory. At the supported scale its working set is several times the reference pool. Revisit only with a pool an order of magnitude larger. |
| A staging tree for fused vectors and cluster shards | rejected | It required a shared filesystem at the same URI from every node, a run-id isolation contract, a preserve-on-failure asymmetry, and manifest validation for stray files. One lineage with a shuffle needs none of it, at the cost of node-local spill. |
| One held GPU pool for the whole run | rejected | The three GPU steps share no cross-step state, so a held pool would sit idle through the CPU assignment pass while still costing a placement-group teardown and an NCCL bootstrap, and on a shared cluster it risks another job taking the GPUs. Discrete whole-GPU tasks release each GPU the moment its step finishes. |
| A corpus-wide `clip_id` uniqueness preflight | rejected | ~17 GB of digests to detect a condition that is harmless across fragments. The real hazard is intra-fragment and is checked for free in the write worker. |
| An oversized-cluster capacity **preflight** | rejected | The skew is a property of the data that no operator can predict, so a check before the fit would be warning about a number it cannot know. The guard that landed instead is at the allocation site, where the row count and the device are both facts: it costs two arithmetic expressions, refuses rather than warns, and needs no prediction. |
| Splitting an oversized cluster instead of refusing | rejected | Retention is the maximum similarity over all strictly earlier rows *within* one cluster, so a split silently changes verdicts — it converts an honest failure into a wrong answer. |
| A config field for the GEMM tile | rejected | It is a memory knob, not a policy choice: the only correct value is a function of the device and the cluster, both of which the stage can read. An operator-set tile could only be wrong. |
| Validating the environment before the run | rejected | Node topology, scratch space and GPU availability all fail loudly and belong to whoever chose the infrastructure. Input **contracts** are validated, because those fail silently. |
| Semantic clustering as the dedup partition | rejected | Would change which pairs are compared and therefore the duplicate decision; the two geometries answer different questions. |
| Decorrelating the fit sample (strided or random fragments instead of a manifest prefix) | rejected, on narrower grounds than before | The prefix is deterministic and needs **no data scan** — preflight already counts eligible rows per fragment — so it reproduces across re-runs without adding a seed to the identity surface. A decorrelated sampler must either scan or give that up. That ground is consumer-owned code and cannot be invalidated from outside, which is why the rejection now rests on it alone. Two earlier arguments are **withdrawn**: the producer pins `max_rows_per_file` to each publish batch, so a 250M-row corpus holds ~31,250 fragments and "randomizing over a set of size one is the identity function" no longer applies to anything; and the **6%** gap-closure and 2%-budget figures come from a synthetic study that disclaims being quoted as production. What a decorrelated sampler *would* buy is dataset coverage the prefix cannot reach — see [Why the fit runs on a sample](#why-the-fit-runs-on-a-sample). |
| Reducing the locality-cluster / level-2-cell overlap | documentation only | The overlap is real and measured (normalized mutual information 0.50; ~55% of the fairness key fixed by the locality cluster), and its cause is structural: the fused vector weights subtask text at 0.6 while the level-2 cell partitions that same embedding. But it has no consumer. Locality clustering balances nothing, and the quota is built from survivor counts over `(canonical task, subtask cell)` and never reads `curate_cluster_id`. Every candidate remedy changes the fused-vector weights, hence the persisted centroids and every verdict, to correct a statistic nothing depends on. Recorded under [Three geometries, not two](#what-must-stay-separate) instead, with the wording constrained: the geometries are distinct and not interchangeable, **not** statistically independent. |
| Deriving the level-2 cell count per task automatically | deferred | Blocked by an ordering conflict rather than by taste: the cell count is consumed by the **fit**, which runs before the scan, while a fractional target is resolved only **after** de-duplication tells the run how many survivors there are. A fully automatic value therefore cannot depend on a fractional target without either a second pass or an estimate of the survivor count. Meanwhile the unconditional unfunded metric already tells an operator that their chosen value is too high for their target, which is the actionable half of what auto-derivation would have provided. |
| Reporting cluster-size skew | deferred | Worth recording *why* it is the right signal if it is built. The same measurement that rejected the sampler change established that **duplicate recall cannot rank a centroid basis**: within-cluster de-duplication is exhaustive, so a coarser partition compares more pairs and therefore finds *more* duplicates — the metric rewards the worse basis. What can rank a basis is cluster-size skew and within-cluster spread, and spread is already reported by the radius histogram. Skew is the missing half, not a second opinion on a question the radius already answers. |
| Funding the largest groups when the budget cannot reach every group | rejected | The exact bias uniform fairness exists to remove: the biggest tasks would always be the ones represented. |
| Largest-remainder (Hamilton) allocation for the scarce-budget residual | rejected | It reverses the documented invariant that capacity is consulted only for saturation and never as an order key, and it contradicts the coverage property that at a target below the group count the funded groups are chosen by the level's residual order **whatever their sizes** — a seeded digest that is deliberately independent of the corpus at both levels. The decisive objection is arithmetic rather than philosophical: under integer max-min a group's quota is `min(capacity, L)`, an integer, so there is **no fractional remainder to sort by**. "Descending remainder" can only mean descending unmet demand, which is descending capacity. In the regime the proposal targets the fill line is 0, so every quota is 0 and every remainder is the group's full capacity — making "largest remainder" identical to "fund the largest groups", the row directly above. The observation that motivated it (a task of 8,541 clips receiving 3 slots while a 1-clip task receives 1) is uniform fairness working as specified, not a defect. The residual complaint that survives — key order is arbitrary, so the funded set skews toward labels that sort early — is real, and was **wrongly** answered by reporting the unfunded share. Reporting it cannot work: within a pass the count is identical under any residual order, so it is bit-identical in the skewed case and the balanced one. That complaint is now answered at **both** levels by the seeded digest above — over the task label at level 1 and over the `(task, cell)` pair at level 2 — which removes the correlation without re-introducing size. |
| Funding the level-2 residual by ascending cell id | superseded | Held on the argument that a fitted cell id is an arbitrary index with no correlation to fold. Arbitrary it is; **neutral** it is not, because the ids are *global*, so ascending cell id is the same order under every parent and a cell that loses its task's last place loses it under all of them at once. On a synthetic corpus where every task occupies every cell at a per-task budget well below the cell count, ascending order left **most of the partition with zero clips corpus-wide** while a digest of the `(task, cell)` pair covered every cell. Hashing the cell alone would not fix it — the rank has to differ per task, which is why the pair is the key. |
| Reporting the unfunded share as the *sole* answer to key-order skew | superseded | Held until the effect was measured. Within one pass the unfunded count is a function of the capacity multiset and the target alone, so every residual order starves the same number of groups — on a synthetic verb-prefixed corpus, key order and a digest order reported a **bit-identical** unfunded count while delivering different numbers of action families. (Nested, the count moves when a swapped parent occupies a different number of cells — measured at a span of 14 groups over twelve seeds — which only widens the gap between what the metric measures and what the skew was.) A metric that cannot distinguish the two outcomes cannot be the mitigation for one of them. The share is still reported — it is the right read-out for *scarcity* — but the skew is fixed in the comparator instead. |
| A per-block gate warning ("these weights make duplicates impossible") | rejected | Arithmetically impossible to trigger. Under the fusion identity a duplicate needs `sum of w_m * (1 - cos_m) < eps`, so block `m` alone must satisfy `cos_m > 1 - eps/w_m`; with `eps > 0` and `w_m > 0` that bound is always strictly below 1, so **no** weight and eps combination makes duplicates unreachable by construction. There is nothing to warn on. The formula is genuinely useful as an explanatory aid, so it lives in the [runbook](../guides/curate-runbook.md) where an operator calibrating a threshold will read it. |
| A distance range or threshold filter before selection | rejected | The distance distribution shifts with `k`, the fit sample and the corpus, so a fixed threshold means something different every run; and filtering before group sizes are counted can silently empty a fairness group. |

---

## Limitations

Stated plainly, because each one is a real property of the design rather than an oversight.

- **A degenerate TASK label merge commits successfully and leaves no trace.** This applies to
  the task merge alone, which is the only merge there is; level 2 partitions an embedding and
  has no threshold to miscalibrate. `merge_theta_task`
  defaults to 0.95, and the only similarity measurement available puts *unrelated* short
  strings around 0.90 — so 0.95 sits close to the background floor. Leader assignment
  prevents chaining but not **absorption**: the highest-count label absorbs everything within
  its threshold, and the realistic degenerate outcome is one dominant representative
  swallowing most labels. Fairness then flattens to "take the first N by ordering" while
  every row still carries a plausible `curate_selection_reason`. Because merged-group
  identity is not persisted, there is nothing on the table to reveal it — the commit's digest
  names the threshold that was used but cannot say the merge collapsed under it. This is the
  one place in the design where a wrong config value produces a silent, plausible, incorrect
  result with no automated backstop, and it is a deliberate choice over a guard whose own
  thresholds would be unverifiable guesses. The mitigation is operational: treat the
  threshold as calibration on a new corpus.
- **A committed selection names its rules but does not carry their values.** The commit
  stamps the config's digest, so two versions can be told apart and a version can be matched
  to a config file. Reading *what* the rules were means resolving the digest against the
  archived `resolved_config` in the centroids artifact, or against the operator's own file.
- **The identity covers the rules, not the code that applied them.** Equal digests mean two
  runs were configured identically, which is not the same as saying they would produce the
  same verdicts: a change to this leg that alters selection without changing the config
  surface leaves the digest untouched. `schema_version` is inside the digest and moves when
  the surface changes, which catches the largest class, but it is not a build identifier. A
  separate behavior version was considered and declined — it would need a discipline to keep
  honest that nothing in the repository enforces, and a stale one is worse than none. Runs
  spanning a change to the leg should pin the code version externally.
- **`below_quota` and `unfunded` are not auditable from the row.** Both refer to a fairness
  group whose identity is not persisted and is not stable across appends.
- **Comparing verdicts across versions is still the operator's own work.** Within one table
  version there is no staleness — the write is total, so every non-NULL value is the latest
  run's — and the digest makes "were these two versions selecting for the same thing" a
  mechanical check. What the digest cannot do is *merge* two selections or say which is
  better. See [Rerun, rebuild, and comparing targets](#rerun-rebuild-and-comparing-targets).
- **Compound annotations are distinct groups.** "open book" and "wipe book, open book" are
  separate labels, so a two-action sequence competes for share as its own group. Separating
  instructions by step count was considered and is not implemented: parsing steps from
  punctuation is a heuristic over annotation text that curation does not control, and its
  benefit is unmeasured.
- **The number of fairness groups is bounded by the annotation vocabulary at level 1 only.**
  `G <= merged_tasks × (subtask_clusters + 1)`, so level 2 contributes a fixed factor
  whatever the annotators wrote. What is still not enforced is a limit on distinct **task**
  labels: that vocabulary is a property of the annotation model rather than of the row count,
  and a corpus with pathologically inconsistent task labels can still exhaust driver memory
  or the quota broadcast. No limit is enforced on it.
- **An oversized cluster fails the run, and the failure cannot be predicted.** De-duplication
  loads one whole cluster at a time onto one card. `target_mean_cluster_rows` controls the
  **mean** cluster size, and k-means imposes no size constraint, so the largest cluster
  depends on how the corpus happens to cluster — a property of the data that cannot be known
  before the fit. The first run on a new corpus is effectively a calibration run. What the
  derived tile and the refusal change is the *failure mode*, not the exposure: a cluster
  between the cap's reach and the ceiling now runs on a smaller tile instead of dying, and
  one past the ceiling refuses with an actionable message instead of an out-of-memory error.
  Past the ceiling the remedy is still to lower `target_mean_cluster_rows` and re-run, and at
  the supported scale that re-run costs hours.
- **`k == 1` on a small corpus makes `curate_cluster_id` carry no information.** The
  de-duplication result is strictly better, but the column is constant and the centroids
  artifact holds one centroid. The run warns — and warns harder when the single cluster
  would also exceed the per-group device ceiling, since at `k == 1` there is no second
  cluster to spread the work over.
- **The centroid basis is fit on a fraction of the corpus.** A very small and very tight
  cluster may not receive its own centroid and will be absorbed by a neighbour. Locality
  clustering tolerates this because a cluster is only a computational partition — it would
  not be acceptable if clusters carried meaning.
- **`curate_cluster_id` is not stable across runs.** Adding rows changes the sample, moves the
  centroids, and changes group sizes. Each run pins one Lance version, so it is internally
  reproducible with respect to that version; a later run intentionally sees newer versions
  and may produce different ids.
- **A different target means a full re-run.** Nothing is persisted that would let a new target
  be applied to an existing set of `curate_*` columns.
- **Selection does not guarantee coverage of the fused space.** Because balancing groups are
  semantic, a fairness group's selected rows can concentrate in a few locality clusters.
  De-duplication has already removed near-duplicates and the default within-group order
  prefers rows far from their centroid, which mitigates but does not guarantee spread.
- **The shuffle spills, and node-local scratch is the scaling limit.** A deployment must size
  it accordingly, and curation does not check that it is large enough.
- **GPU requirement.** Clustering and de-duplication require a CUDA GPU with the cuML runtime
  available. There is no CPU fallback.

---

## When to revisit

| Signal | Revisit |
|---|---|
| A selection looks flattened and the reported group count is implausibly small | the merge thresholds, and whether the merge needs a persisted trace after all |
| Reported group count is far above the number of tasks the corpus actually contains | the merge thresholds in the other direction, or normalizing annotations upstream |
| Reported per-task selected counts diverge from each other where capacity does not bind | the allocation implementation — under uniform fairness this ratio should be near 1 |
| Retention falls well below 10%, or the corpus becomes small | the within-group ordering default: the prototypicality result reverses in the scarce-data regime, favouring `nearest` |
| A measured comparison shows one ordering mode produces better training results | making that mode the default, which is a one-line change since nothing is persisted |
| Reported group count grows large enough to strain the driver or the quota broadcast | an explicit cardinality limit, or computing quotas distributively instead of broadcasting them |
| Evidence that step count changes selection quality, or upstream step metadata appears | separating compound instructions before grouping |
| Selected clips concentrate in few locality clusters within a fairness group | adding locality spread to the within-group order |
| Operators repeatedly need the config's *values* from the table rather than its digest | widening the commit's properties or archiving the config beside the table under a version key, weighed against the audit surface a growing key set re-introduces |
| Duplicate count, rate, score histogram and counterfactual ladder all prove insufficient to choose `dedup_eps` | an offline inspection tool over a re-run, which is the only surface that can show *which* pairs were marginal |
| A run at the 250M envelope reports a duplicate-score distribution whose upper tail sits well away from `1 - eps` at the shipped default | the shipped `dedup_eps` default itself. The default has only ever been calibrated against corpora orders of magnitude smaller, and a large run's histogram is the intended trigger for changing it |
| The per-fragment content-homogeneity measurement shows a corpus whose fragments are segregated by dataset or task | the fit sampler, which is deferred *pending exactly that measurement* |
| Unfunded share is persistently high at a target the operator cannot raise | the level-2 cell count first, and only then whether nested uniform max-min is still the right allocator for that corpus's shape |
| Measured full-run time is prohibitive | resume, or persisting the fused population across runs |
| A GPU allocation an order of magnitude larger becomes available | fitting the centroid basis on the whole population again, which removes the shuffle |
| Real runs hit the oversized-cluster refusal | lowering `target_mean_cluster_rows`, the fit sample's coverage, and whether an approximate within-cluster neighbour search would lift the ceiling |
| A labelled duplicate ground-truth set becomes available | the `0.6 / 0.2 / 0.2` weights, which have never been ranked against an alternative |

---

## Related documents

- [Curator Next Embeddings](curator-next-embeddings.md) — how the `embedding_*` columns get
  onto the clips table, the per-modality scale contract, and why the text leg stores two
  vectors.
- [Curate runbook](../guides/curate-runbook.md) — how to launch a run and read its output.
- [Ray Data Design](ray-data.md) — the execution layer.
- [Cosmos Curator Next](curator-next.md) — product boundary and toolkit direction.
