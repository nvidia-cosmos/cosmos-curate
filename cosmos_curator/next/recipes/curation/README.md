# `recipes/curation/` - the Curate leg (Curator Next)

Shrink a large, redundant clip corpus into a **compact, de-duplicated, and
task-balanced** selection. Curate runs **after** embedding: it reads the wide
clips table, fuses three embedding blocks into one distance, clusters,
optionally drops near-duplicates, balances the survivors across fairness
groups, and writes the verdict back onto the rows it read.

**Wide table in, wide table out.** There is no side table, no staging tree, and
no report sidecar.

```text
clips.lance @ v
  clip_id | task_name | subtask_name
          | embedding_text_task | embedding_text_subtask
          | embedding_image     | embedding_action
        |
        v   fit -> gather -> scan/assign -> [dedup] -> fairness -> write-back
        |        [dedup] runs only when dedup_eps is set
        |
clips.lance @ v+1
  ... + curate_selection_reason (string, nullable)
      + curate_cluster_id       (int32,  nullable)
```

Full rationale, semantics, scale numbers, and rejected alternatives live in the
design doc:
**[docs/curator/design/curator-next-curation.md](../../../../docs/curator/design/curator-next-curation.md)**.

## What one run does

Six passes over one table, then one commit. Two of the six are **bounded** rather
than row-scale: the fit reads only a fragment prefix, and the gather reads every
fragment but projects two columns and emits `O(distinct tasks)` rows. Six is
therefore the number to budget, not four.

```text
preflight   data contracts only: the fused columns exist, clip_id type matches
   |        what update_columns will demand, and each consumed embedding group
   |        records EXACTLY ONE producer in every one of its provenance columns,
   |        as soon as any vector this run reads from it is filled - two
   |        same-width producers would fuse into a cosine space with no shared
   |        meaning, and none at all leaves the fused geometry unattributed.
   |        Fails on the driver in seconds.
   v
fit         bounded fragment prefix -> two cuML k-means in ONE task      (GPU)
   |          locality  fused vectors, k = ceil(eligible_rows /
   |                    target_mean_cluster_rows)
   |          level 2   subtask text vectors, k = subtask_clusters
   |        runs on every run, including when dedup is skipped: the
   |        locality basis is what curate_cluster_id and the default
   |        within-group ordering are read from
   v
gather      one Ray branch per fragment id: a bounded SECOND read, projecting
   |        task_name and embedding_text_task only and reducing each fragment
   |        to one row per distinct canonical TASK label, so the reduction is
   |        O(distinct tasks) and never O(rows)
   |        the driver then merges near-duplicate TASK labels over that
   |        vocabulary and broadcasts the label-to-representative map; the
   |        map is APPLIED after dedup, but it is built here, before the scan
   v
scan        one Ray branch per fragment id; scanner pushes down
   |        "every weighted vector IS NOT NULL"; canonicalize the task
   |        label, fuse, assign nearest locality centroid, assign the
   |        subtask cell
   |          eligible     -> __dedup_key = cluster id, reason NULL
   |          non-finite   -> __dedup_key = -1, 'invalid_embedding'
   |          zero-norm    -> __dedup_key = -2, 'invalid_embedding'
   v
dedup       CONDITIONAL: only when dedup_eps is not None          (GPU)
   |        groupby(__dedup_key), one whole-GPU task per cluster
   |        negative key passes through untouched (no GPU math)
   |        the fused vector is dropped here; a distance and a
   |        transient similarity score survive
   |
   |        when dedup_eps is None the stage is replaced by an
   |        explicit narrow projection that drops the fused vector,
   |        so the schema past this point is identical either way
   v
            materialize()   <- barrier 1 of 2: fairness reads twice
   v
fairness    apply the label-merge map built by gather, count survivors per
   |        merged (task, subtask cell), water-fill quotas on the driver,
   |        rank in group
   v
            materialize()   <- barrier 2 of 2: the verdicts are counted for
   v                           the pre-commit gate and then written back
write-back  groupby(__frag) -> update_columns(left_on="clip_id") per fragment
            -> ONE LanceOperation.Update
```

The commit is the **sole completion signal** and it is **all-or-nothing**: any
fragment failure aborts before the commit, so no verdict is published. This
differs from the embeddings leg, which skips a failed fragment and commits the
rest - a Curate verdict is decided against a corpus-global quota, so a partial
commit would publish verdicts derived from a population that was never written.

A run against a table it is the FIRST to curate makes two commits, not one: the
preflight schema widening, then this one. The widening is metadata-only, stamps
no `kind`, and happens only once per table, so a failure between the two leaves a
version whose `curate_*` columns are all NULL and which no reader counts as a
curate run.

The commit stamps its **identity** and nothing more: `kind`, the config's
`schema_version`, a `config_digest` hashing every result-defining config field
together with the code-defined contract (fused block order and widths, reason
vocabulary) those fields are interpreted against,
and the `centroids_fingerprint` naming the basis the cluster ids were assigned
against. That is what lets two versions holding the same two columns be told
apart. No run id and no per-row provenance is written.

The centroids artifact is keyed by the **hash of its own bytes** and published
**before** the commit: `{clips_lance_uri}__curate_centroids/{sha256}.npz`. The
commit is what makes it locatable, by recording that hash, which is also how a
reader verifies the bytes it fetched. Publishing first is what keeps that commit
the only boundary at which a verdict becomes visible - nothing it references is
still being written when it lands. It archives the config as text next to the same digest,
so hashing the text confirms the artifact belongs to the version beside it.

## Run it

Curate is the `curate` pipeline kind, so the launch surface is the generic
pipeline CLI over a config file - there is no recipe-specific runner:

```bash
cosmos-curator pipeline template curate > curate.yaml
cosmos-curator pipeline validate curate.yaml
pixi run --as-is -e cuml run-pipeline curate.yaml
```

`run-pipeline` is a Pixi task rather than an installed script, hence the prefix;
`cuml` is the environment holding the GPU stack the fit and dedup stages need.
See the [runbook](../../../../docs/curator/guides/curate-runbook.md) for the
multi-node form.

The Python API is one call. Every symbol is imported from the module that owns
it: the package `__init__.py` re-exports nothing, because naming a `config` or
`pipeline` symbol there would put pydantic - or Lance and Ray - on the import
path of every module in the package, including the `pipeline_kind` adapter the
CLI imports at startup.

```python
from cosmos_curator.next.recipes.curation.config import CurateConfig, SelectionTarget
from cosmos_curator.next.recipes.curation.pipeline import run_curate

config = CurateConfig(
    schema_version=1,
    kind="curate",
    clips_lance_uri="s3://bucket/run/clips.lance",  # read AND written
    storage_profile="default",
    target=SelectionTarget(target_count=50_000),
)

result = run_curate(config)
```

`run_curate` returns a `CurateResult` and nothing is persisted alongside the
columns, so that object is the run's whole report. Every field is a scalar or a
small tuple, which is what keeps it from becoming a back door around the
`O(k) + O(G)` driver contract. Two fields answer questions no other surface
answers: `unfunded_groups`, how many fairness groups got no budget at this
target (read as a share of `fairness_groups`), and `merge_stats.clips_moved`,
how many clips changed fairness group because their task label was absorbed by a
representative. The label counts alone cannot show that - folding twenty labels
is unremarkable unless one of them held a third of the corpus.

No **distribution** reaches the result. The cluster-radius histogram, the
duplicate-score histogram and its counterfactual ladder over candidate `eps`
values, and the fit sample's fraction of eligible rows are all log-only, on the
same contract: `O(bins)` on the driver, never persisted, never a field. Capture
the log if you intend to calibrate from them.

For the operational path - single-node and managed Ray-on-Slurm, calibrating a
new corpus, and reading the outcome - see the
[Curate Runbook](../../../../docs/curator/guides/curate-runbook.md).

### Configuration

`CurateConfig` is **strict, frozen, and `extra="forbid"`**, so a config still
naming a removed key fails at parse time with that key named. It is one **flat**
model plus two sub-models, and a sub-model exists if and only if it carries a
cross-field invariant. See the docstrings in [`config.py`](config.py).

`schema_version: 1` and `kind: curate` are required on every config. They are
required rather than defaulted so a file written for a later generation, or one
naming another pipeline, is rejected instead of being read under this
generation's field meanings.

| Field | Owns | Default |
|---|---|---|
| `clips_lance_uri` | the table to read and widen | required |
| `storage_profile` | named storage credentials / endpoint | `default` |
| `weights` | per-block fusion weights; must sum to 1 | `0.6` subtask / `0.2` image / `0.2` action |
| `target` | keep-count or keep-fraction of the **survivor** population | keep every survivor |
| `within_group_order` | which survivors win inside a funded group | `farthest` |
| `target_mean_cluster_rows` | the sole cluster-count control | `200_000` |
| `fit_sample_rows` | row budget for the single-GPU fit | `4_000_000` |
| `kmeans_random_state` | fit seed for both centroid bases | `42` |
| `fairness_residual_seed` | which groups win the remainder, at both fairness levels | `0` |
| `dedup_eps` | duplicate above `1 - eps` similarity, or `None` to skip the stage | `0.01` |
| `dedup_concurrency` | cap on concurrent whole-GPU dedup tasks | `None` (every free GPU) |
| `merge_theta_task` | similarity above which two task labels are one group | `0.95` |
| `subtask_clusters` | level-2 fairness cells: k-means clusters over the subtask text embedding | `16` |

`dedup_eps` is `float | None`, and the two branches mean different things.
A float must be strictly positive, because the retention test is a strict
`> 1 - eps` and a byte-identical pair scores exactly `1.0`, so `eps = 0` would
drop nothing while still paying for every GPU comparison. It must in fact clear
`2**-24`, one float32 step below `1.0`, because the comparison happens in float32
and an `eps` far under that step rounds `1 - eps` back to exactly `1.0f` — inert
for the same reason `0` is, just less visibly. The floor is the step rather than
the exact round-off boundary an octave lower, so every value it turns away is
within an ulp of inert. And it is at most `1.0`: the
cosine gap can reach `2`, but a larger `eps` would mark almost every pair a
duplicate, so the field caps at `1` by policy. `None` skips the stage
outright. Skipping **changes verdicts**: a row that would have carried
`duplicate` becomes a selection candidate, competes for budget, and changes every
fairness group's survivor count.

`subtask_cluster_k` was **renamed** to `subtask_clusters` with no alias. Because
the model is `extra="forbid"`, a config still naming the old key fails at parse
time with that key named rather than silently taking the new field's default.

There is **one** merge threshold, and it applies to task labels only. Level 2 is
not a label: a row's level-2 group is the index of the `subtask_clusters`-way
k-means cell its `embedding_text_subtask` falls in, so the level-2 group count is
bounded by `merged tasks x (subtask_clusters + 1)` at any corpus size.
`subtask_name` is not read by curation at all - its vocabulary tracks the row
count (0.816 distinct labels per row measured), which is exactly the unbounded key
a cell index replaces.

Two ceilings bound `subtask_clusters`, and compute is neither - the fit sample is
`min(fit_sample_rows, k * 100_000)` so its cost does grow with `k`, but the
semantic ceiling stops the field before it could ever bind. The two are
`k <= SelectionTarget.target_count / merged tasks` — knowable before a run only
when `target_count` is set — above which the seed rather than the corpus picks a
task's funded cells, and `k` well below the subtask spellings inside one task,
estimated at ~40, above which the partition enumerates wordings instead of
grouping them. With `target_fraction` or neither target form set, the keep-count
is not fixed until after de-duplication: `SelectionTarget.resolve` divides
survivors, not the corpus, so that arithmetic ceiling is a post-run check
against the logged survivor count, not a pre-run quotient. Compute the first
ceiling only for an absolute `target_count` — and read it as an upper bound
rather than an exact figure: `resolve` clamps `target_count` to the survivor
count, so a run that de-duplicates heavily has a smaller effective target and
therefore a lower true ceiling than the quotient predicts. The degeneracy
warning is a backstop with a silent band, and the second bound is an estimate
the run cannot check, since curation never reads `subtask_name`. **The only
supported direction is down**, when a scarce absolute `target_count` pushes the
arithmetic ceiling below the default of 16. See
[the runbook](../../../../docs/curator/guides/curate-runbook.md#choosing-subtask_clusters-before-the-run).

Changing any fusion weight re-clusters the whole corpus, so the weights are an
**interface**, not a tuning knob. They are also chosen together with
`dedup_eps`, because a duplicate needs the weighted sum of per-block distances
to fall below `eps`.

## Output: two columns on `clips.lance`

| Column | Meaning |
|---|---|
| `curate_selection_reason` | `selected` / `duplicate` / `below_quota` / `unfunded` / `invalid_embedding`; NULL when the run did not claim the row |
| `curate_cluster_id` | the locality partition the row was de-duplicated within; NULL when the run did not claim the row, and NULL for a claimed row that bypassed clustering as `invalid_embedding` |

Both are nullable, which is what makes the widening a metadata-only
`add_columns`. They are atomic siblings - the table carries both or neither.

There is deliberately no `missing_embedding` reason: a row whose required vector
is NULL is never claimed, and `embedding_action IS NULL` already answers "why is
this row uncurated", so a stored value would be a mirror field.
`invalid_embedding` earns a value for the opposite reason - discovering it means
reading and testing the vector, which no predicate can do.

`curate_cluster_id` is a **computational locality partition, never a semantic
category**. Two rows sharing it are near each other in the fused space and
nothing more.

## Environment and re-run safety

- **GPU.** The fit and the per-cluster dedup are the only GPU steps; both name
  the `cuml` pixi environment through their runtime environment. Everything else
  is CPU Ray Data. A GPU is still required at `dedup_eps = None`, because the
  fit runs on every run.
- **No shared staging.** There is no `staging_root` and no shared-filesystem
  prerequisite. The `__dedup_key` shuffle spills to **node-local** Ray object
  store scratch, so that is what a deployment sizes. That shuffle is the dedup
  stage's own grouping, so a skipped stage does not pay it.
- **Re-run.** Curate is a full recomputation with an idempotent commit.
  Re-running overwrites both columns; each run commits a new table version, and
  a prior selection stays readable through Lance version time-travel.
- **The write is total.** Every row of every fragment is written on every run,
  NULL where the run did not claim it, so a NULL means exactly "the run that
  committed last did not claim this row". A re-run under narrower weights blanks
  the rows it no longer curates rather than leaving them holding a verdict
  decided against a population that is no longer on the table. No pre-run drop,
  no detector query.

## Layout

The import direction is the contract: `columns`, `vectors`,
`dedup` and `fairness` must import cleanly with `ray`, `lance`, `cuml` and
`cupy` all absent, so only `pipeline` may touch a driver dependency.
`pipeline_kind` sits outside that ordering: it is imported at CLI startup, so
every import it needs lives inside the callback that needs it.

```text
recipes/curation/
  __init__.py       # docstring only; nothing is re-exported
  columns.py        # the column contract: CurateReason, the two persisted names
                    #   and schemas, the four transient routing names,
                    #   FUSED_BLOCKS, the eligibility predicate
  config.py         # CurateConfig + the two invariant-carrying sub-models,
                    #   and resolve_config, which every config surface shares
  vectors.py        # pure: eligibility classification, fused vectors, assignment
  dedup.py          # the pure retention kernel plus the one GPU UDF
  fairness.py       # pure: task canonicalization, task label merge, water-fill,
                    #   top-k
  pipeline.py       # Lance + Ray: preflight, schema widening, the fit task,
                    #   the scan, the dedup/fairness wiring, the write-back commit
  pipeline_kind.py  # lazy CLI adapter: CURATE_KIND for the pipeline registry
```

## Tests

CPU (default suite): `pytest tests/cosmos_curator/next/recipes/curation`.

Those tests drive the real scan, dedup, fairness and write-back passes against
a synthetic wide-table fixture, with host stand-ins for the GPU numerics. Real
cuML fit and cuPy dedup numerics run only in the GPU gate.
