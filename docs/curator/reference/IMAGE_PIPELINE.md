# Cosmos-Curate - Reference Image Pipeline

- [Cosmos-Curate - Reference Image Pipeline](#cosmos-curate---reference-image-pipeline)
  - [Annotate Pipeline](#annotate-pipeline)
    - [Annotate Pipeline Stages](#annotate-pipeline-stages)
    - [Annotate Pipeline Output Format](#annotate-pipeline-output-format)
    - [Already-captioned skip (resume behavior)](#already-captioned-skip-resume-behavior)
    - [Annotate Pipeline Configurable Options](#annotate-pipeline-configurable-options)

The image reference pipeline provides:

- **Annotate pipeline** — Loads images from local disk or S3, optionally generates captions using a vision-language model (VLM), and writes images and metadata to an output path. Supports skipping images that already have a caption in the output (resume / do-not-recaption behavior).

## Annotate Pipeline

### Annotate Pipeline Stages

The annotate pipeline includes the following stages:

- **Ingest (Image Load)**: Loads image files from the input path (local or S3) into task payloads for downstream stages.
- **Captioning (optional)**:
  - **Caption prep**: Decodes image bytes, resizes within pixel bounds, and builds VLM-specific model input (e.g. for Qwen, Cosmos-R1/R2, Nemotron). Runs on CPU with configurable workers per node.
  - **Caption**: Runs the selected VLM to generate a text caption per image. Runs on GPU.
- **Output (Image Writer)**: Writes each image to `images/{id}{ext}` and per-image metadata (source path, dimensions, timestamps, caption status, caption if present) to `metas/{id}.json`. Writes a top-level `summary.json` with run statistics and resize settings.

One task corresponds to one image; the pipeline does not split or batch images across multiple output files.

### Annotate Pipeline Output Format

The annotate pipeline produces the following artifacts under the path specified by `--output-path`:

```text
{output_path}/
├── images/                    # one file per input image
│   ├── {output_id}.jpg       # output_id = stable hash of input path (first 16 hex chars)
├── metas/
│   ├── {output_id}.json      # metadata: source_path, relative_path, width, height, has_caption, align_timestamp_ns, sensor_timestamp_ns[, caption]
├── summary.json              # run summary: num_input_images, num_output_tasks, resize_min_pixels, resize_max_pixels, captioned_images, etc.
```

Each `metas/{output_id}.json` includes:

- `source_path`: full input path of the image
- `relative_path`: path relative to input root
- `width`, `height`: dimensions after prep resize (or null if captioning was skipped)
- `has_caption`: whether a caption was generated
- `align_timestamp_ns`: sampled/reference timestamp from `image_data`, or `null` if unavailable
- `sensor_timestamp_ns`: sampled native sensor timestamp from `image_data`, or `null` if unavailable
- `caption_status`: normalized caption outcome such as `success`, `truncated`, `error`, or `null` if captioning was skipped
- `caption_failure_reason`: failure reason when `caption_status == "error"`, otherwise `null`
- `token_counts`: per-model token usage keyed by model variant
- `caption`: present only when `has_caption` is true

### Already-captioned skip (resume behavior)

When the same output path is used across runs, the pipeline treats images that **already have a caption** in that output as completed and does not recaption them:

- At **input extraction**, the pipeline checks for existing captions using `summary.json` (field `captioned_images`) or, if that is missing, by scanning `metas/*.json` for `has_caption: true`. Any image whose output ID is in that set is excluded from the task list.
- In the **caption stage**, any task that already has a non-empty caption is skipped (failsafe).

This matches the video pipeline’s “already processed” skip so that re-runs only process new or uncaptioned images.

### Annotate Pipeline Configurable Options

A summary of important options is below. For the full list, run:

```bash
cosmos-curate local launch \
  --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
  -- pixi run python3 -m cosmos_curate.pipelines.image.run_pipeline annotate --help
```

**Required**

- `--input-image-path`: path (local or `s3://`) to a directory of input images.
- `--output-path`: path (local or `s3://`) for output; `images/`, `metas/`, and `summary.json` are written under this path.

**Input/Output**

- `--input-s3-profile-name`: S3 profile for `--input-image-path` when using S3; default `"default"`.
- `--output-s3-profile-name`: S3 profile for `--output-path` when using S3; default `"default"`.
- `--limit`: if greater than 0, process at most this many images (after filtering and sorting).

**Captioning**

- `--no-generate-captions`: disable captioning; only load and write images (and metadata without captions).
- `--captioning-algorithm`: VLM to use. Supported values include `qwen`, `qwen3_vl_30b`, `qwen3_vl_30b_fp8`, `qwen3_vl_235b`, `qwen3_vl_235b_fp8`, `nemotron`, `cosmos_r1`, `cosmos_r2`. Default `qwen`.
- `--caption-num-gpus`: GPUs per node for the caption stage; default `1`.
- `--num-caption-prep-workers-per-node`: CPU workers per node for caption prep; default `2`.
- `--caption-prep-min-pixels`: minimum total pixels for prep resize; default uses video-style `128*28*28`.
- `--caption-prep-max-pixels`: maximum total pixels for prep resize; default uses video-style `768*28*28`.
- `--caption-batch-size`: batch size for the vLLM caption stage; default `16`.
- `--caption-max-output-tokens`: max output tokens for caption generation; default `8192`.
- `--caption-prompt-variant`: prompt variant (e.g. `image`, `default`); default `image`.
- `--caption-prompt-text`: optional custom prompt text (overrides variant).

**Performance and logging**

- `--num-ingest-workers-per-node`: image load workers per node; default `4`.
- `--num-output-workers-per-node`: image writer workers per node; default `8`.
- `--verbose`: enable verbose logging.
- `--perf-profile`: write per-stage performance stats.

**Example**

```bash
cosmos-curate local launch \
  --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
  -- pixi run python3 -m cosmos_curate.pipelines.image.run_pipeline annotate \
  --input-image-path /path/to/images \
  --output-path /path/to/output \
  --captioning-algorithm qwen \
  --limit 10
```
