# Slim Image Design

> **Note:** "Image" throughout this document refers to the Docker/OCI container image built by
> `cosmos-curate image build`.

> **Phasing note:** Phases 1-2 (removing `ffmpeg_gpu`, switching to conda-forge FFmpeg) and Phase 4 (cleanup) are
> valuable independently of the slim image work — they unblock CUDA 13, simplify compliance, and reduce image size for
> all modes. Only Phase 3 depends on the slim mode design and its shared storage assumptions.

## Motivation

1. **Local dev velocity**: A slim image rebuilds in seconds (just source code, no conda install). Mounting the host's
   pixi environments via `--pixi-path` avoids rebuilding the image entirely when iterating on code or dependencies.
2. **Image size**: Baked images with all environments pre-installed are enormous, causing slow push/pull on GitLab CI,
   NVCF, Slurm, and internal registries (throttling, bandwidth waste, layer timeout failures).
3. **CUDA 13 unblocked**: The custom FFmpeg source build was blocking the CUDA 13 upgrade. Replacing it with conda-forge
   FFmpeg removes that blocker.
4. **Compliance & security**: Replacing the custom FFmpeg source build with conda-forge packages eliminates
   custom-compiled binaries — smaller SBOMs, minimal license obligations, cleaner scans.
5. **Scaling**: With `.pixi` on shared storage (e.g. PVC on k8s), a pre-warm step installs environments once and all
   workers use them directly — no per-worker install overhead.

## High-Level Design

**Three image modes** via `cosmos-curate image build --mode <mode>`:

- `full` (default) — pre-installs a curated set of pixi environments at build time. Best for platforms without
  shared/persistent storage (NVCF, air-gapped deployments).
- `slim` — lockfile (`pixi.toml` + `pixi.lock`) and source code only. `pixi run --frozen` auto-installs the exact
  environment on first use. Combined with `--pixi-path .`, this is ideal for local development (near-instant rebuilds,
  no conda install in the image). Cluster use (Slurm/k8s with shared storage) is promising but needs validation.
- `custom --envs env1,env2,...` — pre-installs only the specified pixi environments. Useful for teams that need a
  subset of environments (e.g. `default,unified`) without the full image size.

**Dependency strategy:**

- FFmpeg from conda-forge (LGPL) via the `av` (PyAV) conda package. Currently `av` is a PyPI dependency in `pixi.toml`,
  which bundles its own FFmpeg build. To get the conda-forge FFmpeg (with NVENC support), `av` must be switched to a
  conda dependency so it pulls in conda-forge FFmpeg as a transitive dependency. conda-forge FFmpeg is a strict superset
  of the current source build's codecs (libopenh264, libdav1d, libaom, libvpx, libwebp, libvorbis, vaapi, plus extras
  like libsvtav1, libmp3lame, libopus, libjxl). The conda-forge LGPL build also includes `h264_nvenc` and `hevc_nvenc`
  via the `ffnvcodec-headers` package (MIT-licensed, LGPL-compatible). Verified: `h264_nvenc` encoding works on RTX GPUs
  with the LGPL conda-forge build. **Important:** conda-forge ships both GPL (`ffmpeg=*=gpl_*`) and LGPL
  (`ffmpeg=*=lgpl_*`) variants — pin the LGPL variant in `pixi.toml` (e.g. `ffmpeg = "=*=lgpl_*"`).
- GPU video decode via PyNvVideoCodec (`pynvc` mode, already the preferred path in `VideoFrameExtractionStage`).
  The `ffmpeg_gpu` decode mode is removed — benchmarks showed it performed the same as `ffmpeg_cpu`, while `pynvc`
  is measurably faster. Removing it eliminates the only usage of `scale_npp` (libnpp) and the need for a custom
  GPU FFmpeg build.
- GPU FFmpeg transcoding (`h264_nvenc`) retained for teams with NVENC-capable GPUs (e.g. RTX in data center). CPU
  `libopenh264` remains the default encoder for GPUs without NVENC hardware (A100, H100).

**Shared environments (slim mode):** The `.pixi` directory (containing both the package cache and installed environments)
is mounted on shared storage (PVC on k8s, Lustre on Slurm). A pre-warm step runs `pixi install --frozen` on the shared
mount before Ray workers start — workers then use the pre-populated environments directly with no per-worker install.

## Limitations and risks

1. **Slim mode is unconventional.** The standard container practice — including pixi's own documentation — is to install
   dependencies at build time and ship a self-contained image. Deferring installation to runtime is not a well-trodden
   path. It trades image size for startup complexity and requires infrastructure (shared storage, pre-warm scripts) that
   most deployments don't have.

2. **Shared storage dependency.** For cluster use, slim mode only makes sense when shared persistent storage is available
   (PVC on k8s, Lustre on Slurm). Without it, every worker downloads and installs independently, which is slower than
   pulling a pre-built image. Platforms like NVCF have no shared storage, so they must use full/custom mode — and those
   images are still large. (Local dev avoids this issue by mounting the host `.pixi` directory directly.)

3. **Network access at runtime.** Slim mode requires network access to conda-forge (and PyPI for some packages) during
   the pre-warm step. Air-gapped environments cannot use slim mode at all.

4. **Pre-warm adds orchestration complexity.** The pre-warm script must run before Ray workers start, on the same shared
   storage, and must complete successfully. Failures (network issues, disk full, permission errors) block the entire job.
   This is a new failure mode that doesn't exist with pre-built images.

5. **Untested at scale.** Sharing `.pixi` across many concurrent workers has not been validated at the scale
   cosmos-curate typically runs (hundreds of workers). File locking, NFS/Lustre metadata performance, and concurrent
   access patterns are potential issues.

6. **Full/custom modes don't solve the size problem.** For platforms that need pre-built images (NVCF, air-gapped), the
   image size remains large. Phases 1-2 (removing the FFmpeg source build) help, but the bulk of the image size comes
   from conda environments and model dependencies, which this design does not address.

## Task List

### Phase 1: Remove `ffmpeg_gpu` decode path

Benchmarks showed `ffmpeg_gpu` performed the same as `ffmpeg_cpu` (GPU decode savings negated by CPU scaling),
while `pynvc` is measurably faster. This removal is justified independently of the conda-forge switch.

- [x] **1a. Remove `ffmpeg_gpu` decoder mode from `VideoFrameExtractionStage`**
    - Drop the `ffmpeg_gpu` choice from `--transnetv2-frame-decoder-mode` CLI arg (`splitting_pipeline.py`)
    - Remove the `use_gpu=True` branch in `get_frames_from_ffmpeg()` (`frame_extraction_stages.py`)
    - Update `VideoFrameExtractionStage` to remove the `ffmpeg_gpu` resource/logic path
    - Eliminates the only usage of `scale_npp` (libnpp) — the sole reason for a custom GPU FFmpeg build
    - `pynvc` remains as the GPU decode option; `ffmpeg_cpu` remains as CPU fallback

### Phase 2: Replace source-built FFmpeg with conda-forge

- [ ] **2a. Switch `av` from PyPI to conda-forge and verify FFmpeg codec parity**
    - Move `av` from `[pypi-dependencies]` to `[dependencies]` in `pixi.toml` so it pulls conda-forge FFmpeg
    - Pin `ffmpeg = "=*=lgpl_*"` to ensure the LGPL variant is used (not GPL)
    - The PyPI `av` package bundles its own FFmpeg and does not include conda-forge FFmpeg or NVENC support
    - Build a test image without the source FFmpeg build, relying only on conda-forge FFmpeg
    - Run the CPU video pipeline end-to-end (download, split, transcode with libopenh264, write)
    - Confirm `ffprobe` is available from conda-forge FFmpeg
    - Verify all codecs used in practice: libopenh264 encode, libdav1d/libaom decode, remux
    - Verify `h264_nvenc`/`hevc_nvenc` are present (conda-forge includes them via `ffnvcodec-headers`)
    - Test GPU transcoding on an NVENC-capable GPU (e.g. RTX)

- [ ] **2b. Remove `--ffmpeg-cuda` build flag and GPU FFmpeg source build**
    - Remove `--ffmpeg-cuda` option from `image_app.py`
    - Remove all `ffmpeg_cuda` conditional blocks from `default.dockerfile.jinja2`
    - No longer needed: conda-forge FFmpeg includes `h264_nvenc`/`hevc_nvenc` in the LGPL build

- [ ] **2c. Remove FFmpeg source build from Dockerfile**
    - Remove the entire FFmpeg source build block from `default.dockerfile.jinja2`
    - Remove apt dependencies only needed for FFmpeg compilation (autoconf, automake, cmake, yasm, nasm, libtool, etc.)
    - Keep system deps still needed elsewhere (libsm6, libxext6 for OpenCV)

### Phase 3: Slim image (lockfile-only)

- [ ] **3a. Get pixi approved for open-source release and bundle source in image**
    - Get pixi (MIT-licensed, single static Rust binary) approved for inclusion in the shipped image
    - Bundle pixi source tarball at pinned version (e.g.
      `ADD https://github.com/prefix-dev/pixi/archive/refs/tags/${PIXI_VERSION}.tar.gz /opt/oss-sources/pixi.tar.gz`)
    - ~17MB compressed, source archival only — no build needed

- [ ] **3b. Add `--pixi-path` flag to `cosmos-curate local launch`**
    - Mount the host `.pixi` directory into the container (envs, cache, config)
    - For local dev, `--curator-path . --pixi-path .` mounts both source and `.pixi` from the project root
    - For cluster deployments, `--pixi-path /mnt/shared/pixi` can point to shared storage independently
    - Enables near-instant iteration: slim image + host envs + host source, no image rebuild needed

- [ ] **3c. Add `--mode` flag to `cosmos-curate image build`**
    - `--mode slim`: skip `pixi install`, image contains only lockfile + source
    - `--mode full`: pre-install a curated set of pixi environments at build time
    - `--mode custom --envs env1,env2,...`: pre-install only the specified environments
    - Full and custom modes retain the NVIDIA wheel pre-download hack and retry logic

- [ ] **3d. Switch entrypoint to `pixi run --frozen`**
    - Update `CMD` in Dockerfile to use `pixi run --frozen` instead of `pixi run`
    - Ensure `pixi run --frozen` triggers auto-install when envs are missing
    - Verify this works for all modes: slim (installs on first run), full/custom (already installed, no-op)

- [ ] **3e. Configure shared `.pixi` mount for cluster deployments**
    - Update `sbatch.sh.j2` to mount `.pixi` from shared storage (Lustre)
    - Update `launch_local.py` to support `--pixi-path` for mounting `.pixi` from an arbitrary location
    - Document shared storage setup for cluster deployments

- [ ] **3f. Add pre-warm script for Slurm**
    - Add a head-node step in `sbatch.sh.j2` that runs `pixi install --frozen` for all required envs before `srun` fans
      out workers
    - Populates the shared `.pixi` directory before Ray starts

- [ ] **3g. Validate slim image on Slurm with shared storage**
    - End-to-end test: slim image + shared `.pixi` mount + multi-node Ray cluster
    - Measure cold-start time (empty cache) and warm-start time (pre-warmed)
    - Verify workers use pre-populated environments with no per-worker install

### Phase 4: Cleanup and optimization

- [ ] **4a. Remove deprecated `remux_to_mp4` stage**
    - Already marked for removal by 2026-04-30
    - Removes another FFmpeg subprocess callsite

- [ ] **4b. Slim down system apt dependencies**
    - Audit which apt packages are still needed without the FFmpeg source build
    - Remove build-only tools that conda-forge FFmpeg doesn't need

### Future (lower priority)

- [ ] **Pixi base image**
    - Switch from `nvcr.io/nvidia/cuda:*-devel-ubuntu24.04` to `ubuntu:24.04` with pixi managing the CUDA toolkit
    - Move CUDA toolkit into the pixi environment itself
    - Would further reduce image size and simplify the Dockerfile
