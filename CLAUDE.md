# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cosmos-Curate is a video curation system for processing, analyzing, and organizing video content using AI models and distributed computing. It powers training data generation for NVIDIA Cosmos. The system is built on [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna), a framework for GPU-accelerated streaming pipelines using Ray.

## Architecture

### Core Components

The codebase follows a modular architecture with three main layers:

1. **Client Layer** (`cosmos_curate/client/`): Multiple deployment options
   - `local_cli/`: Launch pipelines in local Docker containers
   - `slurm_cli/`: Launch pipelines on Slurm clusters
   - `nvcf_cli/`: Deploy to NVIDIA Cloud Functions
   - `image_cli/`: Docker image management

2. **Core Layer** (`cosmos_curate/core/`):
   - `interfaces/`: Base classes for pipelines, stages, and models
     - `PipelineTask`: Base class for data passed between stages
     - `CuratorStage`: Base class for pipeline stages (inherits from Cosmos-Xenna `Stage[T, V]`)
     - `ModelInterface`: Base class for AI model integrations
   - `managers/`: CLIs for model and database management inside containers
   - `cf/`: Cloud function deployment utilities
   - `utils/`: Shared utilities (storage, database, config, hardware info)

3. **Pipeline Layer** (`cosmos_curate/pipelines/`):
   - `video/`: Video processing pipelines (splitting, captioning, filtering, embedding, deduplication)
   - `av/`: Autonomous vehicle multi-camera/LiDAR pipelines
   - `examples/`: Minimal example pipelines (start with `hello_world_pipeline.py`)

### Pipeline Execution Model

Pipelines use a streaming architecture with Ray actors:
- Each stage has a pool of Ray actors as workers (some CPU-only, some GPU-accelerated)
- Workers can run in different Pixi conda environments (solving dependency conflicts)
- Data streams through stages using Ray's distributed object store (passing references, not data)
- Auto-scaling balances throughput across stages based on resource availability
- Dynamic chunking handles variable-length inputs (e.g., 1-min vs 5-hour videos)

## Development Commands

### Environment Setup

```bash
# Initialize cosmos-xenna submodule (required)
git submodule update --init --recursive

# If you previously cloned the repo, sync the submodule URL
git submodule sync

# Create virtual environment (using micromamba as an example)
micromamba create -n cosmos-curate -c conda-forge python=3.10.18 poetry
micromamba activate cosmos-curate

# Install development dependencies
poetry install --extras=local

# Setup development tools (mypy types + pre-commit hooks)
./devset.sh
```

### Code Quality

Run before committing:

```bash
ruff format --check   # Check formatting
ruff check           # Lint code
mypy                 # Type checking
```

Configuration in `pyproject.toml`:
- Target: Python 3.10
- Line length: 120 characters
- Excludes: `cosmos-xenna/`, `cosmos_curate/models/internvideo2_multi_modality/`

### Testing

```bash
# Unit tests (CPU-only, no container needed)
pytest

# Model/stage tests (requires GPU + Docker container)
cosmos-curate local launch --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
  -- pixi run -e default pytest -m env tests/cosmos_curate/pipelines/

# For stages requiring vllm/unified environment
cosmos-curate local launch --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
  -- pixi run -e unified pytest -m env tests/cosmos_curate/pipelines/
```

Test marker: `@pytest.mark.env("unified")` indicates tests requiring GPU Pixi environments.

**Testing Guidelines**:
- Place new tests in `tests/` alongside the component they exercise
- Keep test filenames aligned with the target module (e.g., `test_motion_filter.py` for `motion_filter.py`)
- Use `pytest.mark.env("unified")` only when a Pixi GPU environment is required
- Guard slower tests with clear docstrings
- Document expected datasets or artifacts in `docs/curator/` when introducing new pipelines
- Update example configs as part of the same change

### Building

```bash
# Build client package wheel
poetry build

# Install standalone client
pip3 install dist/cosmos_curate*.whl

# Build Docker image (see packages/cosmos_curate/)
cosmos-curate image build --help
```

### CLI Usage

The main entry point is the `cosmos-curate` command with several subcommands:

```bash
# View available commands
cosmos-curate --help

# Subcommands for different deployment targets:
# - local: Run pipelines locally in Docker containers
# - slurm: Run pipelines on a Slurm cluster
# - nvcf: Run pipelines on NVIDIA Cloud Functions
# - image: Build and manage Docker images

# Example: explore local deployment options
cosmos-curate local --help
```

## Working with Pixi Environments

Pixi manages Python environments inside Docker images. Available environments defined in `pixi.toml`:

- `default`: Core dependencies + basic models
- `unified`: Advanced models (vllm, flash-attention, cvcuda)
- `phi`: Phi model family support
- `cuml`: cuML for GPU-accelerated deduplication
- `model-download`: Minimal environment for downloading model weights

```bash
# List all environments
pixi info

# View packages in an environment
pixi list -e unified

# Run command in specific environment
pixi run -e unified python -c "import torch; print(torch.cuda.is_available())"

# After editing pixi.toml, update lock file
pixi lock
```

**Important**: Each stage can specify `conda_env_name` property to run in a specific Pixi environment, enabling conflicting model dependencies in the same pipeline.

## Creating New Pipelines and Stages

### Defining a Pipeline Task

```python
import attrs
from cosmos_curate.core.interfaces.stage_interface import PipelineTask

@attrs.define
class MyTask(PipelineTask):
    input_path: str
    output: str | None = None

    @property
    def weight(self) -> float:
        # Used for load balancing (default: 1.0)
        return 1.0
```

### Defining a Stage

```python
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.interfaces.model_interface import ModelInterface

class MyStage(CuratorStage):
    def __init__(self) -> None:
        self._model = MyModel()  # Optional

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=2.0, gpus=1.0)

    @property
    def model(self) -> ModelInterface | None:
        return self._model  # Or None if no model

    @property
    def conda_env_name(self) -> str | None:
        return "unified"  # Or None for default environment

    def process_data(self, tasks: list[MyTask]) -> list[MyTask] | None:
        # Process tasks and return results
        for task in tasks:
            # Do work...
            pass
        return tasks
```

### Running a Pipeline

```python
from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.stage_interface import CuratorStageSpec

def my_pipeline(args) -> None:
    stages = [
        CuratorStageSpec(stage=MyStage(), max_actors=4),
        CuratorStageSpec(stage=AnotherStage(), max_actors=8),
    ]

    input_tasks = [MyTask(input_path=path) for path in paths]

    run_pipeline(
        stages=stages,
        input_tasks=input_tasks,
        execution_mode="streaming",  # or "batch_simple"
    )
```

## Model Integration

Models inherit from `ModelInterface` (`cosmos_curate/core/interfaces/model_interface.py`):

```python
from cosmos_curate.core.interfaces.model_interface import ModelInterface

class MyModel(ModelInterface):
    @property
    def model_name(self) -> str:
        return "my-model-name"

    def load(self, download_dir: str, device: str) -> None:
        # Load model weights
        pass

    def predict(self, inputs):
        # Run inference
        pass
```

Models are automatically downloaded via `model_cli.py` before pipeline execution. Register new models in `cosmos_curate/models/all_models.py`.

## Key Patterns

### Dynamic Chunking

Stages can output different numbers of tasks than input to handle variable-sized data:

```python
def process_data(self, tasks: list[MyTask]) -> list[MyTask] | None:
    output_tasks = []
    for task in tasks:
        # Split large task into smaller chunks
        chunks = chunk_data(task, max_size=16)
        output_tasks.extend(chunks)
    return output_tasks
```

### Multi-Environment Pipelines

Stages requiring different dependencies can run in separate Pixi environments:

```python
# Stage 1: Uses default environment
class Stage1(CuratorStage):
    @property
    def conda_env_name(self) -> str | None:
        return None  # Uses default

# Stage 2: Requires vllm
class Stage2(CuratorStage):
    @property
    def conda_env_name(self) -> str | None:
        return "unified"
```

### Efficient Resource Allocation

The auto-scaler balances stages based on:
1. Per-actor throughput (tasks/sec)
2. Resource requirements (CPUs, GPUs)
3. Total cluster resources

Specify fractional GPUs for models that can share GPUs:
```python
return CuratorStageResource(cpus=1.0, gpus=0.8)
```

## Coding Conventions

### Style Guidelines
- Follow PEP 8 with 4-space indentation
- Use `snake_case` for functions and modules, `CamelCase` for classes
- Prefer explicit type hints and dataclasses where applicable
- Use `ruff format` for formatting and keep imports ordered
- Configuration files and sample manifests live under `examples/`—mirror naming schemes such as `video_filtering_*.yaml` when adding new assets

## Commit and PR Guidelines

### Commit Style

Use Conventional Commits style with sign-off:

```bash
git commit -s -m "fix: correct motion filter threshold calculation"
git commit -s -m "feat: add support for Qwen2-VL model"
git commit -s -m "chore: update pixi.lock for unified environment"
```

### Pull Request Guidelines

PRs should:
- Summarize functional intent in the description
- List validation commands run to verify the changes
- Note any documentation or submodule updates
- Link tracking issues
- Provide screenshots or logs for UX changes

## Submodule Management

The `cosmos-xenna` submodule contains the core pipeline framework. When updating:

```bash
cd cosmos-xenna
git checkout <desired-version>
cd ..
git add cosmos-xenna
git commit -s -m "chore: update cosmos-xenna to v0.1.7"
```

Pre-commit hooks will warn before committing submodule changes.

## Documentation

- End user guide: `docs/client/END_USER_GUIDE.md`
- Architecture details: `docs/curator/ARCHITECTURE_GUIDE.md`
- Pipeline design walkthrough: `docs/curator/PIPELINE_DESIGN_GUIDE.md`
- Reference pipelines: `docs/curator/REFERENCE_PIPELINES_VIDEO.md`, `docs/curator/REFERENCE_PIPELINES_AV.md`
- NVCF deployment: `docs/client/NVCF_GUIDE.md`
