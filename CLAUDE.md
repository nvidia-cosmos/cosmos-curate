# CLAUDE.md

## Project Overview

Cosmos-Curate is a video curation system for AI training data generation, built on [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna) (GPU-accelerated streaming pipelines using Ray).

**Architecture**: Three-layer modular design
- `cosmos_curate/client/`: Deployment CLIs (local, Slurm, NVCF, Docker image management)
- `cosmos_curate/core/`: Base interfaces (`PipelineTask`, `CuratorStage`, `ModelInterface`), managers, utilities
- `cosmos_curate/pipelines/`: Video/AV pipelines + examples (start with `hello_world_pipeline.py`)

**Key concepts**: Ray-based streaming with multi-environment support (Pixi), auto-scaling, dynamic chunking

## Development

**Setup**: `git submodule update --init --recursive && ./devset.sh`

**Code Quality** (run globally before committing, not just on changed files):
```bash
ruff format && ruff check --fix && mypy
```

**Testing**:
- CPU tests: `pytest`
- GPU tests: `cosmos-curate local launch --curator-path . -- pixi run -e [default|unified] pytest -m env tests/`
- Use `@pytest.mark.env("unified")` for GPU-required tests
- Place tests in `tests/` aligned with module names

**Building**: `poetry build` (client wheel) or `cosmos-curate image build` (Docker)

**CLI**: `cosmos-curate [local|slurm|nvcf|image] --help` for deployment options

## Pixi Environments

Defined in `pixi.toml`: `default` (core), `unified` (vllm/advanced models), `phi`, `cuml`, `model-download`

Stages specify `conda_env_name` property to run in specific environments, enabling dependency isolation. Commands: `pixi info`, `pixi list -e ENV`, `pixi run -e ENV COMMAND`

## Creating Pipelines

**Tasks**: Inherit from `PipelineTask` (attrs dataclass). Define `weight` property for load balancing.

**Stages**: Inherit from `CuratorStage`. Implement:
- `resources` property: `CuratorStageResource(cpus=X, gpus=Y)` (fractional GPUs allowed)
- `conda_env_name` property: Environment name or None for default
- `process_data()`: Process batch of tasks, can return different count (dynamic chunking)
- Optional `model` property: Return `ModelInterface` instance

**Models**: Inherit from `ModelInterface`. Implement `model_name`, `load()`, `predict()`. Register in `cosmos_curate/models/all_models.py`.

**Running**: Use `run_pipeline(stages=[CuratorStageSpec(stage=MyStage(), max_actors=N)], input_tasks=[...], execution_mode="streaming")`

See `cosmos_curate/pipelines/examples/hello_world_pipeline.py` and `docs/curator/PIPELINE_DESIGN_GUIDE.md`

## Code Style

PEP 8 (4-space indent, `snake_case`/`CamelCase`), type hints, ruff formatting. Config: `pyproject.toml` (Python 3.10, 120 chars)

## Commits & PRs

**Commits**: Conventional Commits with sign-off: `git commit -s -m "fix: description"`

**Merge Requests**:
- Create using `glab mr create` targeting branch `nvidia/main`, assign to author, enable "Delete source branch"
- Set MR title and description properly
- Summarize intent, list validation commands, note doc/submodule updates, link issues
- Do NOT include "Generated with Claude Code" in descriptions
- Use GitLab MCP server (if available) to retrieve CodeRabbit and Greptile review comments

**Submodule updates**: Pre-commit warns before committing. Update via `cd cosmos-xenna && git checkout VERSION && cd .. && git add cosmos-xenna`

## Documentation

See `docs/client/END_USER_GUIDE.md`, `docs/curator/ARCHITECTURE_GUIDE.md`, `docs/curator/PIPELINE_DESIGN_GUIDE.md`, reference pipelines, and NVCF guide
