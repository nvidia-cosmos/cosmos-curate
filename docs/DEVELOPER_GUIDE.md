# Cosmos-Curate - Developer Guide

- [Cosmos-Curate - Developer Guide](#cosmos-curate---developer-guide)
  - [Development Environment Setup](#development-environment-setup)
    - [Working with Pixi Environments](#working-with-pixi-environments)
      - [Installing Pixi](#installing-pixi)
      - [Adding New Dependencies](#adding-new-dependencies)
      - [Updating Pixi Environments](#updating-pixi-environments)
      - [Viewing Available Environments](#viewing-available-environments)
      - [Viewing Packages in Pixi Environments](#viewing-packages-in-pixi-environments)
      - [Running Commands in Pixi Environments](#running-commands-in-pixi-environments)
  - [Code Quality Checks](#code-quality-checks)
  - [Building the Client package](#building-the-client-package)
  - [Testing](#testing)
    - [Unit Tests](#unit-tests)
    - [Model and Stage Tests](#model-and-stage-tests)
    - [End-to-End Pipeline Tests](#end-to-end-pipeline-tests)
  - [Best Practices](#best-practices)
  - [Contributing](#contributing)
  - [Troubleshooting](#troubleshooting)
  - [Responsible Use of AI Models](#responsible-use-of-ai-models)
  - [Support](#support)

## Development Environment Setup

Please refer to the following section in [End User Guide](./client/END_USER_GUIDE.md):
- [Prerequisites](./client/END_USER_GUIDE.md#prerequisites) for hardware and software requirements.
- [Initial Setup](./client/END_USER_GUIDE.md#initial-setup) for preparaing configurations files and workspace directories, etc.
- [Setup Environment and Install Dependencies](./client/END_USER_GUIDE.md#setup-environment-and-install-dependencies) for setting up Cosmos-Curate.

For developers to contribute back to the repo, the following additional steps are needed:

To help ease with setup, from within your virtual environment you can run
```bash
./devset.sh
```

Alternatively, you should perform these additional steps.
```bash
# Install mypy types
mypy --install-types

# Set up pre-commit hooks
pre-commit install
```

### Working with Pixi Environments

Cosmos-Curate uses [Pixi](https://pixi.sh) to manage Python environments inside Docker images. As a developer, you may
need to inspect or work with these environments locally.

The `pixi.toml` file at the repository root defines all Pixi environments used by Cosmos-Curate.

#### Installing Pixi

Follow the [Pixi installation instructions](https://pixi.sh/latest/installation/), or run the following commands:

```bash
# Assuming your current user has write access to /usr/local/bin.
# If not, you can change ownership of /usr/local to your user: sudo chown -R $USER:$USER /usr/local
wget -qO- https://pixi.sh/install.sh | PIXI_HOME=/usr/local PIXI_NO_PATH_UPDATE=1 sh
```

This installs Pixi to `/usr/local/bin/pixi`, which matches the location in the Docker image.
You can verify the installation by running `pixi --version`.

#### Adding New Dependencies

When adding new dependencies to a pixi environment:

1. Edit the `pixi.toml` file to add your dependency under the appropriate environment
2. Run `pixi lock` to resolve the new dependency
3. Rebuild the Docker image
4. Test your changes inside the Docker container to ensure compatibility
5. Commit both `pixi.toml` and the updated `pixi.lock` file to version control to ensure reproducibility

Note: Environment names in pixi use hyphens (e.g., `video-splitting`) rather than underscores, as per pixi conventions.

#### Updating Pixi Environments

To update pixi environments:

1. Run `pixi update` to update the environments
2. Rebuild the Docker image
3. Test your changes inside the Docker container to ensure compatibility

#### Viewing Available Environments

To see all available pixi environments defined in `pixi.toml`:

```bash
# Show detailed information about all environments
pixi info

# Or list the environments only
pixi workspace environment list
```

#### Viewing Packages in Pixi Environments

To see all the packages and their versions in a pixi environment:

```bash
# For the default environment
pixi list

# For a specific environment
pixi list -e <env-name>

# For a specific package
pixi list -e <env-name> <package-name>

# Example: see the pytorch packages in the 'video-splitting' environment
pixi list -e video-splitting pytorch
```

#### Running Commands in Pixi Environments

While most pipeline execution happens inside Docker containers, you may need to run commands in pixi environments:

```bash
# Run in the default environment
pixi run python -m <module>

# Run in a specific environment
pixi run -e <env-name> python -m <module>

# Example: check if PyTorch is CUDA enabled in the video-splitting environment
pixi run -e video-splitting python -c "import torch; print(torch.cuda.is_available())"
```

Note: For pipeline execution, always use the Docker container as shown in the testing section.

## Code Quality Checks

This project uses the following development tools:
1. **ruff**: For code formatting and linting
2. **mypy**: For static type checking
3. **poetry**: For dependency management

Before submitting any changes, run the following checks from the repository root:

```bash
ruff format --check
ruff check
mypy
```

## Building the Client package
   - The `cosmos-curate` client can be built as a wheel and installed in a standalone mode, without the need for the rest of the source environment
```bash
poetry build
pip3 install dist/cosmos_curate*.whl
```

## Testing

Tests under the [tests/](../tests/) directory can be categorized into 3 levels:
- Unit tests: for testing critical/complex function which are typically CPU-only and can run in default conda environment.
- Model/stage tests: for testing functional correctness of a model, a pipeline stage, and a combination of a few stages, which typically require GPU and should run inside the container.
- End-to-end pipeline tests: for testing the functionality of reference pipelines.

### Unit Tests

Simply run `pytest` from the repository root:

```bash
pytest

================ test session starts ================
configfile: pytest.ini
testpaths: tests
collected 155 items / 26 deselected / 129 selected

tests/client/slurm_cli/test_slurm.py .....................................                      [ 36%]
tests/client/slurm_cli/test_start_ray.py ...................................                    [ 70%]
tests/cosmos_curate/pipelines/video/filtering/motion/test_motion_filter.py .                    [ 71%]
tests/cosmos_curate/pipelines/video/utils/test_decoder_utils.py .............................   [100%]
================ 102 passed, 6 deselected, 2 warnings in 2.98s ================
```

### Model and Stage Tests

Launch the docker container locally and simply run `pytest` command:

```bash
for conda_env in default video-splitting unified; do
   cosmos-curate local launch --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
   -- pixi run -e $conda_env pytest -m env tests/cosmos_curate/pipelines/;
done

================ test session starts ================
configfile: pytest.ini
collected 58 items / 40 deselected / 18 selected

tests/cosmos_curate/pipelines/video/clipping/test_fixed_stride_extraction.py ........           [ 44%]
tests/cosmos_curate/pipelines/video/clipping/test_transnetv2_extraction.py .....                [ 72%]
tests/cosmos_curate/pipelines/video/filtering/motion/test_motion_filter.py .                    [100%]
================ 18 passed, 40 deselected, 3 warning in 14.44 ================

...

================ test session starts ================
configfile: pytest.ini
collected 58 items / 52 deselected / 6 selected

tests/cosmos_curate/pipelines/video/captioning/test_t5_embedding.py .                           [ 16%]
tests/cosmos_curate/pipelines/video/filtering/aesthetics/test_aesthetic_filter.py .....         [100%]
================ 6 passed, 52 deselected, 2 warning in 30.02 ================
```

### End-to-End Pipeline Tests

Run the reference video pipeline based on instructions in [Run the Split-Annotate Pipeline](./client/END_USER_GUIDE.md#run-the-reference-video-pipeline) section to make sure everything works.

The CI will test more scenarios.

## Best Practices

1. **Virtual Environment**:
   - Always work in a virtual environment
   - Avoid using `$HOME/.local` for Python packages

2. **Dependencies**:
   - Maintain the `pyproject.toml` file
   - Document any new dependencies
   - Use `poetry install --extras=local` to install packages in your virtual environment
   > Note: You may need to do the following : `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` before running `poetry` to avoid getting stuck due to a keyring pop-up when you are running in a headless display environment
   > Note: From within your virtual environment, you may execute `./devset.sh` to complete inital setup of environment.

3. **Code Quality**:
   - Write clean, well-documented code
   - Use ruff for formatting and linting
   - Ensure type hints are properly used and checked with mypy
   - Write meaningful commit messages

4. **Testing**:
   - Write tests for new features
   - Run existing tests before submitting changes
   - Ensure all tests pass

5. **Documentation**:
   - Update documentation when adding new features
   - Keep the README files up to date
   - Document any API changes

## Contributing

1. Create a new branch for your feature/fix
2. Make your changes
3. Run all code quality checks
4. Submit a pull request with a clear description of changes

## Troubleshooting

If you encounter issues during development:

1. **Environment Issues**:
   - Ensure you're using the correct Python version
   - Verify all dependencies are installed
   - Check virtual environment activation

2. **Build Issues**:
   - Clear any cached files
   - Rebuild the environment if necessary
   - Check for conflicting dependencies

## Responsible Use of AI Models
[Responsible Use](./RESPONSIBLE_USE.md)

## Support

For development-related questions or issues:
- Create an issue in the repository
- Contact the development team
- Check existing documentation and issues 
