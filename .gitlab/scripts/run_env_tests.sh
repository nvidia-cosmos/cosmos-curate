#!/usr/bin/env bash
set -euo pipefail

# Run tests for each environment with unique report files and coverage
for env in default legacy-transformers transformers unified; do
  echo "Running tests for $env environment"
  pixi run -e $env pytest -m env -n "${PYTEST_XDIST_WORKERS}" \
    --junitxml="/config/project/$env-report.xml" \
    --cov=cosmos_curate \
    --cov-report=term \
    --cov-report=xml:/config/project/$env-coverage.xml \
    --cov-report=html:/config/project/$env-htmlcov \
    tests/cosmos_curate/pipelines tests/cosmos_curate/models

  # Save the coverage data file for each environment
  if [ -f .coverage ]; then
    cp .coverage /config/project/.coverage.gpu_$env
  fi
done
