#!/bin/bash

# This script is used to quickly help setup dev environment

poetry install --extras=local
poetry run pre-commit install
mypy --install-types
poetry build
