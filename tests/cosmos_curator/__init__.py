# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests mirroring the ``cosmos_curator`` package layout.

This package marker keeps test modules importable under their full dotted
name. Without it, ``--import-mode=importlib`` names them from the first
directory lacking an ``__init__.py``, so a module under ``next/`` would be
imported as ``next....`` and Ray workers could not unpickle functions
defined in it.
"""
