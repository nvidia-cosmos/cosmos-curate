# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge per-stage pyinstrument CPU profiles into a single combined report.

Standalone CLI tool -- not part of the main pipeline codebase.  Run it
after a pipeline execution to combine all ``.pyisession`` files produced
by ``--profile-cpu`` into one aggregated flame-tree or timeline view.

Usage examples::

    # Default: combine all sessions, write combined HTML next to originals
    python -m benchmarks.merge_cpu_profiles /path/to/profiles/cpu/

    # Custom output path
    python -m benchmarks.merge_cpu_profiles /path/to/profiles/cpu/ -o report.html

    # Text output instead of HTML
    python -m benchmarks.merge_cpu_profiles /path/to/profiles/cpu/ --format text

    # Timeline mode (chronological, no aggregation)
    python -m benchmarks.merge_cpu_profiles /path/to/profiles/cpu/ --timeline

    # Combine all of the above
    python -m benchmarks.merge_cpu_profiles /path/to/profiles/cpu/ -o report.txt --format text --timeline

pyinstrument note:
    ``Session.combine()`` concatenates samples from two sessions.
    Aggregate views (the default) work well on combined data.  Timeline
    views preserve chronological order but interleave samples from
    different stages -- useful for seeing the overall execution shape.
"""

import argparse
import pathlib
import sys
from functools import reduce

from loguru import logger
from pyinstrument.renderers import ConsoleRenderer, HTMLRenderer
from pyinstrument.session import Session


def _discover_sessions(directory: pathlib.Path) -> list[pathlib.Path]:
    """Glob ``*.pyisession`` files in *directory*, sorted by name.

    Args:
        directory: Path to the directory containing ``.pyisession`` files.

    Returns:
        Sorted list of discovered session file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.

    """
    if not directory.is_dir():
        msg = f"Directory does not exist: {directory}"
        raise FileNotFoundError(msg)

    return sorted(directory.glob("*.pyisession"))


def _load_and_combine(paths: list[pathlib.Path]) -> Session:
    """Load all session files and combine them into a single session.

    Uses ``functools.reduce`` to chain ``Session.combine()`` pairwise
    across all discovered sessions.

    Args:
        paths: List of ``.pyisession`` file paths to load and merge.

    Returns:
        A single combined ``Session`` object.

    Raises:
        ValueError: If *paths* is empty.

    """
    if not paths:
        msg = "No .pyisession files to combine"
        raise ValueError(msg)

    sessions = []
    for p in paths:
        logger.info(f"  Loading: {p.name}")
        sessions.append(Session.load(str(p)))

    if len(sessions) == 1:
        return sessions[0]

    return reduce(Session.combine, sessions)


def _render(
    session: Session,
    *,
    fmt: str,
    timeline: bool,
    outfile: pathlib.Path | None,
) -> None:
    """Render the combined session and write the output.

    Args:
        session: The combined pyinstrument session.
        fmt: Output format -- ``"html"`` or ``"text"``.
        timeline: If ``True``, preserve chronological order instead of
            aggregating repeated calls.
        outfile: Explicit output path, or ``None`` to write to stdout
            (text) or a default file (html).

    """
    if fmt == "html":
        renderer = HTMLRenderer()
        output = renderer.render(session)

        if outfile is None:
            outfile = pathlib.Path("combined_profile.html")

        outfile.write_text(output)
        logger.info(f"  HTML report written to: {outfile}")

    elif fmt == "text":
        renderer = ConsoleRenderer(
            unicode=True,
            color=outfile is None,
            timeline=timeline,
        )
        output = renderer.render(session)

        if outfile is not None:
            outfile.write_text(output)
            logger.info(f"  Text report written to: {outfile}")
        else:
            sys.stdout.write(output)
            sys.stdout.write("\n")

    else:
        msg = f"Unknown format: {fmt!r}. Expected 'html' or 'text'."
        raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the merge CLI.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    """
    parser = argparse.ArgumentParser(
        description="Merge per-stage pyinstrument .pyisession files into a combined report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.merge_cpu_profiles ./profiles/cpu/\n"
            "  python -m benchmarks.merge_cpu_profiles ./profiles/cpu/ -o combined.html\n"
            "  python -m benchmarks.merge_cpu_profiles ./profiles/cpu/ --format text --timeline\n"
        ),
    )
    parser.add_argument(
        "directory",
        type=pathlib.Path,
        help="Directory containing .pyisession files to merge.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Output file path. Defaults to 'combined_profile.html' for HTML format, stdout for text.",
    )
    parser.add_argument(
        "--format",
        choices=["html", "text"],
        default="html",
        dest="fmt",
        help="Output format (default: html).",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        default=False,
        help="Preserve chronological order instead of aggregating repeated calls.",
    )
    parser.add_argument(
        "--save-session",
        action="store_true",
        default=False,
        help="Also save the combined .pyisession file for later analysis.",
    )

    args = parser.parse_args(argv)

    # ----------------------------------------------------------------
    # 1. Discover session files
    # ----------------------------------------------------------------
    logger.info(f"Scanning: {args.directory}")
    paths = _discover_sessions(args.directory)

    if not paths:
        logger.error(f"No .pyisession files found in {args.directory}")
        sys.exit(1)

    logger.info(f"Found {len(paths)} session file(s)")

    # ----------------------------------------------------------------
    # 2. Load and combine
    # ----------------------------------------------------------------
    combined = _load_and_combine(paths)
    logger.info(f"Combined {len(paths)} sessions successfully")

    # ----------------------------------------------------------------
    # 3. Optionally save combined session
    # ----------------------------------------------------------------
    if args.save_session:
        session_out = args.directory / "combined.pyisession"
        combined.save(str(session_out))
        logger.info(f"  Combined session saved to: {session_out}")

    # ----------------------------------------------------------------
    # 4. Render output
    # ----------------------------------------------------------------
    _render(
        combined,
        fmt=args.fmt,
        timeline=args.timeline,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
