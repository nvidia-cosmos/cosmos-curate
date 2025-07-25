# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""View and display NVCF clip data."""

import contextlib
import json
import logging
import shutil
import tempfile
import zipfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Annotated, ClassVar

import typer
from typer import Context, Option
from typing_extensions import override

from cosmos_curate.client.nvcf_cli.ncf.common import (
    NvcfBase,
    base_callback,
    register_instance,
    validate_address,
    validate_in,
)


class HttpServerHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler for serving clips and metadata.

    This handler serves files from two separate directories:
    - web_dir: Contains the web UI files (index.html, etc.)
    - base_dir: Contains the actual clip data (clips/ and metas/v0/)
    """

    web_dir: ClassVar[str] = ""
    base_dir: ClassVar[str] = ""
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    @classmethod
    def initialize(cls, web_dir: str, base_dir: str, logger: logging.Logger) -> type["HttpServerHandler"]:
        """Initialize the class variables for the handler.

        Args:
            web_dir: Directory containing web files (index.html, etc.)
            base_dir: Directory containing clips and metas
            logger: Logger instance

        Returns:
            The initialized handler class

        """
        cls.web_dir = web_dir
        cls.base_dir = base_dir
        cls.logger = logger
        return cls

    def end_headers(self) -> None:
        """Add CORS and cache control headers before ending the headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    @staticmethod
    def get_clip_time(clip_name: str, base_dir: str, logger: logging.Logger) -> float:
        """Get start time for sorting clips.

        Args:
            clip_name: Name of the clip
            base_dir: Directory containing metas
            logger: Logger instance

        Returns:
            Start time as a float, or inf if not found

        """
        meta_path = Path(base_dir) / "metas/v0" / f"{clip_name}.json"
        try:
            if meta_path.exists():
                with Path(meta_path).open() as f:
                    metadata = json.load(f)
                    return float(metadata.get("duration_span", [float("inf"), float("inf")])[0])
            logger.warning("No metadata file found for %s", clip_name)
        except (FileNotFoundError, PermissionError):
            logger.exception("Cannot open metadata file '%s': ", meta_path)
        except json.JSONDecodeError:
            logger.exception("Invalid JSON in metadata file '%s': ", meta_path)
        except (ValueError, TypeError):
            logger.exception("Invalid 'duration_span' format in metadata file '%s': ", meta_path)
        return float("inf")

    @override
    def translate_path(self, path: str) -> str:
        """Override translate_path to handle both web files and clips/metas directories.

        Args:
            path: The requested path

        Returns:
            The absolute path to serve

        """
        # If path starts with /, strip it
        if path[0] == "/":
            path = path[1:]
        # If path starts with /clips or /metas, serve from base_dir
        # Otherwise serve from web_dir (where index.html is)
        abs_path = Path(self.base_dir) / path if path.startswith(("clips", "metas")) else Path(self.web_dir) / path
        self.logger.info(abs_path)
        return str(abs_path)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests for clips, metadata, and web files."""
        try:
            if self.path == "/list_clips":
                # Construct the clips_dir from base_dir
                clips_dir = Path(self.base_dir) / "clips"

                self.logger.info("Processing request for /list_clips")
                self.logger.debug("Looking for clips in: %s", clips_dir)

                # Check if directory exists
                if not clips_dir.exists():
                    self.logger.error("Clips directory '%s' does not exist", clips_dir)
                    clip_names = []
                else:
                    # Get clip names without extension
                    clip_names = [f.stem for f in Path(clips_dir).rglob("*") if f.suffix.lower() == ".mp4"]
                    self.logger.info("Found %d clips", len(clip_names))

                    # Sort clips by their start time
                    clip_names.sort(key=lambda x: self.get_clip_time(x, self.base_dir, self.logger))
                    self.logger.info("Sorted clips by start time")

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()

                self.wfile.write(json.dumps(clip_names).encode("utf-8"))
            else:
                try:
                    super().do_GET()
                except ConnectionResetError:
                    self.logger.debug("Client disconnected during file transfer")
                except BrokenPipeError:
                    self.logger.debug("Broken pipe during file transfer")
                except ConnectionAbortedError:
                    self.logger.debug("Connection aborted during file transfer")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            self.logger.exception("Error handling GET request: ")
            # If we can't even send the error, just ignore it
            with contextlib.suppress(ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                self.send_error(500, f"Internal server error: {e}")


class ClipViewer(NvcfBase):
    """Viewer for NVCF clip data.

    Provides methods to serve and display video clips and metadata using a local HTTP server.
    """

    def __init__(self, url: str, nvcf_url: str, key: str, org: str, team: str, timeout: int) -> None:  # noqa: PLR0913
        """Initialize the ClipViewer.

        Args:
            url: Base NGC URL
            nvcf_url: Base NVCF URL
            key: NGC NVCF API Key
            org: Organization ID or name
            team: Team name within the organization
            timeout: Request timeout in seconds

        """
        super().__init__(url=url, nvcf_url=nvcf_url, key=key, org=org, team=team, timeout=timeout)
        self.clip_path: str | None = None

    def __del__(self) -> None:
        """Clean up temporary clip directory on deletion."""
        if self.clip_path is not None and Path(self.clip_path).exists():
            shutil.rmtree(self.clip_path)

    def hdlr_factory(
        self,
        web_dir: str,
        base_dir: str,
        logger: logging.Logger,
    ) -> type[SimpleHTTPRequestHandler]:
        """Create a custom HTTP request handler for serving clips and metadata.

        Args:
            web_dir: Directory containing web files (index.html, etc.)
            base_dir: Directory containing clips and metas
            logger: Logger instance

        Returns:
            A subclass of SimpleHTTPRequestHandler

        """
        return HttpServerHandler.initialize(web_dir, base_dir, logger)

    def serve(self, web_dir: str, base_dir: str, ip: str, port: int, *, no_clip_path: bool) -> None:
        """Start the HTTP server to serve clips and metadata.

        Args:
            web_dir: Directory containing web files
            base_dir: Directory containing clips and metas
            ip: IP address to bind the server
            port: Port to run the server on
            no_clip_path: If True, do not log clip path info

        """
        try:
            # Create server with both web_dir and base_dir attributes
            # web_dir: Directory containing index.html and other web files
            # base_dir: Directory containing clips and metas
            self.hdlr = self.hdlr_factory(web_dir, base_dir, self.logger)
            self.server = HTTPServer((ip, port), self.hdlr)

            self.logger.info("Starting server on http://%s:%d", ip, port)
            self.logger.info("Please connect your browser to the above URL to view the clips and captions")
            if not no_clip_path:
                self.logger.info("Base directory: %s", base_dir)
                self.logger.info("Clips directory: %s", Path(base_dir) / "clips")
                self.logger.info("Metadata directory: %s", Path(base_dir) / "metas/v0")

            self.server.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("Shutting down server...")
            self.server.server_close()
        except Exception:
            self.logger.exception("Server error: ")


clip_viewer = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    callback=base_callback,
)
ins_name = "view"
ins_help = "Browser based Local Clip Viewer"
register_instance(ins_name=ins_name, ins_help=ins_help, ins_type=ClipViewer, ins_app=clip_viewer)


@clip_viewer.command(
    name="clips",
    help="View Clips downloaded from a invoke-function",
    no_args_is_help=True,
)
def nvcf_view_clip(
    ctx: Context,
    ip: Annotated[
        str,
        Option(
            help="IP to which the server binds to",
            rich_help_panel="View",
            callback=validate_address,
        ),
    ] = "localhost",
    port: Annotated[
        int,
        Option(
            help="Port to run the server on",
            rich_help_panel="View",
            callback=validate_in(range(65536)),
        ),
    ] = 8080,
    clip_path: Annotated[
        Path | None,
        Option(
            help="Path to directory containing clips/ and metas/v0/ subdirectories",
            rich_help_panel="View",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    zip_file: Annotated[
        Path | None,
        Option(
            help="Downloaded Zip file name",
            rich_help_panel="View",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Start a local web server to view video clips and their metadata.

    Args:
        ctx: Typer context containing the NVCF handler.
        ip: IP address to bind the server.
        port: Port number to run the server on.
        clip_path: Path to directory containing clips and metadata.
        zip_file: Path to a downloaded zip file containing clips and metadata.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    if clip_path is None and zip_file is None:
        nvcf_hdl.logger.error("Either --clip-path or --zip-file must be provided")
        return

    if clip_path is not None and zip_file is not None:
        nvcf_hdl.logger.error("Only one of --clip-path or --zip-file should be provided")
        return

    if zip_file is not None:
        # Create a temporary directory to extract the zip file
        nvcf_hdl.clip_path = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(nvcf_hdl.clip_path)
            nvcf_hdl.logger.info("Extracted zip file to %s", nvcf_hdl.clip_path)
        except (FileNotFoundError, zipfile.BadZipFile, PermissionError, OSError):
            nvcf_hdl.logger.exception("Failed to extract zip file: ")
            return
        base_dir = nvcf_hdl.clip_path
    else:
        base_dir = str(clip_path)

    # Get the web directory from the package
    web_dir = Path(__file__).parent / "webdocs"
    if not web_dir.exists():
        nvcf_hdl.logger.error("Web directory not found: %s", web_dir)
        return

    nvcf_hdl.serve(web_dir=str(web_dir), base_dir=base_dir, ip=ip, port=port, no_clip_path=zip_file is not None)


if __name__ == "__main__":
    clip_viewer()
