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
r"""Shared CLI settings (S3, execution, model weights, profiling) for video pipelines.

Also provides :class:`CliArgSpec`, :func:`cli`, and :func:`add_settings_cli_arguments` for any
attrs settings class that stores ``metadata[\"cli\"]`` (same pattern as shard-specific settings).

Composite settings (``common`` + pipeline-specific fields) use :func:`composite_to_namespace` for
legacy code that expects a flat ``argparse.Namespace`` (e.g. ``profiling_scope``, ``run_pipeline``).
:func:`sync_common_from_namespace` and :func:`composite_profiling_scope` keep ``settings.common``
aligned when profiling mutates that namespace.
"""

import argparse
import contextlib
import json
import types
from collections.abc import Collection, Generator
from typing import TYPE_CHECKING, Any, Self, Union, cast, get_args, get_origin

import attrs
from attrs import NOTHING, validators

from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curator.core.utils.infra.metrics_push import DEFAULT_DROP_METRIC_NAMES_REGEX
from cosmos_curator.core.utils.infra.profiling import profiling_scope

if TYPE_CHECKING:
    AttrField = attrs.Attribute[Any]
else:
    AttrField = attrs.Attribute

_INFER_CLI_ARG_TYPE: Any = object()


@attrs.frozen(kw_only=True)
class CliArgSpec:
    """Describes one CLI flag derived from a settings field."""

    help: str
    flag: str | None = None
    required: bool = False
    default: Any = NOTHING
    choices: frozenset[str] | None = None
    arg_type: Any = _INFER_CLI_ARG_TYPE
    action: type[Any] | str | None = None


def cli(**kwargs: object) -> dict[str, CliArgSpec]:
    r"""Build ``attrs`` field metadata mapping ``\"cli\"`` → :class:`CliArgSpec`.

    Args:
        **kwargs: Forwarded to :class:`CliArgSpec` (``help``, ``default``, ``action``, etc.).
            Omit ``arg_type`` to infer ``argparse`` ``type=`` from the field annotation.

    """
    return {"cli": CliArgSpec(**cast("Any", kwargs))}


def _resolve_cli_arg_type(field: AttrField, spec: CliArgSpec) -> type[Any]:
    """Resolve the argparse type for a non-action flag from the field annotation or spec."""
    if spec.arg_type is not _INFER_CLI_ARG_TYPE:
        if spec.arg_type is None:
            msg = (
                f"CliArgSpec for {field.name}: non-action flag needs argparse type= "
                f"(omit arg_type to infer from the field annotation, or pass arg_type=int, etc.)"
            )
            raise ValueError(msg)
        return cast("type[Any]", spec.arg_type)

    ann: Any = field.type
    origin = get_origin(ann)
    if origin is Union or origin is types.UnionType:
        args = get_args(ann)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            ann = non_none[0]

    if ann is bool:
        msg = (
            f"CliArgSpec for {field.name}: bool fields must set metadata action= "
            f"(e.g. store_true or BooleanOptionalAction), not scalar type inference"
        )
        raise ValueError(msg)
    if ann is str or ann is int or ann is float:
        return cast("type[Any]", ann)
    msg = (
        f"CliArgSpec for {field.name}: cannot infer argparse type from annotation {field.type!r}; "
        f"pass arg_type= explicitly in cli(...)"
    )
    raise ValueError(msg)


def _finalize_action_argparse(field: AttrField, spec: CliArgSpec, options: dict[str, Any]) -> dict[str, Any]:
    r"""Finish ``add_argument`` kwargs for specs that set ``action=``.

    Args:
        field: attrs field (name used only in validation error messages).
        spec: Flag metadata for the action branch.
        options: Partial ``add_argument`` kwargs (e.g. ``help``); updated in-place and returned.

    """
    if spec.required:
        msg = f"CliArgSpec for {field.name}: action= and required=True is unsupported"
        raise ValueError(msg)
    if spec.default is NOTHING:
        msg = f"CliArgSpec for {field.name}: action= requires default="
        raise ValueError(msg)
    options["action"] = spec.action
    options["default"] = spec.default
    return options


def _finalize_required_argparse(
    field: AttrField,
    spec: CliArgSpec,
    options: dict[str, Any],
) -> dict[str, Any]:
    r"""Finish ``add_argument`` kwargs for a required typed flag (no ``action=``).

    Args:
        field: attrs field (annotation used when ``arg_type`` is inferred).
        spec: Flag metadata (``required=True``).
        options: Partial ``add_argument`` kwargs; updated in-place and returned.

    """
    options["required"] = True
    options["type"] = _resolve_cli_arg_type(field, spec)
    return options


def _finalize_optional_argparse(
    field: AttrField,
    spec: CliArgSpec,
    options: dict[str, Any],
) -> dict[str, Any]:
    r"""Finish ``add_argument`` kwargs for an optional typed flag (``type=`` and ``default=``).

    Args:
        field: attrs field (annotation used when ``arg_type`` is inferred).
        spec: Flag metadata for a non-action optional argument.
        options: Partial ``add_argument`` kwargs; updated in-place and returned.

    """
    if spec.default is NOTHING:
        msg = f"CliArgSpec for {field.name}: optional field needs default= or use required=True"
        raise ValueError(msg)
    options["type"] = _resolve_cli_arg_type(field, spec)
    options["default"] = spec.default
    return options


def _add_argument_options_from_cli_spec(field: AttrField, spec: CliArgSpec) -> dict[str, Any]:
    r"""Build keyword arguments for ``parser.add_argument`` from a :class:`CliArgSpec`.

    Args:
        field: attrs field (``type`` hint used when ``spec.arg_type`` is omitted).
        spec: Parsed or frozen spec describing this CLI flag.

    Returns:
        Keyword dict to pass to ``add_argument``.

    """
    options: dict[str, Any] = {"help": spec.help}
    if spec.choices is not None:
        options["choices"] = sorted(spec.choices)
    if spec.action is not None:
        return _finalize_action_argparse(field, spec, options)
    if spec.required:
        return _finalize_required_argparse(field, spec, options)
    return _finalize_optional_argparse(field, spec, options)


def add_settings_cli_arguments[T: attrs.AttrsInstance](
    parser: argparse.ArgumentParser,
    settings_cls: type[T],
    *,
    only_fields: Collection[str] | None = None,
) -> None:
    r"""Register *parser* arguments from *settings_cls* fields that define ``metadata[\"cli\"]``.

    Args:
        parser: ``ArgumentParser`` to register each derived flag on.
        settings_cls: attrs class whose fields may define ``metadata["cli"]`` as :class:`CliArgSpec`.
        only_fields: When not ``None``, only register flags whose field name is in this collection.

    """
    for field in attrs.fields(settings_cls):
        if only_fields is not None and field.name not in only_fields:
            continue
        raw = field.metadata.get("cli")
        if raw is None:
            continue
        spec = raw if isinstance(raw, CliArgSpec) else CliArgSpec(**cast("Any", raw))
        flag = spec.flag or f"--{field.name.replace('_', '-')}"
        parser.add_argument(flag, dest=field.name, **_add_argument_options_from_cli_spec(field, spec))


_EXECUTION_MODES = frozenset(("AUTO", "BATCH", "STREAMING"))
_STREAMING_SCHEDULERS = frozenset(("FRAGMENTATION_BASED", "SATURATION_AWARE"))

# Dest names registered by :func:`add_profiling_args` only (subset of this class).
# Note: ``otlp_metrics_push*`` fields are intentionally NOT included here -- in-pipeline
# OTLP metrics scraping has negligible overhead and is not a sampled profiling backend.
# Those flags are still registered by :func:`add_common_args`, which exposes the full
# settings surface.
PROFILING_CLI_FIELDS: frozenset[str] = frozenset(
    {
        "perf_profile",
        "profile_tracing",
        "profile_tracing_sampling",
        "profile_tracing_otlp_endpoint",
        "otlp_run_attributes",
        "otlp_run_attributes_map",
        "profile_cpu",
        "profile_memory",
        "profile_gpu",
        "profile_cpu_exclude",
        "profile_memory_exclude",
        "profile_gpu_exclude",
    },
)


@attrs.define
class CommonPipelineSettings:
    """Settings mirrored by :func:`~cosmos_curator.pipelines.pipeline_args.add_common_args`."""

    input_s3_profile_name: str = attrs.field(
        validator=validators.min_len(1),
        metadata=cli(help="S3 profile name to use for input S3 path.", default="default"),
    )
    output_s3_profile_name: str = attrs.field(
        validator=validators.min_len(1),
        metadata=cli(help="S3 profile name to use for output S3 path.", default="default"),
    )
    execution_mode: str = attrs.field(
        validator=validators.in_(_EXECUTION_MODES),
        metadata=cli(
            help=(
                "Execution mode of Cosmos-Curator pipeline. AUTO picks STREAMING when the number of "
                "GPUs requested by stages is at most the number of available GPUs, else BATCH; "
                "STREAMING and BATCH force the respective mode."
            ),
            default="AUTO",
            choices=_EXECUTION_MODES,
        ),
    )
    xenna_streaming_scheduler: str = attrs.field(
        validator=validators.in_(_STREAMING_SCHEDULERS),
        metadata=cli(
            help=(
                "Xenna streaming-mode autoscaler implementation. "
                "FRAGMENTATION_BASED (default) uses the Rust-backed FragmentationBasedAutoscaler; "
                "SATURATION_AWARE uses the pure-Python saturation-aware scheduler. "
                "Ignored in BATCH execution mode."
            ),
            default="FRAGMENTATION_BASED",
            choices=_STREAMING_SCHEDULERS,
        ),
    )
    limit: int = attrs.field(
        validator=validators.ge(0),
        metadata=cli(help="Limit number of input videos to process.", default=0),
    )
    verbose: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help="Whether to print verbose logs.",
            default=False,
            arg_type=None,
            action="store_true",
        ),
    )
    model_weights_path: str = attrs.field(
        validator=validators.min_len(1),
        metadata=cli(
            help=(
                "Local path or S3 prefix for model weights. "
                "Used to download model weights to local cache if they are not already present. "
                "If a unix path is provided, it must be accessible from all nodes."
            ),
            default=MODEL_WEIGHTS_PREFIX,
        ),
    )
    perf_profile: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help="Enable lightweight basic performance profiling (use --no-perf-profile to disable).",
            default=True,
            arg_type=None,
            action=argparse.BooleanOptionalAction,
        ),
    )
    profile_tracing: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help=(
                "Enable distributed tracing (OpenTelemetry) via Ray's tracing hook. "
                "Captures cross-actor spans (task scheduling, actor creation, method "
                "invocations) as NDJSON files in <output-path>/profile/traces/. "
                "Implies --perf-profile. Can also be enabled for a whole deployment "
                "by setting COSMOS_CURATOR_PROFILE_TRACING=1; set it to 0 to opt a "
                "single run out of such a default. "
                "Note: should be set to True by default once Xenna adds proper "
                "tracing support."
            ),
            default=False,
            arg_type=None,
            action="store_true",
        ),
    )
    profile_tracing_sampling: float = attrs.field(
        validator=validators.and_(validators.ge(0.0), validators.le(1.0)),
        metadata=cli(
            help=(
                "Trace sampling rate when --profile-tracing is enabled. "
                "Value between 0.0 (none) and 1.0 (all). Default: 1.0 (all). "
                "The decision is made once at the trace root and inherited by "
                "every span in the run, so a value below 1.0 drops whole runs "
                "rather than thinning spans within a run. "
                "Controls both cosmos-curator and vLLM native span sampling."
            ),
            default=1.0,
        ),
    )
    profile_tracing_otlp_endpoint: str = attrs.field(
        validator=validators.instance_of(str),
        metadata=cli(
            help=(
                "OTLP HTTP collector endpoint for remote span export "
                "(e.g. http://localhost:4318). Empty (default) disables OTLP -- "
                "spans are only written to local .jsonl files. "
                "Can also be set via OTEL_EXPORTER_OTLP_ENDPOINT or "
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (takes precedence) env vars."
            ),
            default="",
        ),
    )
    otlp_metrics_push: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help=(
                "Enable the in-pipeline OTLP metrics scraper. A background thread on the "
                "driver polls each Ray node's Prometheus /metrics endpoint on a configurable "
                "interval and pushes the data as OTLP HTTP metrics (NOT Prometheus "
                "remote-write). Requires --otlp-metrics-push-endpoint (or "
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT / OTEL_EXPORTER_OTLP_ENDPOINT). "
                "Negligible runtime overhead -- not a sampled profiling backend."
            ),
            default=False,
            arg_type=None,
            action="store_true",
        ),
    )
    otlp_metrics_push_endpoint: str = attrs.field(
        validator=validators.instance_of(str),
        metadata=cli(
            help=(
                "OTLP HTTP collector endpoint for the in-pipeline metrics scraper "
                "(e.g. http://localhost:4318). Empty (default) falls back to "
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, then OTEL_EXPORTER_OTLP_ENDPOINT. "
                "When none is set, --otlp-metrics-push is a no-op."
            ),
            default="",
        ),
    )
    otlp_metrics_push_interval: int = attrs.field(
        validator=validators.and_(validators.instance_of(int), validators.ge(1)),
        metadata=cli(
            help=("Scrape + push interval in seconds when --otlp-metrics-push is enabled. Default: 30."),
            default=30,
        ),
    )
    otlp_metrics_push_drop_regex: str = attrs.field(
        validator=validators.instance_of(str),
        metadata=cli(
            help=(
                "Regex matched against each OTel metric name (after the `_total` suffix is "
                "stripped from counters). Matching metrics are dropped before export to "
                "reduce backend cardinality. Default mirrors the helm chart's "
                "`metric_relabel_configs` drop list (Ray scheduler/GCS/pull-manager noise). "
                "Pass an empty string to disable filtering."
            ),
            default=DEFAULT_DROP_METRIC_NAMES_REGEX,
        ),
    )
    otlp_run_attributes: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help=(
                "Attach run metadata (user, Slurm job id, host, etc.) to OTLP traces and "
                "in-pipeline metrics push. Use --no-otlp-run-attributes to omit."
            ),
            default=True,
            arg_type=None,
            action=argparse.BooleanOptionalAction,
        ),
    )
    profile_cpu: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help=(
                "Enable CPU profiling (pyinstrument) for every pipeline stage. "
                "Saves per-task HTML flame-tree reports to <output-path>/profile/cpu/. "
                "Implies --perf-profile."
            ),
            default=False,
            arg_type=None,
            action="store_true",
        ),
    )
    profile_memory: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help=(
                "Enable memory profiling (memray) for every pipeline stage. "
                "Saves per-task .bin captures and HTML flamegraphs to "
                "<output-path>/profile/memory/. Implies --perf-profile."
            ),
            default=False,
            arg_type=None,
            action="store_true",
        ),
    )
    profile_gpu: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help=(
                "Enable GPU profiling (torch.profiler) for every pipeline stage. "
                "Captures CUDA kernel launches, operator breakdown, and GPU "
                "memory allocations.  Saves per-task Chrome Trace JSON to "
                "<output-path>/profile/gpu/.  Silently disabled on CPU-only workers. "
                "Implies --perf-profile."
            ),
            default=False,
            arg_type=None,
            action="store_true",
        ),
    )
    profile_cpu_exclude: str = attrs.field(
        validator=validators.instance_of(str),
        metadata=cli(
            help=(
                "Comma-separated list of scope names to exclude from CPU profiling. "
                "Scope names are stage class names (e.g. VideoDownloader, ClipWriterStage) "
                "or '_root' for the driver process."
            ),
            default="_root",
        ),
    )
    profile_memory_exclude: str = attrs.field(
        validator=validators.instance_of(str),
        metadata=cli(
            help=(
                "Comma-separated list of scope names to exclude from memory profiling. "
                "Scope names are stage class names (e.g. VideoDownloader, ClipWriterStage) "
                "or '_root' for the driver process. "
                "Note: memray may conflict with pyinstrument on long-lived driver processes; "
                "pass '_root' to avoid this."
            ),
            default="_root",
        ),
    )
    profile_gpu_exclude: str = attrs.field(
        validator=validators.instance_of(str),
        metadata=cli(
            help=(
                "Comma-separated list of scope names to exclude from GPU profiling. "
                "Scope names are stage class names (e.g. VideoDownloader, ClipWriterStage) "
                "or '_root' for the driver process. "
                "Default: '_root' (driver process typically has no CUDA context)."
            ),
            default="_root",
        ),
    )
    otlp_run_attributes_map: dict[str, str] = attrs.field(
        factory=dict,
        validator=validators.deep_mapping(
            validators.instance_of(str),
            validators.instance_of(str),
            validators.instance_of(dict),
        ),
        metadata=cli(
            help=(
                "Optional JSON object of literal run-attribute labels to attach to OTLP traces "
                'and metrics push (e.g. \'{"customer":"nvidia","nspect_id":"..."}\').'
            ),
            default={},
            arg_type=json.loads,
        ),
    )

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> Self:
        """Build from parsed CLI namespace (``dest`` names match field names)."""
        kwargs = {f.name: getattr(ns, f.name) for f in attrs.fields(cls)}
        return cls(**kwargs)


def composite_from_namespace[T: attrs.AttrsInstance](settings_cls: type[T], ns: argparse.Namespace) -> T:
    """Build a composite settings type that has common and local fields from a flat CLI namespace.

    Pipeline-specific fields are read from *ns* by name (same ``dest`` as attrs fields); *common* is filled via
    :meth:`CommonPipelineSettings.from_namespace`.
    """
    common = CommonPipelineSettings.from_namespace(ns)
    local = {f.name: getattr(ns, f.name) for f in attrs.fields(settings_cls) if f.name != "common"}
    return settings_cls(common=common, **local)  # type: ignore[call-arg]


def composite_to_namespace(settings: attrs.AttrsInstance) -> argparse.Namespace:
    """Flatten composite settings (``common`` + pipeline-specific fields) for profiling."""
    common = getattr(settings, "common", None)
    if not isinstance(common, CommonPipelineSettings):
        msg = f"{type(settings).__name__!r} must have attribute 'common' of type CommonPipelineSettings"
        raise TypeError(msg)
    flat: dict[str, Any] = {f.name: getattr(common, f.name) for f in attrs.fields(CommonPipelineSettings)}
    for f in attrs.fields(type(settings)):
        if f.name == "common":
            continue
        flat[f.name] = getattr(settings, f.name)
    return argparse.Namespace(**flat)


def sync_common_from_namespace(settings: attrs.AttrsInstance, ns: argparse.Namespace) -> None:
    """Copy every :class:`CommonPipelineSettings` field from *ns* into ``settings.common``.

    ``profiling_scope`` and ``run_pipeline`` (via ``_apply_profiling_config``) may mutate *ns*
    — notably setting ``perf_profile`` when a profiling backend implies it. The nested
    ``settings.common`` object is not the same as *ns*, so call this after mutations to keep
    pipeline code and summaries consistent.
    """
    common = getattr(settings, "common", None)
    if not isinstance(common, CommonPipelineSettings):
        msg = f"{type(settings).__name__!r} must have attribute 'common' of type CommonPipelineSettings"
        raise TypeError(msg)
    updates = {f.name: getattr(ns, f.name) for f in attrs.fields(CommonPipelineSettings)}
    object.__setattr__(settings, "common", attrs.evolve(common, **updates))


@contextlib.contextmanager
def composite_profiling_scope(
    settings: attrs.AttrsInstance,
    *,
    stage_name: str = "_root",
    label: str = "main",
) -> Generator[argparse.Namespace]:
    """Enter :func:`~cosmos_curator.core.utils.infra.profiling.profiling_scope` with a flat namespace from *settings*.

    Builds the flat namespace via :func:`composite_to_namespace`, then syncs ``settings.common``
    after profiling applies CLI side effects (e.g. implied ``perf_profile``). Yields the same
    namespace to pass to ``run_pipeline(..., args=...)`` and :func:`sync_common_from_namespace`
    after ``run_pipeline`` returns.
    """
    ns = composite_to_namespace(settings)
    with profiling_scope(ns, stage_name=stage_name, label=label):
        sync_common_from_namespace(settings, ns)
        yield ns
