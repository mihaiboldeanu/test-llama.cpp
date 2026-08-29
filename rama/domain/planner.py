"""Launch planner — pure function: profile + config → argv.

This module is intentionally free of subprocess, PID files, and network calls.
It produces a LaunchPlan that ProcessRunner executes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .models import FamilyCapabilities, ModelInfo, SamplerSettings


@dataclass(frozen=True)
class LaunchPlan:
    """Immutable launch plan — every argv value is traceable to a profile field.

    Produced by LaunchPlanner (pure). Executed by ProcessRunner (side-effects).
    """

    argv: list[str]
    server_bin: Path
    port: int | None  # None means "pick at execution time"
    host: str
    model: ModelInfo
    backend: str
    ctx_size: int
    seed: int | None
    health_url: str
    profile: object  # ResolvedModelProfile — forward ref avoids cycle


@dataclass
class PlannerInput:
    """All inputs needed to build a LaunchPlan."""

    model: ModelInfo
    backend: str = "llama.cpp"
    server_bin: Path | None = None
    port: int | None = None
    host: str = "127.0.0.1"

    # Context
    ctx_size: int | None = None
    preferred_ctx: int | None = None
    ctk: str | None = None
    ctv: str | None = None

    # Sampler (from profile)
    sampler: SamplerSettings | None = None

    # Memory (from profile)
    fit_target_mib: int = 128
    n_gpu_layers: int | None = None
    use_mmap: bool = True

    # Capabilities (from profile)
    capabilities: FamilyCapabilities | None = None

    # Launch options
    seed: int | None = None
    skip_chat_parsing: bool = False
    log_disable: bool = False
    foreground: bool = False
    context_shift: bool = False

    # Chat template
    chat_template: str | None = None
    chat_template_file: str | None = None

    # Reasoning
    reasoning: bool | None = None
    reasoning_budget: int = 1024
    reasoning_msg: str = "I have thought enough, let's move to answering."
    reasoning_template_key: str = "enable_thinking"
    reasoning_template_value: object = None

    # Speculative decoding
    draft_model: str | None = None
    draft_max: int = 16
    draft_min: int = 8
    spec_ngram: bool = False
    spec_type: str | None = None
    spec_ngram_mod_size_n: int | None = None
    spec_ngram_mod_n_min: int | None = None
    spec_ngram_mod_n_max: int | None = None

    # MTP
    mtp: bool = False
    mtp_n_max: int = 3
    mtp_n_min: int = 0
    mtp_ctx_ratio: float = 0.8
    mtp_ngl_threshold_gb: float = 20.0

    # Launch settings (from resolve chain)
    cpu_moe: bool = False
    n_cpu_moe: int | None = None
    alias: str | None = None
    batch_size: int = 2048
    ubatch_size: int = 2048
    threads: int = 8
    threads_batch: int = 16
    n_predict: int = 2048
    samplers: str | None = None
    no_mmproj: bool = False
    cache_ram: int | None = None
    chat_template_kwargs: str | None = None

    # Fit mode
    fit_enabled: bool = True
    fit_ctx_enabled: bool = True

    # Speculative decoding thresholds
    small_threshold_gb: float = 10.0

    # Turboquant spec ngram overrides
    draft_min_val: int | None = None
    draft_max_val: int | None = None


def plan(
    inp: PlannerInput,
) -> list[str]:
    """Build argv from planner input. Pure function — no side effects."""
    model = inp.model
    args: list[str] = []

    # Server binary and model path
    args.extend([str(inp.server_bin), "-m", str(model.path)])

    # Host and port (port may be None for dry-run)
    port_str = str(inp.port) if inp.port is not None else "0"
    args.extend(["--host", inp.host, "--port", port_str])

    # Alias
    args.extend(["--alias", inp.alias or model.name])

    # Parallel and context
    args.extend(["-np", "1"])

    # Context size (already resolved by caller)
    ctx = inp.ctx_size or inp.preferred_ctx or 0
    if inp.mtp:
        ctx = int(ctx * inp.mtp_ctx_ratio)
    args.extend(["--ctx-size", str(ctx)])

    # KV cache types
    qk_type = inp.ctk or "q8_0"
    qkv_type = inp.ctv or "q8_0"
    args.extend(["-ctk", qk_type, "-ctv", qkv_type])

    # Flash attention and batch sizes
    args.extend(["--flash-attn", "on"])
    bs = inp.batch_size
    ubs = inp.ubatch_size
    if inp.mtp:
        bs = min(bs, 2048)
        ubs = min(ubs, 512)
    args.extend(["-b", str(bs), "-ub", str(ubs)])

    # Threads
    args.extend(
        ["--threads", str(inp.threads), "--threads-batch", str(inp.threads_batch)]
    )

    # Sampler settings
    if inp.sampler is not None:
        s = inp.sampler
        args.extend(
            [
                "--temp",
                str(s.temperature),
                "--top-k",
                str(s.top_k),
                "--top-p",
                str(s.top_p),
                "--min-p",
                str(s.min_p),
                "--repeat-penalty",
                str(s.repeat_penalty),
                "--presence-penalty",
                str(s.presence_penalty),
            ]
        )

    args.extend(["--perf", "--kv-unified"])

    # Logging
    if inp.log_disable:
        args.append("--log-disable")

    # Chat parsing
    if inp.skip_chat_parsing:
        args.append("--skip-chat-parsing")

    # GPU layers
    if inp.n_gpu_layers is not None:
        args.extend(["-ngl", str(inp.n_gpu_layers)])

    # mmap
    no_mmap = model.is_moe or (not inp.use_mmap)
    if no_mmap:
        args.append("--no-mmap")

    # Chat template — skip for MoE with avoid_fit_ctx_for_moe
    skip_chat_template_moe = (
        model.is_moe
        and inp.capabilities is not None
        and inp.capabilities.avoid_fit_ctx_for_moe
    )
    use_chat_template = not skip_chat_template_moe
    if use_chat_template and inp.chat_template:
        args.extend(["--chat-template", str(inp.chat_template)])
    if use_chat_template and inp.chat_template_file:
        args.extend(["--chat-template-file", str(inp.chat_template_file)])

    # Cache RAM
    if inp.cache_ram is not None:
        args.extend(["--cache-ram", str(inp.cache_ram)])

    # Seed
    if inp.seed is not None:
        args.extend(["-s", str(inp.seed)])

    # Context shift
    if not inp.context_shift:
        args.append("--no-context-shift")

    # Fit mode
    skip_chat_template_moe = (
        model.is_moe
        and inp.capabilities is not None
        and inp.capabilities.avoid_fit_ctx_for_moe
    )

    if inp.mtp:
        model_fits_gpu = model.size_gb < inp.mtp_ngl_threshold_gb
        if model_fits_gpu:
            args.extend(["-ngl", "-1"])
            fit_args = ["--fit", "off"]
        else:
            fit_args = [
                "--fit",
                "on",
                "--fit-target",
                str(inp.fit_target_mib),
                "--fit-ctx",
                str(ctx),
            ]
    elif skip_chat_template_moe:
        fit_args = ["--fit", "on", "--fit-target", str(inp.fit_target_mib)]
    else:
        fit_args = [
            "--fit",
            "on" if inp.fit_enabled else "off",
            "--fit-target",
            str(inp.fit_target_mib),
        ]
        if inp.fit_ctx_enabled:
            fit_args.extend(["--fit-ctx", str(ctx)])
    args.extend(fit_args)

    # Draft model (speculative decoding)
    if inp.draft_model:
        draft_spec_type = (
            inp.spec_type
            if inp.spec_type
            in {"draft-simple", "draft-eagle3", "draft-dflash", "draft-dspark"}
            else "draft-simple"
        )
        args.extend(
            [
                "--model-draft",
                inp.draft_model,
                "--spec-type",
                draft_spec_type,
                "--spec-draft-n-max",
                str(inp.draft_max),
                "--spec-draft-n-min",
                str(inp.draft_min),
            ]
        )

    # MTP
    if inp.mtp:
        mtp_args = [
            "--cache-type-k-draft",
            "q8_0",
            "--cache-type-v-draft",
            "q8_0",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-p-min",
            "0.75",
            "--spec-draft-n-max",
            str(inp.mtp_n_max),
        ]
        if inp.mtp_n_min > 0:
            mtp_args.extend(["--spec-draft-n-min", str(inp.mtp_n_min)])
        args.extend(mtp_args)

    # CPU MoE
    if inp.cpu_moe:
        args.append("--cpu-moe")
    elif inp.n_cpu_moe is not None:
        args.extend(["--n-cpu-moe", str(inp.n_cpu_moe)])

    # Speculative ngram decoding
    is_small = model.size_gb < inp.small_threshold_gb
    spec_allowed_family = (
        inp.capabilities is not None and inp.capabilities.supports_ngram_speculation
    )
    spec_explicit = inp.spec_type is not None
    if (
        inp.spec_ngram
        and inp.spec_type is not None
        and not inp.draft_model
        and not is_small
        and (spec_allowed_family or spec_explicit)
    ):
        spec_args = ["--spec-type", str(inp.spec_type)]

        if inp.spec_type == "ngram-mod":
            if inp.backend == "turboquant":
                if inp.spec_ngram_mod_size_n is not None:
                    spec_args.extend(
                        ["--spec-ngram-size-n", str(inp.spec_ngram_mod_size_n)]
                    )
                if inp.draft_min_val is not None:
                    spec_args.extend(["--draft-min", str(inp.draft_min_val)])
                if inp.draft_max_val is not None:
                    spec_args.extend(["--draft-max", str(inp.draft_max_val)])
            else:
                if inp.spec_ngram_mod_size_n is not None:
                    spec_args.extend(
                        ["--spec-ngram-mod-n-match", str(inp.spec_ngram_mod_size_n)]
                    )
                if inp.spec_ngram_mod_n_min is not None:
                    spec_args.extend(
                        ["--spec-ngram-mod-n-min", str(inp.spec_ngram_mod_n_min)]
                    )
                if inp.spec_ngram_mod_n_max is not None:
                    spec_args.extend(
                        ["--spec-ngram-mod-n-max", str(inp.spec_ngram_mod_n_max)]
                    )

        args.extend(spec_args)

    # Reasoning (always present)
    enable_reasoning = inp.reasoning if inp.reasoning is not None else True
    args.extend(
        [
            "-n",
            str(inp.n_predict),
            "--reasoning",
            "on" if enable_reasoning else "off",
            "--reasoning-budget",
            str(inp.reasoning_budget),
            "--reasoning-budget-message",
            inp.reasoning_msg,
        ]
    )

    # Family-specific reasoning additions (declarative capabilities)
    caps = inp.capabilities
    if caps is not None:
        if caps.reasoning_preserve_thinking_ctk:
            ctk_args = {"preserve_thinking": True}
            if inp.chat_template_kwargs:
                try:
                    existing = json.loads(inp.chat_template_kwargs)
                    ctk_args.update(existing)
                except (json.JSONDecodeError, TypeError):
                    ctk_args = {"preserve_thinking": True}
            args.extend(["--chat-template-kwargs", json.dumps(ctk_args)])

        if caps.reasoning_use_no_mmproj and inp.no_mmproj:
            args.append("--no-mmproj")
        if caps.reasoning_use_samplers and inp.samplers:
            args.extend(["--samplers", inp.samplers])
        if caps.reasoning_use_preserve_flag and enable_reasoning:
            args.append("--reasoning-preserve")

    # Custom reasoning template key
    if inp.reasoning_template_key != "enable_thinking":
        args.extend(
            [
                "--chat-template-kwargs",
                json.dumps({inp.reasoning_template_key: inp.reasoning_template_value}),
            ]
        )

    return args
