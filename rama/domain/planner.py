"""Launch planner — pure function: profile + runtime inputs → argv.

The resolver owns every merge decision.  This module only emits the merged
flag map in a stable order and applies the documented runtime blocks: MTP,
MoE ``--load-mode none``, automatic placeholders, and the conditional n-gram
spec block.  It performs no subprocess, PID-file, network, or model-dir search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .resolver import DROP, _is_auto, _parse_bool_word, _parse_float, _parse_int

if TYPE_CHECKING:
    from .models import ModelInfo, ResolvedProfile

_ALLOWED_DRAFT_SPEC_TYPES = {
    "draft-simple",
    "draft-eagle3",
    "draft-dflash",
    "draft-dspark",
}


@dataclass(frozen=True)
class PlanRequest:
    """Runtime inputs needed alongside an already-resolved profile."""

    server_bin: Path | None
    model: ModelInfo
    profile: ResolvedProfile
    host: str = "127.0.0.1"
    port: int | None = None
    backend: str = "llama.cpp"
    mtp: bool = False
    mtp_n_max: int | None = None
    mtp_n_min: int | None = None
    mtp_draft_model: str | None = None
    template_dir: str | Path | None = None
    small_threshold_gb: float = 10.0


def _emit(args: list[str], flag: str, value: Any) -> None:
    if value is None or value is DROP:
        return
    args.append(flag)
    if value is not True:
        args.append(str(value))


def _pop_flag(values: dict[str, Any], flag: str, default: Any = None) -> Any:
    value = values.pop(flag, default)
    return None if value is DROP else value


def _resolve_template_file(value: Any, template_dir: str | Path | None) -> Any:
    if not isinstance(value, str) or not template_dir:
        return value
    candidate = Path(str(value))
    if candidate.is_absolute():
        return value
    return str(Path(template_dir) / candidate)


def _mtp_settings(profile: ResolvedProfile, request: PlanRequest) -> dict[str, Any]:
    mtp = dict(profile.mtp or {})
    n_max = request.mtp_n_max if request.mtp_n_max is not None else mtp.get("n_max", 3)
    n_min = request.mtp_n_min if request.mtp_n_min is not None else mtp.get("n_min", 0)
    n_max = max(1, _parse_int(n_max, "mtp n_max"))
    n_min = min(max(0, _parse_int(n_min, "mtp n_min")), n_max)
    p_min = mtp.get("p_min", 0.75)
    p_min_value = _parse_float(p_min, "mtp p_min") if p_min is not None else 0.0
    ngl = mtp.get("ngl")
    ngl_value = None if ngl is None else _parse_int(ngl, "mtp ngl")
    return {
        "ratio": _parse_float(mtp.get("ctx_ratio", 0.8), "mtp ctx_ratio"),
        "n_max": n_max,
        "n_min": n_min,
        "p_min": p_min_value,
        "batch_size": mtp.get("batch_size"),
        "ubatch_size": mtp.get("ubatch_size"),
        "ngl_threshold_gb": _parse_float(
            mtp.get("ngl_threshold_gb", 20), "mtp ngl_threshold_gb"
        ),
        "ngl": ngl_value,
    }


def plan(request: PlanRequest) -> list[str]:
    """Build argv from a resolved profile.  Pure function — no side effects."""
    profile = request.profile
    model = request.model
    values = dict(profile.args)

    host = _pop_flag(values, "--host", request.host) or request.host
    alias = _pop_flag(values, "--alias", profile.alias) or profile.alias
    port = "0" if request.port is None else str(request.port)
    argv = [str(request.server_bin), "-m", str(model.path)]
    argv.extend(["--host", str(host), "--port", port, "--alias", str(alias)])

    raw_ctx = _pop_flag(values, "--ctx-size", profile.ctx)
    ctx = profile.ctx if raw_ctx is None or _is_auto(raw_ctx) else _parse_int(raw_ctx, "--ctx-size")

    mtp_enabled = bool(request.mtp or profile.mtp.get("enabled", False))
    mtp = _mtp_settings(profile, request) if mtp_enabled else None
    if mtp is not None:
        ctx = int(ctx * mtp["ratio"])

    def pop(name: str, default: Any = None) -> Any:
        return _pop_flag(values, name, default)

    # Core ordered flags.
    _emit(argv, "-np", pop("-np", 1))
    _emit(argv, "--ctx-size", ctx)
    _emit(argv, "-ctk", pop("-ctk", "q8_0"))
    _emit(argv, "-ctv", pop("-ctv", "q8_0"))
    _emit(argv, "--flash-attn", pop("--flash-attn", "on"))

    raw_batch = pop("-b", 2048)
    raw_ubatch = pop("-ub", 2048)
    if mtp is not None:
        batch = mtp["batch_size"] if mtp["batch_size"] is not None else min(
            _parse_int(raw_batch, "-b"), 2048
        )
        ubatch = mtp["ubatch_size"] if mtp["ubatch_size"] is not None else min(
            _parse_int(raw_ubatch, "-ub"), 512
        )
    else:
        batch = raw_batch
        ubatch = raw_ubatch
    _emit(argv, "-b", batch)
    _emit(argv, "-ub", ubatch)
    _emit(argv, "--threads", pop("--threads", 8))
    _emit(argv, "--threads-batch", pop("--threads-batch", 16))

    for flag in (
        "--temp",
        "--top-k",
        "--top-p",
        "--min-p",
        "--repeat-penalty",
        "--presence-penalty",
    ):
        _emit(argv, flag, pop(flag))

    _emit(argv, "--perf", pop("--perf"))
    _emit(argv, "--kv-unified", pop("--kv-unified"))

    raw_ngl = pop("-ngl")
    load_mode = pop("--load-mode")
    lazy_mode = pop("--lazy-mode")
    no_mmap = pop("--no-mmap")
    if raw_ngl is not None:
        ngl: Any = raw_ngl
    elif mtp is not None and mtp["ngl"] is not None:
        ngl = mtp["ngl"]
    elif mtp is not None and mtp["ngl"] is None and model.size_gb < mtp["ngl_threshold_gb"]:
        ngl = -1
    else:
        ngl = None
    _emit(argv, "-ngl", ngl)
    # llama.cpp removed --no-mmap (now --load-mode none). MoE weights stay
    # unmapped by default; an explicit --load-mode wins. A merged legacy
    # --no-mmap flag is consumed here so it keeps canonical order.
    if load_mode is None and (model.is_moe or no_mmap is not None):
        load_mode = "none"
    _emit(argv, "--load-mode", load_mode)
    _emit(argv, "--lazy-mode", lazy_mode)

    _emit(argv, "--chat-template", pop("--chat-template"))
    _emit(
        argv,
        "--chat-template-file",
        _resolve_template_file(pop("--chat-template-file"), request.template_dir),
    )
    _emit(argv, "--cache-ram", pop("--cache-ram"))
    _emit(argv, "-s", pop("-s"))
    _emit(argv, "--no-context-shift", pop("--no-context-shift"))

    raw_fit = pop("--fit", "on")
    fit_target = pop("--fit-target")
    raw_fit_ctx = pop("--fit-ctx")
    if mtp is not None and mtp["ngl"] is None:
        if model.size_gb < mtp["ngl_threshold_gb"]:
            fit_value: Any = "off"
            fit_target_value: Any = None
            fit_ctx_value: Any = None
        else:
            fit_value = "on"
            fit_target_value = fit_target
            fit_ctx_value = ctx
    else:
        fit_value = raw_fit
        fit_target_value = fit_target
        if raw_fit_ctx is None:
            fit_ctx_value = None
        elif _is_auto(raw_fit_ctx):
            fit_ctx_value = profile.fit_ctx
        else:
            fit_ctx_value = _parse_int(raw_fit_ctx, "--fit-ctx")
    _emit(argv, "--fit", fit_value)
    _emit(argv, "--fit-target", fit_target_value)
    _emit(argv, "--fit-ctx", fit_ctx_value)

    draft_model = pop("--model-draft")
    draft_spec_type = pop("--spec-type")
    draft_n_max = pop("--spec-draft-n-max")
    draft_n_min = pop("--spec-draft-n-min")
    if draft_model is not None:
        if draft_spec_type not in _ALLOWED_DRAFT_SPEC_TYPES:
            draft_spec_type = "draft-simple"
        _emit(argv, "--model-draft", draft_model)
        _emit(argv, "--spec-type", draft_spec_type)
        _emit(argv, "--spec-draft-n-max", draft_n_max if draft_n_max is not None else 16)
        _emit(argv, "--spec-draft-n-min", draft_n_min if draft_n_min is not None else 8)

    if mtp is not None:
        if request.mtp_draft_model:
            _emit(argv, "--model-draft", request.mtp_draft_model)
        elif profile.mtp.get("draft_model") is not None:
            # Launch resolves external MTP heads before planning; retain the
            # configured reference if resolution was intentionally skipped.
            _emit(argv, "--model-draft", profile.mtp["draft_model"])
        argv.extend(["--cache-type-k-draft", "q8_0", "--cache-type-v-draft", "q8_0"])
        argv.extend(["--spec-type", "draft-mtp"])
        if mtp["p_min"] > 0.0:
            _emit(argv, "--spec-draft-p-min", mtp["p_min"])
        _emit(argv, "--spec-draft-n-max", mtp["n_max"])
        if mtp["n_min"] > 0:
            _emit(argv, "--spec-draft-n-min", mtp["n_min"])

    cpu_moe = pop("--cpu-moe")
    # ``--cpu-moe`` without a count still enables CPU MoE offload.
    if cpu_moe is not None:
        argv.append("--cpu-moe")
    _emit(argv, "--n-cpu-moe", pop("--n-cpu-moe"))

    # Conditional n-gram speculation: enabled only by an explicit marker, for
    # large models, and when no draft model is already paired.
    spec_ngram = pop("--spec-ngram")
    spec_type = pop("--spec-type")
    spec_match = pop("--spec-ngram-mod-n-match")
    spec_min = pop("--spec-ngram-mod-n-min")
    spec_max = pop("--spec-ngram-mod-n-max")
    turbo_size_n = pop("--spec-ngram-size-n")
    turbo_min = pop("--draft-min")
    turbo_max = pop("--draft-max")
    if (
        spec_ngram is not None
        and draft_model is None
        and spec_type is not None
        and model.size_gb >= request.small_threshold_gb
    ):
        argv.extend(["--spec-type", str(spec_type)])
        if spec_type == "ngram-mod":
            if request.backend == "turboquant":
                _emit(argv, "--spec-ngram-size-n", turbo_size_n)
                _emit(argv, "--draft-min", turbo_min)
                _emit(argv, "--draft-max", turbo_max)
            else:
                _emit(argv, "--spec-ngram-mod-n-match", spec_match)
                _emit(argv, "--spec-ngram-mod-n-min", spec_min)
                _emit(argv, "--spec-ngram-mod-n-max", spec_max)

    # Any other merged flags preserve their merge order.  This keeps the
    # emitter forward-compatible with new llama-server options.
    for flag, value in values.items():
        if flag in (
            "-n",
            "--reasoning",
            "--reasoning-budget",
            "--reasoning-budget-message",
            "--reasoning-effort",
            "--reasoning-preserve",
            "--chat-template-kwargs",
            "--no-mmproj",
            "--samplers",
        ):
            continue
        _emit(argv, flag, value)

    raw_predict = _pop_flag(values, "-n", 2048)
    raw_reasoning = _pop_flag(values, "--reasoning", profile.reasoning)
    if isinstance(raw_reasoning, bool):
        reasoning = raw_reasoning
    elif _is_auto(raw_reasoning):
        reasoning = profile.reasoning
    else:
        parsed_reasoning = _parse_bool_word(raw_reasoning)
        reasoning = profile.reasoning if parsed_reasoning is None else parsed_reasoning
    _emit(argv, "-n", raw_predict)
    argv.extend(["--reasoning", "on" if reasoning else "off"])

    reasoning_budget = _pop_flag(values, "--reasoning-budget")
    reasoning_message = _pop_flag(values, "--reasoning-budget-message")
    reasoning_effort = _pop_flag(values, "--reasoning-effort")
    reasoning_preserve = _pop_flag(values, "--reasoning-preserve")
    template_kwargs = _pop_flag(values, "--chat-template-kwargs")
    no_mmproj = _pop_flag(values, "--no-mmproj")
    samplers = _pop_flag(values, "--samplers")
    if reasoning_budget is not None:
        _emit(argv, "--reasoning-budget", reasoning_budget)
        _emit(argv, "--reasoning-budget-message", reasoning_message)
    elif reasoning_effort is not None:
        _emit(argv, "--reasoning-effort", reasoning_effort)
    # ``--reasoning-preserve`` is meaningful only while reasoning is on.
    if reasoning and reasoning_preserve is not None:
        argv.append("--reasoning-preserve")
    _emit(argv, "--chat-template-kwargs", template_kwargs)
    _emit(argv, "--no-mmproj", no_mmproj)
    _emit(argv, "--samplers", samplers)

    return argv
