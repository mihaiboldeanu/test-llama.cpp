"""Profile resolver for Rama.

Merges literal llama-server flag maps from ``rama.yaml`` with explicit
precedence and provenance:

CLI override > batch override > model override > family config >
global config.
"""

from __future__ import annotations

from typing import Any

from .models import FamilyDefinition, ModelInfo, ResolvedProfile

GLOBAL_SOURCE = "global config"
FAMILY_SOURCE = "family config"
MODEL_SOURCE = "model override"
BATCH_SOURCE = "batch override"
CLI_SOURCE = "CLI override"

# Marker for an explicit ``null`` override.  It removes an inherited flag.  It
# is never emitted in the final ``args`` map.
DROP = object()

# Safe fallbacks used only when a partial mapping (usually in tests) does not
# provide the documented ``rules:`` block.  Production values come from YAML.
DEFAULT_RULES: dict[str, Any] = {
    "small_ctx": 32768,
    "large_ctx": 65536,
    "small_threshold_gb": 10,
    "mtp_ngl_threshold_gb": 20,
    "mtp_ctx_ratio": 0.8,
}

_TRUE_WORDS = {"on", "true", "1", "yes", "y", "enable", "enabled"}
_FALSE_WORDS = {"off", "false", "0", "no", "n", "disable", "disabled"}

# Legacy semantic keys accepted in any override layer.  They are translated to
# the literal llama-server flag named in the value.
_FLAG_ALIASES = {
    "temperature": "--temp",
    "temp": "--temp",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "repeat_penalty": "--repeat-penalty",
    "repetition_penalty": "--repeat-penalty",
    "presence_penalty": "--presence-penalty",
    "ctx": "--ctx-size",
    "ctx_size": "--ctx-size",
    "n_parallel": "-np",
    "parallel": "-np",
    "batch_size": "-b",
    "ubatch_size": "-ub",
    "threads": "--threads",
    "threads_batch": "--threads-batch",
    "fit": "--fit",
    "fit_target": "--fit-target",
    "fit_target_mib": "--fit-target",
    "fit_ctx": "--fit-ctx",
    "n_predict": "-n",
    "seed": "-s",
    "alias": "--alias",
    "host": "--host",
    "n_gpu_layers": "-ngl",
    "ngl": "-ngl",
    "n_cpu_moe": "--n-cpu-moe",
    "load_mode": "--load-mode",
    "tensor_read_lazy": "--lazy-mode",
    "chat_template": "--chat-template",
    "chat_template_file": "--chat-template-file",
    "chat_template_kwargs": "--chat-template-kwargs",
    "cache_ram": "--cache-ram",
    "reasoning": "--reasoning",
    "reasoning_budget": "--reasoning-budget",
    "reasoning_budget_message": "--reasoning-budget-message",
    "reasoning_effort": "--reasoning-effort",
    "samplers": "--samplers",
    "no_mmproj": "--no-mmproj",
    "no_mmap": "--no-mmap",
    "kv_unified": "--kv-unified",
    "perf": "--perf",
    "flash_attn": "--flash-attn",
    "skip_chat_parsing": "--skip-chat-parsing",
    "draft_model": "--model-draft",
    "draft_max": "--spec-draft-n-max",
    "draft_min": "--spec-draft-n-min",
    "spec_type": "--spec-type",
    "spec_ngram_mod_size_n": "--spec-ngram-mod-n-match",
    "spec_ngram_mod_n_min": "--spec-ngram-mod-n-min",
    "spec_ngram_mod_n_max": "--spec-ngram-mod-n-max",
    "draft_min_val": "--draft-min",
    "draft_max_val": "--draft-max",
}

# Non-flag keys handled as context/rules/env/MTP data or ignored as structural.
_NON_ARG_KEYS = {
    "args",
    "env",
    "mtp",
    "rules",
    "families",
    "models",
    "small_ctx",
    "large_ctx",
    "preferred_ctx",
    "small_threshold_gb",
    "reasoning_preserve",
    "moe_fit_ctx_skip",
    "model",
    "model_dir",
    "llama_cpp_dir",
    "turbo_dir",
    "llama_cpp_repo",
    "turboquant_repo",
    "template_dir",
    "cuda_root",
    "cuda_arch",
    "build_jobs",
    "vram_total_gb",
    "vram_headroom_gb",
    "default_port",
    "port_range_end",
    "port",
    "backend",
    "foreground",
    "log_disable",
    "dry_run",
    "config",
    "config_item",
    "family_params",
    "max_tokens",
    "ctx_sizes",
    "kv_quants",
    "batch",
}

_MTP_ALIASES = {
    "enabled": "enabled",
    "mtp": "enabled",
    "ctx_ratio": "ctx_ratio",
    "ratio": "ctx_ratio",
    "mtp_ctx_ratio": "ctx_ratio",
    "n_max": "n_max",
    "mtp_n_max": "n_max",
    "n_min": "n_min",
    "mtp_n_min": "n_min",
    "p_min": "p_min",
    "mtp_p_min": "p_min",
    "batch_size": "batch_size",
    "mtp_batch_size": "batch_size",
    "ubatch_size": "ubatch_size",
    "mtp_ubatch_size": "ubatch_size",
    "ngl": "ngl",
    "mtp_ngl": "ngl",
    "ngl_threshold_gb": "ngl_threshold_gb",
    "mtp_ngl_threshold_gb": "ngl_threshold_gb",
    "draft_model": "draft_model",
    "draft": "draft_model",
    "mtp_draft_model": "draft_model",
}


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _parse_bool_word(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


def _parse_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise ValueError(f"{field} must be an integer, got {value!r}")


def _parse_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError(f"{field} must be a number, got {value!r}")


def _normalize_flag_value(flag: str, value: Any) -> Any:
    """Normalize one literal flag value.

    ``True`` means a bare presence flag and ``DROP`` removes an inherited flag.
    Reasoning/fit/flash-attn booleans become explicit ``on``/``off`` values.
    """
    if value is None:
        return DROP
    if isinstance(value, bool):
        if flag in ("--reasoning", "--fit", "--flash-attn"):
            return "on" if value else "off"
        return True if value else DROP
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "auto":
            return "auto"
        return text
    return value


def _normalize_semantic_value(flag: str, value: Any) -> Any:
    if value is None:
        return None
    if flag == "--no-mmap":
        # ``use_mmap: false`` disables mmap; ``use_mmap: true`` is the default
        # mapped behavior and carries no flag.
        return True if value is False else None
    if flag == "--no-context-shift":
        # ``context_shift: true`` removes the inherited opt-out flag.
        return DROP if value is True else True if value is False else None
    if flag in ("--kv-unified", "--perf", "--no-mmproj", "--skip-chat-parsing"):
        return True if value else DROP
    if flag == "--cpu-moe":
        return True if value else DROP
    if flag == "--spec-ngram":
        return True if value else DROP
    return _normalize_flag_value(flag, value)


def _normalize_layer_args(section: Any) -> dict[str, Any]:
    """Translate one override layer to literal flag values.

    Accepts an ``args:`` map of literal flags and/or legacy semantic keys.
    ``None`` under ``args:`` is an explicit tombstone; ``None`` for a legacy
    semantic key means “no override”.
    """
    normalized: dict[str, Any] = {}
    if not isinstance(section, dict):
        return normalized

    nested = section.get("args")
    if isinstance(nested, dict):
        for raw_flag, value in nested.items():
            if not isinstance(raw_flag, str) or not raw_flag.startswith("-"):
                continue
            normalized[raw_flag] = _normalize_flag_value(raw_flag, value)

    for key, value in section.items():
        if key in _NON_ARG_KEYS or key in ("use_mmap", "context_shift"):
            if key == "use_mmap":
                mapped = _normalize_semantic_value("--no-mmap", value)
                if mapped is not None:
                    normalized["--no-mmap"] = mapped
            elif key == "context_shift":
                mapped = _normalize_semantic_value("--no-context-shift", value)
                if mapped is not None:
                    normalized["--no-context-shift"] = mapped
            continue
        flag = _FLAG_ALIASES.get(key)
        if flag is None:
            if isinstance(key, str) and key.startswith("-"):
                normalized[key] = _normalize_flag_value(key, value)
            continue
        mapped = _normalize_semantic_value(flag, value)
        if mapped is not None:
            normalized[flag] = mapped

    return normalized


def _merge_args(
    merged: dict[str, Any],
    override: dict[str, Any],
    source: str,
    provenance: dict[str, str],
) -> dict[str, Any]:
    merged = dict(merged)
    for flag, value in override.items():
        merged[flag] = value
        provenance[f"args.{flag}"] = (
            f"{source} (removed)" if value is DROP else source
        )
    return merged


def _merge_named_map(
    merged: dict[str, Any], override: Any, source: str, provenance: dict[str, str], prefix: str
) -> dict[str, Any]:
    if not isinstance(override, dict):
        return dict(merged)
    merged = dict(merged)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
        provenance[f"{prefix}.{key}"] = source
    return merged


def _merge_mtp(merged: dict[str, Any], override: Any) -> dict[str, Any]:
    if isinstance(override, bool):
        merged = dict(merged)
        merged["enabled"] = override
        return merged
    if not isinstance(override, dict):
        return dict(merged)
    merged = dict(merged)
    for key, value in override.items():
        canonical = _MTP_ALIASES.get(key)
        if canonical is None or value is None:
            continue
        merged[canonical] = value
    return merged


def _layer_parts(section: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split one layer into args, env, mtp, and context/rule values."""
    args = _normalize_layer_args(section)
    env: dict[str, Any] = {}
    mtp: dict[str, Any] = {}
    context: dict[str, Any] = {}
    if not isinstance(section, dict):
        return args, env, mtp, context

    nested_env = section.get("env")
    if isinstance(nested_env, dict):
        env.update({k: v for k, v in nested_env.items() if v is not None})
    nested_mtp = section.get("mtp")
    mtp = _merge_mtp(mtp, nested_mtp)
    # A bare ``mtp: true`` in a batch item enables the runtime MTP block.
    if section.get("mtp") is True:
        mtp["enabled"] = True

    nested_rules = section.get("rules")
    if isinstance(nested_rules, dict):
        for key in (
            "small_ctx",
            "large_ctx",
            "small_threshold_gb",
            "mtp_ctx_ratio",
            "mtp_ngl_threshold_gb",
        ):
            if nested_rules.get(key) is not None:
                context[key] = nested_rules[key]
    for key in (
        "small_ctx",
        "large_ctx",
        "preferred_ctx",
        "small_threshold_gb",
        "reasoning_preserve",
        "moe_fit_ctx_skip",
    ):
        if section.get(key) is not None:
            context[key] = section[key]
    for key, value in section.items():
        if key in _MTP_ALIASES and key not in ("mtp", "enabled"):
            mtp = _merge_mtp(mtp, {key: value})
    return args, env, mtp, context


def _global_parts(global_config: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract only the global layer from a full config mapping."""
    if not isinstance(global_config, dict):
        return {}, {}, {}, {}
    section = {
        key: value
        for key, value in global_config.items()
        if key not in ("families", "models")
    }
    return _layer_parts(section)


def _resolve_context_value(
    size_gb: float | None,
    context: dict[str, Any],
    rules: dict[str, Any],
) -> int:
    threshold = _parse_float(
        context.get("small_threshold_gb", rules.get("small_threshold_gb", 10)),
        "small_threshold_gb",
    )
    small_ctx = _parse_int(
        context.get("small_ctx", rules.get("small_ctx", 32768)), "small_ctx"
    )
    large_ctx = _parse_int(
        context.get("large_ctx", rules.get("large_ctx", 65536)), "large_ctx"
    )
    if size_gb is None:
        return large_ctx
    return small_ctx if size_gb < threshold else large_ctx


def _resolve_reasoning_value(
    raw: Any, size_gb: float | None, threshold: float
) -> bool:
    if raw is None or _is_auto(raw):
        if size_gb is None:
            return True
        return size_gb >= threshold
    parsed = _parse_bool_word(raw)
    if parsed is None:
        raise ValueError(f"--reasoning must be on/off/auto, got {raw!r}")
    return parsed


class ProfileResolver:
    """Deterministic resolver for merged llama-server argument maps."""

    def __init__(self, family_definition: FamilyDefinition | None = None):
        self.family = family_definition

    def resolve(
        self,
        model: ModelInfo | None = None,
        global_config: dict | None = None,
        family_override: dict | None = None,
        model_override: dict | None = None,
        batch_override: dict | None = None,
        cli_override: dict | None = None,
    ) -> ResolvedProfile:
        """Resolve merged args plus the documented runtime values."""
        if model is None:
            raise ValueError("ProfileResolver.resolve() requires a model")
        provenance: dict[str, str] = {}
        args: dict[str, Any] = {}
        env: dict[str, Any] = {}
        mtp: dict[str, Any] = {}
        context: dict[str, Any] = {}
        rules = dict(DEFAULT_RULES)

        global_args, global_env, global_mtp, global_context = _global_parts(global_config)
        args = _merge_args(args, global_args, GLOBAL_SOURCE, provenance)
        env = _merge_named_map(env, global_env, GLOBAL_SOURCE, provenance, "env")
        mtp = _merge_mtp(mtp, global_mtp)
        context.update(global_context)
        rules.update(
            {
                key: global_context[key]
                for key in (
                    "small_ctx",
                    "large_ctx",
                    "small_threshold_gb",
                    "mtp_ctx_ratio",
                    "mtp_ngl_threshold_gb",
                )
                if key in global_context
            }
        )

        layers = (
            (family_override, FAMILY_SOURCE),
            (model_override, MODEL_SOURCE),
            (batch_override, BATCH_SOURCE),
            (cli_override, CLI_SOURCE),
        )
        for section, source in layers:
            layer_args, layer_env, layer_mtp, layer_context = _layer_parts(section)
            args = _merge_args(args, layer_args, source, provenance)
            env = _merge_named_map(env, layer_env, source, provenance, "env")
            mtp_before = dict(mtp)
            mtp = _merge_mtp(mtp, layer_mtp)
            for key in layer_mtp:
                provenance[f"mtp.{key}"] = source
            if mtp == mtp_before and layer_mtp:
                for key in layer_mtp:
                    provenance[f"mtp.{key}"] = source
            context.update(layer_context)
            for key in layer_context:
                provenance[f"context.{key}"] = source

        size_gb = model.size_gb
        threshold = _parse_float(
            context.get("small_threshold_gb", rules.get("small_threshold_gb", 10)),
            "small_threshold_gb",
        )

        raw_ctx = args.get("--ctx-size", "auto")
        ctx_auto = raw_ctx is DROP or _is_auto(raw_ctx)
        if ctx_auto:
            ctx = _resolve_context_value(size_gb, context, rules)
            ctx_rule_key = (
                "context.small_ctx" if size_gb < threshold else "context.large_ctx"
            )
            provenance["args.--ctx-size"] = provenance.get(
                ctx_rule_key, GLOBAL_SOURCE
            )
        else:
            ctx = _parse_int(raw_ctx, "--ctx-size")
        args["--ctx-size"] = ctx

        moe_skip = bool(context.get("moe_fit_ctx_skip", False))
        if self.family is not None and self.family.moe_fit_ctx_skip:
            moe_skip = True
        raw_fit_ctx = args.get("--fit-ctx", "auto")
        if raw_fit_ctx is DROP:
            fit_ctx: int | None = None
        elif _is_auto(raw_fit_ctx):
            fit_ctx = (
                None
                if model is not None and model.is_moe and moe_skip
                else ctx
            )
            if fit_ctx is not None:
                provenance["args.--fit-ctx"] = provenance.get(
                    "args.--ctx-size", GLOBAL_SOURCE
                )
        else:
            fit_ctx = _parse_int(raw_fit_ctx, "--fit-ctx")
        if fit_ctx is None:
            args.pop("--fit-ctx", None)
        else:
            args["--fit-ctx"] = fit_ctx

        raw_reasoning = args.get("--reasoning", "auto")
        if raw_reasoning is DROP:
            raw_reasoning = "auto"
        reasoning = _resolve_reasoning_value(raw_reasoning, size_gb, threshold)
        args["--reasoning"] = "on" if reasoning else "off"
        if raw_reasoning is None or _is_auto(raw_reasoning):
            provenance["args.--reasoning"] = "auto rule (10GB size split)"

        if bool(context.get("reasoning_preserve", False)) and reasoning:
            args["--reasoning-preserve"] = True
            provenance["args.--reasoning-preserve"] = provenance.get(
                "context.reasoning_preserve", FAMILY_SOURCE
            )
        else:
            # The preserve flag is meaningful only while reasoning is on.
            args.pop("--reasoning-preserve", None)

        if model is not None and model.is_moe and moe_skip:
            # MoE models in skip families keep their native template handling
            # and do not receive an explicit template override.
            args.pop("--chat-template", None)
            args.pop("--chat-template-file", None)

        mtp.setdefault("enabled", False)
        mtp.setdefault("ctx_ratio", rules.get("mtp_ctx_ratio", 0.8))
        mtp.setdefault("ngl_threshold_gb", rules.get("mtp_ngl_threshold_gb", 20))
        mtp.setdefault("n_max", 3)
        mtp.setdefault("n_min", 0)

        alias_value = args.get("--alias", model.name if model is not None else "model")
        alias = str(alias_value) if alias_value is not None else (
            model.name if model is not None else "model"
        )
        args["--alias"] = alias
        if "args.--host" not in provenance:
            provenance["args.--host"] = "built-in default"

        final_args = {flag: value for flag, value in args.items() if value is not DROP}
        return ResolvedProfile(
            model=model,
            args=final_args,
            ctx=ctx,
            fit_ctx=fit_ctx,
            reasoning=reasoning,
            mtp=mtp,
            env=dict(env),
            alias=alias,
            provenance=provenance,
        )


def resolve_context_size(
    size_gb: float,
    family_key: str | None = None,
    global_config: dict | None = None,
    family_override: dict | None = None,
    model_override: dict | None = None,
    batch_override: dict | None = None,
    cli_override: dict | None = None,
) -> int:
    """Resolve preferred context without building a full profile."""
    context: dict[str, Any] = {}
    rules = dict(DEFAULT_RULES)
    global_args, _env, _mtp, global_context = _global_parts(global_config)
    context.update(global_context)
    if (
        family_override is None
        and isinstance(global_config, dict)
        and family_key is not None
    ):
        families = global_config.get("families")
        if isinstance(families, dict) and isinstance(families.get(family_key), dict):
            family_override = families[family_key]
    for section in (family_override, model_override, batch_override, cli_override):
        _args, _env, _mtp, layer_context = _layer_parts(section)
        context.update(layer_context)

    explicit = global_args.get("--ctx-size")
    for section in (family_override, model_override, batch_override, cli_override):
        layer_args = _normalize_layer_args(section)
        if "--ctx-size" in layer_args:
            explicit = layer_args["--ctx-size"]
    if explicit is not None and explicit is not DROP and not _is_auto(explicit):
        return _parse_int(explicit, "--ctx-size")
    if context.get("preferred_ctx") is not None:
        return _parse_int(context["preferred_ctx"], "preferred_ctx")
    return _resolve_context_value(size_gb, context, rules)
