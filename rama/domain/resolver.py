"""Profile resolver for Rama.

Resolves settings from multiple sources with explicit precedence and provenance:
CLI/API > batch item > model override > family default > global config > built-in default.
"""

from __future__ import annotations

from dataclasses import asdict, fields

from .models import (
    ContextPolicy,
    FamilyCapabilities,
    FamilyDefaults,
    FamilyDefinition,
    LaunchSettings,
    MemorySettings,
    ModelInfo,
    ResolvedModelProfile,
    SamplerSettings,
)

# All settings classes and their provenance keys.
_SETTINGS_CLASSES = [
    ("sampler", SamplerSettings),
    ("context", ContextPolicy),
    ("memory", MemorySettings),
    ("launch", LaunchSettings),
]


def _collect_keys(*classes: type) -> set[str]:
    """Return the union of all field names from the given dataclass types."""
    keys: set[str] = set()
    for cls in classes:
        keys.update(cls.__dataclass_fields__)
    return keys


_ALL_OVERRIDE_KEYS = _collect_keys(
    SamplerSettings, ContextPolicy, MemorySettings, LaunchSettings
)


# Historical key aliases (config uses short names, domain uses canonical names)
_KEY_ALIASES: dict[str, str] = {
    "temp": "temperature",
}


def _split_override(raw: dict) -> dict[str, dict]:
    """Split a flat or nested override dict into per-settings-class dicts.

    Accepts:
      - Flat: {"temperature": 0.5} -> routed by field name match
      - Nested: {"sampler": {"temperature": 0.5}} -> used directly
      - Mixed: both forms combined
      - Aliased: {"temp": 0.5} -> routed as "temperature"
    """
    result: dict[str, dict] = {}
    consumed: set[str] = set()

    # First pass: handle nested keys (e.g. "sampler": {...})
    for prov_key, cls in _SETTINGS_CLASSES:
        if prov_key in raw and isinstance(raw[prov_key], dict):
            result[prov_key] = dict(raw[prov_key])
            consumed.add(prov_key)

    # Second pass: route flat keys to their settings class
    remaining = {k: v for k, v in raw.items() if k not in consumed}
    for prov_key, cls in _SETTINGS_CLASSES:
        matched = {}
        for k, v in remaining.items():
            if k == "no_mmap" and "use_mmap" in cls.__dataclass_fields__:
                matched["use_mmap"] = not bool(v)
                continue
            canonical = _KEY_ALIASES.get(k, k)
            if canonical in cls.__dataclass_fields__:
                matched[canonical] = v
        if matched:
            existing = result.get(prov_key, {})
            # Nested wins over flat if both provided same key
            existing.update(matched)
            result[prov_key] = existing

    return result


def _merge_dict_into_dataclass(
    base,
    override_dict: dict | None,
    provenance_key: str,
    source: str,
):
    """Merge a partial dictionary into a frozen dataclass, tracking provenance."""
    if not override_dict:
        return base, {}

    base_dict = asdict(base)
    merged = dict(base_dict)
    prov = {}

    for k, v in override_dict.items():
        if v is not None:
            merged[k] = v
            prov[f"{provenance_key}.{k}"] = source

    return type(base)(**merged), prov


def _merge_dataclasses(base, override, provenance_key: str, source: str):
    """Merge two frozen dataclasses, tracking provenance for overridden fields."""
    if override is None:
        return base, {}
    return _merge_dict_into_dataclass(base, asdict(override), provenance_key, source)


def _apply_override(
    sampler: SamplerSettings,
    context: ContextPolicy,
    memory: MemorySettings,
    launch: LaunchSettings,
    raw_override: dict | None,
    source: str,
) -> tuple[
    SamplerSettings, ContextPolicy, MemorySettings, LaunchSettings, dict[str, str]
]:
    """Route a flat or nested override dict through all settings classes."""
    if not raw_override:
        return sampler, context, memory, launch, {}

    split = _split_override(raw_override)
    prov: dict[str, str] = {}

    sampler, p = _merge_dict_into_dataclass(
        sampler, split.get("sampler"), "sampler", source
    )
    prov.update(p)
    context, p = _merge_dict_into_dataclass(
        context, split.get("context"), "context", source
    )
    prov.update(p)
    memory, p = _merge_dict_into_dataclass(
        memory, split.get("memory"), "memory", source
    )
    prov.update(p)
    launch, p = _merge_dict_into_dataclass(
        launch, split.get("launch"), "launch", source
    )
    prov.update(p)

    return sampler, context, memory, launch, prov


class ProfileResolver:
    """Deterministic resolver for model profiles."""

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
    ) -> ResolvedModelProfile:
        """Resolve a complete model profile with provenance tracking.

        Each override dict can be flat ({"temperature": 0.5}) or nested
        ({"sampler": {"temperature": 0.5}}). Keys are routed to the matching
        settings class by field name.
        """
        provenance = {
            f"{prefix}.{field.name}": "built-in default"
            for prefix, settings_type in _SETTINGS_CLASSES
            for field in fields(settings_type)
        }
        provenance.update(
            {
                f"capabilities.{field.name}": "built-in default"
                for field in fields(FamilyCapabilities)
            }
        )

        # 1. Built-in safe defaults
        sampler = SamplerSettings()
        context = ContextPolicy()
        memory = MemorySettings()
        launch = LaunchSettings()
        capabilities = FamilyCapabilities()

        # 2. Global config (weakest user setting)
        sampler, context, memory, launch, p = _apply_override(
            sampler,
            context,
            memory,
            launch,
            global_config,
            "global config",
        )
        provenance.update(p)

        # 3. Family defaults (more specific than global)
        if self.family:
            caps = self.family.capabilities
            capabilities = caps
            for field in fields(FamilyCapabilities):
                provenance[f"capabilities.{field.name}"] = "family default"

            defaults = self.family.defaults
            sampler, p = _merge_dataclasses(
                sampler, defaults.sampler, "sampler", "family default"
            )
            provenance.update(p)
            context, p = _merge_dataclasses(
                context, defaults.context, "context", "family default"
            )
            provenance.update(p)
            if caps.default_no_mmap:
                memory, p = _merge_dict_into_dataclass(
                    memory,
                    {"use_mmap": False},
                    "memory",
                    "family default",
                )
                provenance.update(p)
            # Muse Glimmer prefers f16 KV cache for quality and full context
            if self.family.key == "muse-glimmer" and launch.ctk is None:
                launch = LaunchSettings(
                    backend=launch.backend,
                    ctk="f16",
                    ctv="f16",
                    mtp=launch.mtp,
                    mtp_n_max=launch.mtp_n_max,
                    mtp_n_min=launch.mtp_n_min,
                    seed=launch.seed,
                )
                provenance["launch.ctk"] = "family default"
                provenance["launch.ctv"] = "family default"

        # 3b. User-configured family settings refine registry defaults.
        sampler, context, memory, launch, p = _apply_override(
            sampler,
            context,
            memory,
            launch,
            family_override,
            "family config",
        )
        provenance.update(p)

        # 4. Named model override
        sampler, context, memory, launch, p = _apply_override(
            sampler,
            context,
            memory,
            launch,
            model_override,
            "model override",
        )
        provenance.update(p)

        # 5. Batch override
        sampler, context, memory, launch, p = _apply_override(
            sampler,
            context,
            memory,
            launch,
            batch_override,
            "batch override",
        )
        provenance.update(p)

        # 6. CLI override (strongest)
        sampler, context, memory, launch, p = _apply_override(
            sampler,
            context,
            memory,
            launch,
            cli_override,
            "CLI override",
        )
        provenance.update(p)

        return ResolvedModelProfile(
            model=model,
            context=context,
            sampler=sampler,
            memory=memory,
            launch=launch,
            capabilities=capabilities,
            provenance=provenance,
        )
