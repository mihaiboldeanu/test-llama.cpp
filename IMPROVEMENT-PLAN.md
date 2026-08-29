# Rama Architecture Improvement Plan

## Purpose

Make Rama easy to extend, safe to operate, and straightforward to test without
turning a small local-model tool into an over-engineered framework. The target
is a codebase where adding a model family is mostly data entry, a launch can be
explained before it runs, and benchmark results are reproducible evidence rather
than a pile of ad-hoc files.

This plan deliberately separates **architecture work** from per-model tuning.
Adding one more family entry is useful; it is not an architectural solution if
the same change still requires edits in discovery, KV sizing, launch flags, API
requests, and batch code.

## What Is Wrong Today

The current changes improve coverage and add useful family settings, but the
source of truth remains split across several modules:

* `core.py` detects families and estimates KV cache size.
* `launch.py` decides sampler, reasoning, mmap, fit, and family-specific flags.
* `rama.yaml` contains a partial set of family and model overrides.
* `testing.py` and `batch_runner.py` independently decide request-time sampling
  and reasoning template values.

That creates predictable failures:

* A new family is easy to detect but easy to launch with the wrong template.
* Configuration precedence differs by setting or caller.
* A CLI command can display one resolved value and launch another.
* A local machine's model paths and tuning get mixed with portable defaults.
* Tests cover individual fixes but do not prove the complete launch contract.

The goal is not a class hierarchy for its own sake. The goal is one resolved,
typed **model profile** that every layer consumes.

## Design Principles

1. **One source of truth per concern.** Family capability lives in a family
   definition, user tuning lives in configuration, and runtime state lives in a
   launch plan/result.
2. **Data first, code only for real behavior.** A context limit or sampler
   default should be declarative. Use a small strategy object only when a
   family needs a genuinely different algorithm.
3. **Resolve once at the boundary.** Merge global, family, model, batch, and
   CLI settings once. Internal code receives a resolved object, never a mixture
   of raw dictionaries.
4. **Keep process management boring.** The launcher builds an argv from a plan,
   starts a process, waits for health, and cleans up. It must not know model
   brands.
5. **Make unsafe behavior explicit.** Killing unrelated processes, forcing
   `--no-mmap`, or assuming a model has MTP heads must require an explicit
   configuration choice.
6. **Prefer compatibility during migration.** Keep current YAML keys and
   Python entry points working with deprecation warnings until callers have
   moved.

## Target Architecture

```text
CLI / batch YAML / Python API
             |
             v
      Config loader + validation
             |
             v
   Model discovery + GGUF metadata
             |
             v
      ProfileResolver ------------------- FamilyRegistry
             |                                  |
             v                                  v
      ResolvedModelProfile <---------- FamilyDefinition
             |
             +------------------+
             |                  |
             v                  v
        LaunchPlanner      RequestPlanner
             |                  |
             v                  v
         llama-server       chat completions
             |
             v
      LaunchResult + run manifest
```

The important boundary is `ResolvedModelProfile`. It records every effective
choice (context limit, cache types, sampler values, reasoning template,
capabilities, and memory policy) before Rama starts a process or sends a
request. This makes `rama explain MODEL` practical and turns most bugs into
unit-testable resolution bugs rather than expensive failed launches.

## Module Ownership

| Module | Owns | Must not own |
|---|---|---|
| `config.py` | YAML loading, schema validation, path expansion, precedence | model-name string matching, argv creation |
| `models.py` | GGUF discovery, split-file grouping, metadata extraction, `ModelInfo` | sampler/reasoning decisions |
| `families.py` | family matching rules, capabilities, defaults, KV strategy | process management, CLI parsing |
| `profiles.py` | deterministic merge into `ResolvedModelProfile` | filesystem scans, HTTP calls |
| `launch.py` | `LaunchPlan`, argv construction, process lifecycle, health checks | `if family == ...` branches |
| `requests.py` | OpenAI-compatible request body construction | launch flags, model discovery |
| `testing/` | fixtures, evaluation protocol, reports | server lifecycle details beyond a small interface |
| `cli.py` | user interaction and rendering | business rules |

The existing files can migrate into these boundaries gradually; a big-bang
directory rewrite is not required.

### Dependency Direction (No Import Cycles)

Shared domain dataclasses and protocols must not live in `core.py`: the family
registry needs `ModelInfo`, while discovery needs the registry. Put neutral
types such as `ModelInfo`, `ModelMetadata`, `FamilyCapabilities`, and
`ResolvedModelProfile` in `models.py` or `domain.py`. Do **not** call this file
`types.py`, because that can shadow Python's standard-library `types` module.

Imports must flow one way:

```text
domain/models  <-  config, families, discovery
domain/models  <-  profiles  <-  launch, requests, testing, CLI
```

`domain/models` imports neither application services nor CLI code. Enforce this
with a small import-boundary test or linter rule before migrating behavior.

## The Core Data Model

Use frozen dataclasses for data passed between layers. Raw dictionaries are
allowed only at the YAML edge.

```python
@dataclass(frozen=True)
class FamilyDefinition:
    key: str
    matchers: tuple[Pattern[str], ...]
    capabilities: FamilyCapabilities
    defaults: FamilyDefaults
    kv_estimator: KvEstimator


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path
    size_bytes: int
    family_key: str
    quantization: str | None
    metadata: ModelMetadata | None = None


@dataclass(frozen=True)
class ResolvedModelProfile:
    model: ModelInfo
    context: ContextPolicy
    sampler: SamplerSettings
    reasoning: ReasoningSettings
    memory: MemorySettings
    launch: LaunchSettings
    provenance: Mapping[str, str]
```

`provenance` is essential: `{"sampler.temperature": "model override"}` is
more useful than silently picking a value. Expose it in a future `rama explain`
command and include it in result manifests.

### Resolver Safety Is a First-Class Requirement

`ProfileResolver` is the highest-leverage component in this design and therefore
a potential single point of silent failure. A wrong merge must never look like a
valid successful launch. Provenance is a required field for every resolved
setting, not optional debug logging: the resolver returns both the value and the
source that won it.

Protect the resolver with a compact precedence table covering every setting
class. Each case asserts the final value **and** provenance for conflicts among
global, family, model, batch, and CLI inputs. Include explicit cases for
`False`, `0`, empty collections, and `None`; truthiness-based merging is a
common source of silent override bugs. `rama explain MODEL` and the dry-run
launch output are acceptance tests for this contract, not polish deferred to a
later phase.

### Family Definitions: Declarative by Default

Most families should be records, not subclasses. For example:

```python
FamilyDefinition(
    key="qwen",
    matchers=(re.compile(r"qwen", re.I),),
    capabilities=FamilyCapabilities(
        supports_reasoning=True,
        reasoning_template_key="enable_thinking",
        supports_ngram_speculation=True,
    ),
    defaults=FamilyDefaults(
        sampler=SamplerSettings(temperature=0.7, top_k=20, top_p=0.8),
        context=ContextPolicy(small=24_576, large=151_552),
    ),
    kv_estimator=ArchitectureKvEstimator(...),
)
```

Use a strategy protocol only when declarative data is insufficient:

```python
class KvEstimator(Protocol):
    def bytes_per_token(self, model: ModelInfo, cache: CacheTypes) -> int: ...
```

This avoids a proliferation of one-method `QwenFamily`, `GemmaFamily`, and
similar classes while still giving unusual architectures an escape hatch.

### Registry Decision: Facts in Families, Decisions in Planners

The registry pattern is the right direction, but a `ModelFamily.get_launch_args`
method that accepts an untyped `opts: dict` would only move the existing
conditionals into many small opaque classes. That is neither DRY nor KISS.

Use this division instead:

| Concern | Home | Example |
|---|---|---|
| Family facts | `FamilyDefinition` | template key, supported speculation modes, default sampler |
| User intent and precedence | `ProfileResolver` | model override beats family default |
| A real exceptional algorithm | narrow strategy protocol | metadata-based KV estimator |
| Backend argument syntax | `LaunchPlanner` | `CacheTypes` becomes `-ctk`/`-ctv` |
| Request-body syntax | `RequestPlanner` | reasoning template value becomes JSON kwargs |

The planner consumes capabilities, not family names. For example, it sees
`reasoning.template_key="reasoning_effort"`, not `family == "gpt-oss"`.
Consequently, a normal family addition is a registry entry and tests—not edits
to `launch.py`, `testing.py`, and batch orchestration.

Treat `families.py` as a registry of compact deltas, not a second monolith. An
ordinary family may declare matchers, capabilities, and defaults; it may not
reimplement process startup, generic sampling, config merging, or request
construction. When the file grows, split only by stable concern (for example
`family_definitions.py` and `kv_estimators.py`), while preserving one public
registry interface.

### New-Family Checklist

Adding an ordinary family must be a small, repeatable change:

1. add a specific ordered matcher and a false-positive test;
2. declare capabilities, portable defaults, and a documented KV policy;
3. add a resolved-profile fixture showing precedence and provenance;
4. add launch-plan and request-body snapshots for reasoning on and off;
5. add a real-model integration case only when the model is available locally;
6. document any non-default requirement in the family definition, not in a
   scattered code comment or personal batch YAML.

If that checklist needs a new conditional in process management, improve the
profile/capability model first.

### Matching and Aliases

Family matching must be ordered, tested, and explainable. A matcher returns a
match plus its reason; the resolver selects the highest-priority match. Specific
families such as `qwen-coder` must precede `qwen`. Do not use broad substrings
like `"north"` without a documented reason and test fixtures for false positives.

Model aliases belong in user configuration, not on `ModelInfo` derived from a
filename. That keeps portable discovery separate from personal naming.

## Configuration Contract

Split portable defaults from local overrides:

```text
config/
  defaults.yaml          # versioned, portable family definitions
  local.example.yaml     # versioned example, no real paths
~/.config/rama/config.yaml  # user paths and personal defaults
~/.config/rama/models.yaml  # per-model overrides and aliases
```

The repository may retain `rama.yaml` as a convenience example, but it must not
contain a developer-specific `/mnt/...` path or assume a local CUDA install.

### Explicit Precedence

Resolve settings in exactly this order:

```text
built-in safe default
  < global user config
  < family default
  < named model override
  < batch item
  < explicit CLI/API option
```

No subsystem may bypass this order with a direct `dict.get`. The resolver should
produce a type-checked profile after the merge. Use `None` to mean “not
specified”; do not use truthiness because `False`, `0`, and empty collections
are legitimate overrides.

This means the effective order from strongest to weakest is **CLI/API > batch
item > model override > family default > global config > built-in default**.
Family defaults are deliberately more specific than global user settings. A
user who wants a one-off exception uses a model, batch, or CLI override; they do
not need to fork a family definition.

### Schema Validation

Validate nested family and model settings as well as top-level values. Reject
unknown keys by default, with an `extra` namespace for forward-compatible
llama.cpp flags. Cross-field validation belongs here:

* cache type is supported by the selected backend;
* `ubatch_size <= batch_size`;
* ports form a valid range;
* MTP and draft-model options are mutually compatible;
* `skip_chat_parsing` is enabled only for a documented template/backend pair;
* a model override cannot request both `n_gpu_layers` and an incompatible
  auto-fit policy.

Support the historical `repetition_penalty` spelling temporarily by translating
it to `repeat_penalty` and emitting one clear warning. Documentation, templates,
and serialized manifests must use only the canonical name.

## Launch Planning

Replace a large `start_model` function with two pure steps and one side-effectful
step:

```python
profile = resolver.resolve(model, overrides)
plan = launch_planner.plan(profile, backend)
result = process_runner.start(plan)
```

`LaunchPlan` contains argv, environment, working directory, health URL,
effective port, and a human-readable explanation. `launch_planner.plan` must be
pure and unit-tested with snapshots. `process_runner` owns PID files, retries,
timeouts, and cleanup.

This produces clear invariants:

* Every argv value has a corresponding profile field or explicit backend rule.
* `fit_target_mib` is resolved from configuration, never silently hard-coded.
* The displayed auto-context is the context passed to the launcher.
* Template arguments are emitted once as one JSON object, never as competing
  duplicate flags.
* `--no-mmap`, `-ngl`, and `--n-cpu-moe` come only from an explicit memory
  policy, not filename heuristics.

### Memory and KV Cache Policy

Do not infer architecture from quantized file size when metadata is available.
Read GGUF architecture fields once and cache them in `ModelMetadata`; calculate
KV bytes from layers, KV heads, head dimension, cache types, and K/V precision.

Until metadata support is complete, use named, documented empirical estimates
that include a safety margin and record that estimate in the profile. Never
pretend an estimate is exact. Context selection should be a policy:

1. honor an explicit context option;
2. honor an explicit model preferred context;
3. calculate a safe context from available memory and KV estimate;
4. clamp to the model's declared maximum and a global safety cap.

The user-facing output must state both the requested and final values, plus the
reason for any clamp.

## Request Planning and Reasoning

Server launch settings and request-time settings are related but not identical.
Create `RequestPlan` from the same resolved profile:

```python
request = request_planner.chat(profile, prompt, sampling_override)
```

It owns `chat_template_kwargs`, request sampler fields, token budgets, and
reasoning switches. Family capability describes the template key and its values
(`enable_thinking: false`, `reasoning: false`, or `reasoning_effort: low`) so
callers never special-case a family.

Define the contract for “reasoning off” per family and test the exact JSON body.
Avoid sending a server-level setting and a conflicting request-level setting.

## Testing and Evaluation Architecture

Keep three test layers separate:

1. **Unit tests:** family selection, config merge/provenance, KV calculations,
   launch argv, request JSON, fixture runner, and result aggregation. No model
   server required.
2. **Contract tests:** a fake llama-server verifies health checks, PID cleanup,
   port conflicts, timeouts, and requests. These run in CI.
3. **Optional integration tests:** real local models, explicitly marked and
   skipped unless the user opts in. They validate a family/template combination
   against the installed llama.cpp version.

### Testing Lanes: Fast Experiments Without Lowering the Bar

Rama needs rapid model experimentation, so not every change needs every test
layer immediately. Use explicit lanes instead of quietly skipping checks:

| Lane | Intended use | Required evidence |
|---|---|---|
| Experimental | a new local model/card or tentative tuning | unit tests for resolver/registry changes plus a reviewed `rama start --dry-run` profile and argv |
| Stable | supported family, config key, or launch behavior | unit tests and fake-server contract tests |
| Release | a versioned supported workflow | stable evidence plus opt-in real-model integration when hardware/model is available |

The fake server stays intentionally small: it verifies request bodies,
readiness, failure handling, and cleanup. It must not try to emulate llama.cpp
generation, template rendering, or model behavior. This keeps contract tests
fast enough for normal development while preserving a reliable boundary check.

Benchmark fixtures should state their evaluation contract explicitly:

* exact output vs order-independent output;
* a named validator when multiple correct answers exist;
* timeout and sandbox limits;
* the test category and expected language.

### Reproducible Result Manifests

Each batch writes one atomic JSON manifest and one derived CSV. The JSON should
include:

* app and llama.cpp version/commit;
* model path, size, GGUF metadata hash, family key, and quant;
* resolved profile plus provenance;
* command invocation, seed, fixtures hash, and timing;
* per-test response, execution outcome, and evaluator version.

CSV is for sorting; JSON is the evidence. Do not delete per-seed detail merely
to make the results directory tidy.

## Migration Plan

Each phase is independently reviewable and leaves Rama usable.

### Phase 0 — Stabilize Current Behavior

* Restore config-driven `fit_target_mib` resolution.
* Pass `--config` through every CLI path, including NIHS batch mode.
* Standardize `repeat_penalty` in code, documentation, and generated config.
* Fix lint and add regression tests for these three bugs.
* Add a compatibility test for every public CLI command and Python entry point
  changed by the dataclass refactor.

**Exit criteria:** current commands work with both the checked-in sample config
and a temporary custom config; `ruff`, unit tests, and CLI help pass.

### Phase 1 — Establish Typed Resolution

* Add typed config models and a single `ProfileResolver` without changing the
  existing YAML format.
* Add `ResolvedModelProfile` and provenance.
* Move config precedence into resolver tests before moving launch behavior.
* Add `rama explain MODEL [options]` in dry-run mode.
* Generate a small golden matrix from the current working behavior, then review
  every case before accepting it as a baseline. Preserve intentional behavior;
  correct known bugs explicitly rather than fossilizing them.

**Exit criteria:** old and new paths generate identical launch arguments for a
reviewed golden matrix of representative existing models, except for documented
bug fixes.

### Phase 2 — Introduce the Family Registry

* Move matchers, capabilities, context defaults, and temporary KV estimates
  into `families.py`.
* Replace family conditionals in discovery, launch, and request generation with
  profile fields.
* Keep the legacy family branches behind assertions until the golden matrix is
  stable, then remove them.
* Keep the matrix semantic and small: representative family/profile/argv/request
  cases, not every downloaded model, quantization filename, or llama.cpp build.

**Exit criteria:** adding a declarative family requires one registry entry and
tests; `launch.py` contains no model-family names.

### Phase 3 — Split Planning from Execution

* Extract pure `LaunchPlanner` and side-effectful `ProcessRunner`.
* Add `rama start --dry-run` to print the profile and argv without starting.
* Make port cleanup scoped to Rama-owned PID records; never broadly kill every
  `llama-server` process.
* Write atomic PID and result files with structured error states.

**Exit criteria:** launch-plan tests cover normal, MTP, draft, MoE, mmap, and
  port-conflict cases without spawning llama.cpp.

### Phase 4 — Metadata-Backed Memory Decisions

* Implement GGUF metadata reading behind a small adapter.
* Replace file-size architecture heuristics where metadata is present.
* Report explicit fallback estimates where it is absent.
* Add known-model fixtures for KV/cache calculations.

**Exit criteria:** a profile explains the source and safety margin of every
context and cache decision.

### Phase 5 — Make Evaluation Reproducible

* Move batch orchestration onto a small runner interface using launch/request
  planners.
* Record resolved profiles and fixture hashes in manifests.
* Add resumable batches keyed by immutable run identity, not mutable filenames.
* Keep live-model integration testing opt-in.

**Exit criteria:** two runs can be compared without guessing which model
settings, template, or fixtures were used.

### Phase 6 — Launch-Display Contract and Repo Hygiene (2026-08-29)

* Close the "a CLI command can display one resolved value and launch another"
  failure: `get_preferred_ctx()` now passes family config overrides to the
  resolver; the `rama start` banner applies the MTP context ratio and reports
  the 256-aligned context the server actually allocates (llama.cpp pads with
  `GGML_PAD(n_ctx, 256)`).
* `mtp_ctx_ratio` resolves through the normal config cascade
  (model > family > global > default) and is applied exactly once, in the
  planner.
* Housekeeping tracked in CLEANUP.md: stray files, legacy `.pids/` directory,
  `pytest` dev dependency, and committing the previously untracked domain
  layer, families registry, and test suite so a fresh clone is buildable.

**Exit criteria:** the `rama start` banner, the `--dry-run` argv, and the
server's `n_ctx_slot` log line all report the same context value; a fresh
clone passes ruff and the full test suite.

## Non-Goals

* Do not build a plugin system before a registry and resolver prove insufficient.
* Do not create one subclass per model; family-level declarative definitions are
  the default.
* Do not make real GGUF/model downloads a CI requirement.
* Do not silently change sampling or memory policy during refactoring; any
  behavior change needs a profile/argv regression test and release-note entry.

## Definition of Done

The architecture work is complete when these statements are true:

* A maintainer can answer “why did Rama use this flag?” with `rama explain`.
* A new ordinary family needs one definition, one config entry if desired, and
  a small test matrix—no edits to process-management code.
* A custom config affects every relevant command consistently.
* The same resolved profile drives both server launch and API requests.
* Launch planning, family detection, and result aggregation are covered without
  real models.
* Documentation, templates, config validation, and runtime keys use the same
  vocabulary.
* A failed launch preserves enough structured context to reproduce it.
