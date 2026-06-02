# Rama Codebase Improvement Plan

## Priority 1: Batch runner seed restart elimination

**Problem:** `batch_runner._run_single_code_model` stops and restarts the model for each seed run. With 6 seeds and a 20s model load, that's 120s wasted per model.

**Fix:** Use context reset between seeds instead of full restarts, matching what `multi_seed.py` already does.

**Files:** `rama/testing_modules/batch_runner.py`

**Changes:**
- In `_run_single_code_model`, start model ONCE before the seed loop
- Between seeds, call `reset_context(port, ...)` instead of `stop_model` + `start_model`
- Stop model ONCE after all seeds complete
- Remove the per-seed port incrementing (`use_port += 1`)
- Estimated time savings: ~80% for multi-seed code batches

---

## Priority 2: Config validation

**Problem:** `load_config()` accepts any values. Invalid config silently propagates to llama.cpp, causing crashes or undefined behavior.

**Fix:** Add `validate_config(cfg: dict) -> list[str]` that returns validation errors.

**Files:** `rama/__init__.py` (new function), `rama/__main__.py` (call on init)

**Validation rules:**
- `ctx_size`, `batch_size`, `ubatch_size`, `threads` — must be positive integers
- `temp` — range 0.0 to 2.0
- `top_p` — range 0.0 to 1.0
- `top_k` — positive integer or -1
- `min_p` — range 0.0 to 1.0
- `n_gpu_layers` — positive integer
- `ctk`, `ctv` — must be in `{f16, q8_0, q4_0, turbo3, turbo4, q4_K_M, q5_0, q5_K_M, q6_K}`
- `backend` — must be in `{llama.cpp, turboquant}`
- `cuda_arch` — must be in `{50, 52, 60, 61, 70, 75, 80, 86, 87, 89, 90}`
- `default_port` — range 1024 to 65535
- Warn (non-fatal): `large_ctx` not power of 2, `temp == 0.0` with `top_p < 1.0`

**Call from:** `rama init` command and `Config.__init__`

---

## Priority 3: Config dataclasses for wide function signatures

**Problem:** `run_tests` has 16 params, `start_model` has 17. Callers pass `None` for most.

**Fix:** Introduce config dataclasses, keep backward-compatible signatures with `**kwargs`.

**Files:** `rama/testing.py`, `rama/launch.py`

```python
@dataclass
class TestRunConfig:
    categories: list[str] = field(default_factory=lambda: ["code", "debugging"])
    system_prompt: dict[str, str] | str | None = None
    warmup_rounds: int = 3
    context_text: str | None = None
    context_files: list[str] | None = None
    context_placement: str = "before"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    conversation: list[dict] | None = None
    verbose: bool | None = None
    max_tokens: int | None = None
    extra_body: dict | None = None

@dataclass
class ModelStartConfig:
    backend: str = "llama.cpp"
    port: int | None = None
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    ctk: str | None = None
    ctv: str | None = None
    n_ctx: int | None = None
    seed: int | None = None
    n_batch: int | None = None
    prompt_cache: str | None = None
    prompt_cache_all: bool = False
    log_disable: bool = False
    extra_args: list[str] | None = None
```

Keep old signatures working: `def run_tests(port, categories=None, config=None, **kwargs)` → build `TestRunConfig(**kwargs)`.

---

## Priority 4: Fix KV cache calculation

**Problem:** Quantization multipliers in `calc_kv_cache` are wrong — they don't account for both K and V caches.

**Fix:** Either implement the correct formula or document the current values as empirical estimates.

**Files:** `rama/core.py` lines 129-140

**Correct formula:**
```
kv_bytes_per_token = 2 * n_layers * n_head_kv * head_dim * quant_multiplier
```

The `2` accounts for K + V. Current code is missing this factor, so all estimates are ~50% of actual.

**Options:**
A. Fix the multipliers: `q4_0: 0.50`, `q8_0: 1.0`, `f16: 2.0`
B. Add a docstring explaining these are per-side estimates and the function returns half the actual cache
C. Full formula using model metadata (requires parsing GGUF header)

Recommend **A** — simplest fix, matches what the Granite override already discovered.

---

## Priority 5: Remove dead code and deduplicate

**Problem:** ~200 lines of dead/duplicated code across 4 files.

**Files and actions:**

| Location | What | Action |
|----------|------|--------|
| `launch.py:143-166` | `start_server()` | **Delete** — never called |
| `core.py:247-253` | `detect_backend()` | **Move** to `batch_runner.py` (only caller) |
| `batch_runner.py:746-1134` | `_run_single_code_model` | **Refactor** — extract seed loop, reuse `multi_seed.py` pattern |
| `batch_runner.py:702-718` | `_kill_leftover_servers` | **Replace** with `core._sweep_stale_pids()` |
| `system_prompt.py` templates | Duplicated in batch_runner | **Import** from `system_prompt` module |

---

## Priority 6: Specific exception handling

**Problem:** ~30 bare `except Exception` blocks catch everything including `KeyboardInterrupt` and `SystemExit`.

**Fix:** Replace with specific exceptions where the failure mode is known.

**Pattern:**
```python
# Before
except Exception as e:
    logger.debug(f"failed to stop model on port {port}: {e}")

# After
except (ConnectionError, OSError, ProcessLookupError) as e:
    logger.debug(f"failed to stop model on port {port}: {e}")
```

**Files affected:** `launch.py` (8), `batch_runner.py` (6), `kv_quant.py` (3), `context_difficulty.py` (3), `testing.py` (4), `core.py` (2), `multi_seed.py` (2)

---

## Priority 7: Incremental JSON writes → batched

**Problem:** `batch_runner` writes the full JSON results file after every single result. For 50 models × 6 seeds = 300 file writes of an ever-growing file.

**Fix:** Write after each model completes (not each seed). Or write to a temp file and rename at end.

**Files:** `rama/testing_modules/batch_runner.py`

**Changes:**
- Remove `nihs_results_file.write_text(...)` from inside the ctx_size loop (line ~496)
- Remove `self.results_file.write_text(...)` from inside the seed loop (line ~1042)
- Keep the per-model write, remove the per-seed write

---

## Priority 8: Add basic unit tests

**Problem:** Zero self-tests. The testing framework has no tests for its own logic.

**Fix:** Add pytest tests for pure functions:

```
tests/
  test_calc_kv_cache.py      # calc_kv_cache with known models
  test_find_free_port.py     # port allocation logic
  test_config_validation.py  # validate_config catches bad values
  test_needle_insertion.py   # insert_needle at all locations
  test_code_runner.py        # run_code_task with known inputs
  test_resolve.py            # resolve() config fallback logic
```

**Files:** New `tests/` directory

---

## Execution Order

1. **P3** (dataclasses) — reduces complexity, makes other changes easier
2. **P1** (seed restarts) — biggest performance win
3. **P2** (validation) — prevents future bugs
4. **P5** (dead code) — low risk, clean up before bigger changes
5. **P4** (KV cache) — correctness fix
6. **P6** (exceptions) — reliability
7. **P7** (file writes) — performance
8. **P8** (unit tests) — safety net for future changes

Each item is independently mergeable. No item requires another to be done first (except P3 makes P1 cleaner).
