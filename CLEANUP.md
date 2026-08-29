# Rama Cleanup Plan

Findings from the 2026-08-29 audit (after the MTP / qwen3.8 session).
Baseline: 119/119 tests pass, ruff clean.

## P0 — Fresh clone is broken (untracked load-bearing files)

The Phase 3-5 refactor exists only on this machine. A fresh clone imports
missing modules and has no tests.

- [x] Committed in 4 chunks (2026-08-29): `202b700` pihole removal, `c852032` domain-layer +
  families registry refactor (29 files), `c5c2fae` pytest suite + fixtures + batch configs
  (36 files), `14f9480` docs + pytest dev dep. Tree clean, 119/119 tests pass on HEAD.
- [x] (superseded by the chunking above) Commit `rama/domain/` (7 files) + `rama/families.py` (imported by launch.py, testing.py, core.py)
- [x] (superseded) Commit test suite: 13 `tests/test_*.py` + `tests/__init__.py` + `tests/fixtures/`
- [x] (superseded) Commit `config/` (only `batch_code_test.yaml`, `batch_code_test_reasoning.yaml` are committable; the other 3 are machine-tuning and gitignored)
- [x] (superseded) Commit the 43 modified files in logical chunks, not one blob:
  1. removal of `pihole-backup-2026-05-24.tar.gz`
  2. domain-layer + families registry refactor
  3. test suite + fixtures
  4. banner / mtp-ctx-ratio / qwen3.8 config fixes
  5. chore: pyproject, .gitignore, docs

## P1 — Launch banner correctness

- [x] Banner prints the 256-aligned effective ctx (matches server `n_ctx_slot`) — was: it printed the requested ctx, but llama.cpp pads it: `cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256)` (llama-context.cpp:288). Server logged `n_ctx_slot=144128` while rama printed 143974. Make the banner print the 256-aligned effective ctx so it always matches the server log.
- [x] Fixed "using the configured full context" wording (now "using the configured ctx instead") — was wrong when `mtp_ctx_ratio < 1.0`.

## P2 — Repo hygiene

- [x] Deleted `new_test_func.py` — was a root-level scratch copy of the `test()` command with a `-c` short-flag collision (config_path and categories both use `-c`)
- [x] Deleted stale `.pids/` — legacy PID location; `get_pid_dir()` (core.py:96) uses `~/.local/state/rama/pids`. Contained a stale PID for a model that no longer exists
- [x] `pyproject.toml`: added `pytest` to the dev dependency group (suite ran but the dev group only listed `ruff`, so `uv sync` on a new machine broke testing)
- [ ] Keep `pihole-backup-2026-05-24.tar.gz` deleted (binary that was committed to this repo)

## P3 — Docs and memory

- [x] `IMPROVEMENT-PLAN.md`: added Phase 6 (Launch-Display Contract + Repo Hygiene) with exit criteria
- [x] Refreshed Serena memories: `rama/architecture`, `rama/families` (now 22 families incl. qwen3.8), `rama/config-chain` (mtp_ctx_ratio chain), `rama/spec-decoding` (MTP path, current spec_type)
- [x] README: removed phantom `llama-bench` command (real one is `rama bench <port>`), documented `context`, `explain`, and MTP startup

## Notes

- Fixed this session: `mtp_ctx_ratio` double-apply (launch.py multiplied, then planner multiplied again → 0.8² = 0.64). Planner is now the single application point. Side effect: other MTP families now get 0.8 of configured ctx, not 0.64.
- Fixed this session: `get_preferred_ctx()` (core.py) never passed family overrides to ProfileResolver, so banner and `calc_kv_cache` used registry defaults (131072) instead of rama.yaml values (151552).
- llama.cpp aligns requested ctx up to multiples of 256; 151552 and 131072 are already aligned, 143974 → 144128.
