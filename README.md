# rama

CLI for running and comparing local GGUF models through `llama.cpp` and `turboquant`.

Backend repos:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
- [`llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)

The goal is practical comparison on your own hardware. We care about which backend feels better to use, how well it handles prompts, and how it behaves at long context lengths. This is intentionally not a perplexity/KL-divergence project.

## Install

```bash
cd test-llama.cpp
uv sync
```

Run from anywhere within the repo:

```bash
uv run rama <command>
```

## Configure for Your Device

Create a starter config (clones backend repos if missing):

```bash
uv run rama init
```

The config file is created at `rama.yaml` in the repo root. You can also put it at `~/.config/rama.yaml` or pass `--config path` to any command.

### Key Settings

```yaml
# Where your GGUF models live
model_dir: ~/ollama

# Backend repos (auto-cloned by `init` if missing)
llama_cpp_dir: ~/Projects/llama.cpp
turbo_dir: ~/Projects/llama-cpp-turboquant

# CUDA toolkit path
cuda_root: ~/cuda

# Your GPU VRAM (auto-detected via nvidia-smi if not set)
vram_total_gb: 24

# How much VRAM to reserve for OS/display (headroom)
vram_headroom_gb: 2

# Default context size when auto-calculating (128k)
default_ctx: 131072

# CPU threads for inference
threads: 8
threads_batch: 16
```

### Per-Device Tips

**RTX 4090 (24GB):**
```yaml
vram_total_gb: 24
vram_headroom_gb: 1
threads: 16
threads_batch: 32
```

**RTX 3090/4080 (16-20GB):**
```yaml
vram_total_gb: 16
vram_headroom_gb: 1
threads: 12
threads_batch: 24
```

**Multiple GPUs:** VRAM totals across all GPUs. Set `vram_total_gb` to the sum.

### Model Families

Each family gets its own `small_ctx`/`large_ctx` context tiers and sampling defaults:

```yaml
families:
  qwen:
    small_ctx: 24576
    large_ctx: 112640
    temp: 0.6
    top_k: 20
    top_p: 0.95
    min_p: 0.0
    repetition_penalty: 1.0

  gemma:
    small_ctx: 24576
    large_ctx: 65536
    temp: 1.0
    top_k: 64
    top_p: 0.95
    min_p: 0.05

  bonsai:
    small_ctx: 24576
    large_ctx: 65536
    temp: 0.7
    top_k: 40
    top_p: 0.95
    min_p: 0.05

  devstral:
    small_ctx: 24576
    large_ctx: 81920
    temp: 0.5
    top_k: 40
    top_p: 0.9
    min_p: 0.05

  mistral-small:
    small_ctx: 24576
    large_ctx: 81920
    temp: 0.15
    top_k: 40
    top_p: 0.9
    min_p: 0.05

  glm:
    small_ctx: 24576
    large_ctx: 102400
    temp: 1.0
    top_k: 64
    top_p: 0.95
    min_p: 0.01
    repetition_penalty: 1.0

  nemotron:
    small_ctx: 24576
    large_ctx: 65536
    temp: 0.6
    top_k: 40
    top_p: 0.95
    min_p: 0.05

  phi4:
    small_ctx: 24576
    large_ctx: 65536
    temp: 0.8
    top_k: 50
    top_p: 0.95
    min_p: 0.0

  granite:
    small_ctx: 32768
    large_ctx: 65536
    temp: 1.0
    top_k: 64
    top_p: 0.95
    min_p: 0.05
```

Models below 10GB get `small_ctx`, otherwise `large_ctx` (uniform threshold across families).

### Per-Model Overrides

Override any setting for a specific model (merges on top of family config):

```yaml
models:
  MySpecialModel:
    temp: 0.8
    preferred_ctx: 65536
```

## VRAM Management

All models use `--fit on` by default, which automatically distributes layers between VRAM and RAM to fit your GPU. The target VRAM margin is controlled by `fit_target_mib` (default: 1024 MiB headroom).

```yaml
# Reserve 1GB VRAM for safety margin
fit_target_mib: 1024
```

To disable auto-fit: `--no-fit`.

## Running Models

### Start a Model

```bash
# Basic start (auto-detects context, fits to VRAM)
uv run rama start qwen27b

# Run in foreground (Ctrl+C to stop)
uv run rama start qwen27b -f

# Custom context size
uv run rama start qwen27b --ctx 65536

# Custom KV cache types
uv run rama start qwen27b --ctk q8_0 --ctv q4_0

# TurboQuant KV cache
uv run rama start qwen27b --turbo3
uv run rama start qwen27b --turbo4

# Specific port
uv run rama start qwen27b --port 11436
```

### Reasoning Models (Qwen, Bonsai)

Qwen models auto-detect reasoning based on size (≥10GB = on, <10GB = off). Override with `--reasoning`/`--no-reasoning`:

```bash
# Force reasoning on/off
uv run rama start qwen27b --reasoning
uv run rama start qwen-small --no-reasoning

# Custom reasoning budget (default: 1024 tokens)
uv run rama start qwen27b --reasoning-budget 2048
```

### Speculative Decoding

Two modes available, mutually exclusive:

**N-gram speculative decoding** (set in rama.yaml):
```yaml
spec_type: ngram-mod
spec_ngram_mod_size_n: 24
spec_ngram_mod_n_min: 48
spec_ngram_mod_n_max: 64
```

**Draft model speculative decoding** (CLI):
```bash
uv run rama start qwen27b --draft-model qwen08b --draft-max 64 --draft-min 48
```

Disable speculative decoding: set `spec_type: null` in rama.yaml and omit `--draft-model`.

### Manage Running Models

```bash
# List discovered models
uv run rama list

# Show running servers
uv run rama running

# Show estimated max context per model
uv run rama ctxinfo

# Stop a server by port
uv run rama stop 11435
```

## Testing

### Task Tests

Run the test suite against a running model:

```bash
# Start a model first
uv run rama start qwen27b -f

# In another terminal, run tests
uv run rama test 11435

# Specific categories
uv run rama test 11435 --categories code,debugging

# JSON output
uv run rama test 11435 --format json

# Markdown report
uv run rama test 11435 --format markdown --output report.md
```

Tests live in `tests/` and cover:
- **code/** - coding tasks (LRU cache, graph cycles, tree serialization, etc.)
- **debugging/** - bug-fixing tasks (logic bugs + syntax errors)

### End-to-End Run

Start a model, run tests, then stop automatically:

```bash
uv run rama run qwen27b
uv run rama run qwen27b --turbo3
uv run rama run qwen27b --categories code
```

### Needle-in-Haystack (Unified `nihs` command)

Test how well a model retrieves information from long contexts. The `rama nihs` command supports four modes:

```bash
# Context mode (running model on port)
uv run rama nihs 11435 large_file.txt
uv run rama nihs 11435 large_file.txt -n beginning,25%,middle,75%,end
uv run rama nihs 11435 large_file.txt --seed 42

# Batch mode (multiple models, KV quants, context sizes)
uv run rama nihs config/batch_large.yaml wikitext-2/wiki.train.tokens
uv run rama nihs config/batch_large.yaml text.txt -s 8192,32768 -q f16,q8_0

# Enhanced mode (multi-needle + distractors, auto-starts model)
uv run rama nihs --enhanced -m qwen27b --ctx 65536
uv run rama nihs --enhanced -m qwen27b -n 10 -d lexical,topical,irrelevant

# Difficulty mode (context injection, auto-starts model)
uv run rama nihs --difficulty all -m qwen27b
uv run rama nihs --difficulty medium -m qwen27b
```

The test:
1. Generates a unique token and hides it at specified positions in the text
2. Asks the model to find the hidden token
3. Reports whether the model found it at each position
4. Checks for repetition/looping and response quality degradation

### Throughput Benchmark

Measure tokens-per-second:

```bash
uv run rama bench 11435
uv run rama bench 11435 -p 1024 -n 256
```

## Batch Runs

Compare multiple model/backend combinations automatically. Batch config files live in `config/`.

```bash
# Large models (>10B)
uv run rama batch-test config/batch_large.yaml

# Small models (<=10B)
uv run rama batch-test config/batch_small.yaml

# Perplexity testing
uv run rama batch-test config/batch_ppl.yaml --type perplexity --text wikitext-2/wiki.valid.tokens
```

The batch runner:
1. Starts each model sequentially
2. Runs the test suite
3. Saves JSON + CSV results to `results/`
4. Stops the model before moving to the next

### Batch Config Format

```yaml
# Simple model name
- model: qwen27b

# With custom options
- model: qwen27b
  ctx: 65536
  ctk: q8_0
  ctv: q8_0

# TurboQuant
- model: qwen27b
  turbo3: true

- model: qwen27b
  turbo4: true

# Custom context
- model: qwen27b
  ctx: 131072
```

### Batch Options

```bash
# Custom starting port (models run on sequential ports)
uv run rama batch-test config/batch_large.yaml --start-port 11440

# Custom output directory
uv run rama batch-test config/batch_large.yaml --output-dir my_results

# Fixed seed across all runs
uv run rama batch-test config/batch_large.yaml --seed 42

# Specific categories
uv run rama batch-test config/batch_large.yaml --categories code
```

### Needle-in-a-Haystack Testing

Single unified command for all NIHS testing needs:

```bash
# Context mode (running model on port)
rama nihs 11435 book.txt -n beginning,middle,end
rama nihs 11435 book.txt -n beginning,25%,middle,75%,end

# Batch mode (multiple models, KV quants, context sizes)
rama nihs config/batch_large.yaml wikitext-2/wiki.train.tokens
rama nihs config/batch_large.yaml text.txt -s 8192,32768,65536 -q f16,q8_0

# Enhanced mode (multi-needle + distractors, auto-starts model)
rama nihs --enhanced -m qwen27b -n 10 -d lexical,topical,irrelevant

# Difficulty mode (context difficulty injection, auto-starts model)
rama nihs --difficulty all -m qwen27b
rama nihs --difficulty medium -m qwen27b
```

## Build

Build backends from source (auto-builds on `start` if binary is missing):

```bash
uv run rama build llama.cpp
uv run rama build turboquant
uv run rama build all --force
uv run rama build llama.cpp --cpu-only
```

Build targets CUDA arch 8.9 (RTX 4090) with flash attention for all quant types.

## Useful Commands

```bash
# List available tests
uv run rama tests

# Check if a port is in use
uv run rama status 11435

# Run perplexity benchmark
uv run rama perplexity qwen27b path/to/text.txt

# Run llama-bench (native benchmark, no server needed)
uv run rama llama-bench qwen27b
uv run rama llama-bench qwen27b -p 1024 -n 256 -r 3
uv run rama llama-bench qwen27b -o json
```
