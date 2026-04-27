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
cuda_root: /opt/cuda-13.1

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

Override settings per model family:

```yaml
families:
  qwen:
    max_ctx: 262144        # Hard cap on context
    preferred_ctx: 131072  # Default when auto-calculating
  gemma:
    max_ctx: 131072
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

# Custom quantization types
uv run rama start qwen27b --ctk q8_0 --ctv q4_0

# TurboQuant KV cache
uv run rama start qwen27b --turbo3
uv run rama start qwen27b --turbo4

# Specific port
uv run rama start qwen27b --port 11436
```

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

# All categories (code, debugging, creative)
uv run rama test 11435 --categories all

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

### Needle-in-Haystack (Context Tests)

Test how well a model retrieves information from long contexts:

```bash
# Basic test at default positions (beginning, middle, end)
uv run rama context 11435 large_file.txt

# Test at specific positions
uv run rama context 11435 large_file.txt -n beginning,25%,middle,75%,end

# Custom seed for reproducibility
uv run rama context 11435 large_file.txt --seed 42

# Multiple files
uv run rama context 11435 file1.txt,file2.txt -n beginning,middle,end
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
uv run rama batch config/batch_qwen27b.yaml
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

### Example Batch Files

Pre-configured batch files in `config/`:

- `batch_qwen27b.yaml` - Qwen 27B comparisons
- `batch_gemma31b.yaml` - Gemma 31B comparisons
- `batch_template.yaml` - Template for custom batches

### Batch Options

```bash
# Custom starting port (models run on sequential ports)
uv run rama batch config/batch.yaml --start-port 11440

# Custom output directory
uv run rama batch config/batch.yaml --output-dir my_results

# Fixed seed across all runs
uv run rama batch config/batch.yaml --seed 42

# Specific categories
uv run rama batch config/batch.yaml --categories code
```

## Build

Build backends from source:

```bash
uv run rama build llama.cpp
uv run rama build turboquant
uv run rama build all --force
uv run rama build llama.cpp --cpu-only
```

## Useful Commands

```bash
# List available tests
uv run rama tests

# Check if a port is in use
uv run rama status 11435

# Run perplexity benchmark
uv run rama perplexity qwen27b path/to/text.txt
```
