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

## Configure

`rama` looks for config in `rama.yaml` in the repo root first, then in `~/.config/rama.yaml`. Pass a specific file with `--config` on any command.

Create a starter config:

```bash
uv run rama init
```

Key config paths:

- `model_dir`: where your GGUF models live
- `llama_cpp_dir`: path to your local `llama.cpp` checkout
- `turbo_dir`: path to your local `llama-cpp-turboquant` checkout
- `cuda_root`: CUDA install root used for builds

Example:

```yaml
model_dir: ~/ollama
llama_cpp_dir: ~/Projects/llama.cpp
turbo_dir: ~/Projects/llama-cpp-turboquant
cuda_root: /opt/cuda-13.1
```

## VRAM Management

All models use `--fit on` by default, which automatically distributes layers between VRAM and RAM to fit your GPU. The target VRAM margin is controlled by `fit_target_mib` (default: 1024 MiB headroom).

To disable auto-fit: `--no-fit`.

## Model Management

```bash
# List discovered models
uv run rama list

# Show running models
uv run rama running

# Show estimated max context for each model
uv run rama ctxinfo

# Start a model (auto-fits to VRAM)
uv run rama start qwen27b

# Start with TurboQuant KV cache
uv run rama start qwen27b --turbo3
uv run rama start qwen27b --turbo4

# Start with custom context
uv run rama start qwen27b --ctx 65536

# Run in foreground (Ctrl+C to stop)
uv run rama start qwen27b -f

# Stop a running server
uv run rama stop 11435
```

## Testing

### Task Tests

Run the test suite against a running model:

```bash
uv run rama test 11435
```

Tests cover:
- **code/** - coding tasks (LRU cache, graph cycles, tree serialization, etc.)
- **debugging/** - bug-fixing tasks (logic bugs + syntax errors)

### End-to-End Run

Start a model, run tests, then stop automatically:

```bash
uv run rama run qwen27b
uv run rama run qwen27b --turbo3
```

### Needle-in-Haystack Tests

Hide a unique token in a text file and see if the model can find it:

```bash
# Test at beginning, middle, and end (default)
uv run rama context 11435 large_file.txt

# Test at specific positions
uv run rama context 11435 large_file.txt -n beginning,25%,middle,75%,end
```

Checks for:
- Whether the model finds the hidden token at each location
- Repetition/looping behavior
- Response quality degradation

### Throughput Bench

Measure tokens-per-second:

```bash
uv run rama bench 11435
uv run rama bench 11435 -p 1024 -n 256
```

## Batch Runs

Compare multiple model/backend combinations automatically:

```bash
uv run rama batch batch_qwen27b.yaml
```

The batch runner:
1. Starts each model sequentially
2. Runs the test suite
3. Saves JSON + CSV results to `results/`
4. Stops the model before moving to the next

Example batch config:

```yaml
- model: qwen27b
  ctx: 65536
  ctk: q8_0
  ctv: q8_0

- model: qwen27b
  ctx: 65536
  turbo3: true

- model: qwen27b
  ctx: 65536
  turbo4: true
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
